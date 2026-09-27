import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import copy


def set_seed(seed=42):
    """Sets random seed for reproducibility across numpy, torch, and random."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# 1. Configuration Management

class Args:
    def __init__(self, **kwargs):
        # Default hyperparameters
        self.batch_size = 128
        self.lr = 0.001
        self.hidden_units = 128
        self.num_blocks = 2
        self.num_heads = 2
        self.dropout = 0.2
        self.emb_dropout = 0.2
        self.epochs = 50
        self.maxlen = 100
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.patience = 5
        self.num_negatives = 1
        # Override defaults
        for k, v in kwargs.items():
            setattr(self, k, v)

# 2. Data Preprocessing

def load_data():
    df = pd.read_csv(
        "ratings.dat",
        sep="::",
        engine="python",
        names=["uid", "iid", "rating", "ts"]
    )
    # Convert explicit ratings to binary implicit feedback
    df = df[df["rating"] >= 4]
    # Sort interactions chronologically
    df = df.sort_values(["uid", "ts"])
    # Filter users with fewer than 5 interactions
    cnt = df.groupby("uid").size()
    df = df[df["uid"].isin(cnt[cnt >= 5].index)]

    u_map = {u: i + 1 for i, u in enumerate(df["uid"].unique())}
    i_map = {i: j + 1 for j, i in enumerate(df["iid"].unique())}
    df["uid"] = df["uid"].map(u_map)
    df["iid"] = df["iid"].map(i_map)

    # Calculate item popularity for negative sampling
    item_freq = df["iid"].value_counts().to_dict()
    item_list = list(range(1, len(i_map) + 1))
    weights = np.array([item_freq.get(i, 1) for i in item_list], dtype=np.float32)
    weights = weights / weights.sum()

    user_seq = df.groupby("uid")["iid"].apply(list).to_dict()
    num_items = len(i_map)

    # Leave-one-out split
    train, val, test = {}, {}, {}
    for u, seq in user_seq.items():
        train[u] = seq[:-2]
        val[u] = seq[-2]
        test[u] = seq[-1]

    return train, val, test, num_items, item_list, weights


def sample_negative(item_list, seq_set):
    """Uniform negative sampling as per the original SASRec paper."""
    while True:
        neg = random.choice(item_list)
        if neg not in seq_set:
            return neg


class SASRecDataset(Dataset):
    def __init__(self, train, num_items, item_list, maxlen, num_negatives=1):
        self.users = list(train.keys())
        self.train = train
        self.item_list = item_list
        self.maxlen = maxlen
        self.num_negatives = num_negatives
        self.all_items = torch.tensor(item_list)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        u = self.users[idx]
        seq = self.train[u]
        seq_set = set(seq)

        actual_seq = seq[-self.maxlen:]
        length = len(actual_seq)

        inp = np.zeros(self.maxlen, dtype=np.int32)
        pos = np.zeros(self.maxlen, dtype=np.int32)
        neg = np.zeros((self.maxlen, self.num_negatives), dtype=np.int32)

        for i in range(length - 1):
            idx_in_arr = self.maxlen - length + i
            inp[idx_in_arr] = actual_seq[i]
            pos[idx_in_arr] = actual_seq[i + 1]
            for n in range(self.num_negatives):
                neg_candidate = random.choice(self.item_list)
                while neg_candidate in seq_set or neg_candidate == actual_seq[i + 1]:
                    neg_candidate = random.choice(self.item_list)
                neg[idx_in_arr, n] = neg_candidate

        return torch.tensor(inp).long(), torch.tensor(pos).long(), torch.tensor(neg).long()

# 3. SASRec Model Architecture

class SASRecBlock(nn.Module):
    def __init__(self, hidden, heads, dropout):
        super().__init__()
        self.h, self.d = heads, hidden // heads
        self.scale = self.d ** -0.5
        # Self-attention layers
        self.q = nn.Linear(hidden, hidden)
        self.k = nn.Linear(hidden, hidden)
        self.v = nn.Linear(hidden, hidden)
        self.o = nn.Linear(hidden, hidden)
        # Normalization and FFN
        self.ln1 = nn.LayerNorm(hidden)
        self.ln2 = nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(
            nn.Linear(hidden, hidden * 4),
            nn.GELU(),
            nn.Linear(hidden * 4, hidden)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        # x shape: [Batch, Seq, Hidden]
        B, L, D = x.shape
        q = self.q(x).view(B, L, self.h, self.d).transpose(1, 2)
        k = self.k(x).view(B, L, self.h, self.d).transpose(1, 2)
        v = self.v(x).view(B, L, self.h, self.d).transpose(1, 2)

        # Multi-head attention with causal mask
        score = (q @ k.transpose(-2, -1)) * self.scale
        score = score.masked_fill(mask, -1e9)

        attn = torch.softmax(score, dim=-1)
        out = (self.dropout(attn) @ v).transpose(1, 2).contiguous().view(B, L, D)

        x = self.ln1(x + self.dropout(self.o(out)))
        x = self.ln2(x + self.dropout(self.ffn(x)))
        return x

class SASRec(nn.Module):
    def __init__(self, num_items, args):
        super().__init__()
        # Item and Positional Embeddings
        self.item_emb = nn.Embedding(num_items + 1, args.hidden_units, padding_idx=0)
        self.pos_emb = nn.Embedding(args.maxlen, args.hidden_units)
        self.emb_dropout = nn.Dropout(args.emb_dropout)
        self.blocks = nn.ModuleList([
            SASRecBlock(args.hidden_units, args.num_heads, args.dropout)
            for _ in range(args.num_blocks)
        ])
        self.ln = nn.LayerNorm(args.hidden_units)

    def forward(self, seq):
        B, L = seq.shape
        # Positional embedding logic [cite: 47]
        pos = torch.arange(L, device=seq.device).unsqueeze(0)
        x = self.item_emb(seq) + self.pos_emb(pos)
        x = self.emb_dropout(x)

        # Standard Causal Mask: each position only attends to previous items
        mask = torch.triu(torch.ones((L, L), device=seq.device), diagonal=1).bool()
        mask = mask.unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, L, L] for multi-head compatibility

        for blk in self.blocks:
            x = blk(x, mask)
        return self.ln(x)

    def predict(self, seq, items):
        """
        Final prediction layer scoring candidate items based on sequence representation[cite: 51].
        Returns: Tensor of shape [Batch, Num_Items]
        """
        # Extract representation of the last item in the sequence
        seq_features = self.forward(seq)[:, -1, :]  # Shape: [Batch, Hidden]
        item_embs = self.item_emb(items)  # Shape: [Num_Items, Hidden]

        # Calculate scores via dot product across the item catalog
        scores = torch.matmul(seq_features, item_embs.T)
        return scores

# 4. Evaluation

def evaluate(model, train, val, test, num_items, args, mode='test'):
    model.eval()
    recalls, ndcgs = {10: [], 20: []}, {10: [], 20: []}
    # Pre-calculate all item indices for full-rank scoring
    all_items = torch.arange(1, num_items + 1).to(args.device)

    # Note: For strict assignment requirements, we still iterate per user
    for u in train.keys():
        if mode == 'valid':
            history, target = train[u], val[u]
        else:
            history, target = train[u] + [val[u]], test[u]

        # Prepare input sequence with right-alignment
        seq = np.zeros(args.maxlen, dtype=np.int32)
        history_clipped = history[-args.maxlen:]
        seq[args.maxlen - len(history_clipped):] = history_clipped
        seq_t = torch.from_numpy(seq).long().unsqueeze(0).to(args.device)

        with torch.no_grad():
            # 1. Get prediction scores for all items
            # Returns [1, Num_Items] due to no-squeeze predict()
            scores = model.predict(seq_t, all_items)
            user_scores = scores[0]  # Get scores for this specific user

            # 2. Mask items already seen in history to ensure fair evaluation
            seen_items = set(history)
            for item_id in seen_items:
                user_scores[item_id - 1] = -1e9

            # 3. Rank items and calculate metrics
            _, top_idx = torch.topk(user_scores, 20)
            top_idx = (top_idx + 1).cpu().numpy()  # Convert to 1-based item IDs

        # 4. Metric Calculation for Recall@K and NDCG@K
        for k in [10, 20]:
            if target in top_idx[:k]:
                recalls[k].append(1)
                rank = np.where(top_idx[:k] == target)[0][0] + 1
                ndcgs[k].append(1 / np.log2(rank + 1))
            else:
                recalls[k].append(0)
                ndcgs[k].append(0)

    return {k: np.mean(recalls[k]) for k in recalls}, {k: np.mean(ndcgs[k]) for k in ndcgs}

# 5. Training Execution

def run_single_experiment(args, data_pack):
    train, val, test, num_items, item_list, weights = data_pack
    dataset = SASRecDataset(train, num_items, item_list, args.maxlen, num_negatives=args.num_negatives)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = SASRec(num_items, args).to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)  # Adam optimizer [cite: 59]
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=1)

    best_ndcg, best_state, wait = -1.0, None, 0

    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm(loader, desc=f"Epoch {epoch + 1}")
        for seq, pos, neg in pbar:
            seq, pos, neg = seq.to(args.device), pos.to(args.device), neg.to(args.device)

            logits = model(seq)
            pos_emb = model.item_emb(pos)
            neg_emb = model.item_emb(neg)

            pos_scores = (logits * pos_emb).sum(dim=-1)
            neg_scores = (logits.unsqueeze(2) * neg_emb).sum(dim=-1)

            istarget = (pos != 0).float()

            # BCE loss for negative sampling
            pos_loss = F.binary_cross_entropy_with_logits(
                pos_scores, torch.ones_like(pos_scores), reduction='none'
            )

            neg_loss = F.binary_cross_entropy_with_logits(
                neg_scores, torch.zeros_like(neg_scores), reduction='none'
            )

            loss = (pos_loss.unsqueeze(-1) + neg_loss)
            loss = loss * istarget.unsqueeze(-1)

            final_loss = loss.sum() / istarget.sum()

            opt.zero_grad()
            final_loss.backward()
            opt.step()

            pbar.set_postfix(loss=final_loss.item())

        # Early stopping based on validation NDCG@10
        _, val_n = evaluate(model, train, val, test, num_items, args, 'valid')
        scheduler.step(val_n[10])

        if val_n[10] > best_ndcg:
            best_ndcg = val_n[10]
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        elif wait >= args.patience:
            break
        else:
            wait += 1

    if best_state is None:
        print("Warning: best_state is None, using current model weights")
        best_state = model.state_dict()

    model.load_state_dict(best_state)

    test_recalls, test_ndcgs = evaluate(model, train, val, test, num_items, args, 'test')
    return test_recalls, test_ndcgs

# 6. Automated Multi-Parameter Runner

if __name__ == "__main__":
    data = load_data()

    # Experimental configurations (Ablation Study)

    BASELINE = {
        "hidden_units": 128,
        "num_heads": 2,
        "maxlen": 100,
        "num_negatives": 1
    }

    experiments = []

    # 0. Baseline
    experiments.append({
        **BASELINE,
        "label": "BASELINE"
    })
    # 1. hidden_units ablation
    for h in [64, 128, 256]:
        cfg = BASELINE.copy()
        cfg["hidden_units"] = h
        cfg["label"] = f"hidden={h}"
        experiments.append(cfg)

    # 2. num_heads ablation
    for h in [1, 2, 4]:
        cfg = BASELINE.copy()
        cfg["num_heads"] = h
        cfg["label"] = f"heads={h}"
        experiments.append(cfg)

    # 3. maxlen ablation
    for m in [50, 100, 200]:
        cfg = BASELINE.copy()
        cfg["maxlen"] = m
        cfg["label"] = f"maxlen={m}"
        experiments.append(cfg)

    # 4. num_negatives ablation
    for n in [1, 3, 5]:
        cfg = BASELINE.copy()
        cfg["num_negatives"] = n
        cfg["label"] = f"neg={n}"
        experiments.append(cfg)

    # 5. number of self-attention blocks ablation
    for b in [1, 2, 3]:
        cfg = BASELINE.copy()
        cfg["num_blocks"] = b
        cfg["label"] = f"blocks={b}"
        experiments.append(cfg)

    all_results = []
    for cfg in experiments:
        label = cfg.pop("label")
        set_seed(42)
        print(f"\n>>> Running: {label}")
        current_args = Args(**cfg)
        test_r, test_n = run_single_experiment(current_args, data)

        res = {
            "Experiment": label,
            "Recall@10": test_r[10], "NDCG@10": test_n[10],
            "Recall@20": test_r[20], "NDCG@20": test_n[20]
        }
        all_results.append(res)
        print(f"Result for {label}: NDCG@10 = {test_n[10]:.4f}")

    # Generate summary table for LaTeX report
    print("\n" + "=" * 60 + "\nFINAL EXPERIMENT COMPARISON\n" + "=" * 60)
    summary_df = pd.DataFrame(all_results)
    print(summary_df)