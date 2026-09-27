
import random
import numpy as np
import pandas as pd
import scipy.linalg
import torch
import torch.nn as nn
from scipy.sparse import csr_matrix
from torch.utils.data import Dataset, DataLoader

# ── reproducibility ────────────────────────────────────────────────────
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# ── hyperparameters ────────────────────────────────────────────────────
MAXLEN      = 20
D_MODEL     = 128
N_HEADS     = 2
N_LAYERS    = 2
DROPOUT     = 0.2
L2          = 1e-4
LR          = 0.005
BATCH_SIZE  = 128
NUM_EPOCHS  = 500
TOP_K       = 10
EVAL_EVERY  = 2
PATIENCE    = 10
EASE_LAM    = 500.0
DEVICE      = ("mps" if torch.backends.mps.is_available()
               else "cuda" if torch.cuda.is_available()
               else "cpu")
print(f"device: {DEVICE}")

# ── load & map ids ─────────────────────────────────────────────────────
print("Loading data ...")
train_df   = pd.read_csv("train.csv")
sample_sub = pd.read_csv("sample_submission.csv")

all_items  = sorted(train_df["item_id"].unique())
iid2idx    = {it: i + 1 for i, it in enumerate(all_items)}
idx2iid    = {i: it for it, i in iid2idx.items()}
NUM_ITEMS  = len(all_items) + 1

uid2idx    = {u: i for i, u in enumerate(train_df["user_id"].unique())}
idx2uid    = {v: k for k, v in uid2idx.items()}
train_df["u"] = train_df["user_id"].map(uid2idx)
train_df["i"] = train_df["item_id"].map(iid2idx)

U = len(uid2idx)
I = len(all_items)

seqs = (train_df.sort_values("timestamp")
        .groupby("u")["i"]
        .apply(list)
        .to_dict())

test_users = sample_sub["user_id"].unique()
test_uidx  = [uid2idx[u] for u in test_users if u in uid2idx]

print(f"users={U}  items={I}  test users={len(test_uidx)}")

# ── validation split ──────────────────────────────────────────────────
N_HOLD    = 10
val_truth = {}
fit_seqs  = {}
for u, seq in seqs.items():
    if u in set(test_uidx) and len(seq) >= 2:
        n = min(N_HOLD, len(seq) // 2)
        fit_seqs[u]  = seq[:-n]
        val_truth[u] = set(seq[-n:])
    else:
        fit_seqs[u] = seq

# ── helpers ───────────────────────────────────────────────────────────
def pad_seq(seq, maxlen=MAXLEN):
    seq = seq[-maxlen:]
    return [0] * (maxlen - len(seq)) + seq

def minmax_norm(arr):
    arr  = arr.astype(np.float64)
    out  = np.full_like(arr, -np.inf)
    fin  = np.isfinite(arr)
    tmp  = np.where(fin, arr, np.inf)
    rmin = tmp.min(axis=1, keepdims=True)
    tmp2 = np.where(fin, arr, -np.inf)
    rmax = tmp2.max(axis=1, keepdims=True)
    rng  = np.maximum(rmax - rmin, 1e-9)
    out  = np.where(fin, (arr - rmin) / rng, -np.inf)
    return out

# ── dataset ───────────────────────────────────────────────────────────
class SeqDataset(Dataset):
    def __init__(self, seqs):
        self.samples = []
        for u, seq in seqs.items():
            if len(seq) < 2:
                continue
            self.samples.append((set(seq),
                                 pad_seq([0] + seq[:-1]),
                                 pad_seq(seq)))

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        seen, inp, pos = self.samples[idx]
        neg = []
        for p in pos:
            if p == 0:
                neg.append(0)
                continue
            j = random.randint(1, NUM_ITEMS - 1)
            while j in seen:
                j = random.randint(1, NUM_ITEMS - 1)
            neg.append(j)
        return (torch.tensor(inp, dtype=torch.long),
                torch.tensor(pos, dtype=torch.long),
                torch.tensor(neg, dtype=torch.long))

# ── model ─────────────────────────────────────────────────────────────
class PointWiseFFN(nn.Module):
    def __init__(self, d, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, d * 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d * 4, d), nn.Dropout(dropout))
        self.norm = nn.LayerNorm(d)
    def forward(self, x): return self.norm(x + self.net(x))

class SASRecBlock(nn.Module):
    def __init__(self, d, n_heads, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, n_heads, dropout=dropout,
                                          batch_first=True)
        self.norm = nn.LayerNorm(d)
        self.ffn  = PointWiseFFN(d, dropout)
        self.drop = nn.Dropout(dropout)
    def forward(self, x, mask):
        res = x
        x, _ = self.attn(x, x, x, attn_mask=mask, need_weights=False)
        return self.ffn(self.norm(res + self.drop(x)))

class SASRec(nn.Module):
    def __init__(self, num_items, d=D_MODEL, n_heads=N_HEADS,
                 n_layers=N_LAYERS, maxlen=MAXLEN, dropout=DROPOUT):
        super().__init__()
        self.item_emb = nn.Embedding(num_items, d, padding_idx=0)
        self.pos_emb  = nn.Embedding(maxlen, d)
        self.emb_drop = nn.Dropout(dropout)
        self.blocks   = nn.ModuleList(
            [SASRecBlock(d, n_heads, dropout) for _ in range(n_layers)])
        self.norm     = nn.LayerNorm(d)

    def encode(self, x):
        B, L = x.shape
        pos  = torch.arange(L, device=x.device).unsqueeze(0)
        h    = self.emb_drop(self.item_emb(x) + self.pos_emb(pos))
        mask = torch.triu(torch.full((L, L), float("-inf"),
                                     device=x.device), diagonal=1)
        for blk in self.blocks: h = blk(h, mask)
        return self.norm(h)

    def forward(self, inp, pos, neg):
        h = self.encode(inp)
        return ((h * self.item_emb(pos)).sum(-1),
                (h * self.item_emb(neg)).sum(-1))

    def predict_all(self, inp):
        h = self.encode(inp)[:, -1, :]
        return h @ self.item_emb.weight.T

# ── training ──────────────────────────────────────────────────────────
def train_epoch(model, loader, opt):
    model.train()
    total, crit = 0.0, nn.BCEWithLogitsLoss(reduction="none")
    for inp, pos, neg in loader:
        inp, pos, neg = inp.to(DEVICE), pos.to(DEVICE), neg.to(DEVICE)
        pl, nl = model(inp, pos, neg)
        mask   = (pos != 0).float()
        loss   = ((crit(pl, torch.ones_like(pl)) +
                   crit(nl, torch.zeros_like(nl))) * mask).sum() / mask.sum()
        opt.zero_grad(); loss.backward(); opt.step()
        total += loss.item()
    return total / len(loader)

def evaluate(model, seqs, truth, users, k=TOP_K):
    model.eval(); recalls = []
    with torch.no_grad():
        for u in users:
            seq = seqs.get(u, [])
            if not seq: continue
            inp    = torch.tensor([pad_seq(seq)], dtype=torch.long,
                                  device=DEVICE)
            scores = model.predict_all(inp)[0].cpu().numpy()
            seen   = set(seq)
            scores[0] = -np.inf
            for it in seen: scores[it] = -np.inf
            top = np.argpartition(-scores, k)[:k]
            top = top[np.argsort(-scores[top])]
            ts  = truth.get(u, set())
            if ts:
                recalls.append(len(ts & set(top.tolist())) / min(len(ts), k))
    return float(np.mean(recalls)) if recalls else 0.0

# ── EASE ──────────────────────────────────────────────────────────────
def fit_ease(train_df, lam=EASE_LAM):
    print(f"Fitting EASE (lambda={lam}) ...")
    d   = train_df.drop_duplicates(subset=["u", "i"])
    mat = csr_matrix((np.ones(len(d), dtype=np.float64),
                      (d["u"], d["i"])), shape=(U, NUM_ITEMS))
    G   = (mat.T @ mat).toarray()
    G[np.diag_indices(NUM_ITEMS)] += lam
    c, low = scipy.linalg.cho_factor(G, lower=True,
                                     overwrite_a=True, check_finite=False)
    P = scipy.linalg.cho_solve((c, low), np.eye(NUM_ITEMS),
                               check_finite=False)
    B = P / (-np.diag(P))[None, :]
    B[np.diag_indices(NUM_ITEMS)] = 0.0
    print(f"  B finite={np.isfinite(B).all()}, max|B|={np.abs(B).max():.3e}")
    return B, mat
# ── TRAIN ─────────────────────────────────────────────────────────────
print("=" * 60)
print(f"Training SASRec  epochs={NUM_EPOCHS}  d={D_MODEL}  "
      f"layers={N_LAYERS}  maxlen={MAXLEN}")
print("=" * 60)

dataset = SeqDataset(fit_seqs)
loader  = DataLoader(dataset, batch_size=BATCH_SIZE,
                     shuffle=True, num_workers=0)
model   = SASRec(NUM_ITEMS).to(DEVICE)
opt     = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=L2)
sched   = torch.optim.lr_scheduler.CosineAnnealingLR(
              opt, T_max=NUM_EPOCHS, eta_min=LR * 0.1)

val_users   = list(val_truth.keys())
best_recall = 0.0
best_state  = None
no_improve  = 0

for epoch in range(1, NUM_EPOCHS + 1):
    loss = train_epoch(model, loader, opt)
    sched.step()
    if epoch % EVAL_EVERY == 0 or epoch == 1:
        r        = evaluate(model, fit_seqs, val_truth, val_users)
        improved = r > best_recall
        print(f"epoch {epoch:>3}  loss={loss:.4f}  "
              f"Recall@10={r:.4f}{' <- best' if improved else ''}")
        if improved:
            best_recall = r
            best_state  = {k: v.clone()
                           for k, v in model.state_dict().items()}
            no_improve  = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break

print(f"\nbest Recall@10 = {best_recall:.4f}")
model.load_state_dict(best_state)
model.eval()

# ── INFERENCE: collect full SASRec scores ─────────────────────────────
print("=" * 60)
print("Computing full score matrices ...")

# SASRec
sas_scores = np.full((len(test_uidx), NUM_ITEMS), -np.inf, dtype=np.float32)
with torch.no_grad():
    for r, u in enumerate(test_uidx):
        seq = seqs.get(u, [])
        inp = torch.tensor([pad_seq(seq)], dtype=torch.long, device=DEVICE)
        sas_scores[r] = model.predict_all(inp)[0].cpu().numpy()
        sas_scores[r, 0] = -np.inf
        for it in set(seq):
            sas_scores[r, it] = -np.inf

# EASE
with np.errstate(all="ignore"):
    B_ease, mat_full = fit_ease(train_df)
    X_test  = np.asarray(mat_full[test_uidx].todense())
    ease_scores = X_test @ B_ease
for r, u in enumerate(test_uidx):
    ease_scores[r, 0] = -np.inf
    for it in set(seqs.get(u, [])):
        ease_scores[r, it] = -np.inf

# ── GENERATE SUBMISSIONS ───────────────────────────────────────────────
print("=" * 60)
print("Generating submissions ...")

order    = {u: i for i, u in enumerate(sample_sub["user_id"].tolist())}
pop      = np.asarray(mat_full.sum(axis=0)).ravel()
fallback = list(np.argsort(-pop)[1:TOP_K + 1])

def scores_to_sub(score_mat, fname):
    rows = []
    for r, u in enumerate(test_uidx):
        row = score_mat[r].astype(np.float64)
        top = np.argpartition(-row, TOP_K)[:TOP_K]
        top = top[np.argsort(-row[top])]
        uid_raw = idx2uid[u]
        rows.append({"ID": uid_raw, "user_id": uid_raw,
                     "item_id": ",".join(str(idx2iid[i])
                                        for i in top.tolist())})
    sub = pd.DataFrame(rows)
    sub = sub.sort_values("user_id", key=lambda s: s.map(order))
    sub.to_csv(fname, index=False, quoting=1)
    sample_items = sub.iloc[0]["item_id"].split(",")
    print(f"{fname} written  sample[0]={sample_items[:3]}")

# 1. SASRec only
scores_to_sub(sas_scores, "submission_sasrec.csv")

# 2. Weighted fusion for w_sas in {0.80, 0.90, 0.95}
sas_norm  = minmax_norm(sas_scores.astype(np.float64))
ease_norm = minmax_norm(ease_scores)

for w_sas in [0.80,0.90, 0.95]:
    combined = w_sas * sas_norm + (1 - w_sas) * ease_norm
    scores_to_sub(combined,
                  f"submission_fusion_w{int(w_sas*100)}.csv")

print("\nSample (SASRec only):")
print(pd.read_csv("submission_sasrec.csv").head(3).to_string(index=False))
