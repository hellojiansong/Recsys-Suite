import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# 0. Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# 1. Device Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
print(f"Using device: {DEVICE}")

# 2. Hyperparameters
PARAMS = {
    'embed_dim': 32,
    'batch_size': 1024,
    'lr': 0.001,
    'epochs': 20,
    'neg_ratio': 4,  #
    'top_k': 10,  # [cite: 55, 56]
    'early_stop_patience': 3,  # [cite: 52]
    'mlp_layers': [64, 32, 16],
}

# 3. Data Loading & Preprocessing
def load_data(path):
    # Load full dataset to get global IDs [cite: 22, 23, 24]
    full_df = pd.read_csv(path, sep="::", engine="python",
                          names=["user", "item", "rating", "timestamp"])

    # Global ID mapping to ensure negative sampling covers all items
    user2id = {u: i for i, u in enumerate(full_df.user.unique())}
    item2id = {i: j for j, i in enumerate(full_df.item.unique())}
    num_users, num_items = len(user2id), len(item2id)

    # Filter for positive interactions (rating >= 4) [cite: 26, 37]
    pos_df = full_df[full_df['rating'] >= 4].copy()
    pos_df['user'] = pos_df['user'].map(user2id)
    pos_df['item'] = pos_df['item'].map(item2id)
    pos_df['label'] = 1.0

    return pos_df, num_users, num_items


def split_data(df):
    # Random split: 70% train, 15% val, 15% test
    train_val, test = train_test_split(df, test_size=0.15, random_state=SEED)
    train, val = train_test_split(train_val, test_size=0.15 / 0.85, random_state=SEED)
    return train, val, test

# 4. Negative Sampling
def get_train_instances(train_df, num_items, neg_ratio):
    users, items, labels = [], [], []
    train_set = set(zip(train_df['user'], train_df['item']))

    # Positive samples
    u_list = train_df['user'].values
    i_list = train_df['item'].values
    users.extend(u_list)
    items.extend(i_list)
    labels.extend([1.0] * len(u_list))

    # Negative sampling [cite: 38, 39]
    neg_users = np.repeat(u_list, neg_ratio)
    neg_items = np.random.randint(0, num_items, size=len(neg_users))

    for idx in range(len(neg_items)):
        u, i = neg_users[idx], neg_items[idx]
        while (u, i) in train_set:
            i = np.random.randint(0, num_items)
        neg_items[idx] = i

    users.extend(neg_users)
    items.extend(neg_items)
    labels.extend([0.0] * len(neg_users))

    return users, items, labels


class NCFDataset(Dataset):
    def __init__(self, users, items, labels):
        self.users = torch.LongTensor(users)
        self.items = torch.LongTensor(items)
        self.labels = torch.FloatTensor(labels)

    def __len__(self): return len(self.users)

    def __getitem__(self, idx): return self.users[idx], self.items[idx], self.labels[idx]


# 5. NCF Model
class NCF(nn.Module):
    def __init__(self, num_users, num_items, embed_dim, layers):
        super(NCF, self).__init__()
        # GMF and MLP use separate embeddings [cite: 43, 44, 45]
        self.embed_user_GMF = nn.Embedding(num_users, embed_dim)
        self.embed_item_GMF = nn.Embedding(num_items, embed_dim)
        self.embed_user_MLP = nn.Embedding(num_users, embed_dim)
        self.embed_item_MLP = nn.Embedding(num_items, embed_dim)

        mlp_modules = []
        input_size = embed_dim * 2
        for output_size in layers:
            mlp_modules.append(nn.Linear(input_size, output_size))
            mlp_modules.append(nn.ReLU())
            input_size = output_size
        self.mlp_layers = nn.Sequential(*mlp_modules)

        # Fusion layer [cite: 46, 47]
        self.fc_final = nn.Linear(embed_dim + layers[-1], 1)
        self.sigmoid = nn.Sigmoid()
        self._init_weights()

    def _init_weights(self):
        # Academic standard weight initialization
        nn.init.normal_(self.embed_user_GMF.weight, std=0.01)
        nn.init.normal_(self.embed_item_GMF.weight, std=0.01)
        nn.init.normal_(self.embed_user_MLP.weight, std=0.01)
        nn.init.normal_(self.embed_item_MLP.weight, std=0.01)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
        nn.init.xavier_uniform_(self.fc_final.weight)

    def forward(self, user_indices, item_indices):
        user_gmf = self.embed_user_GMF(user_indices)
        item_gmf = self.embed_item_GMF(item_indices)
        gmf_out = user_gmf * item_gmf

        user_mlp = self.embed_user_MLP(user_indices)
        item_mlp = self.embed_item_MLP(item_indices)
        mlp_out = self.mlp_layers(torch.cat([user_mlp, item_mlp], dim=-1))

        prediction = self.fc_final(torch.cat([gmf_out, mlp_out], dim=-1))
        return self.sigmoid(prediction).view(-1)


# 6. Full-Ranking Evaluation
def evaluate(model, exclude_df, target_df, num_items, k=10):
    model.eval()
    recalls, ndcgs = [], []
    # Mask items seen in training/validation to ensure true recommendation [cite: 54]
    exclude_dict = exclude_df.groupby('user')['item'].apply(set).to_dict()
    target_dict = target_df.groupby('user')['item'].apply(list).to_dict()

    item_tensor = torch.arange(num_items).to(DEVICE)
    with torch.no_grad():
        for u in target_dict.keys():
            u_tensor = torch.full((num_items,), u, dtype=torch.long).to(DEVICE)
            scores = model(u_tensor, item_tensor)

            # Masking all seen items (Train + Val if evaluating Test)
            seen = list(exclude_dict.get(u, set()))
            if seen: scores[seen] = -1e9

            _, indices = torch.topk(scores, k)
            recommends = indices.cpu().numpy().tolist()
            true_items = set(target_dict[u])

            hits = len(set(recommends) & true_items)
            recalls.append(hits / len(true_items))

            dcg = sum([1 / np.log2(i + 2) for i, item in enumerate(recommends) if item in true_items])
            idcg = sum([1 / np.log2(i + 2) for i in range(min(len(true_items), k))])
            ndcgs.append(dcg / idcg if idcg > 0 else 0)
    return np.mean(recalls), np.mean(ndcgs)

# 7. Training Loop with Early Stopping on Validation Loss
if __name__ == "__main__":
    DATA_PATH = "ratings.dat"  # Ensure this file exists

    # Experiment with different MLP layer configurations
    mlp_configs = [[128, 64, 32], [64, 32, 16], [32, 16, 8]]

    # Load and preprocess data
    pos_df, num_users, num_items = load_data(DATA_PATH)
    train_df, val_df, test_df = split_data(pos_df)

    for layers in mlp_configs:
        print(f"\n--- Model Config: MLP Layers {layers} ---")

        # Initialize model, optimizer, and loss function
        model = NCF(num_users, num_items, PARAMS['embed_dim'], layers).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=PARAMS['lr'])
        criterion = nn.BCELoss()  # Binary Cross-Entropy for implicit feedback

        # Early stopping initialization
        best_val_loss = float('inf')  # track lowest validation loss
        patience = 0  # counter for early stopping

        for epoch in range(PARAMS['epochs']):

            # 1. Dynamic Negative Sampling for Training
            u_train, i_train, l_train = get_train_instances(train_df, num_items, PARAMS['neg_ratio'])
            train_dataset = NCFDataset(u_train, i_train, l_train)
            train_loader = DataLoader(train_dataset, batch_size=PARAMS['batch_size'], shuffle=True)

            model.train()
            loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{PARAMS['epochs']}", leave=False)
            for u_b, i_b, l_b in loop:
                u_b, i_b, l_b = u_b.to(DEVICE), i_b.to(DEVICE), l_b.to(DEVICE)
                optimizer.zero_grad()
                pred = model(u_b, i_b)
                loss = criterion(pred, l_b)
                loss.backward()
                optimizer.step()
                loop.set_postfix(loss=loss.item())

            # 2. Compute Validation Loss
            model.eval()
            with torch.no_grad():
                # Generate validation instances
                # Note: neg_ratio=0 to only use positives in validation for BCE loss
                u_val, i_val, l_val = get_train_instances(val_df, num_items, neg_ratio=0)
                val_dataset = NCFDataset(u_val, i_val, l_val)
                val_loader = DataLoader(val_dataset, batch_size=PARAMS['batch_size'], shuffle=False)

                val_losses = []
                loop = tqdm(val_loader, desc="Validation", leave=False)
                for u_b, i_b, l_b in loop:
                    u_b, i_b, l_b = u_b.to(DEVICE), i_b.to(DEVICE), l_b.to(DEVICE)
                    pred = model(u_b, i_b)
                    val_loss = criterion(pred, l_b)
                    val_losses.append(val_loss.item())
                    loop.set_postfix(val_loss=val_loss.item())

                v_loss = np.mean(val_losses)  # average BCE loss on validation set

            print(f"Epoch {epoch} | Val BCE Loss: {v_loss:.4f}")

            # 3. Early Stopping Logic
            if v_loss < best_val_loss:
                best_val_loss = v_loss
                patience = 0
                # Save best model based on validation loss
                torch.save(model.state_dict(), f"best_model_{layers[0]}.pt")
            else:
                patience += 1
                if patience >= PARAMS['early_stop_patience']:
                    print(f"Early stopping triggered at epoch {epoch}")
                    break

        # 4. Final Test Evaluation
        model.load_state_dict(torch.load(f"best_model_{layers[0]}.pt"))
        exclude_test = pd.concat([train_df, val_df])
        t_recall, t_ndcg = evaluate(model, exclude_test, test_df, num_items, PARAMS['top_k'])
        print(f"FINAL TEST RESULT | Recall@10: {t_recall:.4f} | NDCG@10: {t_ndcg:.4f}")