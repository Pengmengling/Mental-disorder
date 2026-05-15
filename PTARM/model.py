import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
import os
import copy

class MotionDataset(Dataset):
    def __init__(self, X, y, ids):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.ids = ids

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.ids[idx]


def reshape_input(batch_x):
    B = batch_x.shape[0]
    batch_x = batch_x.view(B, 24, 8)
    return batch_x

class Transformer(nn.Module):
    def __init__(self, input_dim=8, d_model=16, nhead=2, num_layers=1):
        super().__init__()

        self.proj = nn.Linear(input_dim, d_model)

        self.pos_embedding = nn.Parameter(torch.randn(1, 24, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.attn_pool = nn.Sequential(
            nn.Linear(d_model, 1)
        )

        self.fc = nn.Sequential(
            nn.Linear(d_model, 8),
            nn.ReLU(),
            nn.BatchNorm1d(8),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        x = self.proj(x) + self.pos_embedding

        x = self.transformer(x)

        weights = torch.softmax(self.attn_pool(x), dim=1)
        x = torch.sum(weights * x, dim=1)

        out = self.fc(x)
        return out

def train_val_for_disease(disease_name, train_df, val_df, epochs=20):

    print(f"\n====== {disease_name} ======")

    feature_cols = [str(i) for i in range(192)]

    X_train = train_df[feature_cols].values
    y_train = train_df[disease_name].values
    ids_train = train_df["Participant ID"].values

    X_val = val_df[feature_cols].values
    y_val = val_df[disease_name].values
    ids_val = val_df["Participant ID"].values

    train_loader = DataLoader(
        MotionDataset(X_train, y_train, ids_train),
        batch_size=64,
        shuffle=True
    )

    val_loader = DataLoader(
        MotionDataset(X_val, y_val, ids_val),
        batch_size=64,
        shuffle=False
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Transformer().to(device)

    pos = np.sum(y_train == 1)
    neg = np.sum(y_train == 0)
    print(f"positive sample: {pos} negative sample: {neg}")

    pos_weight = torch.tensor([neg / pos]).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_auc = 0
    best_model_wts = None

    for epoch in range(epochs):
        model.train()

        for batch_x, batch_y, _ in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device).unsqueeze(1)

            x = reshape_input(batch_x)

            outputs = model(x)
            loss = criterion(outputs, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        all_probs, all_labels = [], []

        with torch.no_grad():
            for batch_x, batch_y, _ in val_loader:
                batch_x = batch_x.to(device)

                x = reshape_input(batch_x)
                outputs = model(x)

                probs = torch.sigmoid(outputs).squeeze(1).cpu().numpy()
                all_probs.extend(probs)
                all_labels.extend(batch_y.numpy())

        val_auc = roc_auc_score(all_labels, all_probs)

        print(f"Epoch {epoch+1}/{epochs}  AUC: {val_auc:.4f}")
        print(f"score: {np.min(all_probs):.4f} ~ {np.max(all_probs):.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            best_model_wts = copy.deepcopy(model.state_dict())

    print(f"\n✅ Best AUC: {best_auc:.4f}")

    os.makedirs("model", exist_ok=True)
    #model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), f"model/new_model8/transformer_{disease_name}.pth")

    
    os.makedirs("result", exist_ok=True)
    result_df = pd.DataFrame({
        "id": ids_val,
        "label": all_labels,
        "score": all_probs
    })
    result_df.to_csv(f"result/transformer_{disease_name}.csv", index=False)


# =========================
# main
# =========================
if __name__ == "__main__":

    target_diseases = ['F00', 'F01', 'F02', 'F03', 'F05', 'F06', 'F10', 'F17', 'F32', 'F40', 'F41', 'F43']

    train_df = pd.read_csv("icd_england_train.csv")
    val_df = pd.read_csv("icd_england_val.csv")
    label_df = pd.read_csv("all_icd_f_group_1000_10y_wear.csv")

    t_df = pd.merge(train_df, label_df, on="Participant ID")
    v_df = pd.merge(val_df, label_df, on="Participant ID")

    feature_cols = [str(i) for i in range(192)]

    for disease in target_diseases:
        print(f"\n====== {disease} ======")

        t_df_d = t_df[t_df[disease] != 2].copy()
        v_df_d = v_df[v_df[disease] != 2].copy()

        t_df_d = t_df_d.dropna(subset=feature_cols + [disease])
        v_df_d = v_df_d.dropna(subset=feature_cols + [disease])

        train_val_for_disease(disease, t_df_d, v_df_d, epochs=30)