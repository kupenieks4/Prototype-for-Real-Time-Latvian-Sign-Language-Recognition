import os
import re
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, f1_score

#iestatijumi
DATA_DIR = "dataset_landmarks_trimmed"
METADATA_PATH = os.path.join(DATA_DIR, "landmarks_metadata.csv")

OUTPUT_DIR = "training_outputs_grup_krust_validac"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEQUENCE_LENGTH = 48
BATCH_SIZE = 32
EPOCHS = 40
LR = 0.001

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

#normalizacija
def normalize_sequence_oldnorm(seq):
    wrist = seq[:, 0:1, :]
    seq = seq - wrist

    scale = np.max(np.linalg.norm(seq, axis=2))
    if scale < 1e-6:
        scale = 1.0

    seq = seq / scale
    return seq.astype(np.float32)

#ierakstu parveidosana uz 48 kadriem
def resample_sequence(seq, target_len=48):
    T = len(seq)

    if T == target_len:
        return seq

    if T == 1:
        return np.repeat(seq, target_len, axis=0)

    idxs = np.linspace(0, T - 1, target_len)
    idxs = np.round(idxs).astype(np.int32)
    return seq[idxs]

#datu kopa, klase GestureDataset sagatavo datus priekš PyTorch
class GestureDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

#Bi-LSTM modelis
class BiLSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.3):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.norm = nn.LayerNorm(hidden_size * 2)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.norm(out)
        out = self.classifier(out)
        return out

#grupētā sadalījuma logika
#----------------------------------
#1-80 = group 0, lokacija A
#81-999 = group 1, lokacija B
#ja 120 ieraksti klasē, tad:
#1-80 ir pirmā lokācija, 81-120 otrā
#BET, ja "plane", tad ir savadak. līdz 149, tad 81-149 arī būs otrā lokācija
PLANE_FOAJEE_NUMBERS = {10, 19, 36, 47, 51, 52, 56, 60}
PLANE_FOAJEE_MIRRORS = {"plane_0088_m", "plane_0090_m", "plane_0102_m"}

def extract_number_from_filename(name: str):
    nums = re.findall(r"\d+", name)
    if len(nums) == 0:
        return None
    return int(nums[-1])

def get_group_id(label: str, npz_name: str):
    """
    group 0 = pirmie 80 (pamatlokācija / sesija)
    group 1 = pēdējie 40 (otra lokācija / sesija)

    plane gadījumā:
    group 1 = foajē specifiskie ieraksti (40 gab.)
    """

    n = extract_number_from_filename(npz_name)
    if n is None:
        raise ValueError(f"Cannot parse number from filename: {npz_name}")

    #PLANE sacakareta mapite
    if label == "plane":
        base = npz_name.replace(".npz", "").replace(".avi", "")

        #foaje range
        if 121 <= n <= 149:
            return 1

        #foaje specifiskie skaitli
        if n in PLANE_FOAJEE_NUMBERS:
            return 1

        #foaje specifiskie mirror faili
        if base in PLANE_FOAJEE_MIRRORS:
            return 1

        #parejais
        return 0

    if n <= 80:
        return 0
    else:
        return 1

#datu kopas ielade
df = pd.read_csv(METADATA_PATH)
df = df[df["status"].isin(["good"])].copy()

labels = sorted(df["label"].unique())
label_to_idx = {l: i for i, l in enumerate(labels)}

print("Labels:", label_to_idx)

X_all = []
y_all = []
groups_all = []

for _, row in df.iterrows():
    label = row["label"]
    npz_name = row["npz_name"]

    npz_path = os.path.join(DATA_DIR, label, npz_name)
    data = np.load(npz_path, allow_pickle=True)
    seq = data["sequence"]

    seq = normalize_sequence_oldnorm(seq)
    seq = resample_sequence(seq, SEQUENCE_LENGTH)
    seq = seq.reshape(SEQUENCE_LENGTH, -1)

    X_all.append(seq)
    y_all.append(label_to_idx[label])
    groups_all.append(get_group_id(label, npz_name))

X_all = np.array(X_all, dtype=np.float32)
y_all = np.array(y_all, dtype=np.int64)
groups_all = np.array(groups_all, dtype=np.int64)

print("Loaded dataset:", X_all.shape)
print("Groups distribution:", np.bincount(groups_all))

#grupeta krusta validacija
unique_groups = np.unique(groups_all)
n_splits = len(unique_groups)

print(f"\nDetected {n_splits} unique groups -> using GroupKFold(n_splits={n_splits})")

gkf = GroupKFold(n_splits=n_splits)

fold_results = []

for fold, (train_idx, val_idx) in enumerate(gkf.split(X_all, y_all, groups_all), start=1):
    print("\n==============================")
    print(f"GROUP FOLD {fold}/{n_splits}")
    print("==============================")
    print("Train size:", len(train_idx), "| Val size:", len(val_idx))

    X_train, y_train = X_all[train_idx], y_all[train_idx]
    X_val, y_val = X_all[val_idx], y_all[val_idx]

    train_loader = DataLoader(GestureDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(GestureDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)

    class_counts = np.bincount(y_train)
    weights = 1.0 / class_counts
    weights = weights / weights.sum()
    class_weights = torch.tensor(weights, dtype=torch.float32).to(DEVICE)

    model = BiLSTMClassifier(
        input_size=63,
        hidden_size=128,
        num_layers=2,
        num_classes=len(labels)
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    #train loops
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0

        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)

            logits = model(xb)
            loss = criterion(logits, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} | train_loss={train_loss:.4f}")

    #parbaude uz validacijas grupas
    model.eval()
    all_preds = []
    all_true = []

    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(DEVICE)
            logits = model(xb)
            pred = torch.argmax(logits, dim=1).cpu().numpy()

            all_preds.extend(pred)
            all_true.extend(yb.numpy())

    fold_acc = accuracy_score(all_true, all_preds)
    fold_f1_macro = f1_score(all_true, all_preds, average="macro")
    fold_f1_weighted = f1_score(all_true, all_preds, average="weighted")

    print("\nFold evaluation:")
    print("Accuracy:", fold_acc)
    print("F1 macro:", fold_f1_macro)
    print("F1 weighted:", fold_f1_weighted)

    fold_results.append({
        "fold": fold,
        "accuracy": float(fold_acc),
        "f1_macro": float(fold_f1_macro),
        "f1_weighted": float(fold_f1_weighted),
        "train_size": int(len(train_idx)),
        "val_size": int(len(val_idx))
    })

#rezultati
accs = [r["accuracy"] for r in fold_results]
f1_macros = [r["f1_macro"] for r in fold_results]
f1_weighteds = [r["f1_weighted"] for r in fold_results]

summary = {
    "fold_results": fold_results,
    "accuracy_mean": float(np.mean(accs)),
    "accuracy_std": float(np.std(accs)),
    "f1_macro_mean": float(np.mean(f1_macros)),
    "f1_macro_std": float(np.std(f1_macros)),
    "f1_weighted_mean": float(np.mean(f1_weighteds)),
    "f1_weighted_std": float(np.std(f1_weighteds)),
}

print("\n==============================")
print("GROUP CROSS-VALIDATION SUMMARY")
print("==============================")
print(f"Accuracy mean={summary['accuracy_mean']:.4f} std={summary['accuracy_std']:.4f}")
print(f"F1 macro mean={summary['f1_macro_mean']:.4f} std={summary['f1_macro_std']:.4f}")
print(f"F1 weighted mean={summary['f1_weighted_mean']:.4f} std={summary['f1_weighted_std']:.4f}")

summary_path = os.path.join(OUTPUT_DIR, "group_crossval_summary.json")
with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print(f"\nSaved summary: {summary_path}")