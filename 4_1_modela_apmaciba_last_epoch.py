import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix

#iestatijumi
TRAIN_DIR = "dataset_landmarks_trimmed"
TEST_DIR  = "dataset_landmarks_test"

TRAIN_METADATA = os.path.join(TRAIN_DIR, "landmarks_metadata.csv")
TEST_METADATA  = os.path.join(TEST_DIR,  "landmarks_metadata.csv")

OUTPUT_DIR = "training_outputs_bilstm_trimmed_externaltest"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, "best_bilstm_trimmed_externaltest.pt")
LABEL_MAP_SAVE_PATH = os.path.join(OUTPUT_DIR, "label_map_trimmed.json")
REPORT_SAVE_PATH = os.path.join(OUTPUT_DIR, "final_report_test.txt")

SEQUENCE_LENGTH = 48
BATCH_SIZE = 32
EPOCHS = 60
LR = 0.001

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

#normalizacija
def normalize_sequence_oldnorm(seq):
    #sekvences forma: (T, 21, 3)

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

#datu ielades funkcija
def load_dataset(data_dir, metadata_path, label_to_idx=None, allowed_status=("good",)):
    df = pd.read_csv(metadata_path)

    df = df[df["status"].isin(list(allowed_status))].copy()

    X = []
    y = []
    labels_found = sorted(df["label"].unique())

    if label_to_idx is None:
        label_to_idx = {l: i for i, l in enumerate(labels_found)}

    for _, row in df.iterrows():
        label = row["label"]

        if label not in label_to_idx:
            raise ValueError(f"[ERROR] Dataset contains unknown label '{label}' not in training labels!")

        npz_path = os.path.join(data_dir, label, row["npz_name"])
        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"[ERROR] Missing file: {npz_path}")

        data = np.load(npz_path, allow_pickle=True)
        seq = data["sequence"]  #(T, 21, 3)

        seq = normalize_sequence_oldnorm(seq)
        seq = resample_sequence(seq, SEQUENCE_LENGTH)
        seq = seq.reshape(SEQUENCE_LENGTH, -1)

        X.append(seq)
        y.append(label_to_idx[label])

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)

    return X, y, label_to_idx

#ieladet abas datu kopas: apmacibas un testu
print("Loading TRAIN dataset...")
X_train, y_train, label_to_idx = load_dataset(TRAIN_DIR, TRAIN_METADATA, label_to_idx=None)
idx_to_label = {v: k for k, v in label_to_idx.items()}
labels_sorted = [idx_to_label[i] for i in range(len(idx_to_label))]

print("Train samples:", len(X_train))
print("Labels:", label_to_idx)

print("\nLoading TEST dataset (external)...")
X_test, y_test, _ = load_dataset(TEST_DIR, TEST_METADATA, label_to_idx=label_to_idx)

print("Test samples:", len(X_test))

#saglabat klasu karti, reallaika prototipam
label_map = {
    "label_to_index": label_to_idx,
    "index_to_label": {str(i): lbl for i, lbl in idx_to_label.items()}
}

with open(LABEL_MAP_SAVE_PATH, "w", encoding="utf-8") as f:
    json.dump(label_map, f, ensure_ascii=False, indent=2)

print(f"\nSaved label map: {LABEL_MAP_SAVE_PATH}")

#datu ieliksana DataLoader objektos priekš PyTorch
train_loader = DataLoader(
    GestureDataset(X_train, y_train),
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_loader = DataLoader(
    GestureDataset(X_test, y_test),
    batch_size=BATCH_SIZE,
    shuffle=False
)

#klašu svari (optional)
class_counts = np.bincount(y_train)
weights = 1.0 / class_counts
weights = weights / weights.sum()
class_weights = torch.tensor(weights, dtype=torch.float32).to(DEVICE)

#modela apmaciba
model = BiLSTMClassifier(
    input_size=63,
    hidden_size=128,
    num_layers=2,
    num_classes=len(label_to_idx)
).to(DEVICE)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

best_test_acc = 0.0

print("\nStarting training...\n")

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

    #parbaude uz arejas testu kopas, katras epohas beigas
    model.eval()
    correct = 0
    total = 0

    all_preds = []
    all_true = []

    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)

            logits = model(xb)
            pred = torch.argmax(logits, dim=1)

            correct += (pred == yb).sum().item()
            total += yb.size(0)

            all_preds.extend(pred.cpu().numpy())
            all_true.extend(yb.cpu().numpy())

    test_acc = correct / total

    print(f"Epoch {epoch+1}/{EPOCHS} | train_loss={train_loss:.4f} | test_acc={test_acc:.4f}")

    if test_acc > best_test_acc:
        best_test_acc = test_acc
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"  -> Saved best model: {MODEL_SAVE_PATH}")

#gala parskats: klasifikacijas raditaji un sadalijuma matrica
print("\nFINAL TEST REPORT:\n")
report = classification_report(all_true, all_preds, target_names=labels_sorted)
print(report)

cm = confusion_matrix(all_true, all_preds)

with open(REPORT_SAVE_PATH, "w", encoding="utf-8") as f:
    f.write("FINAL TEST REPORT\n\n")
    f.write(report)
    f.write("\n\nCONFUSION MATRIX:\n")
    f.write(str(cm))

print(f"\nSaved final report: {REPORT_SAVE_PATH}")
print(f"Best test accuracy: {best_test_acc:.4f}")