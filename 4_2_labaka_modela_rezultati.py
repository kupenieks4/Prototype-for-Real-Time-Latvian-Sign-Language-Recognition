#4_1 TIKA SAGLABATI REZULTATI LAST EPOHAI, SIS TO DARA LABAKAJAI
import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix

#iestatijumi
TEST_DIR = "dataset_landmarks_test"
TEST_METADATA = os.path.join(TEST_DIR, "landmarks_metadata.csv")

OUTPUT_DIR = "training_outputs_bilstm_trimmed_externaltest"

MODEL_PATH = os.path.join(OUTPUT_DIR, "best_bilstm_trimmed_externaltest.pt")
LABEL_MAP_PATH = os.path.join(OUTPUT_DIR, "label_map_trimmed.json")

REPORT_SAVE_PATH = os.path.join(OUTPUT_DIR, "final_report_test_best_model.txt")
CONFUSION_MATRIX_SAVE_PATH = os.path.join(OUTPUT_DIR, "confusion_matrix_best_model.txt")

SEQUENCE_LENGTH = 48
BATCH_SIZE = 32

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

#datu ielades funkcija, arejai testu kopai
def load_dataset(data_dir, metadata_path, label_to_idx, allowed_status=("good",)):
    df = pd.read_csv(metadata_path)
    df = df[df["status"].isin(list(allowed_status))].copy()

    X = []
    y = []

    for _, row in df.iterrows():
        label = row["label"]

        if label not in label_to_idx:
            raise ValueError(f"[ERROR] Dataset contains unknown label '{label}' not in label map!")

        npz_path = os.path.join(data_dir, label, row["npz_name"])
        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"[ERROR] Missing file: {npz_path}")

        data = np.load(npz_path, allow_pickle=True)
        seq = data["sequence"]

        seq = normalize_sequence_oldnorm(seq)
        seq = resample_sequence(seq, SEQUENCE_LENGTH)
        seq = seq.reshape(SEQUENCE_LENGTH, -1)

        X.append(seq)
        y.append(label_to_idx[label])

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)

    return X, y

#load klasu karti
if not os.path.exists(LABEL_MAP_PATH):
    raise FileNotFoundError(f"Label map not found: {LABEL_MAP_PATH}")

with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
    label_map = json.load(f)

label_to_idx = label_map["label_to_index"]
idx_to_label = {int(k): v for k, v in label_map["index_to_label"].items()}
labels_sorted = [idx_to_label[i] for i in range(len(idx_to_label))]

print("Loaded labels:", idx_to_label)

#load testu kopa
X_test, y_test = load_dataset(TEST_DIR, TEST_METADATA, label_to_idx=label_to_idx)
print("Test samples:", len(X_test))

test_loader = DataLoader(
    GestureDataset(X_test, y_test),
    batch_size=BATCH_SIZE,
    shuffle=False
)

#load labakais modelis no 4_1
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Best model not found: {MODEL_PATH}")

model = BiLSTMClassifier(
    input_size=63,
    hidden_size=128,
    num_layers=2,
    num_classes=len(label_to_idx)
).to(DEVICE)

state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state_dict)
model.eval()

print(f"Loaded BEST model from: {MODEL_PATH}")
print(f"Device: {DEVICE}")

#labaka modela parbaude un labako rezultatu save
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

print("\nBEST MODEL TEST REPORT:\n")
report = classification_report(all_true, all_preds, target_names=labels_sorted)
print(report)

cm = confusion_matrix(all_true, all_preds)
print("\nCONFUSION MATRIX:")
print(cm)

with open(REPORT_SAVE_PATH, "w", encoding="utf-8") as f:
    f.write("BEST MODEL TEST REPORT\n\n")
    f.write(report)
    f.write("\n\nCONFUSION MATRIX:\n")
    f.write(str(cm))
    f.write(f"\n\nBest model test accuracy: {test_acc:.4f}")

with open(CONFUSION_MATRIX_SAVE_PATH, "w", encoding="utf-8") as f:
    f.write(str(cm))

print(f"\nSaved best-model report: {REPORT_SAVE_PATH}")
print(f"Saved best-model confusion matrix txt: {CONFUSION_MATRIX_SAVE_PATH}")
print(f"Best model test accuracy: {test_acc:.4f}")