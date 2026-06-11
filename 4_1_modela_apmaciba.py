import argparse
import json
import os
import re
from collections import Counter
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import Dataset, DataLoader
#iestatijumi
DATA_DIR = "dataset_landmarks_trimmed"
METADATA_PATH = os.path.join(DATA_DIR, "landmarks_metadata.csv")
SEQUENCE_LENGTH = 48
BATCH_SIZE = 32
EPOCHS = 60
LR = 0.001

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
#10 zimes
CLASSES_10 = [
    "A",
    "U",
    "AA_long",
    "plane",
    "labdien",
    "ata",
    "ok",
    "ne",
    "B",
    "paldies",
]
#15 zimes
CLASSES_15 = CLASSES_10 + [
    "es",
    "vajag",
    "nauda",
    "tualete",
    "telefons",
]
#unique lidmasinas mape
PLANE_FOAJEE_NUMBERS = {10, 19, 36, 47, 51, 52, 56, 60}
PLANE_FOAJEE_MIRRORS = {"plane_0088_m", "plane_0090_m", "plane_0102_m"}

#10 zimes or 15 zimju apmaciba
def parse_args():
    parser = argparse.ArgumentParser(
        description="Bi-LSTM apmaciba ar pareizu train/validation sadalijumu bez arejas testu kopas."
    )
    parser.add_argument(
        "--classes",
        type=int,
        choices=[10, 15],
        default=10,
        help="Cik zimes lietot apmaciba: 10 vai 15. Sakuma ieteicams 10, pec tam 15.",
    )
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--data-dir", type=str, default=DATA_DIR)
    return parser.parse_args()

#normalizacija pret plaukstas 0. punktu
def normalize_sequence_oldnorm(seq):
    #(T, 21, 3), T - kadru skaits video
    wrist = seq[:, 0:1, :]
    seq = seq - wrist

    scale = np.max(np.linalg.norm(seq, axis=2))
    if scale < 1e-6:
        scale = 1.0

    seq = seq / scale
    return seq.astype(np.float32)

#sekvences parveidosana uz 48 kadriem
def resample_sequence(seq, target_len=48):
    T = len(seq)

    if T == target_len:
        return seq

    if T == 1:
        return np.repeat(seq, target_len, axis=0)

    idxs = np.linspace(0, T - 1, target_len)
    idxs = np.round(idxs).astype(np.int32)
    return seq[idxs]

#numbers no faila
def extract_number_from_filename(name: str):
    nums = re.findall(r"\d+", name)
    if len(nums) == 0:
        return None
    return int(nums[-1])

#train validation sadalijums
def get_group_id(label: str, npz_name: str):
    """
    group 0 = train grupa
        Parastajam klasem: 1-80 ieraksti.
    group 1 = validation grupa
        Parastajam klasem: 81+ ieraksti.

    Plane gadijuma izmantota manuala logika no grupetas krusta validacijas,
    lai foaje/citas lokacijas ieraksti nonaktu validation grupa.
    """
    n = extract_number_from_filename(npz_name)
    if n is None:
        raise ValueError(f"Cannot parse number from filename: {npz_name}")

    if label == "plane":
        base = npz_name.replace(".npz", "").replace(".avi", "")

        if 121 <= n <= 149:
            return 1
        if n in PLANE_FOAJEE_NUMBERS:
            return 1
        if base in PLANE_FOAJEE_MIRRORS:
            return 1
        return 0

    if n <= 80:
        return 0
    return 1

#pytorch x ievaddati, y pareizas atbildes. sagatavo datus priekš PyTorch
class GestureDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

#neironu tikls
class BiLSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.3):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.norm = nn.LayerNorm(hidden_size * 2)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.norm(out)
        out = self.classifier(out)
        return out

def load_group_split_dataset(data_dir, selected_classes):
    metadata_path = os.path.join(data_dir, "landmarks_metadata.csv")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")

    df = pd.read_csv(metadata_path)

    #tikai labas kvalites ieraksti
    df = df[df["status"].isin(["good"])].copy()

    df = df[df["label"].isin(selected_classes)].copy()

    label_to_idx = {label: i for i, label in enumerate(selected_classes)}
    idx_to_label = {i: label for label, i in label_to_idx.items()}

    X_train, y_train = [], []
    X_val, y_val = [], []

    split_rows = []

    for _, row in df.iterrows():
        label = row["label"]
        npz_name = row["npz_name"]

        if label not in label_to_idx:
            continue

        npz_path = os.path.join(data_dir, label, npz_name)
        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"Missing npz file: {npz_path}")

        data = np.load(npz_path, allow_pickle=True)
        seq = data["sequence"]

        seq = normalize_sequence_oldnorm(seq)
        seq = resample_sequence(seq, SEQUENCE_LENGTH)
        seq = seq.reshape(SEQUENCE_LENGTH, -1)

        group_id = get_group_id(label, npz_name)
        y = label_to_idx[label]

        if group_id == 0:
            X_train.append(seq)
            y_train.append(y)
            split_name = "train"
        elif group_id == 1:
            X_val.append(seq)
            y_val.append(y)
            split_name = "val"
        else:
            raise ValueError(f"Unexpected group_id={group_id} for {label}/{npz_name}")

        split_rows.append({
            "label": label,
            "npz_name": npz_name,
            "group_id": group_id,
            "split": split_name,
            "status": row["status"],
        })

    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.int64)
    X_val = np.array(X_val, dtype=np.float32)
    y_val = np.array(y_val, dtype=np.int64)

    split_df = pd.DataFrame(split_rows)

    return X_train, y_train, X_val, y_val, label_to_idx, idx_to_label, split_df

#parbaude uz validation datiem
def evaluate(model, loader, criterion=None):
    model.eval()

    total_loss = 0.0
    all_preds = []
    all_true = []

    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            logits = model(xb)

            if criterion is not None:
                loss = criterion(logits, yb)
                total_loss += loss.item()

            pred = torch.argmax(logits, dim=1)
            all_preds.extend(pred.cpu().numpy())
            all_true.extend(yb.cpu().numpy())

    acc = accuracy_score(all_true, all_preds)
    f1_macro = f1_score(all_true, all_preds, average="macro", zero_division=0)
    f1_weighted = f1_score(all_true, all_preds, average="weighted", zero_division=0)
    avg_loss = total_loss / max(len(loader), 1) if criterion is not None else None

    return avg_loss, acc, f1_macro, f1_weighted, all_true, all_preds

#summary izvade
def print_split_summary(split_df, selected_classes):
    print("\n==============================")
    print("SPLIT SUMMARY")
    print("==============================")

    summary = (
        split_df.groupby(["label", "split"])
        .size()
        .unstack(fill_value=0)
        .reindex(selected_classes)
        .fillna(0)
        .astype(int)
    )

    print(summary)
    print("\nTotal train:", int((split_df["split"] == "train").sum()))
    print("Total val:  ", int((split_df["split"] == "val").sum()))

    missing = []
    for label in selected_classes:
        train_count = int(summary.loc[label].get("train", 0)) if label in summary.index else 0
        val_count = int(summary.loc[label].get("val", 0)) if label in summary.index else 0
        if train_count == 0 or val_count == 0:
            missing.append((label, train_count, val_count))

    if missing:
        print("\n[WARNING] Some classes have missing train/val samples:")
        for label, tr, va in missing:
            print(f"  {label}: train={tr}, val={va}")

def main():
    args = parse_args()

    selected_classes = CLASSES_10 if args.classes == 10 else CLASSES_15

    output_dir = f"training_outputs_bilstm_group_split_{args.classes}classes"
    os.makedirs(output_dir, exist_ok=True)

    model_save_path = os.path.join(output_dir, "best_bilstm_group_split.pt")
    label_map_save_path = os.path.join(output_dir, "label_map_group_split.json")
    report_save_path = os.path.join(output_dir, "validation_report_best_model.txt")
    split_csv_path = os.path.join(output_dir, "train_val_split_used.csv")
    history_csv_path = os.path.join(output_dir, "training_history.csv")

    print("Device:", DEVICE)
    print("Classes:", args.classes)
    print("Selected classes:", selected_classes)
    print("Data dir:", args.data_dir)
    print("External test dataset: not used in this script")

    X_train, y_train, X_val, y_val, label_to_idx, idx_to_label, split_df = load_group_split_dataset(
        args.data_dir,
        selected_classes,
    )

    split_df.to_csv(split_csv_path, index=False, encoding="utf-8")
    print_split_summary(split_df, selected_classes)
    print(f"\nSaved split file: {split_csv_path}")

    print("\nX_train:", X_train.shape, "y_train:", y_train.shape)
    print("X_val:  ", X_val.shape, "y_val:  ", y_val.shape)

    label_map = {
        "label_to_index": label_to_idx,
        "index_to_label": {str(i): label for i, label in idx_to_label.items()},
        "classes_used": selected_classes,
        "split_logic": "group 0 train, group 1 validation; only status=good; external test not used",
    }
    with open(label_map_save_path, "w", encoding="utf-8") as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)
    print(f"Saved label map: {label_map_save_path}")

    train_loader = DataLoader(
        GestureDataset(X_train, y_train),
        batch_size=args.batch_size,
        shuffle=True,
    )

    val_loader = DataLoader(
        GestureDataset(X_val, y_val),
        batch_size=args.batch_size,
        shuffle=False,
    )

    #klasu svari
    class_counts = np.bincount(y_train, minlength=len(selected_classes))
    if np.any(class_counts == 0):
        raise ValueError(f"Some classes have 0 train samples: {class_counts}")

    weights = 1.0 / class_counts
    weights = weights / weights.sum()
    class_weights = torch.tensor(weights, dtype=torch.float32).to(DEVICE)

    print("\nTrain class counts:", dict(zip(selected_classes, class_counts.tolist())))
    print("Class weights:", class_weights.detach().cpu().numpy())

    model = BiLSTMClassifier(
        input_size=63,
        hidden_size=128,
        num_layers=2,
        num_classes=len(selected_classes),
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_acc = 0.0
    best_epoch = 0
    history_rows = []

    print("\nStarting training...\n")

    for epoch in range(args.epochs):
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

        train_loss = train_loss / max(len(train_loader), 1)

        val_loss, val_acc, val_f1_macro, val_f1_weighted, _, _ = evaluate(
            model,
            val_loader,
            criterion=criterion,
        )

        history_rows.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "val_f1_macro": val_f1_macro,
            "val_f1_weighted": val_f1_weighted,
        })

        print(
            f"Epoch {epoch+1}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_acc={val_acc:.4f} | "
            f"val_f1_macro={val_f1_macro:.4f} | "
            f"val_f1_weighted={val_f1_weighted:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            torch.save(model.state_dict(), model_save_path)
            print(f" Saved best model by validation accuracy: {model_save_path}")

    pd.DataFrame(history_rows).to_csv(history_csv_path, index=False, encoding="utf-8")
    print(f"\nSaved training history: {history_csv_path}")

    print("\nLoading best validation model for final validation report...")
    best_model = BiLSTMClassifier(
        input_size=63,
        hidden_size=128,
        num_layers=2,
        num_classes=len(selected_classes),
    ).to(DEVICE)

    best_model.load_state_dict(torch.load(model_save_path, map_location=DEVICE))
    best_model.eval()

    _, val_acc, val_f1_macro, val_f1_weighted, all_true, all_preds = evaluate(
        best_model,
        val_loader,
        criterion=None,
    )

    labels_sorted = [idx_to_label[i] for i in range(len(idx_to_label))]
    report = classification_report(
        all_true,
        all_preds,
        target_names=labels_sorted,
        zero_division=0,
    )
    cm = confusion_matrix(all_true, all_preds)

    print("\nBEST VALIDATION MODEL REPORT:\n")
    print(report)
    print("CONFUSION MATRIX:")
    print(cm)
    print(f"\nBest epoch: {best_epoch}")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Final loaded best-model val accuracy: {val_acc:.4f}")

    with open(report_save_path, "w", encoding="utf-8") as f:
        f.write("BEST VALIDATION MODEL REPORT\n\n")
        f.write(f"Classes used: {args.classes}\n")
        f.write(f"Best epoch: {best_epoch}\n")
        f.write(f"Best validation accuracy: {best_val_acc:.4f}\n")
        f.write(f"Validation F1 macro: {val_f1_macro:.4f}\n")
        f.write(f"Validation F1 weighted: {val_f1_weighted:.4f}\n\n")
        f.write(report)
        f.write("\n\nCONFUSION MATRIX:\n")
        f.write(str(cm))

    print(f"\nSaved validation report: {report_save_path}")
    print("Done.")

if __name__ == "__main__":
    main()
