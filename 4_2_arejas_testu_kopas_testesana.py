import os
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

#iestatijumi
TEST_DIR = "dataset_landmarks_test"
TEST_METADATA = os.path.join(TEST_DIR, "landmarks_metadata.csv")
SEQUENCE_LENGTH = 48
BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

#same modelis no apmacibas
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

#pytorch x ievaddati, y pareizas atbildes. sagatavo datus priekš PyTorch
class GestureDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

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

#load klasu karti
def load_label_map(label_map_path):
    if not os.path.exists(label_map_path):
        raise FileNotFoundError(f"Label map nav atrasts: {label_map_path}")

    with open(label_map_path, "r", encoding="utf-8") as f:
        label_map = json.load(f)

    label_to_idx = label_map["label_to_index"]
    idx_to_label = {int(k): v for k, v in label_map["index_to_label"].items()}
    labels_sorted = [idx_to_label[i] for i in range(len(idx_to_label))]

    return label_to_idx, idx_to_label, labels_sorted, label_map

def load_external_test_dataset(data_dir, metadata_path, label_to_idx, allowed_status=("good",)):
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Test metadata nav atrasts: {metadata_path}")

    df = pd.read_csv(metadata_path)

    #tikai labas kvalites ieraksti
    df = df[df["status"].isin(list(allowed_status))].copy()

    df = df[df["label"].isin(label_to_idx.keys())].copy()

    if len(df) == 0:
        raise ValueError("Pēc filtrēšanas testu kopā nav neviena ieraksta.")

    X = []
    y = []
    used_rows = []

    for _, row in df.iterrows():
        label = row["label"]
        npz_name = row["npz_name"]

        npz_path = os.path.join(data_dir, label, npz_name)
        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"Trūkst .npz fails: {npz_path}")

        data = np.load(npz_path, allow_pickle=True)
        seq = data["sequence"]

        seq = normalize_sequence_oldnorm(seq)
        seq = resample_sequence(seq, SEQUENCE_LENGTH)
        seq = seq.reshape(SEQUENCE_LENGTH, -1)

        X.append(seq)
        y.append(label_to_idx[label])

        used_rows.append({
            "label": label,
            "npz_name": npz_name,
            "source_folder": row.get("source_folder", ""),
            "video_name": row.get("video_name", ""),
            "status": row.get("status", ""),
            "detected_frames": row.get("detected_frames", ""),
            "total_frames": row.get("total_frames", ""),
            "detection_ratio": row.get("detection_ratio", ""),
        })

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    used_df = pd.DataFrame(used_rows)

    return X, y, used_df

def save_confusion_matrix_images(cm, labels, output_dir):
    #parasta sadalijuma matrica
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm)

    ax.set_title("Confusion Matrix - ārējā testu kopa")
    ax.set_xlabel("Prognozētā zīme")
    ax.set_ylabel("Patiesā zīme")

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    fig.colorbar(im, ax=ax)
    fig.tight_layout()

    cm_path = os.path.join(output_dir, "external_test_confusion_matrix.png")
    fig.savefig(cm_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    #normalizeta sadalijuma matrica, %
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm_norm)

    ax.set_title("Normalizēta Confusion Matrix - ārējā testu kopa")
    ax.set_xlabel("Prognozētā zīme")
    ax.set_ylabel("Patiesā zīme")

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            ax.text(j, i, f"{cm_norm[i, j]:.1%}", ha="center", va="center")

    fig.colorbar(im, ax=ax)
    fig.tight_layout()

    cm_norm_path = os.path.join(output_dir, "external_test_confusion_matrix_normalized.png")
    fig.savefig(cm_norm_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return cm_path, cm_norm_path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--classes",
        type=int,
        choices=[10, 15],
        required=True,
        help="Kuru modeli testēt: 10 vai 15 klases."
    )
    args = parser.parse_args()

    classes = args.classes

    model_dir = f"training_outputs_bilstm_group_split_{classes}classes"
    model_path = os.path.join(model_dir, "best_bilstm_group_split.pt")
    label_map_path = os.path.join(model_dir, "label_map_group_split.json")

    output_dir = os.path.join(model_dir, "external_test_results")
    os.makedirs(output_dir, exist_ok=True)

    print("============================================================")
    print(f"ĀRĒJĀ TESTĒŠANA | {classes} klases")
    print("============================================================")
    print(f"Model dir:      {model_dir}")
    print(f"Model path:     {model_path}")
    print(f"Label map path: {label_map_path}")
    print(f"Test dir:       {TEST_DIR}")
    print(f"Device:         {DEVICE}")
    print("============================================================")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modelis nav atrasts: {model_path}")

    label_to_idx, idx_to_label, labels_sorted, label_map = load_label_map(label_map_path)

    print("\nKlases, kas tiek testētas:")
    for i, label in enumerate(labels_sorted):
        print(f"  {i}: {label}")

    print("\nIelādēta ārējo testu kopa...")
    X_test, y_test, used_df = load_external_test_dataset(
        data_dir=TEST_DIR,
        metadata_path=TEST_METADATA,
        label_to_idx=label_to_idx,
        allowed_status=("good",)
    )

    print(f"Test samples: {len(X_test)}")
    print("\nTest ierakstu skaits pa klasēm:")
    print(used_df["label"].value_counts().sort_index())

    used_split_path = os.path.join(output_dir, "external_test_used_files.csv")
    used_df.to_csv(used_split_path, index=False, encoding="utf-8-sig")

    test_loader = DataLoader(
        GestureDataset(X_test, y_test),
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    model = BiLSTMClassifier(
        input_size=63,
        hidden_size=128,
        num_layers=2,
        num_classes=len(label_to_idx)
    ).to(DEVICE)

    state_dict = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()

    print("\nModelis ielādēts. Sākta testēšana...")

    all_preds = []
    all_true = []
    all_conf = []

    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(DEVICE)

            logits = model(xb)
            probs = torch.softmax(logits, dim=1)

            pred = torch.argmax(probs, dim=1).cpu().numpy()
            conf = torch.max(probs, dim=1).values.cpu().numpy()

            all_preds.extend(pred)
            all_true.extend(yb.numpy())
            all_conf.extend(conf)

    all_preds = np.array(all_preds)
    all_true = np.array(all_true)
    all_conf = np.array(all_conf)

    acc = accuracy_score(all_true, all_preds)
    f1_macro = f1_score(all_true, all_preds, average="macro")
    f1_weighted = f1_score(all_true, all_preds, average="weighted")

    report = classification_report(
        all_true,
        all_preds,
        target_names=labels_sorted,
        digits=4
    )

    cm = confusion_matrix(all_true, all_preds)

    print("\n============================================================")
    print("ĀRĒJĀ TESTA REZULTĀTI")
    print("============================================================")
    print(f"Accuracy:    {acc:.4f}")
    print(f"F1 macro:    {f1_macro:.4f}")
    print(f"F1 weighted: {f1_weighted:.4f}")
    print("\nClassification report:\n")
    print(report)
    print("\nConfusion matrix:\n")
    print(cm)

    #saglabat report
    report_path = os.path.join(output_dir, "external_test_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("EXTERNAL TEST REPORT\n\n")
        f.write(f"Classes used: {classes}\n")
        f.write(f"Model path: {model_path}\n")
        f.write(f"Label map path: {label_map_path}\n")
        f.write(f"External test dir: {TEST_DIR}\n")
        f.write(f"External test metadata: {TEST_METADATA}\n")
        f.write("External test used only here, after training/validation.\n\n")

        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"F1 macro: {f1_macro:.4f}\n")
        f.write(f"F1 weighted: {f1_weighted:.4f}\n")
        f.write(f"Mean softmax confidence: {float(np.mean(all_conf)):.4f}\n")
        f.write(f"Median softmax confidence: {float(np.median(all_conf)):.4f}\n\n")

        f.write("CLASSIFICATION REPORT\n\n")
        f.write(report)
        f.write("\n\nCONFUSION MATRIX:\n")
        f.write(str(cm))
        f.write("\n")

    predictions_df = used_df.copy()
    predictions_df["true_index"] = all_true
    predictions_df["pred_index"] = all_preds
    predictions_df["true_label"] = [labels_sorted[i] for i in all_true]
    predictions_df["pred_label"] = [labels_sorted[i] for i in all_preds]
    predictions_df["confidence"] = all_conf
    predictions_df["correct"] = predictions_df["true_index"] == predictions_df["pred_index"]

    predictions_path = os.path.join(output_dir, "external_test_predictions.csv")
    predictions_df.to_csv(predictions_path, index=False, encoding="utf-8-sig")

    cm_txt_path = os.path.join(output_dir, "external_test_confusion_matrix.txt")
    with open(cm_txt_path, "w", encoding="utf-8") as f:
        f.write(str(cm))

    cm_csv_path = os.path.join(output_dir, "external_test_confusion_matrix.csv")
    pd.DataFrame(cm, index=labels_sorted, columns=labels_sorted).to_csv(
        cm_csv_path,
        encoding="utf-8-sig"
    )

    cm_path, cm_norm_path = save_confusion_matrix_images(cm, labels_sorted, output_dir)

    print("\nSaglabātie faili:")
    print(f"  Report:              {report_path}")
    print(f"  Used test files:     {used_split_path}")
    print(f"  Predictions:         {predictions_path}")
    print(f"  Confusion txt:       {cm_txt_path}")
    print(f"  Confusion csv:       {cm_csv_path}")
    print(f"  Confusion image:     {cm_path}")
    print(f"  Confusion norm img:  {cm_norm_path}")
    print("\nDone.")

if __name__ == "__main__":
    main()
