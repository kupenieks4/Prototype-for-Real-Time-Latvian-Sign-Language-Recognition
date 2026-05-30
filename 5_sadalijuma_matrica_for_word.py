import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

#iestatijumi
REPORT_FILE = "training_outputs_bilstm_trimmed_externaltest/final_report_test_best_model.txt"   #labaka modela parskata fails
OUTPUT_IMAGE = "confusion_matrix.png"
OUTPUT_IMAGE_NORM = "confusion_matrix_normalized.png"

#zimju nosaukumi tada pasa seciba, ka modeli
labels = [
    "A",
    "AA_long",
    "B",
    "U",
    "ata",
    "es",
    "labdien",
    "nauda",
    "ne",
    "ok",
    "paldies",
    "plane",
    "telefons",
    "tualete",
    "vajag",
]

#teksta nolasisana no rezultata faila
with open(REPORT_FILE, "r", encoding="utf-8") as f:
    content = f.read()

#sadalijuma matricas atrasana
matrix_match = re.search(r"CONFUSION MATRIX:\n(.*?)(?=\n\n|\Z)", content, re.DOTALL)
if not matrix_match:
    raise ValueError("Nevarēja atrast confusion matrix tekstā!")

matrix_text = matrix_match.group(1).strip()
lines = matrix_text.split("\n")

cm = []
for line in lines:
    numbers = [int(x) for x in re.findall(r"-?\d+", line)]
    if numbers:
        cm.append(numbers)

cm = np.array(cm)

print(f"Ielādēta confusion matrix ar izmēru {cm.shape}")

#attela sataisisana
plt.figure(figsize=(10, 8))

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
            xticklabels=labels, yticklabels=labels,
            cbar_kws={'label': 'Skaits'})

plt.title("Confusion Matrix\n(Bi-LSTM modelis - 15 latviešu zīmju valodas žesti)", 
          fontsize=14, pad=20)
plt.xlabel("Prognozētā zīme", fontsize=12)
plt.ylabel("Patiesā zīme", fontsize=12)

plt.tight_layout()
plt.savefig(OUTPUT_IMAGE, dpi=300, bbox_inches='tight')
print(f"Saglabāts: {OUTPUT_IMAGE}")

#procentuala/normalizeta matrica
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(10, 8))
sns.heatmap(cm_norm, annot=True, fmt=".1%", cmap="Blues",
            xticklabels=labels, yticklabels=labels,
            cbar_kws={'label': 'Procenti (%)'})

plt.title("Normalizēta Confusion Matrix (procentos)\n(Bi-LSTM modelis - 15 latviešu zīmju valodas žesti)", 
          fontsize=14, pad=20)
plt.xlabel("Prognozētā zīme", fontsize=12)
plt.ylabel("Patiesā zīme", fontsize=12)

plt.tight_layout()
plt.savefig(OUTPUT_IMAGE_NORM, dpi=300, bbox_inches='tight')
print(f"Saglabāts arī normalizētais variants: {OUTPUT_IMAGE_NORM}")