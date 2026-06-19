# NEUROSENSE — FINAL CLINICAL EVALUATION 

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc
)
from sklearn.preprocessing import label_binarize

# CONFIGURE

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR, "model", "stable_v2", "neurosense_stable_v2.pth"
)

DATA_DIR = os.path.join(BASE_DIR, "datasets_new")

OUTPUT_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CLASS_MAP = {
    "Ischemia": 0,
    "Hemorrhage": 1,
    "Tumor": 2,
    "Normal": 3
}

INV_CLASS_MAP = {v: k for k, v in CLASS_MAP.items()}
NUM_CLASSES = 4


# CNN MODEL


class NeuroSENSEStable(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(1, 32, 7, padding=3),
            nn.GELU(),

            nn.Conv1d(32, 64, 5, padding=2),
            nn.GELU(),

            nn.Conv1d(64, 128, 3, padding=1),
            nn.GELU(),

            nn.AdaptiveAvgPool1d(8)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8, 128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, 4)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

# LOAD CNN MODEL

print("\nLoading NeuroSENSE Stable V2 model...")

model = NeuroSENSEStable().to(DEVICE)

state = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
model.load_state_dict(state)

model.eval()
print("✅ Model loaded")

# LOAD DATA INTO MODEL

X_all = []
y_all = []

for cname, label in CLASS_MAP.items():

    file_path = os.path.join(DATA_DIR, f"{cname}_NVS.csv")
    print("Loading:", file_path)

    df = pd.read_csv(file_path)

    X = df.values.astype(np.float32)

    # SAME NORMALIZATION AS TRAINING
    X = (X - X.mean(axis=1, keepdims=True)) / (
        X.std(axis=1, keepdims=True) + 1e-8
    )

    X_all.append(X)
    y_all.extend([label] * len(X))

X_all = np.vstack(X_all)
y_all = np.array(y_all)

print("\nTotal samples:", len(X_all))

# INFERENCE

X_tensor = torch.tensor(X_all).unsqueeze(1).to(DEVICE)

with torch.no_grad():
    logits = model(X_tensor)
    probs = torch.softmax(logits, dim=1).cpu().numpy()
    preds = np.argmax(probs, axis=1)

# BASIC METRICS

accuracy = accuracy_score(y_all, preds)

print("\n===================================")
print("NEUROSENSE FINAL RESULTS")
print("===================================")

print(f"\nAccuracy: {accuracy*100:.2f}%\n")

print(classification_report(
    y_all,
    preds,
    target_names=list(CLASS_MAP.keys()),
    digits=4
))

# CONFUSION MATRIX

cm = confusion_matrix(y_all, preds)

plt.figure()
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.colorbar()

plt.xticks(range(NUM_CLASSES), CLASS_MAP.keys(), rotation=45)
plt.yticks(range(NUM_CLASSES), CLASS_MAP.keys())

for i in range(NUM_CLASSES):
    for j in range(NUM_CLASSES):
        plt.text(j, i, cm[i, j],
                 ha="center", va="center")

plt.xlabel("Predicted")
plt.ylabel("True")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"))
plt.close()

print("✅ Confusion matrix saved")

# CLINICAL METRICS

print("\nClinical Metrics:\n")

for i, cname in INV_CLASS_MAP.items():

    TP = cm[i, i]
    FN = cm[i, :].sum() - TP
    FP = cm[:, i].sum() - TP
    TN = cm.sum() - (TP + FP + FN)

    sensitivity = TP / (TP + FN + 1e-8)
    specificity = TN / (TN + FP + 1e-8)

    print(f"{cname}")
    print(f"  Sensitivity : {sensitivity:.4f}")
    print(f"  Specificity : {specificity:.4f}\n")

# ROC + AUC CURVES

y_bin = label_binarize(y_all, classes=[0,1,2,3])

plt.figure()

for i in range(NUM_CLASSES):

    fpr, tpr, _ = roc_curve(y_bin[:, i], probs[:, i])
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr,
             label=f"{INV_CLASS_MAP[i]} (AUC={roc_auc:.3f})")

plt.plot([0,1],[0,1],'--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves - NeuroSENSE")
plt.legend()

plt.savefig(os.path.join(OUTPUT_DIR, "roc_curves.png"))
plt.close()

print("✅ ROC curves saved")

# PREDICTION DISTRIBUTION

unique, counts = np.unique(preds, return_counts=True)

plt.figure()
plt.bar(
    [INV_CLASS_MAP[u] for u in unique],
    counts
)

plt.title("Prediction Distribution")
plt.ylabel("Samples")

plt.savefig(os.path.join(OUTPUT_DIR, "prediction_distribution.png"))
plt.close()

print("✅ Distribution plot saved")

print("\n✅ FULL EVALUATION COMPLETE")
print("Results saved in:", OUTPUT_DIR)