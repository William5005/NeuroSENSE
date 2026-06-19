# NEUROSENSE — STABLE TRAINING V2

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

torch.backends.cudnn.benchmark = True

# DEVICE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# CONFIGURE

POINTS = 101
BATCH_SIZE = 256
EPOCHS = 80
LR = 3e-4
PATIENCE = 12

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "datasets_nanovna_domain")


# LOAD DATASETS

def load_class(csv_path, label):

    df = pd.read_csv(csv_path)
    X = df.values.astype(np.float32)

    # SAME NORMALIZATION AS BEFORE
    X = (X - X.mean(axis=1, keepdims=True)) / (
        X.std(axis=1, keepdims=True) + 1e-8
    )

    y = np.full(len(X), label)
    print(os.path.basename(csv_path), len(X))

    return X, y


datasets = {
    "Ischemia":0,
    "Hemorrhage":1,
    "Tumor":2,
    "Normal":3
}

X_all, y_all = [], []

for name, label in datasets.items():
    path = os.path.join(DATA_DIR, f"{name}_NV.csv")
    Xc, yc = load_class(path, label)
    X_all.append(Xc)
    y_all.append(yc)

X = np.vstack(X_all)
y = np.concatenate(y_all)

print("Total samples:", len(X))

# SPLIT TRAIN/VAL

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.15, stratify=y, random_state=42
)

# DATASET LOADER

class S21Dataset(Dataset):

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        x = torch.tensor(self.X[i]).unsqueeze(0)
        y = torch.tensor(self.y[i])
        return x.float(), y.long()

train_loader = DataLoader(S21Dataset(X_train,y_train),
                          batch_size=BATCH_SIZE,
                          shuffle=True)

val_loader = DataLoader(S21Dataset(X_val,y_val),
                        batch_size=BATCH_SIZE)

# CNN MODEL ARCHITECTURE

class NeuroSENSEStable(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(1,32,7,padding=3),
            nn.GELU(),

            nn.Conv1d(32,64,5,padding=2),
            nn.GELU(),

            nn.Conv1d(64,128,3,padding=1),
            nn.GELU(),

            nn.AdaptiveAvgPool1d(8)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*8,128),
            nn.GELU(),
            nn.Dropout(0.3),

            nn.Linear(128,4)
        )

    def forward(self,x):
        x = self.features(x)
        return self.classifier(x)

model = NeuroSENSEStable().to(device)

# LOSS + OPTIMIZER

criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=1e-4
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', patience=4
)

# TRAIN LOOP

best_acc = 0
patience_counter = 0

MODEL_DIR = os.path.join(BASE_DIR,"model","stable_v2")
os.makedirs(MODEL_DIR,exist_ok=True)

SAVE_PATH = os.path.join(MODEL_DIR,"neurosense_stable_v2.pth")

for epoch in range(EPOCHS):

    model.train()
    correct,total = 0,0

    for xb,yb in train_loader:

        xb,yb = xb.to(device),yb.to(device)

        optimizer.zero_grad()

        preds = model(xb)
        loss = criterion(preds,yb)

        loss.backward()

        # gradient clipping prevents collapse
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)

        optimizer.step()

        pred_classes = preds.argmax(1)

        # COLLAPSE MONITOR
        unique,counts = torch.unique(pred_classes,
                                     return_counts=True)
        print("Batch distribution:",
              dict(zip(unique.cpu().numpy(),
                       counts.cpu().numpy())))

        correct += (pred_classes==yb).sum().item()
        total += yb.size(0)

    train_acc = correct/total

    # ---------- VALIDATION ----------
    model.eval()
    correct,total = 0,0

    with torch.no_grad():
        for xb,yb in val_loader:
            xb,yb = xb.to(device),yb.to(device)
            preds = model(xb)
            correct += (preds.argmax(1)==yb).sum().item()
            total += yb.size(0)

    val_acc = correct/total
    scheduler.step(val_acc)

    print(f"\nEpoch {epoch+1} | Train {train_acc:.4f} | Val {val_acc:.4f}")

    if val_acc > best_acc:
        best_acc = val_acc
        patience_counter = 0
        torch.save(model.state_dict(), SAVE_PATH)
        print("✅ Best model saved")

    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print("Early stopping")
            break

print("\n✅ TRAINING COMPLETE")