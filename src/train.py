import os
import json
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import argparse
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from collections import Counter

from src.preprocess import preprocess_image
from src.model      import CRNN

MEDICINE_DIR = "datasets/2Doctor's Prescription/Handwritten Medicine Data Dataset"
NIST_DIR     = "datasets/1Handwritten Text/HandWritten-NIST"
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


class MedicineDataset(Dataset):
    def __init__(self, records, label_encoder, augment=False):
        self.records       = records
        self.label_encoder = label_encoder
        self.augment       = augment
        print(f"  Dataset size: {len(self.records)}")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        img_path, identity = self.records[idx]
        try:
            img = preprocess_image(img_path, target_w=224, target_h=224)
        except Exception:
            img = np.ones((224, 224, 1), dtype=np.float32)

        if self.augment:
            img = self._augment(img)

        img   = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)
        label = self.label_encoder.transform([identity])[0]
        return img, torch.tensor(label, dtype=torch.long)

    def _augment(self, img):
        # Brightness
        img = np.clip(img * random.uniform(0.6, 1.4), 0, 1)
        # Noise
        if random.random() > 0.4:
            img = np.clip(
                img + np.random.normal(0, 0.03, img.shape).astype(np.float32),
                0, 1
            )
        # Horizontal shift
        if random.random() > 0.4:
            shift = random.randint(-15, 15)
            img   = np.roll(img, shift, axis=1)
        # Vertical shift
        if random.random() > 0.4:
            shift = random.randint(-8, 8)
            img   = np.roll(img, shift, axis=0)
        # Random erasing
        if random.random() > 0.5:
            h, w  = img.shape[:2]
            x1    = random.randint(0, w - 20)
            y1    = random.randint(0, h - 10)
            img[y1:y1+10, x1:x1+20] = 1.0
        return img


def load_all_records(csv_path, img_dir):
    df      = pd.read_csv(csv_path)
    records = []
    for _, row in df.iterrows():
        fname    = str(row["FILENAME"]).strip()
        identity = str(row["IDENTITY"]).strip().lower()
        fpath    = os.path.join(img_dir, fname)
        if os.path.exists(fpath):
            records.append((fpath, identity))
    return records


def build_label_encoder(all_records):
    labels = sorted(set(r[1] for r in all_records))
    le     = LabelEncoder()
    le.fit(labels)
    os.makedirs("models", exist_ok=True)
    with open("models/label_classes.json", "w") as f:
        json.dump(le.classes_.tolist(), f)
    print(f"  {len(labels)} unique drug names saved")
    return le


def train_classifier(epochs=150, batch_size=32, lr=5e-5):
    print(f"\n── Classification Training ─────────────────────")
    print(f"Device: {DEVICE}")

    img_dir = os.path.join(MEDICINE_DIR, "images")

    # ── Load train + val combined ──────────────────────────────────
    print("\nLoading records...")
    train_records = load_all_records(
        os.path.join(MEDICINE_DIR, "train_clean_final.csv"),
        os.path.join(img_dir, "train")
    )
    val_records = load_all_records(
        os.path.join(MEDICINE_DIR, "validate_clean_final.csv"),
        os.path.join(img_dir, "validate")
    )

    all_records = train_records + val_records
    print(f"Total samples (train+val): {len(all_records)}")

    # ── Build label encoder ────────────────────────────────────────
    le = build_label_encoder(all_records)

    # ── Stratified 90/10 split ─────────────────────────────────────
    identities = [r[1] for r in all_records]
    counts     = Counter(identities)
    single     = [k for k, v in counts.items() if v < 2]

    if single:
        print(f"  {len(single)} classes with 1 sample — using random split")
        random.shuffle(all_records)
        split      = int(len(all_records) * 0.9)
        train_recs = all_records[:split]
        val_recs   = all_records[split:]
    else:
        paths  = [r[0] for r in all_records]
        labels = [r[1] for r in all_records]
        p_tr, p_v, l_tr, l_v = train_test_split(
            paths, labels,
            test_size=0.1,
            stratify=labels,
            random_state=42
        )
        train_recs = list(zip(p_tr, l_tr))
        val_recs   = list(zip(p_v,  l_v))

    print(f"Train: {len(train_recs)}  Val: {len(val_recs)}")

    # ── Datasets and loaders ───────────────────────────────────────
    train_ds = MedicineDataset(train_recs, le, augment=True)
    val_ds   = MedicineDataset(val_recs,   le, augment=False)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size,
        shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size,
        shuffle=False, num_workers=0
    )

    # ── Model ──────────────────────────────────────────────────────
    num_classes = len(le.classes_)
    print(f"\nNum classes: {num_classes}")

    model     = CRNN(num_classes=num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr * 10,
        steps_per_epoch=len(train_loader),
        epochs=epochs
    )

    # ── Training loop ──────────────────────────────────────────────
    weights_out  = "models/phase2_weights.pt"
    best_val_acc = 0.0
    patience_cnt = 0
    patience     = 20

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        t_loss = t_correct = t_total = 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            out  = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            scheduler.step()
            t_loss    += loss.item()
            t_correct += (out.argmax(1) == labels).sum().item()
            t_total   += labels.size(0)

        # Validate
        model.eval()
        v_loss = v_correct = v_total = 0

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                out   = model(imgs)
                loss  = criterion(out, labels)
                v_loss    += loss.item()
                v_correct += (out.argmax(1) == labels).sum().item()
                v_total   += labels.size(0)

        t_acc = t_correct / t_total * 100
        v_acc = v_correct / v_total * 100
        t_avg = t_loss    / len(train_loader)
        v_avg = v_loss    / len(val_loader)

        print(f"Epoch {epoch:3d}/{epochs} "
              f"| Train Loss: {t_avg:.4f} Acc: {t_acc:.1f}% "
              f"| Val Loss: {v_avg:.4f} Acc: {v_acc:.1f}%")

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            patience_cnt = 0
            torch.save(model.state_dict(), weights_out)
            print(f"  ✓ Saved best → {weights_out} "
                  f"(val acc: {v_acc:.1f}%)")
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    print(f"\nDone. Best val acc: {best_val_acc:.1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch",  type=int, default=32)
    parser.add_argument("--lr",     type=float, default=5e-5)
    args = parser.parse_args()
    train_classifier(args.epochs, args.batch, args.lr)