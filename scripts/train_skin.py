#!/usr/bin/env python3
"""Train skin lesion classifier on HAM10000.

Model: MobileNetV3-Small (ImageNet pretrained), backbone FROZEN,
       only classifier head trained -> fast on CPU, small (<20MB).
7 classes: mel, nv, bcc, akiec, bkl, df, vasc
"""
import csv
import json
import os
import random
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models, transforms
from PIL import Image

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR = Path.home() / "workspace" / "data" / "ham10000"
OUT_DIR = Path.home() / "workspace" / "github-fix" / "MedVision-AI" / "models"
IMG_SIZE = 224
BATCH = 16
EPOCHS = 6
LR = 1e-3
SEED = 42

CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
CLS2IDX = {c: i for i, c in enumerate(CLASS_NAMES)}
FULL_NAMES = {
    "akiec": "Actinic keratosis / Bowen's disease",
    "bcc": "Basal cell carcinoma",
    "bkl": "Benign keratosis",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic nevus",
    "vasc": "Vascular lesion",
}

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(OUT_DIR / "skin_train_log.txt", "a") as f:
        f.write(line + "\n")


# ── 1. Extract zips ───────────────────────────────────────────────────────────
def extract():
    for zname in ["HAM10000_images_part_1.zip", "HAM10000_images_part_2.zip",
                  "skin-cancer-mnist-ham10000.zip"]:
        zpath = DATA_DIR / zname
        if zpath.exists():
            log(f"Extracting {zname} ...")
            with zipfile.ZipFile(zpath) as z:
                z.extractall(DATA_DIR)
            log(f"Done extracting {zname}")
    # find image dir + metadata
    img_dirs = [d for d in DATA_DIR.iterdir() if d.is_dir() and "HAM10000_images" in d.name]
    meta = DATA_DIR / "HAM10000_metadata.csv"
    if not meta.exists():
        # maybe nested one level
        for p in DATA_DIR.rglob("HAM10000_metadata.csv"):
            meta = p
            break
    log(f"Image dirs: {[d.name for d in img_dirs]}, metadata: {meta}")
    return img_dirs, meta


# ── 2. Dataset ────────────────────────────────────────────────────────────────
class HamDataset(Dataset):
    def __init__(self, items, transform):
        self.items = items
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


def build_splits(img_dirs, meta):
    # map image_id -> file path
    id2path = {}
    for d in img_dirs:
        for p in d.glob("*.jpg"):
            id2path[p.stem] = p

    rows = []
    with open(meta) as f:
        for r in csv.DictReader(f):
            dx = r["dx"].strip()
            p = id2path.get(r["image_id"].strip())
            if p and dx in CLS2IDX:
                rows.append((str(p), CLS2IDX[dx], dx))

    log(f"Total usable images: {len(rows)}")
    # stratified 80/10/10
    by_cls = {}
    for path, idx, dx in rows:
        by_cls.setdefault(idx, []).append((path, idx))
    train, val, test = [], [], []
    for idx, items in by_cls.items():
        random.shuffle(items)
        n = len(items)
        n_test = max(1, int(n * 0.10))
        n_val = max(1, int(n * 0.10))
        test += items[:n_test]
        val += items[n_test:n_test + n_val]
        train += items[n_test + n_val:]
    for name, s in [("train", train), ("val", val), ("test", test)]:
        counts = {}
        for _, idx in s:
            counts[CLASS_NAMES[idx]] = counts.get(CLASS_NAMES[idx], 0) + 1
        log(f"{name}: {len(s)} {counts}")
    return train, val, test


# ── 3. Model ──────────────────────────────────────────────────────────────────
def build_model():
    m = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    # Freeze backbone
    for p in m.features.parameters():
        p.requires_grad = False
    # Replace classifier head
    in_f = m.classifier[0].in_features
    m.classifier = nn.Sequential(
        nn.Linear(in_f, 256),
        nn.Hardswish(),
        nn.Dropout(0.2),
        nn.Linear(256, len(CLASS_NAMES)),
    )
    return m


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "skin_train_log.txt").write_text("")
    t_start = time.time()

    img_dirs, meta = extract()
    train_items, val_items, test_items = build_splits(img_dirs, meta)

    tf_train = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(25),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    tf_eval = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # Weighted sampler for class imbalance
    labels = [idx for _, idx in train_items]
    cls_counts = np.bincount(labels, minlength=len(CLASS_NAMES)).astype(float)
    cls_w = 1.0 / np.maximum(cls_counts, 1)
    samp_w = [cls_w[idx] for _, idx in train_items]
    sampler = WeightedRandomSampler(samp_w, len(samp_w), replacement=True)

    train_dl = DataLoader(HamDataset(train_items, tf_train), batch_size=BATCH,
                          sampler=sampler, num_workers=0)
    val_dl = DataLoader(HamDataset(val_items, tf_eval), batch_size=BATCH,
                        num_workers=0)
    test_dl = DataLoader(HamDataset(test_items, tf_eval), batch_size=BATCH,
                         num_workers=0)

    device = torch.device("cpu")
    model = build_model().to(device)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    log(f"Model: mobilenet_v3_small | trainable {n_trainable/1e6:.2f}M / total {n_total/1e6:.2f}M params")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val = 0.0
    for epoch in range(EPOCHS):
        model.train()
        run_loss, run_correct, run_n = 0.0, 0, 0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            run_loss += loss.item() * len(xb)
            run_correct += (out.argmax(1) == yb).sum().item()
            run_n += len(xb)
        scheduler.step()

        # val
        model.eval()
        v_correct, v_n = 0, 0
        with torch.no_grad():
            for xb, yb in val_dl:
                out = model(xb.to(device))
                v_correct += (out.argmax(1) == yb.to(device)).sum().item()
                v_n += len(yb)
        v_acc = v_correct / v_n
        log(f"Epoch {epoch+1}/{EPOCHS} | train_loss={run_loss/run_n:.4f} "
            f"train_acc={run_correct/run_n:.4f} | val_acc={v_acc:.4f}")
        if v_acc > best_val:
            best_val = v_acc
            torch.save(model.state_dict(), OUT_DIR / "skin_model.pt")
            log(f"  -> saved best (val_acc={v_acc:.4f})")

    # ── 4. Test evaluation ────────────────────────────────────────────────────
    model.load_state_dict(torch.load(OUT_DIR / "skin_model.pt", map_location=device))
    model.eval()
    all_pred, all_true = [], []
    with torch.no_grad():
        for xb, yb in test_dl:
            out = model(xb.to(device))
            all_pred += out.argmax(1).cpu().tolist()
            all_true += yb.tolist()

    # metrics (no sklearn — compute manually)
    n = len(all_true)
    acc = sum(p == t for p, t in zip(all_pred, all_true)) / n
    cm = [[0] * len(CLASS_NAMES) for _ in CLASS_NAMES]
    for t, p in zip(all_true, all_pred):
        cm[t][p] += 1
    per_class = {}
    for i, cname in enumerate(CLASS_NAMES):
        tp = cm[i][i]
        fp = sum(cm[r][i] for r in range(len(CLASS_NAMES))) - tp
        fn = sum(cm[i]) - tp
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per_class[cname] = {
            "full_name": FULL_NAMES[cname],
            "support": sum(cm[i]),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
        }
    macro_f1 = round(sum(v["f1"] for v in per_class.values()) / len(CLASS_NAMES), 4)

    metrics = {
        "test_accuracy": round(acc, 4),
        "macro_f1": macro_f1,
        "n_test": n,
        "per_class": per_class,
        "confusion_matrix": cm,
        "class_order": CLASS_NAMES,
    }
    with open(OUT_DIR / "skin_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    with open(OUT_DIR / "skin_classes.json", "w") as f:
        json.dump({"classes": CLASS_NAMES, "full_names": FULL_NAMES}, f, indent=2)

    info = {
        "architecture": "mobilenet_v3_small (ImageNet pretrained, backbone frozen)",
        "input_size": [IMG_SIZE, IMG_SIZE],
        "num_classes": len(CLASS_NAMES),
        "class_names": CLASS_NAMES,
        "test_accuracy": round(acc, 4),
        "macro_f1": macro_f1,
        "dataset": "HAM10000 (10,015 dermatoscopic images)",
        "weights_file": "skin_model.pt",
    }
    with open(OUT_DIR / "skin_model_info.json", "w") as f:
        json.dump(info, f, indent=2)

    size_mb = (OUT_DIR / "skin_model.pt").stat().st_size / 1e6
    elapsed = (time.time() - t_start) / 60
    log(f"DONE | test_acc={acc:.4f} macro_f1={macro_f1:.4f} "
        f"model_size={size_mb:.1f}MB time={elapsed:.1f}min")


if __name__ == "__main__":
    main()
