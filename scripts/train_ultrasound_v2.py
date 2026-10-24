#!/usr/bin/env python3
"""Round 2: BUSI ultrasound — train on train+val combined, TTA evaluation.

Round 1 gave test_acc=0.6522 (val 0.7913) — val/test gap + small data.
This round: 665 train images (train+val), same test set (comparable),
horizontal-flip TTA at eval, slightly longer fine-tune.
"""
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models, transforms
from PIL import Image

DATA_DIR = Path.home() / "workspace/github-fix/MedVision-AI/data/ultrasound/busi"
OUT_DIR = Path.home() / "workspace/github-fix/MedVision-AI/models"
IMG_SIZE = 224
BATCH = 8
EPOCHS_HEAD = 8
EPOCHS_FT = 12
LR_HEAD = 1e-3
LR_FT = 1e-4
SEED = 123

CLASS_NAMES = ["benign", "malignant", "normal"]
CLS2IDX = {c: i for i, c in enumerate(CLASS_NAMES)}

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(OUT_DIR / "us_train_log.txt", "a") as f:
        f.write(line + "\n")


class USImageDataset(Dataset):
    def __init__(self, items, transform):
        self.items = items
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


def build_items():
    # Reproduce round-1 split EXACTLY (same seed 42), then merge train+val
    random.seed(42)
    rows = []
    for cls in CLASS_NAMES:
        for p in sorted((DATA_DIR / cls).glob("*.png")):
            if "_mask" not in p.name:
                rows.append((str(p), CLS2IDX[cls]))
    by_cls = {}
    for path, idx in rows:
        by_cls.setdefault(idx, []).append((path, idx))
    train, test = [], []
    for idx, items in sorted(by_cls.items()):
        random.shuffle(items)
        n = len(items)
        n_test = max(2, int(n * 0.15))
        n_val = max(2, int(n * 0.15))
        test += items[:n_test]
        train += items[n_test:]  # train + val combined
    for name, s in [("train+val", train), ("test", test)]:
        counts = {}
        for _, idx in s:
            counts[CLASS_NAMES[idx]] = counts.get(CLASS_NAMES[idx], 0) + 1
        log(f"{name}: {len(s)} {counts}")
    return train, test


def build_model(num_classes):
    m = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    in_f = m.classifier[0].in_features
    m.classifier = nn.Sequential(
        nn.Linear(in_f, 256),
        nn.Hardswish(),
        nn.Dropout(0.2),
        nn.Linear(256, num_classes),
    )
    return m


def run_epochs(model, train_dl, val_dl, optimizer, scheduler, criterion, device, epochs, tag):
    best_val = 0.0
    for ep in range(epochs):
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
        if scheduler is not None:
            scheduler.step()
        model.eval()
        v_correct, v_n = 0, 0
        with torch.no_grad():
            for xb, yb in val_dl:
                out = model(xb.to(device))
                v_correct += (out.argmax(1) == yb.to(device)).sum().item()
                v_n += len(yb)
        v_acc = v_correct / max(v_n, 1)
        log(f"[{tag}] Epoch {ep+1}/{epochs} | train_loss={run_loss/max(run_n,1):.4f} "
            f"train_acc={run_correct/max(run_n,1):.4f} | val_acc={v_acc:.4f}")
        if v_acc > best_val:
            best_val = v_acc
            torch.save(model.state_dict(), OUT_DIR / "us_model.pt")
            log(f"  -> saved best (val_acc={v_acc:.4f})")
    return best_val


def main():
    t_start = time.time()
    log("=== ROUND 2: train+val combined, TTA eval ===")
    train_items, test_items = build_items()
    # Hold out 10% of train as a small val for checkpoint selection
    random.seed(SEED)
    random.shuffle(train_items)
    n_val = int(len(train_items) * 0.10)
    val_items = train_items[:n_val]
    train_items = train_items[n_val:]
    log(f"train: {len(train_items)}, val: {len(val_items)}, test: {len(test_items)}")

    tf_train = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.RandomAffine(0, translate=(0.08, 0.08), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.25, contrast=0.25),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    tf_eval = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    labels = [idx for _, idx in train_items]
    cls_counts = np.bincount(labels, minlength=len(CLASS_NAMES)).astype(float)
    cls_w = 1.0 / np.maximum(cls_counts, 1)
    sampler = WeightedRandomSampler([cls_w[i] for _, i in train_items], len(train_items), replacement=True)
    train_dl = DataLoader(USImageDataset(train_items, tf_train), batch_size=BATCH, sampler=sampler, num_workers=0)
    val_dl = DataLoader(USImageDataset(val_items, tf_eval), batch_size=BATCH, num_workers=0)
    test_dl = DataLoader(USImageDataset(test_items, tf_eval), batch_size=BATCH, num_workers=0)

    device = torch.device("cpu")
    model = build_model(len(CLASS_NAMES)).to(device)
    criterion = nn.CrossEntropyLoss()

    for p in model.features.parameters():
        p.requires_grad = False
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR_HEAD)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS_HEAD)
    best_val = run_epochs(model, train_dl, val_dl, optimizer, scheduler, criterion, device, EPOCHS_HEAD, "head")

    log("Unfreezing backbone for fine-tuning")
    for p in model.features.parameters():
        p.requires_grad = True
    optimizer = torch.optim.Adam(model.parameters(), lr=LR_FT)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS_FT)
    best_val = max(best_val, run_epochs(model, train_dl, val_dl, optimizer, scheduler, criterion, device, EPOCHS_FT, "finetune"))

    # ── Test with TTA (original + horizontal flip) ──
    model.load_state_dict(torch.load(OUT_DIR / "us_model.pt", map_location=device))
    model.eval()
    all_pred, all_true = [], []
    with torch.no_grad():
        for xb, yb in test_dl:
            xb = xb.to(device)
            probs = torch.softmax(model(xb), dim=1)
            probs_flip = torch.softmax(model(torch.flip(xb, dims=[3])), dim=1)
            out = (probs + probs_flip) / 2
            all_pred += out.argmax(1).tolist()
            all_true += yb.tolist()
    all_pred = np.array(all_pred)
    all_true = np.array(all_true)
    acc = (all_pred == all_true).mean()
    per_class, cm = {}, np.zeros((3, 3), dtype=int)
    for i, c in enumerate(CLASS_NAMES):
        tp = int(((all_pred == i) & (all_true == i)).sum())
        fp = int(((all_pred == i) & (all_true != i)).sum())
        fn = int(((all_pred != i) & (all_true == i)).sum())
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        per_class[c] = {"precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4),
                        "n": int((all_true == i).sum())}
        for t in range(3):
            for p in range(3):
                cm[t, p] = int(((all_true == t) & (all_pred == p)).sum())

    metrics = {"test_accuracy": round(float(acc), 4), "per_class": per_class,
               "confusion_matrix": cm.tolist(), "class_order": CLASS_NAMES,
               "n_test": len(all_true), "tta": "hflip", "round": 2}
    (OUT_DIR / "us_metrics.json").write_text(json.dumps(metrics, indent=2))
    log(f"DONE in {(time.time()-t_start)/60:.1f} min | test_acc(TTA)={acc:.4f}")
    log(f"per-class: {json.dumps({c: v['f1'] for c, v in per_class.items()})}")


if __name__ == "__main__":
    main()
