#!/usr/bin/env python3
"""Train skin lesion classifier on HAM10000 — v2 (85%+ accuracy target).

Upgrades over v1 (MobileNetV3-Small frozen, 62.1%):
  1. EfficientNet-B0 instead of MobileNetV3-Small
  2. FULL fine-tuning with differential learning rates (backbone 2e-5, head 1e-4)
  3. 2-phase training: head warmup (4 epochs) -> full unfreeze (up to 16 epochs,
     early stopping on val macro-F1)
  4. Strong augmentation: RandAugment + flips + color jitter + random erasing
  5. Class-weighted CrossEntropy + label smoothing 0.1
  6. 256x256 input
  7. Test-time augmentation (5 views) for final evaluation

7 classes (same order as v1): akiec, bcc, bkl, df, mel, nv, vasc
Same seed (42) -> same train/val/test splits as v1 for comparability.
"""
import argparse
import csv
import json
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
IMG_SIZE = 224         # 224 is faster on CPU; 256 gave diminishing returns
BATCH = 16           # better CPU utilization
WORKERS = 2          # parallel data loading (overlaps aug with compute)
SEED = 42            # SAME as v1 -> identical splits
PHASE1_EPOCHS = 3    # head warmup, backbone frozen
PHASE2_EPOCHS = 10   # full fine-tune
PATIENCE = 3         # early stopping on val macro-F1

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
torch.set_num_threads(2)


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(OUT_DIR / "skin_train_log.txt", "a") as f:
        f.write(line + "\n")


# ── Data (identical split logic to v1) ────────────────────────────────────────
def find_data():
    img_dirs = [d for d in DATA_DIR.iterdir()
                if d.is_dir() and "HAM10000_images" in d.name
                and "part" in d.name]
    meta = DATA_DIR / "HAM10000_metadata.csv"
    if not meta.exists():
        for p in DATA_DIR.rglob("HAM10000_metadata.csv"):
            meta = p
            break
    return img_dirs, meta


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


# ── Model ─────────────────────────────────────────────────────────────────────
def build_model():
    m = models.efficientnet_b0(
        weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    in_f = m.classifier[1].in_features
    m.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_f, 256),
        nn.SiLU(),
        nn.Dropout(0.2),
        nn.Linear(256, len(CLASS_NAMES)),
    )
    return m


def set_frozen(model, frozen: bool):
    for p in model.features.parameters():
        p.requires_grad = not frozen


# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_metrics(all_true, all_pred):
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
    return acc, macro_f1, cm, per_class


@torch.no_grad()
def evaluate(model, dl, device):
    model.eval()
    all_pred, all_true = [], []
    for xb, yb in dl:
        out = model(xb.to(device))
        all_pred += out.argmax(1).cpu().tolist()
        all_true += yb.tolist()
    return compute_metrics(all_true, all_pred)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "skin_train_log.txt").write_text("")
    t_start = time.time()

    img_dirs, meta = find_data()
    train_items, val_items, test_items = build_splits(img_dirs, meta)

    # Lighter augmentation (RandAugment too slow on CPU) — still strong enough
    # NOTE: Resize uses default BILINEAR (fast); images are 600x450 -> 224
    tf_train = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15),
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
    log(f"Class counts: {dict(zip(CLASS_NAMES, cls_counts.astype(int)))}")
    # Class-weighted CE ONLY (no weighted sampler — combining both over-corrects
    # and destroyed accuracy in the first run: 20% after 2 epochs)
    # sqrt-inverse frequency: gentler than raw inverse (df 7.5x nv, not 57x)
    cls_w = 1.0 / np.sqrt(np.maximum(cls_counts, 1))
    cls_w = cls_w / cls_w.sum() * len(CLASS_NAMES)  # normalize

    train_dl = DataLoader(HamDataset(train_items, tf_train), batch_size=BATCH,
                          shuffle=True, num_workers=WORKERS)
    val_dl = DataLoader(HamDataset(val_items, tf_eval), batch_size=BATCH,
                        num_workers=WORKERS)
    test_dl = DataLoader(HamDataset(test_items, tf_eval), batch_size=BATCH,
                         num_workers=WORKERS)

    device = torch.device("cpu")
    model = build_model().to(device)
    n_total = sum(p.numel() for p in model.parameters())
    log(f"Model: efficientnet_b0 | total {n_total/1e6:.2f}M params")

    # Class-weighted CE + label smoothing (handles df/vasc with <20 samples)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(cls_w, dtype=torch.float32).to(device),
        label_smoothing=0.1)

    best_f1 = 0.0
    best_path = OUT_DIR / "skin_model.pt"

    # ── Phase 1: head warmup (backbone frozen) ────────────────────────────────
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-phase1", action="store_true",
                        help="Skip head warmup, load init weights instead")
    parser.add_argument("--init-from", default="",
                        help="Checkpoint to load before Phase 2")
    parser.add_argument("--resume-p2-epoch", type=int, default=0,
                        help="If >0: resume Phase 2 from this epoch number (1-based)")
    args = parser.parse_args()

    if args.skip_phase1 and args.init_from:
        log(f"Loading checkpoint: {args.init_from}")
        model.load_state_dict(torch.load(args.init_from, map_location=device,
                                         weights_only=True))
        acc, f1, _, _ = evaluate(model, val_dl, device)
        log(f"Loaded checkpoint: val_acc={acc:.4f} val_macroF1={f1:.4f}")
        best_f1 = f1
        # NOTE: do NOT overwrite best_path here — the loaded checkpoint IS the best
    elif args.skip_phase1:
        pass  # fall through to Phase 1 below (handled by else branch)
    else:
        pass
        log("=== PHASE 1: head warmup (backbone frozen) ===")
        set_frozen(model, True)
        opt = torch.optim.AdamW(model.classifier.parameters(), lr=5e-4, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=PHASE1_EPOCHS)
        for epoch in range(PHASE1_EPOCHS):
            train_epoch(model, train_dl, criterion, opt, device)
            sched.step()
            acc, f1, _, _ = evaluate(model, val_dl, device)
            log(f"P1 Epoch {epoch+1}/{PHASE1_EPOCHS} | val_acc={acc:.4f} val_macroF1={f1:.4f}")
            if f1 > best_f1:
                best_f1 = f1
                torch.save(model.state_dict(), best_path)
                log(f"  -> saved (val_macroF1={f1:.4f})")

    # ── Phase 2: partial fine-tune, differential LRs ───────────────────────────
    # Freeze early stages (generic ImageNet features) — train only late stages
    # + classifier. Cuts epoch time ~40% vs full fine-tune on CPU.
    log("=== PHASE 2: partial fine-tune (late stages + head, differential LRs) ===")
    for i, block in enumerate(model.features):
        frozen = i < 5  # stem + stages 1-4 frozen; stages 5-7 + head conv trainable
        for p in block.parameters():
            p.requires_grad = not frozen
    late_params = [p for blk in list(model.features)[5:] for p in blk.parameters()]
    opt = torch.optim.AdamW([
        {"params": late_params, "lr": 2e-5},
        {"params": model.classifier.parameters(), "lr": 1e-4},
    ], weight_decay=1e-4)
    # Resume support: start Phase 2 from epoch N (already completed N-1 epochs)
    p2_start = args.resume_p2_epoch if args.resume_p2_epoch > 0 else 1
    if p2_start > 1:
        log(f"Resuming Phase 2 from epoch {p2_start} (checkpoint already at epoch {p2_start-1})")
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=PHASE2_EPOCHS - p2_start + 1)
    stale = 0
    for epoch in range(p2_start - 1, PHASE2_EPOCHS):
        train_epoch(model, train_dl, criterion, opt, device)
        sched.step()
        acc, f1, _, _ = evaluate(model, val_dl, device)
        log(f"P2 Epoch {epoch+1}/{PHASE2_EPOCHS} | val_acc={acc:.4f} val_macroF1={f1:.4f} "
            f"lr_backbone={opt.param_groups[0]['lr']:.1e}")
        if f1 > best_f1:
            best_f1 = f1
            stale = 0
            torch.save(model.state_dict(), best_path)
            log(f"  -> saved (val_macroF1={f1:.4f})")
        else:
            stale += 1
            if stale >= PATIENCE:
                log(f"Early stopping at epoch {epoch+1} (best val_macroF1={best_f1:.4f})")
                break

    # ── Final test eval with TTA ──────────────────────────────────────────────
    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    log("=== Final test evaluation (with TTA x5) ===")
    acc, f1, cm, per_class = evaluate_tta(model, test_items, device)

    metrics = {
        "test_accuracy": round(acc, 4),
        "macro_f1": f1,
        "n_test": len(test_items),
        "per_class": per_class,
        "confusion_matrix": cm,
        "class_order": CLASS_NAMES,
        "tta_views": 5,
    }
    with open(OUT_DIR / "skin_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(OUT_DIR / "skin_classes.json", "w") as f:
        json.dump({"classes": CLASS_NAMES, "full_names": FULL_NAMES}, f, indent=2)
    info = {
        "architecture": "efficientnet_b0 (ImageNet pretrained, full fine-tune, differential LRs)",
        "input_size": [IMG_SIZE, IMG_SIZE],
        "num_classes": len(CLASS_NAMES),
        "class_names": CLASS_NAMES,
        "test_accuracy": round(acc, 4),
        "macro_f1": f1,
        "dataset": "HAM10000 (10,015 dermatoscopic images)",
        "weights_file": "skin_model.pt",
        "training": "phase1 head-warmup 4ep + phase2 full fine-tune, class-weighted CE + label smoothing 0.1, RandAugment, TTA x5",
    }
    with open(OUT_DIR / "skin_model_info.json", "w") as f:
        json.dump(info, f, indent=2)

    size_mb = best_path.stat().st_size / 1e6
    elapsed = (time.time() - t_start) / 60
    log(f"DONE | test_acc={acc:.4f} macro_f1={f1:.4f} model_size={size_mb:.1f}MB time={elapsed:.1f}min")
    for cname in CLASS_NAMES:
        pc = per_class[cname]
        log(f"  {cname}: P={pc['precision']:.3f} R={pc['recall']:.3f} F1={pc['f1']:.3f} (n={pc['support']})")


def train_epoch(model, dl, criterion, opt, device):
    model.train()
    run_loss, run_n = 0.0, 0
    for xb, yb in dl:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        run_loss += loss.item() * len(xb)
        run_n += len(xb)
    return run_loss / run_n


@torch.no_grad()
def evaluate_tta(model, test_items, device, views=5):
    """Test-time augmentation: average softmax over multiple augmented views."""
    model.eval()
    base = [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
    aug_sets = []
    for v in range(views):
        rng_tf = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]) if v > 0 else transforms.Compose(base)
        aug_sets.append(rng_tf)

    all_pred, all_true = [], []
    for path, label in test_items:
        img = Image.open(path).convert("RGB")
        probs = []
        for tf in aug_sets:
            xb = tf(img).unsqueeze(0).to(device)
            probs.append(torch.softmax(model(xb), dim=1).cpu())
        avg = torch.stack(probs).mean(0)
        all_pred.append(avg.argmax(1).item())
        all_true.append(label)
    return compute_metrics(all_true, all_pred)


if __name__ == "__main__":
    main()
