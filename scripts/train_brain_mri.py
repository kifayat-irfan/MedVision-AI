"""Train brain MRI tumor classifier: MobileNetV3-Small, frozen backbone,
classifier head only. CPU-friendly."""
import json
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

BASE = Path("/home/hatch/workspace/github-fix/MedVision-AI")
DATA = BASE / "data" / "brain_mri" / "timri"
MODELS_DIR = BASE / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]  # canonical order
# HF folder uses 'no-tumor' -> map to 'notumor'
FOLDER_MAP = {"glioma": "glioma", "meningioma": "meningioma",
              "no-tumor": "notumor", "pituitary": "pituitary"}

DEVICE = torch.device("cpu")
BATCH = 32
EPOCHS = 6
LR = 1e-3

train_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
eval_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


class MappedFolder(datasets.ImageFolder):
    """ImageFolder that remaps 'no-tumor' -> 'notumor' and enforces class order."""
    def __init__(self, root, transform):
        super().__init__(root, transform=transform)
        # rebuild class mapping
        self.classes = CLASSES
        self.class_to_idx = {c: i for i, c in enumerate(CLASSES)}
        # remap samples
        new_samples = []
        for path, _ in self.samples:
            folder = Path(path).parent.name
            canon = FOLDER_MAP.get(folder, folder)
            new_samples.append((path, self.class_to_idx[canon]))
        self.samples = new_samples
        self.targets = [s[1] for s in new_samples]


def main():
    t_start = time.time()
    log_lines = []

    def log(msg):
        print(msg, flush=True)
        log_lines.append(msg)

    log(f"Device: {DEVICE}, batch={BATCH}, epochs={EPOCHS}")
    train_ds = MappedFolder(str(DATA / "train"), transform=train_tf)
    valid_ds = MappedFolder(str(DATA / "valid"), transform=eval_tf)
    test_ds = MappedFolder(str(DATA / "test"), transform=eval_tf)
    log(f"Train: {len(train_ds)}, Valid: {len(valid_ds)}, Test: {len(test_ds)}")

    train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=2)
    valid_dl = DataLoader(valid_ds, batch_size=BATCH, num_workers=2)
    test_dl = DataLoader(test_ds, batch_size=BATCH, num_workers=2)

    # Model: MobileNetV3-Small pretrained, freeze backbone
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    for p in model.features.parameters():
        p.requires_grad = False
    in_f = model.classifier[0].in_features
    model.classifier = nn.Sequential(
        nn.Linear(in_f, 256),
        nn.Hardswish(),
        nn.Dropout(0.3),
        nn.Linear(256, 4),
    )
    model.to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"Params: {n_params/1e6:.1f}M total")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val = 0.0
    best_state = None
    for epoch in range(EPOCHS):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for x, y in train_dl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            total += x.size(0)
        scheduler.step()
        train_loss = running_loss / total
        train_acc = correct / total

        # validation
        model.eval()
        v_correct, v_total = 0, 0
        with torch.no_grad():
            for x, y in valid_dl:
                x, y = x.to(DEVICE), y.to(DEVICE)
                v_correct += (model(x).argmax(1) == y).sum().item()
                v_total += x.size(0)
        val_acc = v_correct / v_total
        msg = (f"Epoch {epoch+1}/{EPOCHS} | train_loss={train_loss:.4f} "
               f"train_acc={train_acc:.4f} val_acc={val_acc:.4f}")
        log(msg)
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Final evaluation on test set with best weights
    model.load_state_dict(best_state)
    model.eval()
    all_pred, all_true = [], []
    with torch.no_grad():
        for x, y in test_dl:
            x = x.to(DEVICE)
            pred = model(x).argmax(1).cpu().tolist()
            all_pred.extend(pred)
            all_true.extend(y.tolist())

    test_acc = sum(p == t for p, t in zip(all_pred, all_true)) / len(all_true)
    prec, rec, f1, _ = precision_recall_fscore_support(
        all_true, all_pred, labels=[0, 1, 2, 3], zero_division=0)
    cm = confusion_matrix(all_true, all_pred, labels=[0, 1, 2, 3]).tolist()

    log(f"\nTEST ACCURACY: {test_acc:.4f}")
    for i, c in enumerate(CLASSES):
        log(f"  {c}: P={prec[i]:.3f} R={rec[i]:.3f} F1={f1[i]:.3f}")
    log(f"Confusion matrix:\n{cm}")

    train_time = time.time() - t_start
    log(f"\nTraining time: {train_time/60:.1f} min | best val_acc={best_val:.4f}")

    # Save deliverables
    torch.save(best_state, MODELS_DIR / "brain_model.pt")
    size_mb = (MODELS_DIR / "brain_model.pt").stat().st_size / 1e6
    log(f"Model saved: {size_mb:.1f} MB")

    (MODELS_DIR / "brain_classes.json").write_text(json.dumps(CLASSES, indent=1))

    metrics = {
        "test_accuracy": round(test_acc, 4),
        "per_class": {
            c: {"precision": round(float(prec[i]), 4),
                "recall": round(float(rec[i]), 4),
                "f1": round(float(f1[i]), 4)}
            for i, c in enumerate(CLASSES)
        },
        "confusion_matrix": cm,
        "confusion_matrix_labels": CLASSES,
        "best_val_accuracy": round(best_val, 4),
        "training_time_min": round(train_time / 60, 1),
    }
    (MODELS_DIR / "brain_metrics.json").write_text(json.dumps(metrics, indent=1))

    info = {
        "architecture": "mobilenet_v3_small (ImageNet pretrained, backbone frozen, custom 256-unit classifier head)",
        "input_size": 224,
        "num_classes": 4,
        "class_names": CLASSES,
        "test_accuracy": round(test_acc, 4),
        "model_file": "brain_model.pt",
        "model_size_mb": round(size_mb, 1),
    }
    (MODELS_DIR / "brain_model_info.json").write_text(json.dumps(info, indent=1))
    (MODELS_DIR / "brain_train_log.txt").write_text("\n".join(log_lines) + "\n")
    log("All deliverables saved.")


if __name__ == "__main__":
    main()
