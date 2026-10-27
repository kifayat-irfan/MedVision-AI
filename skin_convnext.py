"""R4 ConvNeXt-Tiny skin model integration — LIVE since 2026-10-27.

Status: INTEGRATED. The R4 model file was retrieved 2026-10-27 via the
Kaggle CLI (kaggle kernels output, Version 2, scriptVersionId 362662399)
and verified locally: models/skin_model.pt is 107 MB with the same hash
as the Kaggle download, and models/skin_metrics.json is byte-identical
to an independently downloaded copy (signed-URL route). Old v2
EfficientNet-B0 files are preserved in models/backup-2026-10-27/.
multi_model.py dispatches Dermatology to this module (ConvNeXt head,
Shades-of-Gray preprocessing, TTA x5, contested threshold t=0.40).

Architecture (exact, from train_skin_85_convnext_tiny.ipynb Cell 4):
  ConvNeXt-Tiny (IMAGENET1K_V1) with classifier replaced by:
    Sequential(LayerNorm2d(768), Flatten,
               Dropout(0.3), Linear(768, 256), SiLU,
               Dropout(0.2), Linear(256, 7))
Eval preprocessing (exact, tf_eval):
  Resize(224, 224) -> Shades-of-Gray colour constancy (p=6)
  -> ToTensor -> ImageNet norm
Shipped decision (exact, Cell 6/7):
  TTA x5 = identity + hflip + vflip + hvflip + rot90, mean softmax.
  Contested-zone threshold t=0.40, val-locked: if the top-2 predicted classes
  are {mel, nv} (either order), predict mel when p_mel / (p_mel + p_nv) >= t,
  else nv; all other predictions are plain argmax.
  Shipped Kaggle numbers (TTA x5, t=0.40): test 85.07% (849/998),
  macro F1 0.7574, mel recall 0.7027, mel F1 0.6166.
  Single-view argmax was also reported (85.47% / 0.7609) but its mel gates
  were NOT printed — do NOT claim single-view passes the mel gates until
  per-class numbers for that readout are computed from retrieved prob CSVs.
Grad-CAM: target layer model.features[-1] (last ConvNeXt stage) is
wired in multi_model.py (_gradcam_convnext) and verified by the torch
test run 2026-10-27 (Grad-CAM returns a 512x512 overlay for Dermatology).
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

MODELS_DIR = Path(__file__).resolve().parent / "models"
CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
LOCKED_THRESHOLD = 0.40  # val-locked, R4 Cell 6; frozen, do not tune on test
MEL_IDX = CLASS_NAMES.index("mel")
NV_IDX = CLASS_NAMES.index("nv")


def shades_of_gray(img: Image.Image, p: int = 6) -> Image.Image:
    """Shades-of-Gray (p=6) white balancing, deterministic, as in R4 Cell 4."""
    arr = np.asarray(img).astype(np.float32) / 255.0
    illum = (np.mean(arr ** p, axis=(0, 1)) + 1e-8) ** (1.0 / p)
    gain = illum.mean() / np.maximum(illum, 1e-8)
    out = np.clip(arr * gain, 0.0, 1.0)
    return Image.fromarray((out * 255.0).astype(np.uint8))


def build_convnext_skin(num_classes: int = 7) -> nn.Module:
    """ConvNeXt-Tiny with the exact R4 repo head (768 -> 256 -> classes)."""
    from torchvision import models
    m = models.convnext_tiny(weights=None)
    m.classifier = nn.Sequential(
        m.classifier[0], m.classifier[1],
        nn.Dropout(0.3),
        nn.Linear(768, 256),
        nn.SiLU(),
        nn.Dropout(0.2),
        nn.Linear(256, num_classes),
    )
    return m


def preprocess_convnext(img: Image.Image) -> torch.Tensor:
    """R4 tf_eval: Resize 224 -> Shades-of-Gray -> ToTensor -> ImageNet norm."""
    from torchvision import transforms
    t = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Lambda(lambda im: shades_of_gray(im, p=6)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return t(img.convert("RGB")).unsqueeze(0)


def apply_contested(probs: np.ndarray, t: float = LOCKED_THRESHOLD) -> np.ndarray:
    """Val-locked contested-zone rule on a (n, 7) probability array."""
    pred = probs.argmax(axis=1).copy()
    for i, row in enumerate(probs):
        top2 = np.argsort(row)[-2:]
        if set(top2.tolist()) == {MEL_IDX, NV_IDX}:
            frac = row[MEL_IDX] / (row[MEL_IDX] + row[NV_IDX] + 1e-12)
            pred[i] = MEL_IDX if frac >= t else NV_IDX
    return pred


@torch.no_grad()
def predict_convnext_tta(model: nn.Module, x: torch.Tensor) -> np.ndarray:
    """TTA x5 mean-softmax probabilities, matching R4 Cell 6/7."""
    views = [x, torch.flip(x, dims=[3]), torch.flip(x, dims=[2]),
             torch.flip(x, dims=[2, 3]), torch.rot90(x, k=1, dims=[2, 3])]
    probs = torch.stack(
        [torch.softmax(model(v), dim=1) for v in views]
    ).mean(dim=0).numpy()[0]
    return probs


def load_convnext_skin(path: Path | None = None) -> nn.Module:
    """Load an R4-format checkpoint, failing loudly on any key mismatch.

    Raises FileNotFoundError if no model file is present (the current state:
    the R4 file is lost from Kaggle and must be retrieved or re-run first).
    """
    p = path or (MODELS_DIR / "skin_model.pt")
    if not p.exists():
        raise FileNotFoundError(
            f"R4 ConvNeXt skin model file not found: {p}. "
            "Retrieve it from Kaggle Version 2 output "
            "(kaggle kernels output kifayatirfan/train-skin-85-convnext-tiny-rerun), "
            "then retry."
        )
    model = build_convnext_skin(len(CLASS_NAMES))
    state = torch.load(p, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint did not load cleanly into ConvNeXt-Tiny repo head. "
            f"missing={missing} unexpected={unexpected}. Abort."
        )
    model.eval()
    return model
