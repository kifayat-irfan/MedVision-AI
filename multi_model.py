"""Multi-model router for MedVision AI — all-in-one medical image analysis.

Routes each modality to its trained model:
  X-Ray (chest)  → DenseNet121 (NIH ChestX-ray14, 112k images) — cxr_model.py
  MRI (brain)    → MobileNetV3-Small (trained, 92% test accuracy)
  Dermatology    → EfficientNet-B0 (trained on HAM10000, 82.5% test accuracy, TTA x5)
  Ultrasound     → MobileNetV3-Small (trained on BUSI, 78.3% test accuracy, 4-way TTA)
  CT / other     → VLM-only (clearly labeled AI-assist, no CNN)

Every CNN model also gets Grad-CAM visual explanations.
"""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

MODELS_DIR = Path(__file__).resolve().parent / "models"

# ── Model registry ───────────────────────────────────────────────────────────
# modality -> (model key, human label, accuracy, dataset)
REGISTRY = {
    "X-Ray": ("cxr", "DenseNet121 · NIH ChestX-ray14", "112,120 images", None),
    "MRI": ("brain", "MobileNetV3-Small · Brain MRI", "92.0% test accuracy", "brain"),
    "Dermatology": ("skin", "EfficientNet-B0 · HAM10000", "82.5% test accuracy", "skin"),
    "CT Scan": (None, None, None, None),      # VLM-only
    "Ultrasound": ("us", "MobileNetV3-Small · BUSI breast ultrasound", "78.3% test accuracy (4-way TTA)", "us"),
}

_loaded = {}


def _build_mobilenet(num_classes: int) -> nn.Module:
    from torchvision import models
    m = models.mobilenet_v3_small(weights=None)
    in_f = m.classifier[0].in_features
    m.classifier = nn.Sequential(
        nn.Linear(in_f, 256),
        nn.Hardswish(),
        nn.Dropout(0.2),
        nn.Linear(256, num_classes),
    )
    return m


def _build_efficientnet(num_classes: int) -> nn.Module:
    from torchvision import models
    m = models.efficientnet_b0(weights=None)
    in_f = m.classifier[1].in_features
    m.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_f, 256),
        nn.SiLU(),
        nn.Dropout(0.2),
        nn.Linear(256, num_classes),
    )
    return m


def _load_custom(key: str):
    """Load a trained classifier (brain/us=MobileNet, skin=EfficientNet)."""
    if key in _loaded:
        return _loaded[key]
    info = json.loads((MODELS_DIR / f"{key}_model_info.json").read_text())
    classes = info["class_names"]
    arch = info.get("architecture", "")
    if "efficientnet" in arch:
        model = _build_efficientnet(len(classes))
    else:
        model = _build_mobilenet(len(classes))
    state = torch.load(MODELS_DIR / f"{key}_model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    full_names = {}
    fn_path = MODELS_DIR / f"{key}_classes.json"
    if fn_path.exists():
        data = json.loads(fn_path.read_text())
        if isinstance(data, dict):
            full_names = data.get("full_names", {})
        # if it's a plain list, class names are used as-is
    _loaded[key] = (model, classes, full_names, info)
    return _loaded[key]


def _preprocess_rgb(img: Image.Image) -> torch.Tensor:
    """Standard ImageNet preprocessing for RGB models."""
    from torchvision import transforms
    t = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return t(img.convert("RGB")).unsqueeze(0)


def predict_modality(img: Image.Image, modality: str) -> dict | None:
    """Run the trained CNN for a modality. Returns {label: prob} or None (VLM-only)."""
    key = REGISTRY.get(modality, (None,))[0]
    if key == "cxr":
        from cxr_model import predict as cxr_predict
        return cxr_predict(img)
    if key in ("brain", "skin", "us"):
        model, classes, full_names, _ = _load_custom(key)
        x = _preprocess_rgb(img)
        with torch.no_grad():
            if key == "us":
                # 4-way TTA, matching the 78.26% eval in models/us_metrics.json:
                # average softmax probabilities over original + hflip + vflip + hvflip.
                # Flips are applied to the normalized tensor (per-channel, so
                # spatial flips are unaffected by normalization).
                views = [x, torch.flip(x, dims=[3]), torch.flip(x, dims=[2]),
                         torch.flip(x, dims=[2, 3])]
                probs = torch.stack(
                    [torch.softmax(model(v), dim=1) for v in views]
                ).mean(dim=0).numpy()[0]
            else:
                probs = torch.softmax(model(x), dim=1).numpy()[0]
        out = {}
        for i, c in enumerate(classes):
            label = full_names.get(c, c)
            out[label] = float(probs[i])
        return out
    return None  # VLM-only modality


def model_info(modality: str) -> str:
    """Human-readable model description for the UI."""
    entry = REGISTRY.get(modality)
    if not entry or not entry[0]:
        return "AI-assist mode (vision-language model, no trained CNN for this modality)"
    _, label, metric, _ = entry
    return f"{label} · {metric}"


def gradcam_for(img: Image.Image, modality: str, target_idx: int | None = None) -> Image.Image | None:
    """Grad-CAM heatmap for the modality's CNN. None for VLM-only modalities."""
    key = REGISTRY.get(modality, (None,))[0]
    if key == "cxr":
        from cxr_model import gradcam as cxr_gradcam
        return cxr_gradcam(img, target_idx)
    if key in ("brain", "skin", "us"):
        return _gradcam_mobilenet(img, key, target_idx)
    return None


def _gradcam_mobilenet(img: Image.Image, key: str, target_idx: int | None = None) -> Image.Image:
    import torch.nn.functional as F
    model, classes, _, info = _load_custom(key)
    x = _preprocess_rgb(img)

    activations, gradients = {}, {}

    def fwd_hook(m, inp, out):
        activations["v"] = out.detach()

    def bwd_hook(m, gin, gout):
        gradients["v"] = gout[0].detach()

    # Last conv block: MobileNetV3-Small -> features[-1]; EfficientNet-B0 -> features[-1]
    target_layer = model.features[-1]
    h1 = target_layer.register_forward_hook(fwd_hook)
    h2 = target_layer.register_full_backward_hook(bwd_hook)
    try:
        logits = model(x)
        if target_idx is None:
            target_idx = int(logits.argmax())
        model.zero_grad()
        logits[0, target_idx].backward()
        acts, grads = activations["v"], gradients["v"]
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * acts).sum(dim=1).squeeze())
        cam = cam / (cam.max() + 1e-8)
        cam_np = cam.numpy()
    finally:
        h1.remove()
        h2.remove()

    size = (512, 512)
    base = img.convert("RGB").resize(size, Image.LANCZOS)
    cam_img = Image.fromarray((cam_np * 255).astype(np.uint8)).resize(size, Image.BILINEAR)
    heat = np.array(cam_img, dtype=np.float32) / 255.0
    overlay = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    overlay[..., 0] = (heat * 255).astype(np.uint8)
    overlay[..., 1] = (heat * 180).astype(np.uint8)
    base_np = np.array(base)
    alpha = (heat * 0.65)[..., None]
    return Image.fromarray((alpha * overlay + (1 - alpha) * base_np).astype(np.uint8))
