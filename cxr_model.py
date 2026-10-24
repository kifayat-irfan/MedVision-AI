"""Real chest X-ray analysis: DenseNet121 pretrained on NIH ChestX-ray14
(112,120 frontal X-rays) + Grad-CAM for genuine visual explanations.

This replaces the old approach where the VLM invented heatmap coordinates.
Here the CNN produces REAL pathology probabilities and Grad-CAM shows
where the network actually looked.
"""
import io
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

_model = None
_model_pathologies = None


def get_model():
    """Lazy-load the pretrained DenseNet (6M params, CPU-friendly)."""
    global _model, _model_pathologies
    if _model is None:
        import torchxrayvision as xrv
        _model = xrv.models.DenseNet(weights="densenet121-res224-nih")
        _model.eval()
        _model_pathologies = [p for p in _model.pathologies if p]
    return _model


def preprocess(img: Image.Image) -> torch.Tensor:
    """Prepare a PIL image for the model: grayscale, 224x224, normalized."""
    import torchxrayvision as xrv
    g = img.convert("L").resize((224, 224), Image.LANCZOS)
    arr = np.array(g, dtype=np.float32)  # 0-255
    # normalize() maps 0..maxval -> roughly [-1024, 1024] (model's training range)
    arr = xrv.datasets.normalize(arr, maxval=255)
    return torch.from_numpy(arr)[None, None, :, :]


def predict(img: Image.Image) -> dict:
    """Run the CNN. Returns {pathology: probability} for the 14 pathologies."""
    model = get_model()
    x = preprocess(img)
    with torch.no_grad():
        out = torch.sigmoid(model(x)).numpy()[0]
    return {p: float(out[i]) for i, p in enumerate(_model_pathologies)}


def gradcam(img: Image.Image, target_idx: int | None = None) -> Image.Image:
    """Grad-CAM heatmap for the top pathology (or a given index).

    Returns a PIL RGB image: original grayscale with red-yellow heat overlay.
    """
    model = get_model()
    x = preprocess(img)
    x.requires_grad_()

    # Hook the last conv block's activations
    activations, gradients = {}, {}

    def fwd_hook(m, inp, out):
        activations["v"] = out.detach()

    def bwd_hook(m, gin, gout):
        gradients["v"] = gout[0].detach()

    # DenseNet: last dense block is features.denseblock4
    target_layer = model.features.denseblock4
    h1 = target_layer.register_forward_hook(fwd_hook)
    h2 = target_layer.register_full_backward_hook(bwd_hook)

    try:
        logits = model(x)
        if target_idx is None:
            target_idx = int(torch.sigmoid(logits).argmax())
        score = logits[0, target_idx]
        model.zero_grad()
        score.backward()

        acts = activations["v"]      # [1, C, H, W]
        grads = gradients["v"]       # [1, C, H, W]
        weights = grads.mean(dim=(2, 3), keepdim=True)  # global avg pool
        cam = (weights * acts).sum(dim=1).squeeze()     # [H, W]
        cam = F.relu(cam)
        cam = cam / (cam.max() + 1e-8)
        cam_np = cam.numpy()
    finally:
        h1.remove()
        h2.remove()

    # Resize CAM to original image size and colorize
    orig = img.convert("L").resize((512, 512), Image.LANCZOS)
    cam_img = Image.fromarray((cam_np * 255).astype(np.uint8)).resize(
        orig.size, Image.BILINEAR)

    # Red-yellow colormap overlay
    overlay = np.zeros((orig.size[1], orig.size[0], 3), dtype=np.uint8)
    heat = np.array(cam_img, dtype=np.float32) / 255.0
    overlay[..., 0] = (heat * 255).astype(np.uint8)          # red channel
    overlay[..., 1] = (heat * 180).astype(np.uint8)          # some green -> yellow
    base = np.stack([np.array(orig)] * 3, axis=-1)

    # Blend: strong heat = mostly overlay, weak heat = mostly original
    alpha = (heat * 0.65)[..., None]
    blended = (alpha * overlay + (1 - alpha) * base).astype(np.uint8)
    return Image.fromarray(blended)


def top_findings(predictions: dict, threshold: float = 0.5, top_k: int = 5) -> list[tuple[str, float]]:
    """Pathologies above threshold, sorted by probability."""
    ranked = sorted(predictions.items(), key=lambda kv: kv[1], reverse=True)
    above = [(p, s) for p, s in ranked if s >= threshold]
    return (above or ranked[:1])[:top_k]


def format_for_prompt(predictions: dict) -> str:
    """Human-readable model output to ground the VLM report (anti-hallucination)."""
    lines = []
    for p, s in sorted(predictions.items(), key=lambda kv: kv[1], reverse=True):
        flag = " <-- ELEVATED" if s >= 0.5 else ""
        lines.append(f"- {p}: {s:.2f}{flag}")
    return "\n".join(lines)
