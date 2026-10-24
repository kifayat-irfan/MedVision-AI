"""Tests for the multi-model router (multi_model.py)."""
import sys
from pathlib import Path

import pytest
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from multi_model import predict_modality, model_info, gradcam_for, REGISTRY


def _rgb():
    return Image.fromarray(np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8))


@pytest.mark.slow
def test_brain_mri_predict():
    preds = predict_modality(_rgb(), "MRI")
    assert set(preds) == {"glioma", "meningioma", "notumor", "pituitary"}
    assert abs(sum(preds.values()) - 1.0) < 0.01  # softmax


@pytest.mark.slow
def test_skin_predict():
    preds = predict_modality(_rgb(), "Dermatology")
    assert len(preds) == 7
    assert abs(sum(preds.values()) - 1.0) < 0.01


def test_ct_vlm_only():
    assert predict_modality(_rgb(), "CT Scan") is None
    assert "AI-assist" in model_info("CT Scan")


@pytest.mark.slow
def test_ultrasound_predict_tta():
    # 4-way TTA inference (see multi_model.predict_modality): labels are the
    # full names from models/us_classes.json, probabilities sum to 1.
    preds = predict_modality(_rgb(), "Ultrasound")
    assert set(preds) == {"Benign lesion", "Malignant lesion", "Normal (no lesion)"}
    assert abs(sum(preds.values()) - 1.0) < 0.01
    assert all(0.0 <= v <= 1.0 for v in preds.values())


@pytest.mark.slow
def test_ultrasound_model_shape():
    from multi_model import _load_custom, _preprocess_rgb
    import torch
    model, classes, _, _ = _load_custom("us")
    assert classes == ["benign", "malignant", "normal"]
    with torch.no_grad():
        out = model(_preprocess_rgb(_rgb()))
    assert tuple(out.shape) == (1, 3)


def test_ultrasound_model_info_labels_tta_accuracy():
    # The UI number must match the inference actually used: 4-way TTA = 78.3%.
    info = model_info("Ultrasound")
    assert "78.3%" in info and "4-way TTA" in info


@pytest.mark.slow
def test_gradcam_ultrasound():
    heat = gradcam_for(_rgb(), "Ultrasound")
    assert heat.mode == "RGB" and heat.size == (512, 512)
    heat_target = gradcam_for(_rgb(), "Ultrasound", target_idx=1)
    assert heat_target.mode == "RGB" and heat_target.size == (512, 512)


def test_registry_covers_modalities():
    for mod in ["X-Ray", "MRI", "CT Scan", "Ultrasound", "Dermatology"]:
        assert mod in REGISTRY


@pytest.mark.slow
def test_gradcam_mobilenet():
    heat = gradcam_for(_rgb(), "MRI")
    assert heat.mode == "RGB" and heat.size == (512, 512)
