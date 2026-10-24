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


def test_registry_covers_modalities():
    for mod in ["X-Ray", "MRI", "CT Scan", "Ultrasound", "Dermatology"]:
        assert mod in REGISTRY


@pytest.mark.slow
def test_gradcam_mobilenet():
    heat = gradcam_for(_rgb(), "MRI")
    assert heat.mode == "RGB" and heat.size == (512, 512)
