"""Tests for the real CNN pipeline (cxr_model.py).

Uses the actual pretrained DenseNet121 (downloaded once, cached).
Slow (~10s) — marked accordingly.
"""
import sys
from pathlib import Path

import pytest
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cxr_model import predict, gradcam, top_findings, format_for_prompt, preprocess


def _fake_cxr():
    arr = np.full((512, 512), 40, dtype=np.uint8)
    arr[100:400, 80:220] = 90
    arr[100:400, 290:430] = 90
    return Image.fromarray(arr, "L")


@pytest.mark.slow
def test_predict_returns_14_pathologies():
    preds = predict(_fake_cxr())
    assert len(preds) == 14
    assert all(0.0 <= v <= 1.0 for v in preds.values())
    assert "Pneumonia" in preds and "Cardiomegaly" in preds


@pytest.mark.slow
def test_preprocess_shape():
    t = preprocess(_fake_cxr())
    assert tuple(t.shape) == (1, 1, 224, 224)


@pytest.mark.slow
def test_gradcam_returns_overlay():
    heat = gradcam(_fake_cxr())
    assert isinstance(heat, Image.Image)
    assert heat.mode == "RGB"
    assert heat.size == (512, 512)


def test_top_findings_sorted():
    preds = {"A": 0.9, "B": 0.2, "C": 0.7}
    top = top_findings(preds, threshold=0.5)
    assert [p for p, _ in top] == ["A", "C"]


def test_format_for_prompt_flags_elevated():
    preds = {"Pneumonia": 0.85, "Edema": 0.2}
    txt = format_for_prompt(preds)
    assert "Pneumonia: 0.85" in txt
    assert "ELEVATED" in txt
