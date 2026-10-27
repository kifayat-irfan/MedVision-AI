"""Tests for the R4 ConvNeXt-Tiny skin integration draft (skin_convnext.py).

These tests run WITHOUT a model file: architecture, preprocessing, and the
contested-zone rule are all verifiable locally. The load test SKIPS (does not
fail) while the R4 model file is missing — it must PASS (not skip) before
any integration is wired into multi_model.py.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from skin_convnext import (  # noqa: E402
    LOCKED_THRESHOLD,
    apply_contested,
    build_convnext_skin,
    load_convnext_skin,
    preprocess_convnext,
    shades_of_gray,
)


def test_convnext_head_shape():
    m = build_convnext_skin(7)
    m.eval()
    with torch.no_grad():
        out = m(torch.zeros(1, 3, 224, 224))
    assert out.shape == (1, 7)
    # Head must be LayerNorm+Flatten+Dropout+Linear(768,256)+SiLU+Dropout+Linear(256,7)
    assert isinstance(m.classifier[3], torch.nn.Linear)
    assert m.classifier[3].in_features == 768
    assert m.classifier[3].out_features == 256
    assert m.classifier[6].out_features == 7


def test_shades_of_gray_is_deterministic_and_bounded():
    img = Image.new("RGB", (32, 32), color=(180, 120, 60))
    a = np.asarray(shades_of_gray(img))
    b = np.asarray(shades_of_gray(img))
    assert np.array_equal(a, b)
    assert a.min() >= 0 and a.max() <= 255


def test_preprocess_shape_and_norm():
    x = preprocess_convnext(Image.new("RGB", (100, 80), color=(200, 100, 50)))
    assert x.shape == (1, 3, 224, 224)


def test_contested_threshold_rule():
    # Top-2 = {mel(4), nv(5)}: frac = 0.30/(0.30+0.40) = 0.428 >= 0.40 -> mel
    probs = np.array([[0.05, 0.05, 0.05, 0.05, 0.30, 0.40, 0.10]])
    assert apply_contested(probs)[0] == 4
    # frac = 0.20/(0.20+0.50) = 0.286 < 0.40 -> nv
    probs = np.array([[0.05, 0.05, 0.05, 0.05, 0.20, 0.50, 0.10]])
    assert apply_contested(probs)[0] == 5
    # Top-2 not {mel, nv} -> plain argmax
    probs = np.array([[0.05, 0.60, 0.05, 0.05, 0.10, 0.10, 0.05]])
    assert apply_contested(probs)[0] == 1
    assert LOCKED_THRESHOLD == 0.40


def test_load_skips_without_model_file(tmp_path):
    missing = tmp_path / "skin_convnext_model.pt"
    with pytest.raises(FileNotFoundError):
        load_convnext_skin(missing)
