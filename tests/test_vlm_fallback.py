"""Tests for the 2026-10-24 SenseNova empty-content fix.

Live runs failed with "Model returned empty content" on Ultrasound and
Dermatology, wiping the whole analysis including CNN + Grad-CAM. The fix:
robust message parsing, a fallback model, and a CNN-only report in app.py.
"""
from PIL import Image

import utils


class _FakeSecrets:
    def get(self, key, default=""):
        return "test-key" if key == "SENSENOVA_API_KEY" else default


class _FakeSt:
    secrets = _FakeSecrets()

    @staticmethod
    def write(*args, **kwargs):
        pass


class _Resp:
    def __init__(self, status_code, data):
        self.status_code = status_code
        self._data = data
        self.text = str(data)

    def json(self):
        return self._data


def _msg(content=None, **extra):
    message = {}
    if content is not None:
        message["content"] = content
    message.update(extra)
    return {"choices": [{"message": message}]}


def test_extract_string_content():
    assert utils._extract_message_content({"content": "  hello  "}) == "hello"


def test_extract_content_blocks():
    msg = {"content": [{"type": "text", "text": "part1"}, {"type": "text", "text": "part2"}]}
    assert utils._extract_message_content(msg) == "part1\npart2"


def test_extract_empty_without_structured_reasoning():
    assert utils._extract_message_content({"content": ""}) == ""
    assert utils._extract_message_content(
        {"content": "", "reasoning": "thinking about the image"}
    ) == ""


def test_extract_structured_reasoning_as_last_resort():
    reasoning = "FINDINGS:\n1. Something.\nFINAL IMPRESSION:\nImpression."
    assert utils._extract_message_content(
        {"content": "", "reasoning": reasoning}
    ) == reasoning


def test_fallback_model_used_when_primary_empty(monkeypatch):
    monkeypatch.setattr(utils, "st", _FakeSt())
    calls = []

    def fake_post(url, headers=None, json=None, timeout=None):
        calls.append(json["model"])
        if json["model"] == utils.SENSENOVA_MODEL:
            return _Resp(200, _msg(""))
        return _Resp(200, _msg("TECHNICAL QUALITY:\nOK."))

    monkeypatch.setattr(utils.requests, "post", fake_post)
    img = Image.new("RGB", (32, 32), color="gray")
    content, model = utils.call_ensemble_ai(img, None, "Ultrasound", "ID: T-1")
    assert model == utils.SENSENOVA_FALLBACK_MODELS[0]
    assert "TECHNICAL QUALITY" in content
    assert calls == [utils.SENSENOVA_MODEL, utils.SENSENOVA_FALLBACK_MODELS[0]]


def test_primary_used_when_content_present(monkeypatch):
    monkeypatch.setattr(utils, "st", _FakeSt())
    calls = []

    def fake_post(url, headers=None, json=None, timeout=None):
        calls.append(json["model"])
        return _Resp(200, _msg("TECHNICAL QUALITY:\nOK."))

    monkeypatch.setattr(utils.requests, "post", fake_post)
    img = Image.new("RGB", (32, 32), color="gray")
    content, model = utils.call_ensemble_ai(img, None, "Dermatology", "ID: T-1")
    assert model == utils.SENSENOVA_MODEL
    assert len(calls) == 1


def test_grounding_labels_are_modality_correct():
    import inspect

    src = inspect.getsource(utils.call_ensemble_ai)
    # Grounding must come from MODEL_LABELS, not a hardcoded CXR label.
    assert "MODEL_LABELS.get(modality" in src
    assert utils.MODEL_LABELS["Ultrasound"].startswith("MobileNetV3-Small trained on BUSI")
    assert utils.MODEL_LABELS["Dermatology"].startswith("EfficientNet-B0 trained on HAM10000")
    assert utils.MODEL_LABELS["X-Ray"].startswith("DenseNet121")
