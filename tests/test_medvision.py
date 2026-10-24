"""Tests for MedVision AI core logic (no API calls, no Streamlit runtime needed)."""
import io
import sys
import tempfile
import os
from pathlib import Path

import pytest
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dicom_utils import process_dicom
from report_gen import generate_pdf, generate_docx, generate_txt, sanitize_text, DISCLAIMER
from utils import draw_heatmap, SENSENOVA_MODEL, SENSENOVA_BASE


# ── DICOM ────────────────────────────────────────────────────────────────────

def _make_dicom(pixel_array, photometric="MONOCHROME2"):
    """Build a minimal in-memory DICOM file."""
    import pydicom
    from pydicom.dataset import Dataset, FileDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid
    import datetime

    file_meta = Dataset()
    file_meta.MediaStorageSOPClassUID = generate_uid()
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = FileDataset(None, {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.PatientName = "Test^Patient"
    # Multi-frame: shape is (frames, rows, cols); single frame: (rows, cols)
    frame_shape = pixel_array.shape[-2:]
    ds.Rows, ds.Columns = frame_shape
    if pixel_array.ndim > 2:
        ds.NumberOfFrames = pixel_array.shape[0]
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = photometric
    ds.PixelRepresentation = 0
    ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.is_little_endian = True
    ds.is_implicit_VR = False
    ds.PixelData = pixel_array.astype(np.uint16).tobytes()

    buf = io.BytesIO()
    pydicom.dcmwrite(buf, ds)
    buf.seek(0)
    return buf


def test_dicom_normal():
    arr = np.random.randint(0, 4096, (64, 64))
    img = process_dicom(_make_dicom(arr))
    assert isinstance(img, Image.Image)
    assert img.size == (64, 64)


def test_dicom_constant_no_div_zero():
    """Regression: constant pixel data caused ZeroDivisionError."""
    arr = np.full((32, 32), 1000)
    img = process_dicom(_make_dicom(arr))  # must not raise
    assert isinstance(img, Image.Image)


def test_dicom_multiframe_uses_middle():
    """Multi-frame DICOM (e.g. CT series) should not crash."""
    arr = np.random.randint(0, 4096, (5, 32, 32))
    img = process_dicom(_make_dicom(arr))
    assert img.size == (32, 32)


def test_dicom_monochrome1_inverted():
    """MONOCHROME1 (0=white) should be flipped to standard."""
    arr = np.zeros((16, 16), dtype=np.uint16)  # all black in MONO1 = white visually
    img = process_dicom(_make_dicom(arr, photometric="MONOCHROME1"))
    assert img.getpixel((0, 0)) == 255  # flipped to white


def test_dicom_corrupt_raises():
    with pytest.raises(Exception, match="DICOM Processing Error"):
        process_dicom(io.BytesIO(b"not a dicom file"))


# ── Heatmap ─────────────────────────────────────────────────────────────────

def _blank_img():
    return Image.new("RGB", (200, 200), "black")


def test_heatmap_draws_circle():
    img = draw_heatmap(_blank_img(), "HEATMAP: [50, 50, 10]")
    # Center pixel should be red (the center dot)
    assert img.getpixel((100, 100)) == (255, 0, 0)


def test_heatmap_no_coords_returns_original():
    img = _blank_img()
    out = draw_heatmap(img, "No heatmap line here")
    assert out.getpixel((100, 100)) == (0, 0, 0)


def test_heatmap_zero_zero_returns_original():
    """x=0,y=0 sentinel means 'no finding' — image unchanged."""
    img = _blank_img()
    out = draw_heatmap(img, "HEATMAP: [0, 0, 10]")
    assert out.getpixel((100, 100)) == (0, 0, 0)


# ── Report generation ────────────────────────────────────────────────────────

SAMPLE_REPORT = "TECHNICAL QUALITY:\nGood exposure.\n\nFINDINGS:\n1. Clear lungs."

def test_pdf_has_disclaimer():
    pdf_bytes = generate_pdf(SAMPLE_REPORT, "PAT-001, Age: 25", "X-Ray")
    assert pdf_bytes[:5] == b"%PDF-"
    # PDF streams are compressed — extract text to verify the disclaimer
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(pdf_bytes))
    full = " ".join((p.extract_text() or "") for p in reader.pages).replace("\n", " ")
    assert "NOT a medical diagnosis" in full


def test_docx_has_disclaimer():
    docx_bytes = generate_docx(SAMPLE_REPORT, "PAT-001", "X-Ray")
    from docx import Document
    doc = Document(io.BytesIO(docx_bytes))
    full = "\n".join(p.text for p in doc.paragraphs)
    assert "NOT a medical diagnosis" in full
    assert "PAT-001" in full


def test_txt_roundtrip():
    assert generate_txt(SAMPLE_REPORT).decode("utf-8") == SAMPLE_REPORT


def test_sanitize_unicode():
    # Smart quotes / em-dashes must not crash FPDF (latin-1)
    tricky = "It\u2019s a \u201ctest\u201d \u2014 done\u2026"
    out = sanitize_text(tricky)
    out.encode("latin-1")  # must not raise
    assert "It's" in out


# ── Model config ─────────────────────────────────────────────────────────────

def test_sensenova_model_config():
    assert SENSENOVA_MODEL == "sensenova-6.8-flash-lite"
    assert SENSENOVA_BASE == "https://token.sensenova.ai/v1/chat/completions"


def test_prompt_sections_present():
    """The AI prompt must request all 7 structured sections."""
    import inspect
    import utils
    src = inspect.getsource(utils.call_ensemble_ai)
    for section in ["TECHNICAL QUALITY", "FINDINGS", "TEMPORAL COMPARISON",
                    "FINAL IMPRESSION", "TREATMENT ROADMAP", "EVIDENCE BASE", "HEATMAP"]:
        assert section in src, f"Missing prompt section: {section}"


# ── DB (SQLite local path) ───────────────────────────────────────────────────

def test_db_local_roundtrip(tmp_path, monkeypatch):
    """PatientDB saves and retrieves via local SQLite."""
    monkeypatch.chdir(tmp_path)
    # Streamlit not running: PatientDB must fall back to LOCAL without crashing
    from db import PatientDB
    pdb = PatientDB()
    assert pdb.mode == "LOCAL"
    assert pdb.save_report("PAT-999", "X-Ray", "Test report content")
    hist = pdb.get_history("PAT-999")
    assert len(hist) == 1
    assert hist[0][0] == "X-Ray"
    assert hist[0][1] == "Test report content"
    assert pdb.get_history("NOBODY") == []
