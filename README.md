# MedVision AI — Medical Scan Analysis Assistant

[![Python](https://img.shields.io/badge/Python-3.11+-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-ff4b4b)](https://streamlit.io/)
[![SenseNova](https://img.shields.io/badge/AI-SenseNova-purple)](https://token.sensenova.ai/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

> ⚠️ **Disclaimer:** This is an educational demo project. The AI-generated analysis is a *second-opinion-style draft*,
> **not a medical diagnosis** and not a medical device. Do not use it for clinical decisions.

A Streamlit app that analyzes **chest X-rays** with a real trained neural network —
DenseNet121 pretrained on the **NIH ChestX-ray14 dataset (112,120 images)** — plus
vision-language AI for structured reporting.

## How it works (the real pipeline)

1. **Upload a medical image** — JPG/PNG, or DICOM (`.dcm`) converted automatically.
2. **Modality routing** — each scan type goes to its own trained neural network:

| Modality | Model | Training data | Test accuracy |
|---|---|---|---|
| 🫁 Chest X-ray | DenseNet121 | NIH ChestX-ray14 (112,120 images) | 14 pathologies |
| 🧠 Brain MRI | MobileNetV3-Small | Brain tumor MRI (5,369 images) | **92.0%** |
| 🔬 Skin / Dermatology | MobileNetV3-Small | HAM10000 (10,015 images) | **62.1%** |
| 🩻 CT / Ultrasound | Vision-language AI assist | — | clearly labeled |

3. **Grad-CAM visual explanation** — genuine heatmaps from each model's own gradients.
4. **Grounded AI report** — SenseNova's vision model writes the structured clinical report,
   with every claim grounded in the CNN's real probability outputs (anti-hallucination by design).
5. **Patient vault** — reports saved per patient ID (SQLite locally / Supabase in cloud).
6. **Export** — PDF, DOCX, TXT, each with a medical disclaimer.

## Tech stack

- **Pathology CNNs:**
  - Chest X-ray: DenseNet121 via `torchxrayvision`, pretrained on NIH ChestX-ray14 (112k images)
  - Brain MRI: MobileNetV3-Small, trained on 5,369 brain MRIs — **92.0% test accuracy**
  - Skin lesions: MobileNetV3-Small, trained on HAM10000 (10,015 dermatoscopic images) — **62.1% test accuracy**
- **Explainability:** Grad-CAM — genuine heatmaps from each model's own gradients
- **Report AI:** SenseNova `sensenova-6.8-flash-lite` (vision), grounded in CNN probabilities
- **App:** Streamlit (Python) — clean clinical light theme
- **Imaging:** Pillow, pydicom + NumPy (DICOM → image, handles multi-frame + MONOCHROME1)
- **Storage:** SQLite (local default) / Supabase PostgreSQL (cloud, optional)
- **Export:** fpdf2 (PDF), python-docx (DOCX) — every export carries a medical disclaimer
- **Tests:** pytest — 25 tests (multi-model router, CNN pipelines, Grad-CAM, DICOM, report gen, DB)

## Setup

```bash
git clone https://github.com/kifayat-irfan/MedVision-AI.git
cd MedVision-AI
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Create `.streamlit/secrets.toml`:

```toml
SENSENOVA_API_KEY = "your-sensenova-key-here"   # required
# Optional: cloud patient vault (otherwise local SQLite is used)
# SUPABASE_URL = "https://your-project.supabase.co"
# SUPABASE_KEY = "your-anon-key"
```

```bash
streamlit run app.py
```

## Project structure

```
app.py          # Streamlit UI — upload, history, dashboard, export
utils.py        # OpenRouter ensemble call (3-model fallback) + heatmap drawing
dicom_utils.py  # DICOM → PIL image (pydicom, min-max normalization)
db.py           # Patient vault — SQLite locally, Supabase when secrets exist
report_gen.py   # Report export to PDF / DOCX / TXT
styles.py       # Custom UI styling
```

## Limitations

- Analysis quality depends entirely on the underlying vision-language models; free-tier models can be
  slow or rate-limited, which is why the app tries three in sequence.
- Heatmap coordinates come from the model itself — treat them as approximate.
- DICOM handling is basic (pixel data → image); no Hounsfield-unit windowing or series navigation.

## License

MIT — see [LICENSE](LICENSE).
