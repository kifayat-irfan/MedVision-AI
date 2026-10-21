# MedVision AI — Medical Scan Analysis Assistant

[![Python](https://img.shields.io/badge/Python-3.11+-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-ff4b4b)](https://streamlit.io/)
[![OpenRouter](https://img.shields.io/badge/AI-OpenRouter-orange)](https://openrouter.ai/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

> ⚠️ **Disclaimer:** This is an educational demo project. The AI-generated analysis is a *second-opinion-style draft*,
> **not a medical diagnosis** and not a medical device. Do not use it for clinical decisions.

A Streamlit app that analyzes medical scans (X-ray, MRI, CT) with vision-language models and produces a
structured, readable report — with the key finding highlighted directly on the image.

## What it does

1. **Upload a scan** — JPG/PNG, or DICOM (`.dcm`) which is converted to an image automatically.
2. **Optional prior scan** — upload an older scan of the same patient for temporal comparison.
3. **AI analysis** — the image is sent to vision-language models via OpenRouter. Three free models are tried
   in order with automatic fallback if one fails or rate-limits:
   - `nvidia/nemotron-nano-12b-v2-vl:free`
   - `google/gemma-4-26b-a4b-it:free`
   - `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free`
4. **Structured report** — the model returns fixed sections: technical quality, findings, temporal comparison
   (improved / stable / worsened vs. the prior scan), final impression, treatment roadmap, evidence base,
   and heatmap coordinates for the most critical finding.
5. **Heatmap overlay** — the reported coordinates are drawn on the scan (Pillow) so you can see *where*
   the model is pointing.
6. **Patient vault** — reports are saved per patient ID. Works with local SQLite out of the box; switches to
   Supabase (PostgreSQL) automatically when `SUPABASE_URL` / `SUPABASE_KEY` are provided.
7. **Export** — download the report as PDF, DOCX, or TXT. Past reports can be retrieved by patient ID and
   restored to the dashboard.

## Tech stack

- **App:** Streamlit (Python)
- **AI:** OpenRouter API — free vision-language models with fallback chain
- **Imaging:** Pillow (heatmap overlay), pydicom + NumPy (DICOM → image via min-max normalization)
- **Storage:** SQLite (local default) / Supabase PostgreSQL (cloud, optional)
- **Export:** fpdf2 (PDF), python-docx (DOCX)

## Setup

```bash
git clone https://github.com/kifayat-irfan/MedVision-AI.git
cd MedVision-AI
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Create `.streamlit/secrets.toml`:

```toml
OPENROUTER_API_KEY = "sk-or-your-key-here"   # required — get one free at openrouter.ai
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
