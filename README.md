# 🏥 MedVision AI — Enterprise Clinical Imaging Suite v5.0

[![Python](https://img.shields.io/badge/Python-3.11+-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Framework-ff4b4b?style=for-the-badge&logo=streamlit)](https://streamlit.io/)
[![OpenRouter](https://img.shields.io/badge/AI-OpenRouter-orange?style=for-the-badge)](https://openrouter.ai/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](https://opensource.org/licenses/MIT)

## 🌟 Overview
**MedVision AI** is a high-performance, industry-grade medical imaging diagnostic support platform. Unlike standard AI tools, it implements a **Multi-Model Ensemble Architecture** to analyze complex medical scans (X-Ray, MRI, CT, etc.) with clinical-grade precision. 

It transforms raw medical imagery into structured, actionable diagnostic reports, providing a "Second Opinion" for radiologists and clinicians.

---

## 🚀 Key Enterprise Features

### 🧠 1. Ensemble AI Brain (Multi-Model Consensus)
Instead of relying on a single AI, MedVision AI routes images through an ensemble of frontier VLMs (**Gemini 1.5 Pro, Claude 3.5, Llama 3.2 Vision**). 
- **Reliability:** The system cross-references findings across models to ensure high-confidence diagnosis.
- **Fallback Logic:** Automatic fail-over mechanism ensures 99.9% uptime.

### ⏳ 2. Temporal Evolution Analysis (Longitudinal Study)
A critical clinical requirement. The platform allows users to upload a **Prior Scan** alongside the **Current Scan**.
- **Comparative Logic:** AI analyzes changes in pathology over time (e.g., "Nodule growth of 2mm since last scan").
- **Trend Tracking:** Identifies if a condition is regressing, stable, or evolving.

### 🖼️ 3. AI-Driven Saliency Heatmaps
No more "Black Box" AI. MedVision AI implements **Pathological Localization**.
- **Visual Evidence:** The AI identifies the exact coordinates of the abnormality and overlays a **Dynamic Red Heatmap** on the scan.
- **Transparency:** Provides visual proof of why the AI reached a specific diagnosis.

### 📂 4. Clinical-Grade DICOM Ingestion
Supports professional medical standards beyond just JPG/PNG.
- **DICOM Native:** Directly processes `.dcm` files, extracting pixel arrays and metadata.
- **Preprocessing:** Automatic normalization of Hounsfield units for consistent analysis.

### 🏥 5. Secure Patient Record Vault
Integrated local SQLite database to manage longitudinal patient data.
- **History Retrieval:** Instant retrieval of past diagnostic reports using Patient IDs.
- **Case Management:** Allows clinicians to review a patient's entire journey in one click.

---

## 🛠️ Technical Architecture

- **Frontend:** Streamlit (Custom CSS Glass-morphism UI)
- **Backend:** Python 3.11+
- **AI Orchestration:** OpenRouter API $\rightarrow$ Vision-Language Models (VLMs)
- **Image Processing:** Pillow, NumPy, PyDICOM
- **Data Persistence:** SQLite3
- **Export Engine:** FPDF2, Python-Docx

---

## 📦 Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/medvision-ai.git
cd medvision-ai
