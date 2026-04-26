import requests
import base64
import streamlit as st
import json
import re
from io import BytesIO
from PIL import Image, ImageDraw

OPENROUTER_BASE = "https://openrouter.ai/api/v1/chat/completions"
MODELS = ["google/gemini-pro-1.5", "google/gemini-flash-1.5", "meta-llama/llama-3.2-11b-vision-instruct"]

def image_to_base64(img):
    buf = BytesIO()
    img.convert("RGB").save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()

def call_ensemble_ai(current_img, prior_img, modality, patient_data):
    api_key = st.secrets["OPENROUTER_API_KEY"]
    messages_content = []
    
    # Current Image
    messages_content.append({"type": "text", "text": "CURRENT SCAN (Today's Image):"})
    messages_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_to_base64(current_img)}"}})
    
    # Prior Image (if exists)
    if prior_img:
        messages_content.append({"type": "text", "text": "PRIOR SCAN (Previous Record):"})
        messages_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_to_base64(prior_img)}"}})
    
    prompt = f"""
    You are MedVision AI, a board-certified virtual radiologist. 
    MODALITY: {modality} | PATIENT: {patient_data}
    
    REQUIRED ANALYSIS:
    1. TECHNICAL QUALITY: Assess image resolution and artifacts.
    2. FINDINGS: Detailed anatomical description.
    3. TEMPORAL COMPARISON: If a prior scan is provided, compare them. State if pathology has evolved, regressed, or remained stable. If no prior scan, state 'N/A'.
    4. FINAL IMPRESSION: Definitive diagnosis based on evidence.
    5. TREATMENT ROADMAP: Suggested medications, lifestyle changes, and follow-up timeline.
    6. EVIDENCE BASE: Cite guidelines (ACR, WHO, RSNA).
    7. HEATMAP_COORDS: Identify critical pathology point as [x, y, radius] (0-100%). Example: HEATMAP: [50, 20, 10].
    """
    messages_content.append({"type": "text", "text": prompt})

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    
    for model in MODELS:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": messages_content}]
        }
        try:
            resp = requests.post(OPENROUTER_BASE, headers=headers, json=payload, timeout=60)
            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]["content"], model
        except: continue
    raise Exception("Ensemble Cluster Failure.")

def draw_heatmap(image, report_text):
    try:
        match = re.search(r"HEATMAP: \[(\d+),\s*(\d+),\s*(\d+)\]", report_text)
        if match:
            x, y, r = map(int, match.groups())
            if x == 0 and y == 0: return image
            draw = ImageDraw.Draw(image)
            w, h = image.size
            cx, cy = int(x * w / 100), int(y * h / 100)
            radius = int(r * w / 100)
            draw.ellipse([cx-radius, cy-radius, cx+radius, cy+radius], outline="red", width=5)
            return image
    except: pass
    return image
