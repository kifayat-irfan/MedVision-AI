import requests
import base64
import streamlit as st
import re
from io import BytesIO
from PIL import Image, ImageDraw

SENSENOVA_BASE = "https://token.sensenova.ai/v1/chat/completions"

# SenseNova vision-capable model (tested 2026-10-24: correctly identified image colors)
SENSENOVA_MODEL = "sensenova-6.8-flash-lite"
# Fallback model tried only when the primary returns empty content (2026-10-24
# live runs failed with "Model returned empty content" on both Ultrasound and
# Dermatology). 401/429 errors are NOT retried on a fallback model.
SENSENOVA_FALLBACK_MODELS = ["sensenova-6.7-flash-lite"]

# Honest model label per modality for the grounding prompt. Previously the
# grounding text hardcoded "DenseNet121 / NIH ChestX-ray14" for EVERY modality,
# which is false for MRI, Ultrasound and Dermatology and could mislead the VLM.
MODEL_LABELS = {
    "X-Ray": "DenseNet121 trained on NIH ChestX-ray14 (112,120 images)",
    "MRI": "MobileNetV3-Small trained on brain MRI (5,369 images, 4 classes)",
    "CT Scan": "no trained CNN for this modality (VLM-only)",
    "Ultrasound": "MobileNetV3-Small trained on BUSI breast ultrasound (780 images, 3 classes)",
    "Dermatology": "EfficientNet-B0 trained on HAM10000 (10,015 images, 7 classes)",
}


def _extract_message_content(message):
    """Extract answer text from a SenseNova chat message, tolerating shape
    variations: string content, content-block lists, reasoning fields."""
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("text"):
                parts.append(str(block["text"]))
            elif isinstance(block, str):
                parts.append(block)
        content = "\n".join(parts)
    text = (content or "").strip() if isinstance(content, str) else ""
    if text:
        return text
    # Only use reasoning as a last resort, and only if it already looks like a
    # structured report (section headers) rather than an internal trace.
    for key in ("reasoning_content", "reasoning"):
        reasoning = message.get(key)
        if isinstance(reasoning, str) and "FINDINGS" in reasoning and "FINAL IMPRESSION" in reasoning:
            return reasoning.strip()
    return ""

def image_to_base64(img):
    buf = BytesIO()
    img.convert("RGB").save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()

def call_ensemble_ai(current_img, prior_img, modality, patient_data, cnn_predictions=None):
    # SenseNova API key — clear error message instead of raw KeyError
    api_key = st.secrets.get("SENSENOVA_API_KEY", "").strip()
    if not api_key:
        raise Exception(
            "SENSENOVA_API_KEY is missing. Add it in Streamlit Secrets "
            "(App settings → Secrets)."
        )

    # ── Grounding: real CNN outputs so the VLM explains instead of inventing ──
    grounding = ""
    if cnn_predictions:
        lines = []
        for p, s in sorted(cnn_predictions.items(), key=lambda kv: kv[1], reverse=True):
            flag = " <-- ELEVATED (≥0.50)" if s >= 0.5 else ""
            lines.append(f"- {p}: {s:.2f}{flag}")
        label = MODEL_LABELS.get(modality, "trained CNN for this modality")
        grounding = (
            f"\n\nCNN MODEL OUTPUT ({label}). These are REAL model probabilities — "
            "ground every claim in your report in these numbers. Do NOT invent "
            "findings that contradict them. If no pathology is elevated, say the "
            "scan appears unremarkable:\n"
            + "\n".join(lines)
        )
    
    messages_content = []
    messages_content.append({"type": "text", "text": "CURRENT SCAN (Today's Image):"})
    messages_content.append({
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{image_to_base64(current_img)}"}
    })

    if prior_img:
        messages_content.append({"type": "text", "text": "PRIOR SCAN (Previous Record):"})
        messages_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{image_to_base64(prior_img)}"}
        })

    prompt = f"""
    You are MedVision AI, a clinical decision-support assistant. Analyze the medical scan and return a structured report draft for review by a qualified healthcare professional. This is educational / decision-support output, not a diagnosis.

PATIENT INFO: {patient_data}
MODALITY: {modality}{grounding}

STRICT RULES — YOU MUST FOLLOW THESE:
- Do NOT use markdown symbols like ###, **, --, or *.
- Do NOT write long paragraphs or stories.
- Use ONLY the exact section headers shown below, in ALL CAPS, followed by a colon.
- Each finding should be one clear, concise clinical sentence.
- Be direct and clinical. No filler words.

OUTPUT FORMAT (copy this structure exactly):

TECHNICAL QUALITY:
[One sentence about image resolution, exposure, and artifacts.]

FINDINGS:
[Bullet each finding as a plain numbered point like: 1. Finding here. 2. Finding here.]

TEMPORAL COMPARISON:
[If prior scan provided: compare and state if condition improved, worsened, or stable. If no prior scan: write N/A.]

FINAL IMPRESSION:
[One to two sentence clinical impression for professional review. Do not present this as a definitive diagnosis.]

TREATMENT ROADMAP:
[Numbered list: 1. Medication/action. 2. Lifestyle change. 3. Follow-up timeline.]

EVIDENCE BASE:
[Cite one or two relevant guidelines: ACR, WHO, RSNA, or similar.]

HEATMAP: [x, y, radius] (0-100 scale, identify the most critical finding location)
    """
    messages_content.append({"type": "text", "text": prompt})

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    # Try the primary model first, then a fallback model if the primary returns
    # empty content. This is the fix for the 2026-10-24 live failure where both
    # Ultrasound and Dermatology runs died with "Model returned empty content"
    # and the whole analysis (including the already-computed CNN + Grad-CAM)
    # was lost.
    last_error = None
    for model in [SENSENOVA_MODEL] + SENSENOVA_FALLBACK_MODELS:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": messages_content}],
            "max_tokens": 2000,
        }
        try:
            st.write(f"🔄 Analyzing with `{model}`...")
            resp = requests.post(SENSENOVA_BASE, headers=headers, json=payload, timeout=120)

            if resp.status_code == 200:
                data = resp.json()
                message = data.get("choices", [{}])[0].get("message", {})
                content = _extract_message_content(message)
                if content:
                    st.write("✅ Analysis complete")
                    return content, model
                last_error = Exception(
                    f"Model {model} returned empty content. Full response: {data}"
                )
                continue

            if resp.status_code == 401:
                raise Exception(
                    "❌ Invalid SENSENOVA_API_KEY. Check your key in Streamlit Secrets."
                )
            if resp.status_code == 429:
                raise Exception("⚠️ Rate limited. Please wait a moment and try again.")
            last_error = Exception(
                f"Model {model} failed. Status: {resp.status_code} | "
                f"Response: {resp.text[:300]}"
            )
            continue

        except requests.exceptions.Timeout:
            last_error = Exception("⏱️ Model timed out after 120 seconds. Try again.")
            continue
        except requests.exceptions.ConnectionError as e:
            last_error = Exception(f"🌐 Connection error: {str(e)}")
            continue

    raise last_error if last_error else Exception("Model returned empty content.")


def draw_heatmap(image, report_text):
    try:
        match = re.search(r"HEATMAP: \[(\d+),\s*(\d+),\s*(\d+)\]", report_text)
        if match:
            x, y, r = map(int, match.groups())
            if x == 0 and y == 0:
                return image
            img_copy = image.copy()
            draw = ImageDraw.Draw(img_copy)
            w, h = img_copy.size
            cx = int(x * w / 100)
            cy = int(y * h / 100)
            radius = int(r * w / 100)
            # Outer glow ring
            draw.ellipse(
                [cx - radius - 4, cy - radius - 4, cx + radius + 4, cy + radius + 4],
                outline="yellow", width=2
            )
            # Main circle
            draw.ellipse(
                [cx - radius, cy - radius, cx + radius, cy + radius],
                outline="red", width=5
            )
            # Center dot
            draw.ellipse(
                [cx - 4, cy - 4, cx + 4, cy + 4],
                fill="red"
            )
            return img_copy
    except Exception:
        pass
    return image