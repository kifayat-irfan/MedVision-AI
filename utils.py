import requests
import base64
import streamlit as st
import re
from io import BytesIO
from PIL import Image, ImageDraw

OPENROUTER_BASE = "https://openrouter.ai/api/v1/chat/completions"

# FIXED: Correct OpenRouter model name format
MODELS = [
    "nvidia/nemotron-nano-12b-v2-vl:free",               # FREE - vision-language model
    "google/gemma-4-26b-a4b-it:free",                    # FREE - Gemma 4 multimodal
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free", # FREE - omni vision model
]

def image_to_base64(img):
    buf = BytesIO()
    img.convert("RGB").save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()

def call_ensemble_ai(current_img, prior_img, modality, patient_data):
    api_key = st.secrets["OPENROUTER_API_KEY"]
    
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
    You are MedVision AI, a board-certified radiologist AI. Analyze the medical scan and return a structured clinical report.

PATIENT INFO: {patient_data}
MODALITY: {modality}

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
[One to two sentence definitive clinical diagnosis.]

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
        "HTTP-Referer": "https://medvision-ai24.streamlit.app",
        "X-Title": "MedVision AI"
    }

    last_error = ""

    for model in MODELS:
        try:
            st.write(f"🔄 Trying model: `{model}`...")

            payload = {
                "model": model,
                "messages": [{"role": "user", "content": messages_content}]
            }

            resp = requests.post(
                OPENROUTER_BASE,
                headers=headers,
                json=payload,
                timeout=90
            )

            # ── Verbose error logging (replaces silent except: continue) ──
            if resp.status_code == 200:
                data = resp.json()
                content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                if content:
                    st.write(f"✅ Success with `{model}`")
                    return content, model
                else:
                    last_error = f"Model `{model}` returned empty content. Full response: {data}"
                    st.warning(f"⚠️ {last_error}")

            elif resp.status_code == 401:
                last_error = "❌ Invalid API Key. Go to openrouter.ai and check your OPENROUTER_API_KEY in Streamlit Secrets."
                st.error(last_error)
                break  # No point trying other models with wrong key

            elif resp.status_code == 402:
                last_error = f"❌ Out of credits for `{model}`. Add credits at openrouter.ai/credits"
                st.warning(last_error)

            elif resp.status_code == 429:
                last_error = f"⚠️ Rate limited on `{model}`. Trying next..."
                st.warning(last_error)

            else:
                last_error = f"Model `{model}` failed. Status: {resp.status_code} | Response: {resp.text[:300]}"
                st.warning(f"⚠️ {last_error}")

        except requests.exceptions.Timeout:
            last_error = f"⏱️ Model `{model}` timed out after 90 seconds."
            st.warning(last_error)

        except requests.exceptions.ConnectionError as e:
            last_error = f"🌐 Connection error on `{model}`: {str(e)}"
            st.warning(last_error)

        except Exception as e:
            last_error = f"💥 Unexpected error on `{model}`: {type(e).__name__}: {str(e)}"
            st.warning(last_error)

    # All models failed — raise with full context
    raise Exception(
        f"All 3 AI models failed.\n\n"
        f"Last error: {last_error}\n\n"
        f"Check:\n"
        f"1. openrouter.ai/credits — do you have credits?\n"
        f"2. Is OPENROUTER_API_KEY correct in Streamlit Secrets?\n"
        f"3. Try openrouter.ai/chat to test the model manually."
    )


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