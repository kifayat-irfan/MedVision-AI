import streamlit as st
from PIL import Image
import re

from styles import apply_custom_styles
from utils import call_ensemble_ai, draw_heatmap
from dicom_utils import process_dicom
from db import db
from report_gen import generate_pdf, generate_docx, generate_txt

st.set_page_config(page_title="MedVision AI", page_icon="🏥", layout="wide")
apply_custom_styles()

# ── Session state ────────────────────────────────────────────────────────────
for key, default in [("analysis_done", False), ("report", None),
                     ("final_img", None), ("model", "N/A")]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="mv-header">'
    "<h1>🏥 MedVision AI</h1>"
    "<p>AI-assisted medical scan analysis — structured reports with visual finding localization</p>"
    '<span class="mv-badge">Clinical Decision Support · Educational Demo</span>'
    "</div>",
    unsafe_allow_html=True,
)

if db.mode == "CLOUD":
    st.markdown(
        '<div class="mv-status cloud">🟢 Connected to cloud database</div>',
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        '<div class="mv-status local">🟡 Running on local database</div>',
        unsafe_allow_html=True,
    )

# ── Patient history ──────────────────────────────────────────────────────────
with st.expander("📁 Patient History — retrieve past reports"):
    pid_search = st.text_input("Patient ID", placeholder="e.g. PAT-001", key="hist_search")
    if pid_search:
        history = db.get_history(pid_search)
        if history:
            st.write(f"Found **{len(history)}** report(s) for **{pid_search}**")
            for mod, rep, date in history:
                with st.expander(f"📅 {date} · {mod}"):
                    st.text(rep[:1500] + ("…" if len(rep) > 1500 else ""))
                    if st.button("Restore to dashboard", key=f"rest_{date}_{mod}"):
                        st.session_state.report = rep
                        st.session_state.analysis_done = True
                        st.session_state.final_img = None
                        st.rerun()
        else:
            st.info("No records found for this Patient ID.")

# ── Main layout ──────────────────────────────────────────────────────────────
col_left, col_right = st.columns([1, 1.4], gap="large")

with col_left:
    st.markdown('<div class="mv-card">', unsafe_allow_html=True)
    st.markdown('<div class="mv-card-title">Patient Information</div>', unsafe_allow_html=True)

    p_id = st.text_input("Patient ID", value="PAT-001")
    c1, c2 = st.columns(2)
    with c1:
        age = st.number_input("Age", 0, 120, 25)
    with c2:
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    modality = st.selectbox("Modality", ["X-Ray", "MRI", "CT Scan", "Ultrasound", "Dermatology"])
    history_txt = st.text_area("Clinical history / symptoms",
                               placeholder="Brief clinical context…", height=80)

    st.markdown('<div class="mv-card-title" style="margin-top:8px">Scans</div>', unsafe_allow_html=True)
    current_file = st.file_uploader("Current scan", type=["jpg", "png", "jpeg", "dcm"])
    prior_file = st.file_uploader("Prior scan (optional — for comparison)",
                                  type=["jpg", "png", "jpeg", "dcm"])

    current_img = None
    if current_file:
        current_img = (process_dicom(current_file)
                       if current_file.name.lower().endswith(".dcm")
                       else Image.open(current_file))
        st.image(current_img, caption="Current scan", use_container_width=True)

    analyze = st.button("🔍 Analyze Scan", type="primary", disabled=not current_img)
    if analyze and current_img:
        with st.status("Analyzing scan…", expanded=True) as status:
            try:
                # ── Step 1: Route to the trained CNN for this modality ──
                from multi_model import predict_modality, model_info, gradcam_for
                st.write(f"🧠 {model_info(modality)}")
                predictions = predict_modality(current_img, modality)
                if predictions:
                    st.session_state.predictions = predictions
                    top = sorted(predictions.items(), key=lambda kv: kv[1], reverse=True)[:3]
                    st.write("✅ CNN analysis: " + ", ".join(f"{p} ({s:.0%})" for p, s in top))
                    # ── Step 2: Real Grad-CAM heatmap ──
                    st.write("🔥 Generating Grad-CAM visual explanation…")
                    st.session_state.final_img = gradcam_for(current_img, modality)
                else:
                    st.session_state.predictions = None
                    st.session_state.final_img = None
                    st.write("ℹ️ AI-assist mode — no trained CNN for this modality.")

                # ── Step 3: VLM structured report, grounded in CNN outputs ──
                prior_img = None
                if prior_file:
                    st.write("Comparing with prior scan…")
                    prior_img = (process_dicom(prior_file)
                                 if prior_file.name.lower().endswith(".dcm")
                                 else Image.open(prior_file))

                p_str = f"ID: {p_id}, Age: {age}, Gender: {gender}, History: {history_txt}"
                try:
                    report, model = call_ensemble_ai(current_img, prior_img, modality, p_str,
                                                     cnn_predictions=predictions)
                except Exception as vlm_err:
                    # Do NOT lose the already-computed CNN + Grad-CAM just
                    # because the VLM step failed (2026-10-24 live failure:
                    # "Model returned empty content" wiped the whole analysis).
                    # Show a CNN-only structured report with a clear warning.
                    st.warning(
                        f"VLM report unavailable ({vlm_err}). Showing CNN-only "
                        "results below — retry Analyze Scan for the full report."
                    )
                    if predictions:
                        top_lines = "\n".join(
                            f"{i + 1}. {p} ({s:.0%})"
                            for i, (p, s) in enumerate(
                                sorted(predictions.items(),
                                       key=lambda kv: kv[1], reverse=True))
                        )
                        report = (
                            "TECHNICAL QUALITY:\nCNN analysis completed; "
                            "VLM structured report unavailable in this run.\n\n"
                            f"FINDINGS:\n{top_lines}\n\n"
                            "TEMPORAL COMPARISON:\nN/A.\n\n"
                            "FINAL IMPRESSION:\nCNN-only result — for professional "
                            "review, not a diagnosis.\n\n"
                            "TREATMENT ROADMAP:\n1. Review the CNN findings and "
                            "Grad-CAM with a qualified professional. "
                            "2. Retry the full analysis. 3. Follow up as advised.\n\n"
                            "EVIDENCE BASE:\nCNN model output only; no VLM report "
                            "generated in this run."
                        )
                    else:
                        report = (
                            "TECHNICAL QUALITY:\nVLM report unavailable in this run.\n\n"
                            "FINDINGS:\nNo CNN model for this modality; VLM report "
                            "could not be generated.\n\n"
                            "TEMPORAL COMPARISON:\nN/A.\n\n"
                            "FINAL IMPRESSION:\nNo report generated — retry the analysis.\n\n"
                            "TREATMENT ROADMAP:\n1. Retry the analysis.\n\n"
                            "EVIDENCE BASE:\nN/A."
                        )
                    model = "CNN-only (VLM unavailable)"
                st.session_state.report = report
                st.session_state.analysis_done = True
                st.session_state.model = model

                st.write("Saving report…")
                if not db.save_report(p_id, modality, report):
                    st.warning("Report generated but could not be saved to the database. "
                               "You can still download it below.")
                status.update(label="Analysis complete", state="complete", expanded=False)
            except Exception as e:
                st.error(f"Analysis failed: {e}")
                status.update(label="Analysis failed", state="error", expanded=True)

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        '<div class="mv-disclaimer">'
        "⚠️ <b>Not a medical diagnosis.</b> This AI-generated report is a decision-support "
        "draft for educational purposes only. A qualified healthcare professional must "
        "review all findings before any clinical decision."
        "</div>",
        unsafe_allow_html=True,
    )

with col_right:
    if not st.session_state.analysis_done:
        st.markdown(
            '<div class="mv-empty">'
            "<h3>No analysis yet</h3>"
            "<p>Upload a scan on the left and click <b>Analyze Scan</b>.<br>"
            "The structured report will appear here.</p>"
            "</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown('<div class="mv-card">', unsafe_allow_html=True)
        st.markdown('<div class="mv-card-title">Analysis Report</div>', unsafe_allow_html=True)

        m1, m2 = st.columns(2)
        with m1:
            st.markdown(f"**AI model:** `{st.session_state.model}`")
        with m2:
            st.markdown(f"**Patient:** {p_id} · {modality}")

        # ── CNN predictions (if a trained model exists for this modality) ──
        preds = st.session_state.get("predictions")
        if preds:
            from multi_model import model_info as _mi
            st.markdown(f"**Model outputs** *({_mi(modality)})*")
            top5 = sorted(preds.items(), key=lambda kv: kv[1], reverse=True)[:5]
            for p, s in top5:
                c1, c2 = st.columns([3, 1])
                with c1:
                    st.progress(s, text=str(p).replace("_", " "))
                with c2:
                    st.markdown(f"**{s:.0%}**{' 🔴' if s >= 0.5 else ''}")

        if st.session_state.final_img is not None:
            st.markdown("**Grad-CAM visual explanation** *(where the CNN looked)*")
            st.image(st.session_state.final_img, use_container_width=True)
        else:
            st.info("Heatmap not available for restored reports.")

        # ── Render structured report ──
        def render_report(raw: str):
            clean = raw
            for pat, rep in [(r"#{1,6}\s*", ""), (r"\*{1,3}(.*?)\*{1,3}", r"\1"),
                             (r"_{1,2}(.*?)_{1,2}", r"\1"), (r"`{1,3}(.*?)`{1,3}", r"\1")]:
                clean = re.sub(pat, rep, clean)

            section_re = re.compile(
                r"(TECHNICAL QUALITY|FINDINGS|TEMPORAL COMPARISON|"
                r"FINAL IMPRESSION|TREATMENT ROADMAP|EVIDENCE BASE)",
                re.IGNORECASE,
            )
            parts = section_re.split(clean)
            colors = {
                "TECHNICAL QUALITY": "#0284c7",
                "FINDINGS": "#7c3aed",
                "TEMPORAL COMPARISON": "#d97706",
                "FINAL IMPRESSION": "#dc2626",
                "TREATMENT ROADMAP": "#16a34a",
                "EVIDENCE BASE": "#0891b2",
            }
            if len(parts) <= 1:
                st.markdown(f'<div class="mv-section-body">{clean}</div>', unsafe_allow_html=True)
                return
            for i in range(1, len(parts), 2):
                header = parts[i].strip().upper()
                body = parts[i + 1].strip() if i + 1 < len(parts) else ""
                color = colors.get(header, "#64748b")
                st.markdown(
                    f'<div class="mv-section-head" style="background:{color}14;'
                    f'color:{color};border-left:4px solid {color}">{header}</div>'
                    f'<div class="mv-section-body">{body}</div>',
                    unsafe_allow_html=True,
                )

        render_report(st.session_state.report)

        st.markdown("**Export report**")
        d1, d2, d3 = st.columns(3)
        p_full = f"ID: {p_id}, Age: {age}, Gender: {gender}"
        with d1:
            st.download_button("📄 PDF", generate_pdf(st.session_state.report, p_full, modality),
                               "medvision-report.pdf", use_container_width=True)
        with d2:
            st.download_button("📝 DOCX", generate_docx(st.session_state.report, p_full, modality),
                               "medvision-report.docx", use_container_width=True)
        with d3:
            st.download_button("📃 TXT", generate_txt(st.session_state.report),
                               "medvision-report.txt", use_container_width=True)

        if st.button("↺ New analysis"):
            for k in ("analysis_done", "report", "final_img"):
                st.session_state[k] = False if k == "analysis_done" else None
            st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)
