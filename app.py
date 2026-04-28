import streamlit as st
from PIL import Image
import time

from styles import apply_custom_styles
from utils import call_ensemble_ai, draw_heatmap
from dicom_utils import process_dicom
from db import db
from report_gen import generate_pdf, generate_docx, generate_txt

st.set_page_config(page_title="MedVision AI | All-in-One Suite", layout="wide")
apply_custom_styles()

# ── Session State Initialization ──────────────────────────────────────────────
if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False
if "report" not in st.session_state:
    st.session_state.report = None
if "final_img" not in st.session_state:
    # CRITICAL FIX #4: Guard against KeyError when a report is restored from
    # history (analysis_done=True but final_img was never set in that session).
    st.session_state.final_img = None
if "model" not in st.session_state:
    st.session_state.model = "N/A"

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="command-center">', unsafe_allow_html=True)
st.markdown(
    '<div class="cyber-header">'
    '<div class="cyber-title">MedVision AI</div>'
    '<div style="color: #60a5fa;">The Complete Enterprise Intelligence Suite v5.0</div>'
    "</div>",
    unsafe_allow_html=True,
)

if db.mode == "CLOUD":
    st.markdown(
        '<div style="text-align: center; color: #22c55e; font-size: 0.8rem; margin-bottom: 20px;">'
        "🟢 System Status: Connected to Cloud Database (Supabase)"
        "</div>",
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        '<div style="text-align: center; color: #facc15; font-size: 0.8rem; margin-bottom: 20px;">'
        "🟡 System Status: Running on Local Database (SQLite)"
        "</div>",
        unsafe_allow_html=True,
    )

# ── 1. PATIENT HISTORY (DATABASE) ─────────────────────────────────────────────
with st.expander("📁 Patient Archive & History (Retrieve Past Records)"):
    pid_search = st.text_input(
        "Enter Patient ID to retrieve records", placeholder="e.g. PAT-001"
    )
    if pid_search:
        history = db.get_history(pid_search)
        if history:
            st.markdown(f"Found **{len(history)}** records for **{pid_search}**")
            for mod, rep, date in history:   # Always tuples now — Bug #3 fixed
                with st.expander(f"📅 {date} | 🔬 {mod}"):
                    st.markdown("---")
                    st.markdown("**Full Diagnostic Report:**")
                    st.markdown(
                        f'<div style="background: #0f172a; color: #e2e8f0; padding: 15px; '
                        f'border-radius: 10px; border: 1px solid #3b82f6; font-family: Inter; '
                        f'white-space: pre-wrap;">{rep}</div>',
                        unsafe_allow_html=True,
                    )
                    if st.button("Restore this report to dashboard", key=f"rest_{date}_{mod}"):
                        st.session_state.report = rep
                        st.session_state.analysis_done = True
                        st.session_state.final_img = None   # No heatmap for restored report
                        st.rerun()
        else:
            st.info("No records found for this Patient ID.")

# ── 2. MAIN INTERFACE ─────────────────────────────────────────────────────────
col_left, col_right = st.columns([1, 2], gap="large")

with col_left:
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.markdown(
        '<div style="font-weight: 700; color: #60a5fa; margin-bottom: 20px;">DATA ACQUISITION</div>',
        unsafe_allow_html=True,
    )

    p_id     = st.text_input("Patient ID", value="PAT-001")
    age      = st.number_input("Age", 0, 120, 25)
    gender   = st.selectbox("Gender", ["Male", "Female", "Other"])
    modality = st.selectbox("Modality", ["X-Ray", "MRI", "CT Scan", "Ultrasound", "Dermatology"])
    history  = st.text_area("Clinical History")

    st.markdown(
        '<div style="font-weight: 600; color: #fff; margin: 15px 0 5px 0;">📸 Current Scan</div>',
        unsafe_allow_html=True,
    )
    current_file = st.file_uploader(
        "Upload Current Scan", type=["jpg", "png", "jpeg", "dcm"], key="curr"
    )

    st.markdown(
        '<div style="font-weight: 600; color: #60a5fa; margin: 15px 0 5px 0;">'
        "⏳ Prior Scan (Optional for Comparison)"
        "</div>",
        unsafe_allow_html=True,
    )
    prior_file = st.file_uploader(
        "Upload Previous Scan", type=["jpg", "png", "jpeg", "dcm"], key="prior"
    )

    if current_file:
        current_img = (
            process_dicom(current_file)
            if current_file.name.endswith(".dcm")
            else Image.open(current_file)
        )

        st.markdown(
            '<div class="scan-container"><div class="scan-line"></div>', unsafe_allow_html=True
        )
        st.image(current_img, caption="Current Scan", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        if st.button("🚀 Execute Full Intelligence Analysis"):
            with st.status("Running Comprehensive Analysis...", expanded=True) as status:
                st.write("📡 Synchronizing Ensemble Vision Clusters...")
                time.sleep(1)

                prior_img = None
                if prior_file:
                    st.write("⚖️ Performing Temporal Comparison...")
                    prior_img = (
                        process_dicom(prior_file)
                        if prior_file.name.endswith(".dcm")
                        else Image.open(prior_file)
                    )
                    time.sleep(1)

                try:
                    p_str = f"ID: {p_id}, Age: {age}, Gender: {gender}, History: {history}"
                    report, model = call_ensemble_ai(current_img, prior_img, modality, p_str)

                    final_img = draw_heatmap(current_img, report)

                    st.session_state.report       = report
                    st.session_state.analysis_done = True
                    st.session_state.final_img    = final_img
                    st.session_state.model        = model

                    # ── SAVE TO DATABASE ──────────────────────────────────────
                    st.write("💾 Syncing report to database...")
                    saved = db.save_report(p_id, modality, report)

                    # CRITICAL FIX #5: Was `if not success: pass` — now we warn.
                    if not saved:
                        st.warning(
                            "⚠️ Report was generated successfully but could NOT be saved "
                            "to the database. See the error above for details. "
                            "You can still download the report below."
                        )

                    status.update(
                        label="Diagnostic Complete!", state="complete", expanded=False
                    )
                except Exception as e:
                    st.error(f"Analysis Error: {e}")

    st.markdown("</div>", unsafe_allow_html=True)

# ── 3. RIGHT PANEL — RESULTS ──────────────────────────────────────────────────
with col_right:
    if not st.session_state.analysis_done:
        st.markdown(
            '<div class="glass-panel" style="height: 600px; display: flex; align-items: center; '
            'justify-content: center; text-align: center;">'
            "<h2 style='color:white;'>System Idle</h2>"
            "<p style='color: #64748b;'>Awaiting diagnostic data for full clinical analysis...</p>"
            "</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown('<div class="glass-panel">', unsafe_allow_html=True)

        s1, s2, s3 = st.columns(3)
        with s1:
            st.markdown(
                f'<div class="med-card"><span class="card-title">AI Engine</span>'
                f'<div style="font-weight: 700;">{st.session_state.model}</div></div>',
                unsafe_allow_html=True,
            )
        with s2:
            st.markdown(
                '<div class="med-card"><span class="card-title">Consensus</span>'
                '<div style="font-weight: 700; color: #22c55e;">High Confidence</div></div>',
                unsafe_allow_html=True,
            )
        with s3:
            st.markdown(
                f'<div class="med-card"><span class="card-title">Patient ID</span>'
                f'<div style="font-weight: 700;">{p_id}</div></div>',
                unsafe_allow_html=True,
            )

        # CRITICAL FIX #4: Only render heatmap if one exists (won't exist for
        # restored-from-history reports, which have no image in this session).
        if st.session_state.final_img is not None:
            st.markdown(
                '<div style="text-align:center; margin: 20px 0;">'
                '<span class="card-title">AI Pathological Localization (Heatmap)</span>'
                "</div>",
                unsafe_allow_html=True,
            )
            st.image(st.session_state.final_img, width=500)
        else:
            st.info("ℹ️ Heatmap not available for restored historical reports.")

        st.markdown(
            f'<div style="background: #f8fafc; color: #0f172a; padding: 30px; '
            f'border-radius: 20px; margin: 20px 0; white-space: pre-wrap; '
            f'font-family: Inter; border: 1px solid #e2e8f0;">'
            f"{st.session_state.report}</div>",
            unsafe_allow_html=True,
        )

        d1, d2, d3 = st.columns(3)
        p_full = f"ID: {p_id}, Age: {age}, Gender: {gender}"
        with d1:
            st.download_button(
                "📄 PDF",
                generate_pdf(st.session_state.report, p_full, modality),
                "report.pdf",
            )
        with d2:
            st.download_button(
                "📝 DOCX",
                generate_docx(st.session_state.report, p_full, modality),
                "report.docx",
            )
        with d3:
            st.download_button(
                "📃 TXT",
                generate_txt(st.session_state.report),
                "report.txt",
            )

        if st.button("🔄 Reset Command Center"):
            st.session_state.analysis_done = False
            st.session_state.report        = None
            st.session_state.final_img     = None
            st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)