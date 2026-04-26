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

if 'analysis_done' not in st.session_state: st.session_state.analysis_done = False
if 'report' not in st.session_state: st.session_state.report = None

st.markdown('<div class="command-center">', unsafe_allow_html=True)
st.markdown('<div class="cyber-header"><div class="cyber-title">MedVision AI</div><div style="color: #60a5fa;">The Complete Enterprise Intelligence Suite v5.0</div></div>', unsafe_allow_html=True)

# --- 1. PATIENT HISTORY (DATABASE) - PRESERVED ---
with st.expander("📁 Patient Archive & History (Retrieve Past Records)"):
    pid_search = st.text_input("Enter Patient ID to retrieve records", placeholder="e.g. PAT-001")
    if pid_search:
        history = db.get_history(pid_search)
        if history:
            st.markdown(f"Found {len(history)} records for **{pid_search}**")
            for mod, rep, date in history:
                # Har report ke liye ek alag expander banayein taake poora data dikhe
                with st.expander(f"📅 {date} | 🔬 {mod}"):
                    st.markdown("---")
                    st.markdown("**Full Diagnostic Report:**")
                    st.markdown(f'<div style="background: #0f172a; color: #e2e8f0; padding: 15px; border-radius: 10px; border: 1px solid #3b82f6; font-family: "Inter"; white-space: pre-wrap;">{rep}</div>', unsafe_allow_html=True)
                    
                    # Extra Feature: Restore this report to main dashboard
                    if st.button(f"Restore this report to dashboard", key=f"rest_{date}_{mod}"):
                        st.session_state.report = rep
                        st.session_state.analysis_done = True
                        st.rerun()
        else:
            st.info("No records found for this Patient ID.")


# --- MAIN INTERFACE ---
col_left, col_right = st.columns([1, 2], gap="large")

with col_left:
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.markdown('<div style="font-weight: 700; color: #60a5fa; margin-bottom: 20px;">DATA ACQUISITION</div>', unsafe_allow_html=True)
    
    p_id = st.text_input("Patient ID", value="PAT-001")
    age = st.number_input("Age", 0, 120, 25)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    modality = st.selectbox("Modality", ["X-Ray", "MRI", "CT Scan", "Ultrasound", "Dermatology"])
    history = st.text_area("Clinical History")
    
    # --- DUAL IMAGE UPLOAD (NEW & PRESERVED) ---
    st.markdown('<div style="font-weight: 600; color: #fff; margin: 15px 0 5px 0;">📸 Current Scan</div>', unsafe_allow_html=True)
    current_file = st.file_uploader("Upload Current Scan", type=["jpg", "png", "jpeg", "dcm"], key="curr")
    
    st.markdown('<div style="font-weight: 600; color: #60a5fa; margin: 15px 0 5px 0;">⏳ Prior Scan (Optional for Comparison)</div>', unsafe_allow_html=True)
    prior_file = st.file_uploader("Upload Previous Scan", type=["jpg", "png", "jpeg", "dcm"], key="prior")
    
    if current_file:
        # Process Current Image
        if current_file.name.endswith('.dcm'): current_img = process_dicom(current_file)
        else: current_img = Image.open(current_file)
            
        st.markdown('<div class="scan-container"><div class="scan-line"></div>', unsafe_allow_html=True)
        st.image(current_img, caption="Current Scan", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        if st.button("🚀 Execute Full Intelligence Analysis"):
            with st.status("Running Comprehensive Analysis...", expanded=True) as status:
                st.write("📡 Synchronizing Ensemble Vision Clusters...")
                time.sleep(1)
                
                # Process Prior Image
                prior_img = None
                if prior_file:
                    st.write("⚖️ Performing Temporal Comparison...")
                    if prior_file.name.endswith('.dcm'): prior_img = process_dicom(prior_file)
                    else: prior_img = Image.open(prior_file)
                    time.sleep(1)
                
                try:
                    p_str = f"ID: {p_id}, Age: {age}, Gender: {gender}, History: {history}"
                    report, model = call_ensemble_ai(current_img, prior_img, modality, p_str)
                    
                    # Apply Heatmap
                    final_img = draw_heatmap(current_img, report)
                    
                    st.session_state.report, st.session_state.analysis_done = report, True
                    st.session_state.final_img, st.session_state.model = final_img, model
                    
                    # SAVE TO DATABASE
                    db.save_report(p_id, modality, report)
                    
                    status.update(label="Diagnostic Complete!", state="complete", expanded=False)
                except Exception as e:
                    st.error(f"Error: {e}")
    st.markdown('</div>', unsafe_allow_html=True)

with col_right:
    if not st.session_state.analysis_done:
        st.markdown('<div class="glass-panel" style="height: 600px; display: flex; align-items: center; justify-content: center; text-align: center;"><h2 style="color:white;">System Idle</h2><p style="color: #64748b;">Awaiting diagnostic data for full clinical analysis...</p></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
        
        # Stats Row
        s1, s2, s3 = st.columns(3)
        with s1: st.markdown('<div class="med-card"><span class="card-title">AI Engine</span><div style="font-weight: 700;">Gemini 1.5 Pro</div></div>', unsafe_allow_html=True)
        with s2: st.markdown('<div class="med-card"><span class="card-title">Consensus</span><div style="font-weight: 700; color: #22c55e;">High Confidence</div></div>', unsafe_allow_html=True)
        with s3: st.markdown('<div class="med-card"><span class="card-title">Patient ID</span><div style="font-weight: 700;">'+p_id+'</div></div>', unsafe_allow_html=True)

        # Visual Output
        st.markdown('<div style="text-align:center; margin: 20px 0;"><span class="card-title">AI Pathological Localization (Heatmap)</span></div>', unsafe_allow_html=True)
        st.image(st.session_state.final_img, width=500)

        # Report Box
        st.markdown(f'<div style="background: #f8fafc; color: #0f172a; padding: 30px; border-radius: 20px; margin: 20px 0; white-space: pre-wrap; font-family: "Inter"; border: 1px solid #e2e8f0;">{st.session_state.report}</div>', unsafe_allow_html=True)
        
        # Exports
        d1, d2, d3 = st.columns(3)
        p_full = f"ID: {p_id}, Age: {age}, Gender: {gender}"
        with d1: st.download_button("📄 PDF", generate_pdf(st.session_state.report, p_full, modality), "report.pdf")
        with d2: st.download_button("📝 DOCX", generate_docx(st.session_state.report, p_full, modality), "report.docx")
        with d3: st.download_button("📃 TXT", generate_txt(st.session_state.report), "report.txt")
        
        if st.button("🔄 Reset Command Center"):
            st.session_state.analysis_done = False
            st.session_state.report = None
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
