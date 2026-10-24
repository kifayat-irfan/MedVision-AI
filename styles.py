"""Clean clinical UI theme for MedVision AI — light, professional, trustworthy."""
import streamlit as st


def apply_custom_styles():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    .stApp {
        background: #f1f5f9;
        color: #1e293b;
        font-family: 'Inter', sans-serif;
    }
    #MainMenu, footer { visibility: hidden; }
    header[data-testid="stHeader"] { background: transparent; }

    /* ── Header ── */
    .mv-header {
        background: linear-gradient(135deg, #0c4a6e 0%, #0369a1 60%, #0284c7 100%);
        border-radius: 16px;
        padding: 32px 36px;
        margin-bottom: 24px;
        color: white;
        box-shadow: 0 4px 14px rgba(2, 132, 199, 0.25);
    }
    .mv-header h1 {
        margin: 0;
        font-size: 1.9rem;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    .mv-header p {
        margin: 6px 0 0 0;
        opacity: 0.85;
        font-size: 0.95rem;
    }
    .mv-badge {
        display: inline-block;
        background: rgba(255,255,255,0.18);
        border: 1px solid rgba(255,255,255,0.3);
        border-radius: 20px;
        padding: 3px 12px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-top: 10px;
    }

    /* ── Cards ── */
    .mv-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 24px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
        margin-bottom: 16px;
    }
    .mv-card-title {
        font-size: 0.8rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #0284c7;
        margin-bottom: 16px;
        padding-bottom: 10px;
        border-bottom: 2px solid #e0f2fe;
    }

    /* ── Status pill ── */
    .mv-status {
        text-align: center;
        font-size: 0.82rem;
        font-weight: 500;
        padding: 8px 16px;
        border-radius: 20px;
        margin-bottom: 20px;
    }
    .mv-status.cloud { background: #dcfce7; color: #166534; border: 1px solid #bbf7d0; }
    .mv-status.local { background: #fef9c3; color: #854d0e; border: 1px solid #fde68a; }

    /* ── Report sections ── */
    .mv-section-head {
        margin: 14px 0 4px 0;
        padding: 8px 14px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 0.78rem;
        letter-spacing: 1px;
        text-transform: uppercase;
    }
    .mv-section-body {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-top: none;
        border-radius: 0 0 8px 8px;
        padding: 14px 16px;
        font-size: 0.92rem;
        line-height: 1.7;
        color: #334155;
        margin-bottom: 8px;
        white-space: pre-wrap;
    }

    /* ── Info boxes ── */
    .mv-info {
        background: #eff6ff;
        border: 1px solid #bfdbfe;
        border-radius: 10px;
        padding: 14px 16px;
        font-size: 0.88rem;
        color: #1e40af;
        margin: 12px 0;
    }
    .mv-warn {
        background: #fffbeb;
        border: 1px solid #fde68a;
        border-radius: 10px;
        padding: 14px 16px;
        font-size: 0.88rem;
        color: #92400e;
        margin: 12px 0;
    }

    /* ── Buttons ── */
    .stButton > button {
        background: #0284c7 !important;
        color: white !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        border: none !important;
        padding: 10px 24px !important;
        transition: background 0.2s;
    }
    .stButton > button:hover {
        background: #0369a1 !important;
    }

    /* ── Empty state ── */
    .mv-empty {
        background: white;
        border: 2px dashed #cbd5e1;
        border-radius: 14px;
        padding: 60px 30px;
        text-align: center;
        color: #94a3b8;
    }
    .mv-empty h3 { color: #64748b; margin-bottom: 8px; }

    /* ── Disclaimer ── */
    .mv-disclaimer {
        background: #fef2f2;
        border: 1px solid #fecaca;
        border-radius: 10px;
        padding: 12px 16px;
        font-size: 0.8rem;
        color: #991b1b;
        margin-top: 20px;
        line-height: 1.6;
    }

    @media (max-width: 768px) {
        .mv-header { padding: 24px 20px; }
        .mv-header h1 { font-size: 1.4rem; }
        .mv-card { padding: 16px; }
    }
    </style>
    """, unsafe_allow_html=True)
