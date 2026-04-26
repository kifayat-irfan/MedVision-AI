import streamlit as st

def apply_custom_styles():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
    .stApp { background: #020617; color: #f8fafc; font-family: 'Outfit', sans-serif; }
    #MainMenu, footer, header, [data-testid="stSidebar"] {visibility: hidden !important; display: none !important;}
    
    .command-center { max-width: 1300px; margin: 0 auto; padding: 20px; }
    .cyber-header { text-align: center; padding: 50px 0; border-bottom: 1px solid rgba(59, 130, 246, 0.2); margin-bottom: 40px; }
    .cyber-title { font-size: 4rem !important; font-weight: 800; background: linear-gradient(180deg, #fff 0%, #3b82f6 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    
    .glass-panel { background: rgba(15, 23, 42, 0.7); backdrop-filter: blur(20px); border: 1px solid rgba(59, 130, 246, 0.3); border-radius: 24px; padding: 30px; box-shadow: 0 20px 50px rgba(0,0,0,0.5); }
    
    /* The Scanner Line */
    .scan-container { position: relative; width: 100%; border-radius: 20px; overflow: hidden; border: 1px solid #3b82f6; }
    .scan-line { position: absolute; top: 0; left: 0; width: 100%; height: 4px; background: #06b6d4; box-shadow: 0 0 20px #06b6d4; animation: scan 2s linear infinite; z-index: 10; }
    @keyframes scan { 0% { top: 0%; } 50% { top: 100%; } 100% { top: 0%; } }

    .med-card { background: rgba(255, 255, 255, 0.05); border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 15px; padding: 15px; margin-bottom: 10px; }
    .card-title { color: #60a5fa; font-weight: 700; font-size: 0.7rem; text-transform: uppercase; display: block; margin-bottom: 5px; }
    
    .stButton>button { background: linear-gradient(90deg, #3b82f6, #06b6d4) !important; color: white !important; border-radius: 12px !important; font-weight: 700 !important; width: 100% !important; border: none !important; }
    .chat-bubble-ai { background: rgba(59, 130, 246, 0.1); border: 1px solid #3b82f6; border-radius: 15px 15px 2px 15px; padding: 12px; margin-bottom: 10px; color: #e2e8f0; }
    .chat-bubble-user { background: rgba(255, 255, 255, 0.05); border: 1px solid rgba(255, 255, 255, 0.2); border-radius: 15px 15px 15px 2px; padding: 12px; margin-bottom: 10px; text-align: right; color: #e2e8f0; }
    </style>
    """, unsafe_allow_html=True)
