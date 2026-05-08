import sys
import os
import base64
from pathlib import Path
from dotenv import load_dotenv

# 1. RIGOROUS PATH SETUP
root_path = str(Path(__file__).resolve().parent.parent)
if root_path not in sys.path:
    sys.path.append(root_path)

import asyncio
import logging
import streamlit as st
from app.core.rag_engine import EnterpriseMultimodalRAG

# =========================
# CONFIG & THEME
# =========================
load_dotenv()

# Set Page Config with your custom logo as Favicon
st.set_page_config(
    page_title="Intelligence Studio", 
    page_icon="assets/logo.jpeg", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# Professional High-Contrast CSS
st.markdown("""
    <style>
    .stApp { background-color: #ffffff; color: #1a1a1b; }
    section[data-testid="stSidebar"] { background-color: #f8f9fa !important; border-right: 1px solid #dee2e6; }
    h1, h2, h3, h4, .stMarkdown p { color: #1a1a1b !important; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
    
    /* Sleek Chat Bubbles */
    [data-testid="stChatMessage"]:nth-child(even) { background-color: #f4f5f7 !important; border-radius: 8px; border: 1px solid #e2e8f0; margin-bottom: 15px; }
    [data-testid="stChatMessage"]:nth-child(odd) { background-color: #ffffff !important; border-radius: 8px; border-left: 4px solid #0f172a; box-shadow: 0 1px 3px rgba(0,0,0,0.05); margin-bottom: 15px; }
    
    /* Center and Constrain Chat Width for Readability */
    [data-testid="stChatMessage"], .stChatInputContainer {
        max-width: 950px;
        margin-left: auto !important;
        margin-right: auto !important;
    }

    /* Clean Input & Status */
    .stChatInputContainer { border-top: 1px solid #dee2e6 !important; background-color: white !important; }
    .stStatusWidget { border-radius: 6px; border: 1px solid #e2e8f0; }
    
    /* Button Override */
    .stButton>button { border-radius: 6px; font-weight: 500; }
    </style>
""", unsafe_allow_html=True)

# Helper function to encode image for HTML centering
def get_image_base64(path):
    with open(path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# =========================
# INITIALIZATION & LOADING
# =========================

# Automated Directory Setup for Production
def initialize_workspace():
    """Create necessary directories if they don't exist."""
    folders = ["uploads", "data", "memory", "assets"]
    for folder in folders:
        os.makedirs(folder, exist_ok=True)

initialize_workspace()

if "rag" not in st.session_state:
    startup_screen = st.empty()
    with startup_screen.container():
        st.markdown("<br><br><br><br><br><br><br><br><br>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            with st.spinner("Starting system..."):
                st.session_state.rag = EnterpriseMultimodalRAG()
    startup_screen.empty()

if "history" not in st.session_state:
    st.session_state.history = []
if "inventory" not in st.session_state:
    st.session_state.inventory = []

# =========================
# SIDEBAR: DATA CONTROLS
# =========================
with st.sidebar:
    st.markdown("<br><br><br>", unsafe_allow_html=True)
    st.title("Workspace")
    st.caption("Active Library")
    st.divider()
    
    with st.container(border=False):
        uploaded_file = st.file_uploader("Upload Document", type=["pdf", "csv", "xlsx", "parquet"])
        uploaded_img = st.file_uploader("Upload Visual Reference", type=["png", "jpg", "jpeg"])
    
    if uploaded_file and uploaded_file.name not in st.session_state.inventory:
        with st.spinner(f"Reading {uploaded_file.name}..."):
            path = os.path.join("uploads", uploaded_file.name)
            # Ensure specific upload path exists
            os.makedirs("uploads", exist_ok=True)
            
            with open(path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            try:
                asyncio.run(st.session_state.rag.ingest(path))
                st.session_state.inventory.append(uploaded_file.name)
            except Exception as e:
                st.error("Failed to process the document.")
                logging.error(f"Ingestion Error: {e}", exc_info=True)

    if st.session_state.inventory:
        st.write("Available Files:")
        for item in st.session_state.inventory:
            st.caption(f"- {item}")
            
    st.divider()
    if st.button("Clear History & Library", use_container_width=True):
        st.session_state.history = []
        st.session_state.inventory = []
        asyncio.run(st.session_state.rag.memory.clear_memory())
        st.rerun()

# =========================
# MAIN DASHBOARD
# =========================
if not st.session_state.history:
    # 1. Pull the header high
    st.markdown("<div style='margin-top: -5rem;'></div>", unsafe_allow_html=True)
    
    # 2. Centered Layout
    _, center_col, _ = st.columns([0.5, 2, 0.5])
    
    with center_col:
        logo_path = "assets/logo.jpeg"
        if os.path.exists(logo_path):
            img_64 = get_image_base64(logo_path)
            st.markdown(
                f"""
                <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; text-align: center;">
                    <img src="data:image/jpeg;base64,{img_64}" width="280">
                    <p style="color: #64748b; font-size: 1.1rem; margin-top: 15px; margin-bottom: 20px;">
                        Ask questions or request calculations based on your uploaded files.
                    </p>
                </div>
                """, 
                unsafe_allow_html=True
            )
        
        # 3. System Ready message
        if not st.session_state.inventory:
            st.info("System is ready. Please upload a file in the sidebar to begin.")
else:
    # Active Session Header
    h_col1, h_col2, h_col3 = st.columns([1, 4, 1])
    with h_col2:
        inner_logo_col, inner_text_col = st.columns([0.15, 2])
        with inner_logo_col:
            if os.path.exists("assets/logo.jpeg"):
                st.image("assets/logo.jpeg", width=50)
        with inner_text_col:
            st.markdown(
                "<p style='margin-top: 12px; font-weight: 600; color: #1a1a1b; font-size: 1.1rem;'>Intelligence Studio <span style='color: #64748b; font-weight: 400; font-size: 0.9rem;'>| Active Session</span></p>", 
                unsafe_allow_html=True
            )
        st.divider()

# Display Chat History
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Enter your query...")

if user_input:
    st.session_state.history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        status_placeholder = st.empty()
        with status_placeholder.status("Processing query...", expanded=True) as status:
            try:
                st.write("Reviewing request...")
                image_ctx = None
                if uploaded_img:
                    st.write("Analyzing image...")
                    image_ctx = os.path.join("uploads", uploaded_img.name)
                    with open(image_ctx, "wb") as f:
                        f.write(uploaded_img.getbuffer())

                st.write("Formulating answer...")
                result = asyncio.run(st.session_state.rag.run(
                    query=user_input,
                    history=st.session_state.history[:-1],
                    image=image_ctx
                ))
                success = True
            except Exception as e:
                status.update(label="System Error", state="error", expanded=True)
                st.error("The system encountered an error.")
                logging.error(f"UI Error: {e}", exc_info=True)
                success = False

        if success:
            status_placeholder.empty()
            st.markdown(result.answer)
            if result.citations:
                st.write("") 
                with st.popover("View References"):
                    st.write("**Sources used:**")
                    for cite in result.citations:
                        st.caption(f"- {cite}")
                    st.divider()
                    st.caption(f"System Confidence: **{result.confidence*100:.1f}%**")
            st.session_state.history.append({"role": "assistant", "content": result.answer})