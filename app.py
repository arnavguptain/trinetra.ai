import streamlit as st
import cv2
import numpy as np
from PIL import Image
from google import genai
from google.genai import types

# Page Configurations
st.set_page_config(
    page_title="Project TrineTra - Live AI Lab",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark-themed Premium Academic UI Stylesheet
st.markdown("""
<style>
    /* Global Styles */
    .stApp { background-color: #0E1117; color: #E2E8F0; }
    
    /* Header Component */
    .header-container {
        background: linear-gradient(135deg, #1E1E2F 0%, #11111F 100%);
        padding: 30px;
        border-radius: 12px;
        border-left: 6px solid #FF4B4B;
        box-shadow: 0 4px 20px rgba(0,0,0,0.4);
        margin-bottom: 30px;
    }
    
    /* Telemetry KPI Cards */
    .kpi-box {
        background-color: #1A1C24;
        padding: 20px;
        border-radius: 8px;
        border: 1px solid #2D313E;
        text-align: center;
        box-shadow: 0 4px 10px rgba(0,0,0,0.2);
    }
    .kpi-value { font-size: 1.8rem; font-weight: 800; color: #FF4B4B; margin-top: 5px; }
    
    /* Audio Output Loops */
    .audio-card {
        background-color: #161920;
        padding: 18px;
        border-radius: 8px;
        border-left: 4px solid #00E676;
        margin-top: 15px;
        font-family: 'Courier New', monospace;
    }
</style>
""", unsafe_allow_html=True)

# Application Header Banner
st.markdown("""
<div class="header-container">
    <h1 style="margin:0; color:#FFFFFF; font-size: 2.5rem; letter-spacing: 1px;">PROJECT TRINETRA</h1>
    <p style="margin:8px 0 0 0; color:#94A3B8; font-size: 1.1rem;">
        Smart Wearable Assistive System • <strong>Pacific World School</strong> • Class of 2026-2027
    </p>
</div>
""", unsafe_allow_html=True)

# Secure API Key Management Registry
if "GEMINI_API_KEY" in st.secrets:
    api_key = st.secrets["GEMINI_API_KEY"]
else:
    api_key = st.sidebar.text_input("Enter Gemini API Key", type="password", help="Required for active multi-modal vision processing.")

client = genai.Client(api_key=api_key) if api_key else None

# Sidebar Engineering Metadata
with st.sidebar:
    st.markdown("### 🛠️ Hardware Diagnostic Telemetry")
    pi_power = st.toggle("Power Device Core (Raspberry Pi)", value=True)
    
    st.markdown("---")
    st.markdown("### 📋 Administration Specs")
    st.markdown("**Guide:** Ms. Sapna Shukla")
    st.markdown("**Core Team:**\n* Diksha Bachu\n* Jigya Yadav\n* Arnav Gupta\n* Atharva Parashar")
    st.markdown("---")
    st.caption("Targeting UN Sustainable Development Goals: SDG 3, SDG 9, SDG 10")

if not pi_power:
    st.warning("The simulated Raspberry Pi computing hub is powered off. Toggle system power in the left panel to initialize.")
else:
    # Top Level Hardware Statistics
    col_kpi1, col_kpi2, col_kpi3 = st.columns(3)
    with col_kpi1:
        st.markdown('<div class="kpi-box"><span style="color:#94A3B8;">Edge System Status</span><div class="kpi-value" style="color:#00E676;">ONLINE</div></div>', unsafe_allow_html=True)
    with col_kpi2:
        st.markdown('<div class="kpi-box"><span style="color:#94A3B8;">Camera Array Mode</span><div class="kpi-value">LIVE FEED</div></div>', unsafe_allow_html=True)
    with col_kpi3:
        st.markdown('<div class="kpi-box"><span style="color:#94A3B8;">Latency Index</span><div class="kpi-value" style="color:#00B0FF;">12 ms</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Core Workspace Tabs Split
    tab_vision, tab_ocr = st.tabs(["🚀 Spatial Obstacle Scanner", "📖 OCR Audio Reader"])

    # TAB 1: Live Spatial Proximity Analysis & Hazard Zoning
    with tab_vision:
        st.markdown("### Live Spatial Mapping Desk")
        st.caption("Capture an image below using TrineTra's wearble camera rig to map out spatial navigation sectors.")
        
        live_capture = st.camera_input("TrineTra Spatial Camera Capture Input", key="spatial_cam")
        
        if live_capture:
            # Reconstruct buffer into OpenCV manipulation frame matrices
            file_bytes = live_capture.read()
            cv_frame = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
            h, w, _ = cv_frame.shape
            
            # Divide image into 3 distinct spatial scanning matrices (Left, Center, Right)
            sector_width = w // 3
            center_sector = cv_frame[:, sector_width:sector_width*2]
            
            # Process mean frame luminosity vectors to approximate physical wall proximity
            center_depth_metric = float(np.mean(center_sector))
            
            # Setup bounding canvas overlay mapping graphic lines
            canvas = cv_frame.copy()
            cv2.line(canvas, (sector_width, 0), (sector_width, h), (255, 255, 255), 2)
            cv2.line(canvas, (sector_width*2, 0), (sector_width*2, h), (255, 255, 255), 2)
            
            audio_alert = "Path clear. Proceed straight forward."
            is_hazard = False
            
            # Flag anomalies inside the central walking vector paths
            if center_depth_metric < 110:
                is_hazard = True
                audio_alert = "ALERT: Critical structural obstacle detected directly ahead! Halt forward movement."
                cv2.rectangle(canvas, (sector_width, 0), (sector_width*2, h), (0, 0, 255), 4)
                cv2.putText(canvas, "BLOCKED", (sector_width + 15, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            else:
                cv2.rectangle(canvas, (sector_width, 0), (sector_width*2, h), (0, 255, 0), 2)
                cv2.putText(canvas, "CLEAR SEC", (sector_width + 15, h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
            col_out1, col_out2 = st.columns([2, 1])
            with col_out1:
                st.image(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB), caption="TrineTra Real-Time Hardware Boundary Breakdown Matrix", use_container_width=True)
            with col_out2:
                st.markdown("#### 🔊 Audio Engine Synthesis Feedback Loop")
                status_color = "#FF1744" if is_hazard else "#00E676"
                status_label = "HAZARD THREAT TRIGGERED" if is_hazard else "SAFE TO NAVIGATE"
                
                st.markdown(f"<div style='color:{status_color}; font-weight:bold; font-size:1.1rem; border:1px solid {status_color}; padding:10px; border-radius:6px; text-align:center;'>{status_label}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='audio-card'>🗣️ [Vocal Output Loop]<br><span style='color:#FFFFFF;'>\"{audio_alert}\"</span></div>", unsafe_allow_html=True)

    # TAB 2: Live Document Snapshot Character Processing & TTS Generation
    with tab_ocr:
        st.markdown("### Document Scanning & Text Translation Desk")
        st.caption("Point a textbook, product label, or street sign toward the camera array module to read visual elements aloud.")
        
        ocr_capture = st.camera_input("TrineTra OCR Document Capture Input", key="ocr_cam")
        
        if ocr_capture:
            if not client:
                st.error("Please add a valid Gemini API Key in the left sidebar to unlock text recognition processing.")
            else:
                if st.button("🚀 Process Frame Characters & Transcribe", type="primary"):
                    with st.spinner("🤖 Extracting text elements via multimodal processing pipeline..."):
                        try:
                            # Read raw byte inputs directly for processing execution
                            img_data = ocr_capture.getvalue()
                            image_part = types.Part.from_bytes(data=img_data, mime_type="image/jpeg")
                            
                            prompt = (
                                "You are the edge computing module inside Project TrineTra's smart glasses. "
                                "Extract all visible words from this picture cleanly. Then, summarize the nature of the text "
                                "(e.g., product label, caution sign) and format it into a brief script ready for the Text-to-Speech audio engine."
                            )
                            
                            response = client.models.generate_content(
                                model='gemini-2.5-flash',
                                contents=[image_part, prompt]
                            )
                            
                            st.markdown("### 📊 Transcribed System Data Output")
                            st.markdown(response.text)
                            
                        except Exception as err:
                            st.error(f"Hardware pipeline transcription execution fault: {err}")
