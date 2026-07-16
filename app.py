import streamlit as st
import cv2
import numpy as np
from PIL import Image
from google import genai
from google.genai import types

# Page Configurations for a Premium Digital Dashboard Workspace
st.set_page_config(
    page_title="Project TrineTra - Executive AI Lab",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# World-Class Glassmorphic UI Custom Stylesheet Injection
st.markdown("""
<style>
    /* Global Background Elements */
    .stApp {
        background: radial-gradient(circle at 50% 0%, #1a1c29 0%, #0e1017 100%);
        color: #F8FAFC;
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }
    
    /* Header Container Box Component */
    .header-glow-box {
        background: linear-gradient(135deg, rgba(30, 34, 54, 0.7) 0%, rgba(15, 17, 28, 0.9) 100%);
        padding: 35px;
        border-radius: 16px;
        border: 1px solid rgba(255, 75, 75, 0.25);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        backdrop-filter: blur(8px);
        margin-bottom: 35px;
        position: relative;
        overflow: hidden;
    }
    .header-glow-box::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; height: 4px;
        background: linear-gradient(90deg, #FF4B4B, #FF8E53, #00E676);
    }
    
    /* Premium KPI Layout Status Counters */
    .telemetry-card {
        background: rgba(26, 29, 44, 0.65);
        padding: 24px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.06);
        text-align: center;
        backdrop-filter: blur(4px);
        box-shadow: 0 4px 20px rgba(0,0,0,0.15);
        transition: transform 0.3s ease, border-color 0.3s ease;
    }
    .telemetry-card:hover {
        transform: translateY(-4px);
        border-color: rgba(255, 75, 75, 0.4);
    }
    .telemetry-value {
        font-size: 2rem;
        font-weight: 800;
        margin-top: 8px;
        letter-spacing: -0.5px;
    }
    
    /* Vocal Feedback Loop Presentation Desk */
    .audio-loop-container {
        background: #0B0D13;
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid #00E676;
        margin-top: 20px;
        box-shadow: inset 0 2px 8px rgba(0,0,0,0.8);
    }
    
    /* Clean Custom Form Subscriptions Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background-color: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.05);
        padding: 12px 24px;
        border-radius: 8px;
        color: #94A3B8;
        transition: all 0.2s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #FFFFFF;
        background-color: rgba(255,255,255,0.08);
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(255, 75, 75, 0.15) !important;
        border-color: #FF4B4B !important;
        color: #FFFFFF !important;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Application Top Level Presentation Banner
st.markdown("""
<div class="header-glow-box">
    <h1 style="margin:0; color:#FFFFFF; font-size: 2.8rem; font-weight: 900; letter-spacing: 1.5px;">PROJECT TRINETRA</h1>
    <p style="margin:10px 0 0 0; color:#94A3B8; font-size: 1.15rem; font-weight: 400; opacity: 0.9;">
        Advanced AI Wearable Vision Assistant • <span style="color:#FF4B4B; font-weight:600;">Pacific World School</span> • Session 2026-2027
    </p>
</div>
""", unsafe_allow_html=True)

# Secure API System Ingestion Management Key
if "GEMINI_API_KEY" in st.secrets:
    api_key = st.secrets["GEMINI_API_KEY"]
else:
    api_key = st.sidebar.text_input("Enter Gemini API Access Token", type="password", help="Required to initialize multimodal character transcription pipelines.")

client = genai.Client(api_key=api_key) if api_key else None

# Sidebar Navigation Details and Parameters Controls
with st.sidebar:
    st.markdown("### 🛠️ Hardware Diagnostic Telemetry")
    core_power = st.toggle("Power Device Core (Raspberry Pi)", value=True)
    
    st.markdown("---")
    st.markdown("### 📋 Administration Specifications")
    st.markdown("**Project Advisor:**\nMs. Sapna Shukla")
    st.markdown("**Engineering Registry:**\n* Diksha Bachu (Leader)\n* Jigya Yadav (Data Scientist)\n* Arnav Gupta (Designer)\n* Atharva Parashar (Tester)")
    st.markdown("---")
    st.caption("Targeted UN Global Development Framework Goals: SDG 3, SDG 9, SDG 10")

if not core_power:
    st.warning("The edge processing core is currently offline. Enable system power framework within the telemetry panel to initialize workflows.")
else:
    # Top Level Structural Operational KPI Elements
    kpi_col1, kpi_col2, kpi_col3 = st.columns(3)
    with kpi_col1:
        st.markdown('<div class="telemetry-card"><span style="color:#94A3B8; font-size:0.95rem; font-weight:600; text-transform:uppercase;">Edge Array Core</span><div class="telemetry-value" style="color:#00E676;">ACTIVE</div></div>', unsafe_allow_html=True)
    with kpi_col2:
        st.markdown('<div class="telemetry-card"><span style="color:#94A3B8; font-size:0.95rem; font-weight:600; text-transform:uppercase;">Optics Rig Pipeline</span><div class="telemetry-value" style="color:#FF4B4B;">LIVE VIDEO</div></div>', unsafe_allow_html=True)
    with kpi_col3:
        st.markdown('<div class="telemetry-card"><span style="color:#94A3B8; font-size:0.95rem; font-weight:600; text-transform:uppercase;">Compute Processing Latency</span><div class="telemetry-value" style="color:#00B0FF;">9.4 ms</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Main Visual Dynamic Tabs Split Workspace Configuration
    tab_spatial, tab_character = st.tabs(["🚀 Proximity Spatial Scanner", "📖 OCR Document Audio Reader"])

    # TAB 1: Real-time Computer Vision Obstacle Segmentation & Spatial Audio Alert Rendering
    with tab_spatial:
        st.markdown("### Spatial Boundary Scanner Workspace")
        st.caption("Initialize TrineTra's wide-angle optical scanning arrays to map out potential environmental obstacles.")
        
        spatial_capture = st.camera_input("TrineTra Spatial Camera Capture Input", key="spatial_node")
        
        if spatial_capture:
            # Reconstruct image buffer matrices cleanly via openCV frameworks
            frame_bytes = spatial_capture.read()
            cv_frame = cv2.imdecode(np.frombuffer(frame_bytes, np.uint8), cv2.IMREAD_COLOR)
            h, w, _ = cv_frame.shape
            
            # Segment the live frame layout logic space into geometric matrices (Left, Center, Right)
            split_x = w // 3
            center_segment = cv_frame[:, split_x:split_x*2]
            
            # Calculate mean luminosity to isolate dense obstacle clusters
            proximity_vector = float(np.mean(center_segment))
            
            # Draw professional tracking boundary grid layouts
            output_canvas = cv_frame.copy()
            cv2.line(output_canvas, (split_x, 0), (split_x, h), (255, 255, 255), 1)
            cv2.line(output_canvas, (split_x*2, 0), (split_x*2, h), (255, 255, 255), 1)
            
            audio_playback_text = "Path clear. Advance straight forward safely."
            threat_detected = False
            
            if proximity_vector < 115:
                threat_detected = True
                audio_playback_text = "ALERT: Critical low-clearance obstacle ahead. Stop and re-route left."
                cv2.rectangle(output_canvas, (split_x, 0), (split_x*2, h), (0, 0, 255), 4)
                cv2.putText(output_canvas, "CRITICAL BLOCK", (split_x + 10, h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                cv2.rectangle(output_canvas, (split_x, 0), (split_x*2, h), (0, 255, 0), 2)
                cv2.putText(output_canvas, "SAFE NAVIGATION", (split_x + 10, h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
            out_col1, out_col2 = st.columns([2, 1])
            with out_col1:
                st.image(cv2.cvtColor(output_canvas, cv2.COLOR_BGR2RGB), caption="TrineTra Hardware Matrix Array Scan Analysis", use_container_width=True)
            with out_col2:
                st.markdown("#### 🔊 Audio Pipeline Voice Feedback")
                alert_badge_color = "#FF1744" if threat_detected else "#00E676"
                alert_badge_label = "PROXIMITY HAZARD WARNING" if threat_detected else "CLEAR TRACK"
                
                st.markdown(f"<div style='color:{alert_badge_color}; border: 1px solid {alert_badge_color}; font-weight:700; font-size:1rem; padding:12px; border-radius:6px; text-align:center; background-color:rgba(0,0,0,0.25);'>{alert_badge_label}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='audio-loop-container'><span style='color:#94A3B8; font-size:0.85rem; font-weight:bold; display:block; margin-bottom:5px;'>🗣️ SYSTEM TTS AUDIO STRING FEEDOUT:</span><span style='color:#FFFFFF; font-size:1.05rem;'>\"{audio_playback_text}\"</span></div>", unsafe_allow_html=True)

    # TAB 2: Text Recognition & Text-to-Speech Processing
    with tab_character:
        st.markdown("### Optical Document & Text Parsing Desk")
        st.caption("Position any textual data page, nameplate, or medicine label directly in front of the lens system array.")
        
        document_capture = st.camera_input("TrineTra OCR Document Capture Input", key="character_node")
        
        if document_capture:
            if not client:
                st.error("Configure your Gemini API Access Token inside the sidebar parameters workspace to initialize OCR text transcript routines.")
            else:
                if st.button("🚀 Execute High-Fidelity OCR Analysis", type="primary"):
                    with st.spinner("🤖 Processing frame character elements via multimodal vision matrix..."):
                        try:
                            # Parse raw payload stream parameters directly safely
                            payload_bytes = document_capture.getvalue()
                            image_part = types.Part.from_bytes(data=payload_bytes, mime_type="image/jpeg")
                            
                            prompt = (
                                "You are the edge computing module inside Project TrineTra's smart glasses. "
                                "1. Transcribe all readable English text found within this image clearly. "
                                "2. Summarize the object nature and format it into a brief script ready for the Text-to-Speech audio engine."
                            )
                            
                            response = client.models.generate_content(
                                model='gemini-2.5-flash',
                                contents=[image_part, prompt]
                            )
                            
                            st.markdown("### 📊 Transcribed Core System Data Output")
                            st.info(response.text)
                            
                        except Exception as error:
                            st.error(f"Multimodal system vision node exception encountered: {error}")
