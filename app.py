import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import time
from google import genai
from google.genai import types

# Page Configurations
st.set_page_config(page_title="Project TrineTra - AI Prototype", page_icon="👁️", layout="wide")

# Custom UI Styles
st.markdown("""
<style>
    .header-box { background-color: #121620; padding: 20px; border-radius: 10px; border-left: 6px solid #00E676; margin-bottom: 25px; }
    .status-active { color: #00E676; font-weight: bold; }
    .status-alert { color: #FF1744; font-weight: bold; }
    .feedback-card { background-color: #1E2230; padding: 15px; border-radius: 8px; border: 1px solid #30364D; margin-top: 10px; }
</style>
""", unsafe_allow_html=True)

# Application Header Banner
st.markdown("""
<div class="header-box">
    <h2 style="margin:0; color:#FFFFFF;">👁️ Project TrineTra: AI Wearable Vision Assistant</h2>
    <p style="margin:5px 0 0 0; color:#A0AEC0;">Functional Hardware Emulation Sandbox | Pacific World School (2026-2027)</p>
</div>
""", unsafe_allow_html=True)

# Global API Credentials Authentication Registry
if "GEMINI_API_KEY" in st.secrets:
    api_key = st.secrets["GEMINI_API_KEY"]
else:
    api_key = st.sidebar.text_input("Enter Gemini API Key to unlock Edge Pipeline", type="password")

client = genai.Client(api_key=api_key) if api_key else None

# Sidebar Diagnostics Pane
with st.sidebar:
    st.markdown("### 🛠️ Hardware Diagnostic Telemetry")
    system_power = st.toggle("Power Device Core (Raspberry Pi)", value=True)
    camera_status = "ONLINE" if system_power else "OFFLINE"
    st.markdown(f"Camera Status: <span class='status-active'>{camera_status}</span>", unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 👥 Engineering Logbook Registry")
    st.caption("Team: Diksha Bachu, Jigya Yadav, Arnav Gupta, Atharva Parashar")
    st.caption("Target Framework: SDG 3, SDG 9, SDG 10")

if not system_power:
    st.warning("The simulated Raspberry Pi hardware core is powered off. Toggle system power in the telemetry panel to begin.")
else:
    # Main Navigation Tabs split across core features
    tab_vision, tab_ocr = st.tabs(["🚀 Real-Time Spatial Proximity Scanner", "📖 OCR Document Text Reader"])

    # TAB 1: Real-time Computer Vision Obstacle Segmentation & Spatial Audio Alert Rendering
    with tab_vision:
        st.subheader("Spatial Matrix Mapping & Navigation Alerts")
        st.markdown("This module emulates TrineTra's physical ultrasonic radar matrix and computer vision segmentation layer. Upload an environment scene to calculate safe navigation sectors.")
        
        uploaded_scene = st.file_uploader("Upload Environment Snapshot Capture", type=["jpg", "jpeg", "png"], key="nav_uploader")
        
        if uploaded_scene:
            # Convert asset payload to openCV format matrix
            raw_bytes = uploaded_scene.read()
            cv_img = cv2.imdecode(np.frombuffer(raw_bytes, np.uint8), cv2.IMREAD_COLOR)
            h, w, _ = cv_img.shape
            
            # Simulated Computer Vision Range Processing (Left, Center, Right Sectors)
            sector_w = w // 3
            left_zone = cv_img[:, 0:sector_w]
            center_zone = cv_img[:, sector_w:sector_w*2]
            right_zone = cv_img[:, sector_w*2:w]
            
            # Calculate pixel luminosity values to simulate distance depth approximations
            left_proximity = float(np.mean(left_zone))
            center_proximity = float(np.mean(center_zone))
            right_proximity = float(np.mean(right_zone))
            
            # Map distance values and highlight threats
            alert_triggered = False
            alert_messages = []
            
            # Apply colored spatial safety boundaries over the matrix frame
            processed_canvas = cv_img.copy()
            cv2.line(processed_canvas, (sector_w, 0), (sector_w, h), (255, 255, 255), 2)
            cv2.line(processed_canvas, (sector_w*2, 0), (sector_w*2, h), (255, 255, 255), 2)
            
            # Analyze Center Path Threat Level
            if center_proximity < 100:
                alert_triggered = True
                alert_messages.append("ALERT: Critical obstacle directly ahead! Halt forward movement.")
                cv2.rectangle(processed_canvas, (sector_w, 0), (sector_w*2, h), (0, 0, 255), 4)
                cv2.putText(processed_canvas, "BLOCKED", (sector_w + 20, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            else:
                cv2.rectangle(processed_canvas, (sector_w, 0), (sector_w*2, h), (0, 255, 0), 2)
                cv2.putText(processed_canvas, "CLEAR", (sector_w + 20, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
            # Analyze Left Path Threat Level
            if left_proximity < 90:
                alert_messages.append("WARNING: Low clearance obstacle approaching on your left.")
                cv2.rectangle(processed_canvas, (0, 0), (sector_w, h), (0, 165, 255), 3)
            else:
                cv2.rectangle(processed_canvas, (0, 0), (sector_w, h), (0, 255, 0), 2)

            # Analyze Right Path Threat Level
            if right_proximity < 90:
                alert_messages.append("WARNING: Structure boundary detected on your right.")
                cv2.rectangle(processed_canvas, (sector_w*2, 0), (w, h), (0, 165, 255), 3)
            else:
                cv2.rectangle(processed_canvas, (sector_w*2, 0), (w, h), (0, 255, 0), 2)
                
            # Display output visualization frames
            col1, col2 = st.columns([2, 1])
            with col1:
                st.image(cv2.cvtColor(processed_canvas, cv2.COLOR_BGR2RGB), caption="TrineTra Hardware Spatial Array Mapping", use_container_width=True)
            
            with col2:
                st.markdown("#### 🔊 Wearable Audio Feedback Feed")
                if alert_triggered:
                    st.markdown("<div class='status-alert'>⚠️ WARN STATUS: HAZARD DETECTED</div>", unsafe_allow_html=True)
                else:
                    st.markdown("<div class='status-active'>✅ NAV STATUS: SAFE PATH</div>", unsafe_allow_html=True)
                    
                for msg in alert_messages if alert_messages else ["Path clear. Proceed straight forward safely."]:
                    st.markdown(f"<div class='feedback-card'>🗣️ <em>Audio Loop:</em> {msg}</div>", unsafe_allow_html=True)

    # TAB 2: Text Recognition & Text-to-Speech Processing
    with tab_ocr:
        st.subheader("Optical Text Parsing Desk")
        st.markdown("Simulates TrineTra capturing physical document frames (e.g., product text labels, street signs, textbook pages) and parsing text directly into clear spoken voice output loops.")
        
        uploaded_doc = st.file_uploader("Capture and Upload Document Page Image", type=["jpg", "jpeg", "png"], key="doc_uploader")
        
        if uploaded_doc:
            st.image(uploaded_doc, caption="Ingested Source Document Fragment", width=400)
            
            if not client:
                st.error("Please add a valid Gemini API Key in the sidebar interface to initialize the Optical Parser pipeline.")
            else:
                if st.button("🔍 Execute OCR Processing & Speech Synthesis", type="primary"):
                    with st.spinner("🤖 Executing edge computer vision and character processing matrices..."):
                        try:
                            # Package file data for multimodal text engine execution
                            doc_bytes = uploaded_doc.getvalue()
                            image_part = types.Part.from_bytes(data=doc_bytes, mime_type="image/jpeg")
                            
                            ocr_prompt = (
                                "You are the edge computing module inside Project TrineTra's smart glasses. "
                                "1. Transcribe all readable English textual characters found inside this image cleanly. "
                                "2. Provide a short description summarizing what kind of object this text belongs to (e.g., medicine label, book cover, caution sign). "
                                "3. Format the final read text into a natural spoken dialogue script string for text-to-speech engine conversion."
                            )
                            
                            response = client.models.generate_content(
                                model='gemini-2.5-flash',
                                contents=[image_part, ocr_prompt]
                            )
                            
                            st.success("Analysis Complete!")
                            st.markdown("### 📊 Transcribed Telemetry Output Data")
                            st.write(response.text)
                            
                        except Exception as e:
                            st.error(f"Hardware edge pipeline processing exception: {e}")
