import streamlit as st
import requests
from PIL import Image
import io
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av
import cv2
import numpy as np

# Page Config
st.set_page_config(
    page_title="Face Auth System",
    page_icon="🔒",
    layout="centered"
)

# Custom CSS
st.markdown("""
    <style>
    .stApp {
        background-color: #0f0f13;
        color: white;
    }
    h1 {
        background: linear-gradient(to right, #00f2ff, #7000ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
    }
    .stButton>button {
        background: linear-gradient(135deg, #7000ff, #00f2ff);
        color: white;
        border: none;
        border-radius: 50px;
        padding: 0.5rem 2rem;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'user_name' not in st.session_state:
    st.session_state.user_name = ""

def login_view():
    st.title("Security Access")
    st.markdown("<p style='text-align: center; color: rgba(255,255,255,0.6);'>Identity Verification System</p>", unsafe_allow_html=True)
    
    # Camera Input
    img_file_buffer = st.camera_input("Take a picture to verify identity")

    if img_file_buffer is not None:
        # Show a spinner while processing
        with st.spinner('Verifying identity...'):
            try:
                # Prepare the file for the API
                bytes_data = img_file_buffer.getvalue()
                
                # Send to backend
                # Using host.docker.internal to access the backend running on the host's port 8000
                # from inside the frontend container.
                files = {'file': ('capture.jpg', bytes_data, 'image/jpeg')}
                try:
                    response = requests.post("http://host.docker.internal:8000/predict", files=files)
                except requests.exceptions.ConnectionError:
                    # Fallback for some setups or if running locally without docker
                    response = requests.post("http://localhost:8000/predict", files=files)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    if "error" in data:
                        st.error(f"Error: {data['error']}")
                    else:
                        person = data['person']
                        confidence = data['confidence']
                        if person != "Desconocido" and person != "Unknown":
                            st.success(f"Acceso Concedido: Bienvenido, {person}!")
                            st.metric("Confianza", f"{confidence*100:.1f}%")
                            
                            # Set session state and rerun to advance
                            st.session_state.logged_in = True
                            st.session_state.user_name = person
                            st.rerun()
                        else:
                            st.error(f"Acceso Denegado: Clasificado como '{person}'")
                            st.metric("Confianza", f"{confidence*100:.1f}%")
                            st.info("Intenta acercarte más a la cámara o mejorar la iluminación.")
                else:
                    st.error(f"Server Error: {response.status_code}")
                    
            except Exception as e:
                st.error(f"Connection Error: {e}")
                st.info("Ensure the backend service is running.")

def dashboard_view():
    # Sidebar
    with st.sidebar:
        st.markdown(f"<h2 style='color: #00f2ff;'>User: {st.session_state.user_name}</h2>", unsafe_allow_html=True)
        if st.button("Log Out"):
            st.session_state.logged_in = False
            st.session_state.user_name = ""
            st.rerun()

    # Main Content
    st.markdown("<h1 style='text-align: center; color: #5e66f7;'>Live Surveillance</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: left; color: #a6a6a6;'>Real-time face recognition active.</p>", unsafe_allow_html=True)
    
    # Live Recognition Feature (Real-Time)
    class VideoProcessor(VideoTransformerBase):
        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")
            
            # Compress image for sending to API (Speed optimization)
            _, encoded_img = cv2.imencode('.jpg', img)
            bytes_data = encoded_img.tobytes()
            
            files = {'file': ('live.jpg', bytes_data, 'image/jpeg')}
            
            try:
                # Try connection to backend
                try:
                    response = requests.post("http://host.docker.internal:8000/predict_live", files=files, timeout=1)
                except:
                    response = requests.post("http://localhost:8000/predict_live", files=files, timeout=1)
                
                if response.status_code == 200:
                    result = response.json()
                    faces = result.get("faces", [])
                    
                    for face in faces:
                        name = face['name']
                        conf = face['confidence']
                        box = face['box']
                        x, y, w, h = box
                        
                        color = (0, 255, 0) # Green
                        if name == "Desconocido" or name == "Unknown":
                            color = (0, 0, 255) # Red
                        
                        # Draw Box
                        cv2.rectangle(img, (x, y), (x+w, y+h), color, 3)
                        
                        # Draw Text
                        label = f"{name} {int(conf*100)}%"
                        cv2.putText(img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                        
            except Exception as e:
                print(f"Error in processing: {e}")
            
            return img

    webrtc_streamer(key="live_surveillance", video_processor_factory=VideoProcessor, rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

# Main Flow
if st.session_state.logged_in:
    dashboard_view()
else:
    login_view()

