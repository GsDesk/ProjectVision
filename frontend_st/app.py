import streamlit as st
import requests
import cv2
import numpy as np
import av
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

st.set_page_config(page_title="Sistema ROG Vision", page_icon="👁️", layout="wide")

API_LOGIN = "http://host.docker.internal:8000/predict"
API_LIVE = "http://host.docker.internal:8000/predict_live"

if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
if 'user_name' not in st.session_state:
    st.session_state['user_name'] = ""

def login_page():
    st.title("🔒 Acceso de Seguridad")
    img_file = st.camera_input("Identificación")

    if img_file is not None:
        try:
            bytes_data = img_file.getvalue()
            files = {"file": ("image.jpg", bytes_data, "image/jpeg")}
            with st.spinner('Verificando...'):
                response = requests.post(API_LOGIN, files=files)
            
            if response.status_code == 200:
                data = response.json()
                nombre = data.get("class", data.get("person", "Desconocido"))
                confianza = data.get("confidence", 0.0)

                if confianza > 0.70:
                    st.success(f"✅ Identidad: {nombre}")
                    st.session_state['logged_in'] = True
                    st.session_state['user_name'] = nombre
                    st.rerun()
                else:
                    st.error(f"⛔ No reconocido ({confianza:.1%})")
            else:
                st.error("Error de conexión.")
        except Exception as e:
            st.error(f"Error: {e}")

class VideoProcessor(VideoTransformerBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        try:
            _, encoded_img = cv2.imencode('.jpg', img)
            bytes_data = encoded_img.tobytes()
            files = {"file": ("image.jpg", bytes_data, "image/jpeg")}
            
            response = requests.post(API_LIVE, files=files, timeout=0.5)
            
            if response.status_code == 200:
                data = response.json()
                faces = data.get("faces", [])
                
                for face in faces:
                    name = face["name"]
                    conf = face["confidence"]
                    x, y, w, h = face["box"]
                    
                    # DIBUJAR SOLO RECUADRO VERDE (Sin malla)
                    color = (0, 255, 0) if name != "Desconocido" else (0, 0, 255)
                    cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
                    
                    # Etiqueta simple
                    label = f"{name} {conf:.0%}"
                    cv2.putText(img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        except Exception:
            pass
        return av.VideoFrame.from_ndarray(img, format="bgr24")

def landing_page():
    with st.sidebar:
        st.title(f"👤 {st.session_state['user_name']}")
        if st.button("Cerrar Sesión"):
            st.session_state['logged_in'] = False
            st.rerun()

    st.title("🚀 Detección Activa")
    webrtc_streamer(
        key="live-feed",
        video_processor_factory=VideoProcessor,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False},
    )

if st.session_state['logged_in']:
    landing_page()
else:
    login_page()
