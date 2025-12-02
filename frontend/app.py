import streamlit as st
import requests
from PIL import Image
import io

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
                    
                    st.markdown("---")
                    if person != "Unknown":
                        st.success(f"Access Granted: Welcome, {person}!")
                        st.metric("Confidence", f"{confidence*100:.1f}%")
                    else:
                        st.error("Access Denied: Unknown Person")
                        st.metric("Confidence", f"{confidence*100:.1f}%")
            else:
                st.error(f"Server Error: {response.status_code}")
                
        except Exception as e:
            st.error(f"Connection Error: {e}")
            st.info("Ensure the backend service is running.")

