FROM python:3.9-slim

WORKDIR /app

# Install system dependencies for OpenCV and AV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
RUN pip install streamlit requests streamlit-webrtc opencv-python-headless av

# Copy the app code
COPY app.py .

# Expose Streamlit port
EXPOSE 8501

# Run Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
