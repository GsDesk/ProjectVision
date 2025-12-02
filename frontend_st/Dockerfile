FROM python:3.9-slim

WORKDIR /app

# Dependencias del sistema
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Instalamos dependencias
RUN pip install streamlit requests watchdog streamlit-webrtc opencv-python-headless av

COPY app.py .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]
