FROM python:3.9-slim

WORKDIR /app

# Instalar dependencias de sistema para Gráficos (GL) y OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# INSTALAR OPENCV MANUALMENTE (Crucial para el nuevo código)
RUN pip install opencv-python-headless

# Copiar código
COPY backend/app /app/app
COPY modules /app/modules

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
