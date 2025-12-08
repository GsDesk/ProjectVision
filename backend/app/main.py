from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
import cv2
import tensorflow as tf
import os
import json

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "/app/modules/models/face_recognition_model.h5"
CLASSES_PATH = "/app/modules/models/class_indices.json"

model = None
face_cascade = None
profile_cascade = None

@app.on_event("startup")
async def load_resources():
    global model, face_cascade, profile_cascade
    if os.path.exists(MODEL_PATH):
        try:
            model = tf.keras.models.load_model(MODEL_PATH)
            print(f" MODELO CARGADO (3 CLASES)")
        except Exception as e:
            print(f" Error Modelo: {e}")
    
    try:
        # Load Frontal and Profile Cascades
        frontal_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        profile_path = cv2.data.haarcascades + 'haarcascade_profileface.xml'
        
        face_cascade = cv2.CascadeClassifier(frontal_path)
        profile_cascade = cv2.CascadeClassifier(profile_path)
        
        print(" DETECTORES CARGADOS (Frontal + Perfil)")
    except Exception as e:
        print(f" Error Detector: {e}")

def predecir_imagen(img):
    # Preprocesamiento idéntico al entrenamiento
    img_resized = cv2.resize(img, (128, 128))
    img_normalized = img_resized / 255.0
    img_expanded = np.expand_dims(img_normalized, axis=0)
    
    prediction = model.predict(img_expanded)
    class_index = np.argmax(prediction)
    confidence = float(np.max(prediction))
    
    # Valores por defecto
    nombre_detectado = "Desconocido"
    umbral_personalizado = 0.80 # Fallback por defecto
    
    if os.path.exists(CLASSES_PATH):
        try:
            with open(CLASSES_PATH, 'r') as f:
                config_indices = json.load(f)
                
            # El JSON ahora es: {"0": {"name": "Alex", "threshold": 0.70}, ...}
            key = str(class_index)
            if key in config_indices:
                data = config_indices[key]
                nombre_detectado = data.get("name", "Desconocido")
                umbral_personalizado = data.get("threshold", 0.80)
            else:
                # Fallback para formato antiguo o indices rotos
                nombre_detectado = "Desconocido"

        except Exception as e:
            print(f"Error cargando indices json: {e}")
            nombre_detectado = "Desconocido"
    
    # 0=Alex, 1=Oscar, 2=Unknown
    
    # Lógica de seguridad estricta:
    sorted_prediction = np.sort(prediction[0])
    top_score = sorted_prediction[-1]
    second_score = sorted_prediction[-2]
    margin = top_score - second_score

    UMBRAL_MARGEN = 0.15

    if nombre_detectado == "Unknown" or nombre_detectado == "Desconocido":
        return "Desconocido", confidence
    
    # Verificación estricta con UMBRAL DINÁMICO
    if confidence >= umbral_personalizado and margin >= UMBRAL_MARGEN:
        return nombre_detectado, confidence
    else:
        return "Desconocido", confidence

def detect_faces_multiscale(frame):
    """
    Función auxiliar para detectar caras usando cascadas múltiples (Frontal + Perfil).
    Devuelve una lista de recuadros [x, y, w, h] sin solapamientos.
    """
    if face_cascade is None or profile_cascade is None:
        return []

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Filtros visuales para mejorar detección
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)

    # Parametros de detección
    scale_factor = 1.1
    min_neighbors = 8 
    min_size = (60, 60)

    # 1. Frontal
    faces_frontal = face_cascade.detectMultiScale(gray, scale_factor, min_neighbors, minSize=min_size)
    
    # 2. Perfil Izquierdo
    faces_profile = profile_cascade.detectMultiScale(gray, scale_factor, min_neighbors, minSize=min_size)
    
    # 3. Perfil Derecho (Flip)
    flipped_gray = cv2.flip(gray, 1)
    faces_profile_flipped = profile_cascade.detectMultiScale(flipped_gray, scale_factor, min_neighbors, minSize=min_size)
    
    # Combinar detecciones
    all_faces = []
    
    for (x, y, w, h) in faces_frontal:
        all_faces.append([x, y, w, h])
        
    for (x, y, w, h) in faces_profile:
        all_faces.append([x, y, w, h])
        
    h_img, w_img = gray.shape
    for (x, y, w, h) in faces_profile_flipped:
        x_orig = w_img - x - w
        all_faces.append([x_orig, y, w, h])

    # Non-Maximum Suppression (NMS)
    if len(all_faces) > 0:
        all_faces_rects = list(all_faces)
        all_faces_rects.append(all_faces_rects[0]) # Dummy para groupRectangles
        rects, weights = cv2.groupRectangles(all_faces_rects, 1, 0.2)
        return rects
    else:
        return []

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """ Endpoint para Login (una sola imagen) """
    if model is None: return {"error": "Modelo no listo"}
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # 1. Detectar caras primero
        rects = detect_faces_multiscale(img)
        
        if len(rects) > 0:
            # Seleccionar la cara más grande (mayor área)
            x, y, w, h = max(rects, key=lambda b: b[2] * b[3])
            
            # Recortar ROI
            if w > 0 and h > 0:
                roi = img[y:y+h, x:x+w]
                nombre, conf = predecir_imagen(roi)
                return {"class": nombre, "person": nombre, "confidence": conf}
        
        # Si no se detecta nada, devolvemos Desconocido.
        # Enviar la imagen completa (fondo + cuerpo) al modelo suele dar resultados basura.
        return {"class": "Desconocido", "person": "Desconocido", "confidence": 0.0}

    except Exception as e:
        return {"error": str(e)}

@app.post("/predict_live")
async def predict_live(file: UploadFile = File(...)):
    """ Endpoint para Detección en Vivo (Stream de video) """
    if model is None: return {"error": "Sistema no listo"}
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        rects = detect_faces_multiscale(frame)

        detections = []
        for (x, y, w, h) in rects:
            roi = frame[y:y+h, x:x+w]
            if roi.size == 0: continue
            
            nombre, conf = predecir_imagen(roi)
            detections.append({
                "name": nombre,
                "confidence": conf,
                "box": [int(x), int(y), int(w), int(h)]
            })
            
        return {"faces": detections}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
