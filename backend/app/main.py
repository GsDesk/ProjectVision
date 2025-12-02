from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
import cv2
import tensorflow as tf
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "/app/modules/models/face_recognition_model.h5"
model = None
face_cascade = None

@app.on_event("startup")
async def load_resources():
    global model, face_cascade
    if os.path.exists(MODEL_PATH):
        try:
            model = tf.keras.models.load_model(MODEL_PATH)
            print(f"✅ MODELO IA CARGADO")
        except Exception as e:
            print(f"❌ Error Modelo: {e}")
    
    try:
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(cascade_path)
        print("✅ DETECTOR OPENCV CARGADO")
    except Exception as e:
        print(f"❌ Error Detector: {e}")

# --- LOGIN ---
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None: return {"error": "Modelo no listo"}
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        img = cv2.resize(img, (128, 128))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)
        
        prediction = model.predict(img)
        class_index = np.argmax(prediction)
        confidence = float(np.max(prediction))
        
        # --- CORRECCIÓN DE ETIQUETAS ---
        # Invertimos el orden para corregir el cruce de identidades
        classes = ["Oscar", "Alex"]
        
        # Umbral estricto para Login
        result = classes[class_index] if confidence > 0.85 else "Desconocido"
        
        return {"class": result, "person": result, "confidence": confidence}
    except Exception as e:
        return {"error": str(e)}

# --- VIDEO EN VIVO ---
@app.post("/predict_live")
async def predict_live(file: UploadFile = File(...)):
    if model is None or face_cascade is None: return {"error": "Sistema no listo"}
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Filtros de imagen para ayudar al detector
        gray_blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray_enhanced = clahe.apply(gray_blurred)

        # Detector MUY estricto para eliminar fantasmas
        # minNeighbors=20: Requiere muchas coincidencias para validar una cara
        # minSize=(100, 100): Ignora cosas pequeñas o lejanas que suelen ser ruido
        faces = face_cascade.detectMultiScale(
            gray_enhanced, 
            scaleFactor=1.1, 
            minNeighbors=20, 
            minSize=(100, 100)
        )
        
        detections = []
        for (x, y, w, h) in faces:
            roi_color = frame[y:y+h, x:x+w]
            roi_resized = cv2.resize(roi_color, (128, 128))
            roi_normalized = roi_resized / 255.0
            roi_expanded = np.expand_dims(roi_normalized, axis=0)
            
            prediction = model.predict(roi_expanded)
            class_index = np.argmax(prediction)
            confidence = float(np.max(prediction))
            
            # --- CORRECCIÓN DE ETIQUETAS ---
            classes = ["Oscar", "Alex"]
            
            # Umbral de confianza 85% para filtrar desconocidos
            name = classes[class_index] if confidence > 0.85 else "Desconocido"
            
            detections.append({
                "name": name,
                "confidence": confidence,
                "box": [int(x), int(y), int(w), int(h)]
            })
            
        return {"faces": detections}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
