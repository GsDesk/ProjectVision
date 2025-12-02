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
    global model, face_cascade, profile_cascade, face_cascade_alt
    if os.path.exists(MODEL_PATH):
        try:
            model = tf.keras.models.load_model(MODEL_PATH)
            print(f"✅ MODELO IA CARGADO")
        except Exception as e:
            print(f"❌ Error Modelo: {e}")
    
    try:
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        cascade_alt_path = cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml'
        profile_path = cv2.data.haarcascades + 'haarcascade_profileface.xml'
        
        face_cascade = cv2.CascadeClassifier(cascade_path)
        face_cascade_alt = cv2.CascadeClassifier(cascade_alt_path)
        profile_cascade = cv2.CascadeClassifier(profile_path)
        print("✅ DETECTORES OPENCV CARGADO")
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
        classes = ["Alex", "Oscar"]
        
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

        # 1. Frontal Face (Default)
        faces_frontal = face_cascade.detectMultiScale(
            gray_enhanced, 
            scaleFactor=1.1, 
            minNeighbors=15, 
            minSize=(100, 100)
        )

        # 2. Frontal Face (Alt2) - Better for different angles/looking down
        faces_frontal_alt = face_cascade_alt.detectMultiScale(
            gray_enhanced,
            scaleFactor=1.1,
            minNeighbors=10,
            minSize=(100, 100)
        )
        
        # 3. Profile Face
        faces_profile = profile_cascade.detectMultiScale(
            gray_enhanced,
            scaleFactor=1.1,
            minNeighbors=8,
            minSize=(100, 100)
        )
        
        # 4. Flipped Profile Face
        flipped_gray = cv2.flip(gray_enhanced, 1)
        faces_flipped = profile_cascade.detectMultiScale(
            flipped_gray,
            scaleFactor=1.1,
            minNeighbors=8,
            minSize=(100, 100)
        )
        
        # Combine detections
        all_faces = []
        
        # Add frontal
        for f in faces_frontal:
            all_faces.append(f)

        # Add frontal alt
        for f in faces_frontal_alt:
            all_faces.append(f)
            
        # Add profile
        for f in faces_profile:
            all_faces.append(f)
            
        # Add flipped profile (adjust coords)
        h_img, w_img = gray.shape
        for (x, y, w, h) in faces_flipped:
            x_orig = w_img - x - w
            all_faces.append((x_orig, y, w, h))
            
        # Simple NMS (Non-Maximum Suppression) to remove duplicates
        if len(all_faces) > 0:
            # Use area as score to prefer larger faces
            scores = [float(w * h) for (x, y, w, h) in all_faces]
            
            # nms_threshold=0.3: suppress if IoU > 0.3
            indices = cv2.dnn.NMSBoxes(all_faces, scores, score_threshold=0.0, nms_threshold=0.3)
            
            final_faces = []
            if len(indices) > 0:
                # indices returns a numpy array, flatten it to iterate
                for i in indices.flatten():
                    final_faces.append(all_faces[i])
            all_faces = final_faces
        
        detections = []
        for (x, y, w, h) in all_faces:
            roi_color = frame[y:y+h, x:x+w]
            if roi_color.size == 0: continue
            
            roi_resized = cv2.resize(roi_color, (128, 128))
            roi_normalized = roi_resized / 255.0
            roi_expanded = np.expand_dims(roi_normalized, axis=0)
            
            prediction = model.predict(roi_expanded)
            class_index = np.argmax(prediction)
            confidence = float(np.max(prediction))
            
            # --- CORRECCIÓN DE ETIQUETAS ---
            classes = ["Alex", "Oscar"]
            
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
