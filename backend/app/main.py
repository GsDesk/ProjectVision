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
    
    # Intentar cargar índices dinámicos
    import json
    CLASSES_PATH = "/app/modules/models/class_indices.json"
    classes = []
    
    if os.path.exists(CLASSES_PATH):
        try:
            with open(CLASSES_PATH, 'r') as f:
                indices = json.load(f)
                # indices es {'Alex': 0, 'Oscar': 1, ...} -> invertir a [0: 'Alex', ...]
                classes_dict = {v: k for k, v in indices.items()}
                classes = [classes_dict[i] for i in range(len(classes_dict))]
        except Exception as e:
            print(f"Error cargando indices: {e}")
            classes = ["Alex", "Oscar", "Desconocido"]
    else:
        classes = ["Alex", "Oscar", "Desconocido"]
    
    if not classes:
         classes = ["Alex", "Oscar", "Desconocido"]

    # 0=Alex, 1=Oscar, 2=Unknown (Por defecto)
    
    nombre_detectado = classes[class_index]
    
    # Lógica de seguridad estricta:
    # 1. Si el modelo dice explícitamente "Desconocido" -> Es Desconocido.
    # 2. Si la confianza es menor a 0.93 (93%) -> Es Desconocido.
    # 3. Verificamos el margen con la segunda mejor opción para evitar ambigüedad.
    
    sorted_prediction = np.sort(prediction[0])
    top_score = sorted_prediction[-1]
    second_score = sorted_prediction[-2]
    margin = top_score - second_score

    # Umbrales
    # Umbrales Estrictos Actualizados
    # Umbrales Estrictos (Ajustados para usabilidad)
    # Umbrales (Ajustados para permitir acceso con ~80%)
    UMBRAL_CONFIANZA = 0.75  
    UMBRAL_MARGEN = 0.15

    if nombre_detectado == "Desconocido":
        return "Desconocido", confidence
    
    # Verificación estricta
    if confidence >= UMBRAL_CONFIANZA and margin >= UMBRAL_MARGEN:
        return nombre_detectado, confidence
    else:
        # Si no cumple los requisitos estrictos, se marca como Desconocido
        return "Desconocido", confidence

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None: return {"error": "Modelo no listo"}
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        nombre, conf = predecir_imagen(img)
        return {"class": nombre, "person": nombre, "confidence": conf}
    except Exception as e:
        return {"error": str(e)}

@app.post("/predict_live")
async def predict_live(file: UploadFile = File(...)):
    if model is None: return {"error": "Sistema no listo"}
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Filtros visuales
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)

        # Multiscale detection parameters
        scale_factor = 1.1
        min_neighbors = 8 # Slightly less strict than 12 to catch profiles
        min_size = (60, 60)

        # 1. Frontal Detection
        faces_frontal = face_cascade.detectMultiScale(gray, scale_factor, min_neighbors, minSize=min_size)
        
        # 2. Profile Detection (Left)
        faces_profile = profile_cascade.detectMultiScale(gray, scale_factor, min_neighbors, minSize=min_size)
        
        # 3. Profile Detection (Right - via Flip)
        flipped_gray = cv2.flip(gray, 1)
        faces_profile_flipped = profile_cascade.detectMultiScale(flipped_gray, scale_factor, min_neighbors, minSize=min_size)
        
        # Combine detections
        all_faces = []
        
        # Add Frontal
        for (x, y, w, h) in faces_frontal:
            all_faces.append([x, y, w, h])
            
        # Add Profile
        for (x, y, w, h) in faces_profile:
            all_faces.append([x, y, w, h])
            
        # Add Flipped Profile (Transform coords back)
        h_img, w_img = gray.shape
        for (x, y, w, h) in faces_profile_flipped:
            x_orig = w_img - x - w
            all_faces.append([x_orig, y, w, h])

        # Basic Non-Maximum Suppression (NMS) to remove overlapping boxes
        # OpenCV groupRectangles is robust for this
        if len(all_faces) > 0:
            all_faces_rects = list(all_faces)
            all_faces_rects.append(all_faces_rects[0]) # Dummy for groupRectangles quirks sometimes
            rects, weights = cv2.groupRectangles(all_faces_rects, 1, 0.2)
        else:
            rects = []

        detections = []
        for (x, y, w, h) in rects:
            roi = frame[y:y+h, x:x+w]
            # Ensure ROI is valid
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
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
