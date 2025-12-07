import tensorflow as tf
import cv2
import numpy as np
import os
import glob

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'modules/models/face_recognition_model.h5')
DATA_DIR = os.path.join(BASE_DIR, 'modules/data_collection/dataset_cropped')

def test_model():
    print("--- Verificación Local del Modelo ---")
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Modelo no encontrado en: {MODEL_PATH}")
        return

    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print("✅ Modelo cargado correctamente.")
    except Exception as e:
        print(f"❌ Error cargando modelo: {e}")
        return

    classes = ["Alex", "Oscar", "Unknown"]
    
    # Test one image from each class
    for name in classes:
        class_dir = os.path.join(DATA_DIR, name)
        if not os.path.exists(class_dir):
            print(f"⚠️ No hay carpeta para {name}")
            continue
            
        images = glob.glob(os.path.join(class_dir, "*.jpg"))
        if not images:
            print(f"⚠️ No hay imágenes para {name}")
            continue
            
        # Pick the first image
        img_path = images[0]
        
        # Preprocess
        img = cv2.imread(img_path)
        if img is None:
            print(f"❌ Error leyendo imagen: {img_path}")
            continue
            
        img_resized = cv2.resize(img, (128, 128))
        img_norm = img_resized / 255.0
        img_batch = np.expand_dims(img_norm, axis=0)
        
        # Predict
        pred = model.predict(img_batch, verbose=0)
        idx = np.argmax(pred)
        conf = np.max(pred)
        predicted_raw = classes[idx]
        
        # Apply Strict Logic
        final_decision = predicted_raw
        if predicted_raw == "Unknown":
            final_decision = "Unknown"
        elif conf < 0.93:
            final_decision = "Unknown (Low Conf)"
        
        # Result
        # For Unknown class, we expect Unknown
        expected = name
        if name == "Unknown":
            is_correct = "Unknown" in final_decision
        else:
            is_correct = (final_decision == name)
            
        status = "✅ CORRECTO" if is_correct else "❌ INCORRECTO"
        print(f"Prueba con foto de {name}: Predicción -> {predicted_raw} ({conf:.1%}) -> Final: {final_decision} {status}")

if __name__ == "__main__":
    test_model()
