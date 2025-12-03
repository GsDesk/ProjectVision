import cv2
import os
import numpy as np

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "../../"))
DATASET_UNKNOWN = os.path.join(PROJECT_ROOT, "dataset", "Unknown")

# Images uploaded by user (Step 81 - The "False Positive" images)
# These are confirmed to be the "Unknown" person
IMAGE_PATHS = [
    r"C:/Users/AleX'x/.gemini/antigravity/brain/c107b992-80ef-4865-b6ae-84726426e09c/uploaded_image_0_1764689286839.png",
    r"C:/Users/AleX'x/.gemini/antigravity/brain/c107b992-80ef-4865-b6ae-84726426e09c/uploaded_image_1_1764689286839.png"
]

def add_unknowns():
    if not os.path.exists(DATASET_UNKNOWN):
        os.makedirs(DATASET_UNKNOWN)

    # Load detectors
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)

    count = 0
    for img_path in IMAGE_PATHS:
        if not os.path.exists(img_path):
            print(f"Image not found: {img_path}")
            continue

        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to load: {img_path}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        print(f"Found {len(faces)} faces in {os.path.basename(img_path)}")

        for (x, y, w, h) in faces:
            # Save original
            face_img = img[y:y+h, x:x+w]
            save_name = f"unknown_{count}.jpg"
            cv2.imwrite(os.path.join(DATASET_UNKNOWN, save_name), face_img)
            count += 1
            
            # Augment immediately to increase dataset size
            # Flip
            flipped = cv2.flip(face_img, 1)
            cv2.imwrite(os.path.join(DATASET_UNKNOWN, f"unknown_{count}_flip.jpg"), flipped)
            
            # Brightness/Contrast changes
            for alpha in [0.8, 1.2]: # Contrast
                adjusted = cv2.convertScaleAbs(face_img, alpha=alpha, beta=0)
                cv2.imwrite(os.path.join(DATASET_UNKNOWN, f"unknown_{count}_c{alpha}.jpg"), adjusted)
            
            count += 1

    print(f"Added {count} unknown face images to {DATASET_UNKNOWN}")

if __name__ == "__main__":
    add_unknowns()
