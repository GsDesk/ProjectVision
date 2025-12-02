import cv2
import os
import glob

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, 'dataset')
OUTPUT_DIR = os.path.join(BASE_DIR, 'dataset_cropped')

def process_dataset():
    if not os.path.exists(INPUT_DIR):
        print(f"Input directory {INPUT_DIR} not found.")
        return

    # Load detector
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)

    classes = ['Alex', 'Oscar']
    
    total_processed = 0
    total_faces = 0

    for class_name in classes:
        input_class_dir = os.path.join(INPUT_DIR, class_name)
        output_class_dir = os.path.join(OUTPUT_DIR, class_name)
        
        if not os.path.exists(output_class_dir):
            os.makedirs(output_class_dir)
            
        image_paths = glob.glob(os.path.join(input_class_dir, '*.jpg'))
        print(f"Processing {class_name}: {len(image_paths)} images...")
        
        for img_path in image_paths:
            img = cv2.imread(img_path)
            if img is None: continue
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) > 0:
                # Take the largest face
                faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)
                x, y, w, h = faces[0]
                
                # Crop
                face_img = img[y:y+h, x:x+w]
                
                # Save
                filename = os.path.basename(img_path)
                save_path = os.path.join(output_class_dir, filename)
                cv2.imwrite(save_path, face_img)
                total_faces += 1
            else:
                # If no face detected, maybe save original? 
                # No, better to skip bad data for training.
                pass
            
            total_processed += 1

    print(f"Done. Processed {total_processed} images. Saved {total_faces} cropped faces to {OUTPUT_DIR}")

if __name__ == "__main__":
    process_dataset()
