import cv2
import os
import time

def capture_images(name, num_images=50):
    # Fix: Use absolute path to the root 'dataset' folder
    # Script is in modules/data_collection, so root is ../../
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "../../"))
    base_dir = os.path.join(project_root, "dataset", name)
    
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    print("Attempting to open webcam...")
    # Use CAP_DSHOW for Windows to avoid hanging
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Find the next available index
    existing_files = [f for f in os.listdir(base_dir) if f.startswith(name) and f.endswith(".jpg")]
    start_count = 0
    if existing_files:
        indices = []
        for f in existing_files:
            try:
                # Extract number from "Name_123.jpg"
                idx = int(f.split('_')[-1].split('.')[0])
                indices.append(idx)
            except ValueError:
                continue
        if indices:
            start_count = max(indices) + 1
            
    print(f"Starting capture for {name} from index {start_count}. Press 'q' to quit early.")
    print("Get ready...")
    time.sleep(2)

    count = 0
    while count < num_images:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Display the resulting frame
        cv2.imshow('Frame', frame)

        # Save frame
        current_idx = start_count + count
        img_name = os.path.join(base_dir, f"{name}_{current_idx}.jpg")
        cv2.imwrite(img_name, frame)
        print(f"Saved {img_name}")
        count += 1
        
        # Small delay to allow movement
        time.sleep(0.1)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Finished capturing {count} images for {name}.")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        name = sys.argv[1]
        if name in ["Alex", "Oscar"]:
            capture_images(name)
        else:
            print(f"Unknown name: {name}. Please use 'Alex' or 'Oscar'.")
    else:
        print("Select person to capture images for:")
        print("1. Alex")
        print("2. Oscar")
        choice = input("Enter number (1 or 2): ")
        
        if choice == '1':
            capture_images("Alex")
        elif choice == '2':
            capture_images("Oscar")
        else:
            print("Invalid choice.")
