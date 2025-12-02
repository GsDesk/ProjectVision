import cv2
import os
import time

def capture_images(name, num_images=50):
    base_dir = f"dataset/{name}"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print(f"Starting capture for {name}. Press 'q' to quit early.")
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
        img_name = os.path.join(base_dir, f"{name}_{count}.jpg")
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
