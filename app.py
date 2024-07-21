import hashlib
import subprocess
import cv2
import numpy as np
import os
import time
import tkinter as tk
from PIL import Image, ImageTk
from threading import Timer
from datetime import datetime

# Load the Haar Cascade Classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to capture an image from the webcam
def capture_image():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video capture device.")
        return False

    root = tk.Tk()
    root.title("Image Capture")
    label = tk.Label(root)
    label.pack()
    
    frame = None
    timer_started = False  # Initialize timer_started here

    def update_frame():
        nonlocal frame, timer_started
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from video capture device.")
            return
        
        cv2_im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(cv2_im)
        img_tk = ImageTk.PhotoImage(img)
        label.config(image=img_tk)
        label.image = img_tk
        root.update()

        if not timer_started:
            Timer(2, close_camera).start()
            timer_started = True

        root.after(1, update_frame)

    def on_key_press(event):
        if event.keysym == 'c':  # Press 'c' to capture the image
            if frame is not None:
                # Convert the frame to grayscale for face detection
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Detect faces in the image
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                if len(faces) == 0:
                    image_path = f"images/full_frame_{timestamp}.jpg"
                    cv2.imwrite(image_path, frame)
                    print(f"Full frame captured and saved to {image_path}")
                else:
                    # Crop and save each detected face
                    for i, (x, y, w, h) in enumerate(faces):
                        face_img = frame[y:y+h, x:x+w]
                        face_path = f"images/cropped_face_{timestamp}_{i}.jpg"
                        cv2.imwrite(face_path, face_img)
                        print(f"Face cropped and saved to {face_path}")
                
                root.destroy()
        elif event.keysym == 'q':  # Press 'q' to quit
            root.destroy()

    def close_camera():
        cap.release()
        print("Camera released after 2 seconds.")

    root.bind("<KeyPress>", on_key_press)
    update_frame()
    root.mainloop()
    return True

# Function to read and preprocess the image
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Failed to load image from {image_path}.")
        return None
    
    img = cv2.resize(img, (96, 96))
    img = img[..., ::-1]  # BGR to RGB
    img = np.around(np.transpose(img, (2, 0, 1)) / 255.0, decimals=12)
    img = np.array([img])
    return img

# Function to generate hash from image
def generate_image_hash(image_path):
    img = preprocess_image(image_path)
    if img is None:
        print("Error: Preprocessing failed. Exiting.")
        return None
    
    encoding_str = ','.join(map(str, img.flatten()))
    hash_value = hashlib.sha256(encoding_str.encode()).hexdigest()
    return hash_value

# Capture image and save to specified path
if capture_image():
    # Generate hash from the captured image
    # Find the latest image file for hashing
    image_files = [f for f in os.listdir('images') if f.endswith('.jpg')]
    if image_files:
        latest_image = max(image_files, key=lambda f: os.path.getmtime(os.path.join('images', f)))
        latest_image_path = os.path.join('images', latest_image)
        start_time = time.time()
        image_hash = generate_image_hash(latest_image_path)
        end_time = time.time()
        
        if image_hash is not None:
            if end_time - start_time > 2:
                print("Warning: Hash generation took longer than 2 seconds.")
            
            # Write the hash to a file
            with open('hash_output.txt', 'w') as f:
                f.write(image_hash)
            print("Hash written to hash_output.txt")
            
            # Run the Node.js script to send the hash to the smart contract
            try:
                subprocess.run(['node', 'Interaction.js'], check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error running Node.js script: {e}")
        else:
            print("Hash generation failed.")
    else:
        print("No images found to generate hash.")
else:
    print("Image capture failed.")
