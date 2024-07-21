import hashlib
import subprocess
import cv2
import numpy as np
import os
import time
import tkinter as tk
from PIL import Image, ImageTk
from threading import Timer

# Load the Haar Cascade Classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to generate a unique filename based on the current timestamp
def generate_unique_filename(base_path, extension):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return f"{base_path}_{timestamp}.{extension}"

# Function to capture an image from the webcam
def capture_image():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video capture device.")
        return False

    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error: Camera not accessible.")
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
                
                if len(faces) == 0:
                    print("No face detected. Capturing the full frame.")
                    # Generate a unique filename for the full frame image
                    full_frame_path = generate_unique_filename("images/full_frame", "jpg")
                    cv2.imwrite(full_frame_path, frame)
                    print(f"Full frame captured and saved to {full_frame_path}")
                else:
                    # Crop and save each detected face with a unique filename
                    for i, (x, y, w, h) in enumerate(faces):
                        face_img = frame[y:y+h, x:x+w]
                        face_path = generate_unique_filename("images/cropped_face", "jpg")
                        cv2.imwrite(face_path, face_img)
                        print(f"Face cropped and saved to {face_path}")

                        # Generate hash for this face image
                        face_hash = generate_image_hash(face_path)
                        hash_path = generate_unique_filename("images/face_hash", "txt")
                        with open(hash_path, 'w') as f:
                            f.write(face_hash)
                        print(f"Hash for face saved to {hash_path}")
                
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
    print("Image capture and processing completed.")
else:
    print("Image capture failed.")
