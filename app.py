import hashlib
import subprocess
import cv2
import numpy as np
import os
import tkinter as tk
from PIL import Image, ImageTk
from threading import Timer
from datetime import datetime

# Load the Haar Cascade Classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Ensure the images directory exists
os.makedirs("images", exist_ok=True)

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
    timer_started = False

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
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                if len(faces) > 0:
                    # Crop and save the first detected face
                    x, y, w, h = faces[0]
                    face_img = frame[y:y+h, x:x+w]
                    face_path = f"images/cropped_face_{timestamp}.jpg"
                    cv2.imwrite(face_path, face_img)
                    print(f"Face cropped and saved to {face_path}")

                    # Generate hash from the face image and call Node.js script
                    image_hash = generate_image_hash(face_path)
                    if image_hash:
                        hash_file_path = f"hash_output_{timestamp}.txt"
                        save_hash_to_file(image_hash, hash_file_path)
                        call_node_script(hash_file_path)
                
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

# Function to save the hash to a file
def save_hash_to_file(hash_value, file_path):
    with open(file_path, 'w') as f:
        f.write(hash_value)
    print(f"Hash written to {file_path}")

# Function to call the Node.js script with the hash file path as an argument
def call_node_script(hash_file_path):
    try:
        result = subprocess.run(['node', 'Interaction.js', hash_file_path], check=True, text=True, capture_output=True)
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error calling Node.js script: {e.stderr}")

# Start the image capture process
if capture_image():
    print("Image capture and processing completed.")
else:
    print("Image capture failed.")
