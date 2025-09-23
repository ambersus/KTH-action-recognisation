# -*- coding: utf-8 -*-
"""
Created on Tue Sep 23 17:19:31 2025

@author: Amma
"""

# 

import os
import cv2
import numpy as np
import torch
import mediapipe as mp

# Import the model class from your training script
from train import LSTMClassifier

# -------------------------------------------------
# CONFIGURATION
# -------------------------------------------------
# 🔹 Path to the root folder containing cropped person images
TEST_CROPS_ROOT = r"D:\Deepak\Dataset\crops"
# 🔹 Path where the output visualization videos will be saved
OUTPUT_VIZ_ROOT = r"D:\Deepak\Dataset\predictions_viz"
# 🔹 Path to your trained model weights
MODEL_PATH = "pose_lstm.pth"

# Model & Data Parameters (should match your training setup)
SEQUENCE_LENGTH = 15
FRAMES_PER_VIDEO = 150  # We process the first 150 frames of each video
INPUT_DIM = 33 * 4      # 33 landmarks * 4 coordinates (x, y, z, visibility)

# -------------------------------------------------
# SETUP
# -------------------------------------------------
# Create output directory
os.makedirs(OUTPUT_VIZ_ROOT, exist_ok=True)

# Setup device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")

# MediaPipe Pose setup
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

# -------------------------------------------------
# LOAD TRAINED MODEL
# -------------------------------------------------
# Dynamically get class names from the dataset folder structure
class_dirs = sorted([d for d in os.listdir(TEST_CROPS_ROOT) if os.path.isdir(os.path.join(TEST_CROPS_ROOT, d))])
class_names = {i: name for i, name in enumerate(class_dirs)}
num_classes = len(class_names)

print(f"✅ Found {num_classes} classes: {list(class_names.values())}")

# Instantiate the model with the same architecture as during training
model = LSTMClassifier(
    input_dim=INPUT_DIM,
    hidden_size=256,
    num_classes=num_classes,
    num_layers=2,
    dropout=0.5,
    bidirectional=True
).to(device)

# Load the saved weights
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval() # Set the model to evaluation mode

print(f"✅ Model '{MODEL_PATH}' loaded successfully.")


# -------------------------------------------------
# MAIN VISUALIZATION LOOP
# -------------------------------------------------
# Walk through the directory to find all "Testing" folders
for root, dirs, files in os.walk(TEST_CROPS_ROOT):
    if os.path.basename(root) == "Testing":
        # We are in a "Testing" folder, now look for sample folders inside it
        for sample_folder in dirs:
            sample_path = os.path.join(root, sample_folder)
            
            # Get a sorted list of all frame images in the sample folder
            frame_files = sorted([f for f in os.listdir(sample_path) if f.lower().endswith((".jpg", ".png"))])
            
            if len(frame_files) < FRAMES_PER_VIDEO:
                print(f"⚠️ Skipping {sample_path}, found only {len(frame_files)} frames (less than {FRAMES_PER_VIDEO}).")
                continue

            # Take the first 150 frames
            frames_to_process = frame_files[:FRAMES_PER_VIDEO]
            
            print(f"\nProcessing video: {sample_folder}")

            # Process in 10 chunks of 15 frames each
            for i in range(FRAMES_PER_VIDEO // SEQUENCE_LENGTH):
                start_idx = i * SEQUENCE_LENGTH
                end_idx = start_idx + SEQUENCE_LENGTH
                sequence_files = frames_to_process[start_idx:end_idx]
                
                sequence_features = []
                frame_paths = [os.path.join(sample_path, f) for f in sequence_files]

                # 1. Extract features for the 15-frame sequence
                for img_path in frame_paths:
                    img = cv2.imread(img_path)
                    if img is None:
                        # If an image is missing, append zeros
                        features = np.zeros((33, 4))
                    else:
                        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        results = pose.process(rgb_img)
                        if results.pose_landmarks:
                            landmarks = results.pose_landmarks.landmark
                            features = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in landmarks])
                        else:
                            # If no pose is detected, append zeros
                            features = np.zeros((33, 4))
                    sequence_features.append(features)
                
                # Convert features to the format expected by the model
                features_np = np.stack(sequence_features, axis=0) # Shape: (15, 33, 4)
                features_flat = features_np.reshape(SEQUENCE_LENGTH, -1) # Shape: (15, 132)
                features_tensor = torch.tensor(features_flat, dtype=torch.float32).unsqueeze(0).to(device) # Shape: (1, 15, 132)

                # 2. Get model prediction
                with torch.no_grad():
                    outputs = model(features_tensor)
                    _, preds = torch.max(outputs, 1)
                    predicted_idx = preds.item()
                    predicted_class = class_names[predicted_idx]
                
                # 3. Create visualization video
                # Determine output path, maintaining the folder structure
                rel_path = os.path.relpath(sample_path, TEST_CROPS_ROOT)
                output_dir = os.path.join(OUTPUT_VIZ_ROOT, rel_path)
                os.makedirs(output_dir, exist_ok=True)
                video_out_path = os.path.join(output_dir, f"prediction_seq_{i+1:02d}_{predicted_class}.avi")

                # Get frame dimensions for the video writer
                first_frame = cv2.imread(frame_paths[0])
                height, width, _ = first_frame.shape
                
                # Use 'XVID' codec for .avi files
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                video_writer = cv2.VideoWriter(video_out_path, fourcc, 10.0, (width, height))

                # Loop through frames again to draw and write to video
                for img_path in frame_paths:
                    frame = cv2.imread(img_path)
                    annotated_frame = frame.copy()

                    # Re-run pose detection to get landmarks for drawing
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = pose.process(rgb_frame)
                    
                    # Draw skeleton
                    if results.pose_landmarks:
                        mp_drawing.draw_landmarks(
                            annotated_frame,
                            results.pose_landmarks,
                            mp_pose.POSE_CONNECTIONS,
                            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                            connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
                        )
                    
                    # Draw prediction text
                    cv2.putText(
                        annotated_frame,
                        f"Prediction: {predicted_class}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (55, 225, 205),
                        2,
                        cv2.LINE_AA
                    )
                    video_writer.write(annotated_frame)
                
                video_writer.release()
                print(f"  -> Saved sequence {i+1:02d} prediction '{predicted_class}' to {video_out_path}")

print("\n✅ All test sequences processed and visualizations saved.")