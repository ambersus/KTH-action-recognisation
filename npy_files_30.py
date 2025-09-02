# -*- coding: utf-8 -*-
"""
Created on Thu Aug 28 15:46:57 2025
Upgraded version: process sequence folders and save first 30 frame features
"""

import os
import cv2
import numpy as np
import mediapipe as mp

# -----------------------------
# CONFIG
# -----------------------------
root_dir = r"D:\Deepak\Dataset\crops"   # main root folder
output_feat_root = r"D:\Deepak\Dataset\pose_features"  # output root
os.makedirs(output_feat_root, exist_ok=True)

# MediaPipe Pose setup
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

# -----------------------------
# MAIN LOOP
# -----------------------------
for subdir, dirs, files in os.walk(root_dir):
    # Check if this folder contains image frames (sequence folder)
    img_files = [f for f in files if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    if len(img_files) > 0:  # found a sequence folder
        img_files.sort()  # ensure correct order
        img_files = img_files[:30]  # take first 30 frames only

        sequence_features = []

        for file in img_files:
            img_path = os.path.join(subdir, file)
            img = cv2.imread(img_path)
            if img is None:
                continue

            # Convert to RGB for mediapipe
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                features = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in landmarks])
            else:
                # If no person detected, fill with zeros (33x4)
                features = np.zeros((33, 4))

            sequence_features.append(features)

        # Stack into array (30, 33, 4)
        sequence_features = np.stack(sequence_features, axis=0)

        # Save features (keeping folder structure)
        rel_path = os.path.relpath(subdir, root_dir)
        save_dir = os.path.join(output_feat_root, rel_path)
        os.makedirs(save_dir, exist_ok=True)

        out_path = os.path.join(save_dir, "sequence.npy")
        np.save(out_path, sequence_features)

print("✅ Sequence feature extraction complete! (saved as .npy)")
