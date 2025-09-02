# -*- coding: utf-8 -*-
"""
Created on Thu Aug 28 15:46:57 2025

@author: Amma
"""

import os
import cv2
import mediapipe as mp

# -----------------------------
# CONFIG
# -----------------------------
crops_root = r"D:\Deepak\Dataset\crops"   # folder where cropped persons are stored
output_vis_root = r"D:\Deepak\Dataset\pose_vis"  # where visualizations will be saved
os.makedirs(output_vis_root, exist_ok=True)

# MediaPipe Pose setup
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

# -----------------------------
# MAIN LOOP
# -----------------------------
for subdir, dirs, files in os.walk(crops_root):
    for file in files:
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            img_path = os.path.join(subdir, file)
            img = cv2.imread(img_path)

            # Convert to RGB for mediapipe
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            if results.pose_landmarks:
                # Draw skeleton on image
                annotated = img.copy()
                mp_drawing.draw_landmarks(
                    annotated,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(255,0,0), thickness=2)
                )

                # Save visualization (keeping folder structure)
                rel_path = os.path.relpath(subdir, crops_root)
                save_dir = os.path.join(output_vis_root, rel_path)
                os.makedirs(save_dir, exist_ok=True)

                out_path = os.path.join(save_dir, f"vis_{file}")
                cv2.imwrite(out_path, annotated)

print("✅ Visualization complete!")
