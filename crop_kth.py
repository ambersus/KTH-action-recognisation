# -*- coding: utf-8 -*-
"""
Created on Thu Aug 28 15:26:39 2025

Modified: For video -> frames -> YOLO crops pipeline
"""

import os
import cv2
from ultralytics import YOLO

# -----------------------------
# CONFIG
# -----------------------------
videos_root = r"D:\Deepak\Dataset\KTH"    # 🔹 input dataset root (contains subfolders with videos)
output_frames_root = r"D:\Deepak\Dataset\frames" # 🔹 where extracted frames will be saved
output_crops_root = r"D:\Deepak\Dataset\crops"   # 🔹 where cropped persons will be saved
CROP_SIZE = 224
model = YOLO("yolov8n.pt")
# -----------------------------


def crop_centered(img, x1, y1, x2, y2, crop_size=224):
    """Take a fixed crop of size crop_size x crop_size around the detected box center."""
    h, w = img.shape[:2]

    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    half = crop_size // 2

    x_start = max(0, cx - half)
    y_start = max(0, cy - half)
    x_end = min(w, cx + half)
    y_end = min(h, cy + half)

    crop = img[y_start:y_end, x_start:x_end]

    if crop.shape[0] != crop_size or crop.shape[1] != crop_size:
        crop = cv2.copyMakeBorder(
            crop,
            top=max(0, crop_size - crop.shape[0]),
            bottom=0,
            left=max(0, crop_size - crop.shape[1]),
            right=0,
            borderType=cv2.BORDER_CONSTANT,
            value=(0, 0, 0)
        )
        crop = cv2.resize(crop, (crop_size, crop_size))

    return crop


# -----------------------------
# MAIN LOOP
# -----------------------------
for subdir, dirs, files in os.walk(videos_root):
    for file in files:
        if file.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):  # video formats
            video_path = os.path.join(subdir, file)

            # Maintain relative subfolder structure
            rel_path = os.path.relpath(subdir, videos_root)
            frames_save_dir = os.path.join(output_frames_root, rel_path, os.path.splitext(file)[0])
            crops_save_dir = os.path.join(output_crops_root, rel_path, os.path.splitext(file)[0])
            os.makedirs(frames_save_dir, exist_ok=True)
            os.makedirs(crops_save_dir, exist_ok=True)

            cap = cv2.VideoCapture(video_path)
            frame_idx = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_name = f"frame_{frame_idx:05d}.jpg"
                frame_path = os.path.join(frames_save_dir, frame_name)

                # Save frame
                cv2.imwrite(frame_path, frame)

                # Run YOLO detection
                results = model(frame, verbose=False)

                person_boxes = []
                for box in results[0].boxes:
                    if int(box.cls[0]) == 0:  # class "person"
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        area = (x2 - x1) * (y2 - y1)
                        person_boxes.append((area, (x1, y1, x2, y2)))

                if person_boxes:
                    _, (x1, y1, x2, y2) = max(person_boxes, key=lambda x: x[0])
                    crop = crop_centered(frame, x1, y1, x2, y2, CROP_SIZE)

                    crop_name = f"crop_{frame_idx:05d}.jpg"
                    crop_path = os.path.join(crops_save_dir, crop_name)
                    cv2.imwrite(crop_path, crop)

                frame_idx += 1

            cap.release()

print("✅ Done! Frames and crops saved.")
