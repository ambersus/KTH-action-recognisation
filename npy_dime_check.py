import os
import numpy as np
import cv2

def extract_30_frames_from_sequence(seq_path, save_path):
    # list and sort all frames in the sequence
    frames_list = sorted([f for f in os.listdir(seq_path) if f.lower().endswith((".jpg", ".png"))])
    
    frames = []
    for frame_name in frames_list[:30]:  # take only first 30
        frame_path = os.path.join(seq_path, frame_name)
        img = cv2.imread(frame_path)
        if img is None:
            continue
        frames.append(img)
    
    frames = np.array(frames)  # shape: (N, H, W, C)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, frames)
    print(f"Saved {frames.shape} to {save_path}")

def process_dataset(root_dir, save_root):
    for annotation in os.listdir(root_dir):
        annotation_path = os.path.join(root_dir, annotation)
        if not os.path.isdir(annotation_path):
            continue
        
        for split in ["training", "testing", "other"]:  # go inside training/testing/other
            split_path = os.path.join(annotation_path, split)
            if not os.path.isdir(split_path):
                continue
            
            for seq in os.listdir(split_path):  # each sequence folder
                seq_path = os.path.join(split_path, seq)
                if not os.path.isdir(seq_path):
                    continue
                
                save_dir = os.path.join(save_root, annotation, split)
                save_path = os.path.join(save_dir, f"{seq}.npy")
                
                extract_30_frames_from_sequence(seq_path, save_path)

# Example usage:
# process_dataset("path/to/root", "path/to/save_npy")
