import os
import cv2
import numpy as np
import torch
import mediapipe as mp
from ultralytics import YOLO
from collections import deque

# Import the model class from your training script
from train import LSTMClassifier

# -------------------------------------------------
# CONFIGURATION
# -------------------------------------------------
# 🔹 Path to the new video you want to test
INPUT_VIDEO_PATH = r"D:\Deepak\Dataset\test_real_time\ssvid.net--Running-Routine-for-Beginners_1080p.mp4"
# 🔹 Path to the directory where the output video will be saved
OUTPUT_DIR = r"D:\Deepak\Dataset\real_time_result"
# 🔹 Path to your trained LSTM model weights
MODEL_PATH = "pose_lstm.pth"
# 🔹 Path to the YOLO model for person detection
YOLO_MODEL_PATH = "yolov8n.pt"

# Model & Data Parameters
SEQUENCE_LENGTH = 15
INPUT_DIM = 33 * 4

# -------------------------------------------------
# MAIN INFERENCE SCRIPT
# -------------------------------------------------
if __name__ == "__main__":
    # ---- 1. SETUP ALL MODELS ----
    print("✅ Setting up models...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load YOLO Person Detector
    yolo_model = YOLO(YOLO_MODEL_PATH)
    
    # Load MediaPipe Pose
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    
    # Load Trained LSTM Classifier
    class_dirs_path = r"D:\Deepak\Dataset\crops"
    if not os.path.exists(class_dirs_path):
        print(f"❌ Error: The directory for class names '{class_dirs_path}' was not found.")
        exit()
        
    class_dirs = sorted([d for d in os.listdir(class_dirs_path) if os.path.isdir(os.path.join(class_dirs_path, d))])
    class_names = {i: name for i, name in enumerate(class_dirs)}
    num_classes = len(class_names)

    lstm_model = LSTMClassifier(
        input_dim=INPUT_DIM, hidden_size=256, num_classes=num_classes, num_layers=2, bidirectional=True
    ).to(device)
    lstm_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    lstm_model.eval()
    print("✅ All models loaded successfully.")

    # ---- 2. SETUP VIDEO I/O ----
    cap = cv2.VideoCapture(INPUT_VIDEO_PATH)
    if not cap.isOpened():
        print("❌ Error: Could not open input video.")
        exit()
        
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # Corrected line: Create the full path with a filename
    output_filename = "predicted_video.avi"
    OUTPUT_VIDEO_PATH = os.path.join(OUTPUT_DIR, output_filename)
    
    # Ensure the output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (width, height))
    if not out.isOpened():
        print(f"❌ Error: Could not create output video writer at {OUTPUT_VIDEO_PATH}.")
        exit()
    
    # ---- 3. INITIALIZE VARIABLES ----
    feature_sequence = deque(maxlen=SEQUENCE_LENGTH)
    current_prediction = "Analyzing..."
    no_person_counter = 0 # Counter for frames where no person is detected

    # ---- 4. MAIN PROCESSING LOOP ----
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        print(f"Processing frame {frame_count}...")
        
        annotated_frame = frame.copy()
        
        # Step 1: Use YOLO to check for a person in the frame
        yolo_results = yolo_model(frame, classes=[0], verbose=False) # class 0 is 'person'
        person_detected = len(yolo_results[0].boxes) > 0

        pose_results = None
        if person_detected:
            no_person_counter = 0 # Reset counter if person is found
            
            # Step 2: Run MediaPipe Pose on the full frame since a person is present
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pose_results = pose.process(rgb_frame)
            
            # Extract features if a pose is detected
            if pose_results.pose_landmarks:
                landmarks = pose_results.pose_landmarks.landmark
                features = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in landmarks])
            else:
                features = np.zeros((33, 4)) # Pose not found, even if person was
        else:
            no_person_counter += 1 # Increment counter if no person is found
            features = np.zeros((33, 4)) # No person, so no features
        
        # Add the features (or zeros) to our sequence
        feature_sequence.append(features)

        # Step 3: Predict if we have a full sequence
        if len(feature_sequence) == SEQUENCE_LENGTH:
            features_np = np.stack(list(feature_sequence), axis=0)
            features_flat = features_np.reshape(SEQUENCE_LENGTH, -1)
            features_tensor = torch.tensor(features_flat, dtype=torch.float32).unsqueeze(0).to(device)
            
            with torch.no_grad():
                outputs = lstm_model(features_tensor)
                _, preds = torch.max(outputs, 1)
                current_prediction = class_names[preds.item()]
        
        # Step 4: Visualize
        # Draw the skeleton only if a pose was actually detected in this frame
        if pose_results and pose_results.pose_landmarks:
            mp_drawing.draw_landmarks(
                annotated_frame,
                pose_results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=1)
            )

        # Determine the text to display
        display_text = ""
        # If we haven't seen a person for a while, override the model's prediction
        if no_person_counter > SEQUENCE_LENGTH:
            display_text = "No Person Detected"
        else:
            display_text = f"Action: {current_prediction}"

        # Draw the prediction text
        cv2.putText(annotated_frame, display_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3, cv2.LINE_AA)
        
        out.write(annotated_frame)
        
    # ---- 5. CLEANUP ----
    print("✅ Processing complete.")
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"✅ Output video saved to: {OUTPUT_VIDEO_PATH}")