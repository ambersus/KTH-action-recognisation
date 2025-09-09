# KTH-action-recognisation

This project provides a complete pipeline for human action recognition from raw video to classification using pose features, combining **YOLO person detection**, **MediaPipe Pose landmark extraction**, and **LSTM sequence modeling** for end-to-end action classification.

## Video Processing Pipeline

File: ```crop_kth.py```

- **Frame Extraction**: Each video is split into individual frames for downstream person detection.
- **YOLO Person Detection**: YOLO (default: "yolov8n.pt") detects persons in each frame; the largest detected bounding box is selected.
- **Person Cropping**: Crops a fixed-size (default **224×224**) region centered on the person, maintaining the folder structure for compatibility with subsequent pose analysis.

### Configuration

Before running the script, configure the following variables in `video_to_crops.py`:

- `videos_root`: Path to the root folder containing video subfolders.
- `output_frames_root`: Path to save the extracted frames.
- `output_crops_root`: Path to save the cropped person images.
- `CROP_SIZE`: Size of the square crop (default: 224).
- `model`: Pre-trained YOLO model (default: `"yolov8n.pt"`).


## Pose Estimation & Visualization

File: ```mp_feature_viz.py```

- **MediaPipe Pose Detection**: Applies pose estimation on each cropped person image to extract key body landmarks.
- **Skeleton Visualization**: Draws pose skeletons on cropped images and saves the visualizations, keeping folders consistent for easy tracking and comparison.

### Configuration

Before running the script, configure the following variables in the script:

- `crops_root`: Path to the folder containing cropped person images.
- `output_vis_root`: Path to save pose visualizations.



## Feature Extraction

File: ```npy_files_30.py```

- **Pose Feature Extraction**: Extracts up to 30 frames per image sequence, obtaining **33 landmarks** per frame (x, y, z, visibility), using MediaPipe Pose.
- **Output Format**: Features saved as NumPy arrays of shape (30, 33, 4) for each sequence; missing persons are saved as zeros for robustness.

### Configuration

- `root_dir`: Path to the input dataset containing folders of image sequences (cropped person images).
- `output_feat_root`: Path where the extracted pose features will be saved.


## Action Recognition Pipeline

File: ```train.py```

- **Dataset Preparation**: Loads pose feature sequences from disk, reshaping each to (30, 132) for LSTM input (flattening 33 keypoints × 4 features per frame).
  - Loads pose feature sequences from disk.
  - Reshapes each sample to shape (30, 132) for LSTM input.
- **Model Architecture**: LSTM processes temporal information, followed by a fully connected layer for class logits.
  - Takes input shape (B, 30, 132).
    
    -B is the number of samples in a batch (e.g., 32).
    
    -30 is the number of time steps (frames per sequence).
    
    -132 is the feature dimension per time step (33 keypoints × 4 features: x, y, z, visibility).
    
    -Outputs predicted class logits using a fully connected layer after LSTM layers.
    
- **Training**: Uses cross-entropy loss and Adam optimizer, periodically printing loss/accuracy and saving the trained model as **pose_lstm.pth**.
  - Uses cross-entropy loss and Adam optimizer.
  -Prints training loss and accuracy per epoch.
  -Saves the trained model as pose_lstm.pth.
- **Class Mapping**: Prints class-to-index mapping for easier interpretation of results.

## Model Evaluation

File: ```evaluation.py```

- **Inference**: Evaluates the saved LSTM model using test pose feature sequences; expects input reshaped to (30, 132) per sample.
- **Outputs**: Prints confusion matrix and classification report, displays a heatmap, and provides overall accuracy.
- **Hardware**: Supports both GPU and CPU for inference.
