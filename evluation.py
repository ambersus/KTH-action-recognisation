# -*- coding: utf-8 -*-
"""
Evaluate LSTM on MediaPipe Pose Features (.npy files) with full metrics
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# -------------------------------------------------
# Device setup
# -------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
    torch.cuda.empty_cache()
else:
    print("❌ CUDA not available. Using CPU.")

# -------------------------------------------------
# Dataset class
# -------------------------------------------------
class PoseFeatureDataset(Dataset):
    def __init__(self, root_dir, split="Testing"):
        self.samples = []
        self.class_to_idx = {}

        class_dirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        class_dirs = sorted(class_dirs)  # ensure consistent ordering
        self.class_to_idx = {cls: i for i, cls in enumerate(class_dirs)}

        for cls in class_dirs:
            split_dir = os.path.join(root_dir, cls, split)
            if not os.path.exists(split_dir):
                continue
            for sample in os.listdir(split_dir):
                sample_path = os.path.join(split_dir, sample)
                seq_path = os.path.join(sample_path, "sequence.npy")
                if os.path.exists(seq_path):
                    self.samples.append((seq_path, self.class_to_idx[cls]))

        print(f"✅ Loaded {len(self.samples)} {split} sequences across {len(self.class_to_idx)} classes.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, label = self.samples[idx]
        features = np.load(file_path)              # (30, 33, 4)
        features = features.reshape(features.shape[0], -1)  # (30, 132)
        features = torch.tensor(features, dtype=torch.float32)
        return features, label

# -------------------------------------------------
# LSTM Classifier
# -------------------------------------------------
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim=132, hidden_size=256, num_classes=6, num_layers=4):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        return self.fc(last_output)

# -------------------------------------------------
# Evaluation function
# -------------------------------------------------
def evaluate_model(model, dataloader, device, class_names):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for features, labels in dataloader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix:\n", cm)

    # Classification report
    report = classification_report(all_labels, all_preds, target_names=class_names)
    print("\nClassification Report:\n", report)

    # Overall accuracy
    accuracy = np.sum(all_preds == all_labels) / len(all_labels)
    print(f"✅ Overall Test Accuracy: {accuracy:.4f} ({np.sum(all_preds == all_labels)}/{len(all_labels)})")

    # Optional: plot confusion matrix heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()

# -------------------------------------------------
# Main evaluation script
# -------------------------------------------------
if __name__ == "__main__":
    TEST_DIR = r"D:\Deepak\Dataset\pose_features"  # root folder of classes

    # Load test dataset
    test_dataset = PoseFeatureDataset(TEST_DIR, split="Testing")
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Get model parameters
    sample, _ = test_dataset[0]
    input_dim = sample.shape[1]
    num_classes = len(test_dataset.class_to_idx)
    class_names = list(test_dataset.class_to_idx.keys())

    # Create and load model
    model = LSTMClassifier(input_dim=input_dim, hidden_size=256, num_classes=num_classes).to(device)
    model.load_state_dict(torch.load("pose_lstm.pth", map_location=device))

    # Evaluate
    evaluate_model(model, test_loader, device, class_names)
