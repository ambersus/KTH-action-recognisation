# -*- coding: utf-8 -*-
"""
Train LSTM on MediaPipe Pose Features (.npy files) with train/test split
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# -------------------------------------------------
# Device setup
# -------------------------------------------------
def setup_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
        torch.cuda.empty_cache()
    else:
        device = torch.device("cpu")
        print("❌ CUDA not available. Using CPU.")
    return device

device = setup_device()

# -------------------------------------------------
# Dataset class
# -------------------------------------------------
class PoseFeatureDataset(Dataset):
    def __init__(self, root_dir, split="Training"):
        self.samples = []
        self.class_to_idx = {}

        # Each folder under root_dir is a class (track1, track2, ...)
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
        # x: (B, T, 132)
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]  # last time step
        return self.fc(last_output)

# -------------------------------------------------
# Training function
# -------------------------------------------------
def train_model(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss, total_correct, total_samples = 0, 0, 0

    for features, labels in dataloader:
        features, labels = features.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * features.size(0)
        _, preds = torch.max(outputs, 1)
        total_correct += torch.sum(preds == labels).item()
        total_samples += features.size(0)

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy


# -------------------------------------------------
# Main script
# -------------------------------------------------
if __name__ == "__main__":
    # Now point to the root folder directly (not just track1)
    root_dir = r"D:\Deepak\Dataset\pose_features"

    # Load datasets from ALL tracks
    train_dataset = PoseFeatureDataset(root_dir, split="Training")
    test_dataset = PoseFeatureDataset(root_dir, split="Testing")

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Infer dimensions
    sample, _ = train_dataset[0]
    input_dim = sample.shape[1]  # should be 132
    num_classes = len(train_dataset.class_to_idx)

    model = LSTMClassifier(input_dim=input_dim, hidden_size=256, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Training loop
    for epoch in range(50):
        train_loss, train_acc = train_model(model, train_loader, criterion, optimizer, device)
        print(f"Epoch {epoch+1:02d}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}")
        
    # Assuming train_dataset is already created
    print("Classes used by the model:")
    for cls_name, idx in train_dataset.class_to_idx.items():
        print(f"{idx}: {cls_name}")

        
    # Save model
    save_path = "pose_lstm.pth"
    torch.save(model.state_dict(), save_path)
    print(f"💾 Model saved to {save_path}")
