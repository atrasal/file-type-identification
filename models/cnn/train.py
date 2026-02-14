"""
CNN training script for file-type fragment classification.

Usage:
  python -m models.cnn.train
  python models/cnn/train.py
"""

import os
import sys
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from utils.data_loader import prepare_dataset, load_fragments, CHUNK_SIZE

# ========== CONFIG ==========
TRAIN_DIR = "datasets/train"
TEST_DIR = "datasets/test"
TRAIN_MAPPING = os.path.join(TRAIN_DIR, "fragment_mapping.csv")
TEST_MAPPING = os.path.join(TEST_DIR, "fragment_mapping.csv")

MODEL_SAVE_PATH = "saved_models/cnn/cnn_model.pth"
RESULTS_PATH = "results/cnn_results.json"
EPOCHS = 15
BATCH_SIZE = 64
LR = 0.001


# ========== CNN Model ==========
class FragmentCNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(FragmentCNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool1d(2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * (input_size // 4), 128)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def main():
    print("📦 Loading training data...")
    X_train, y_train, label_enc, class_names = prepare_dataset(
        TRAIN_MAPPING, TRAIN_DIR
    )
    num_classes = len(class_names)
    input_size = X_train.shape[1]

    print("📦 Loading test data...")
    X_test_raw, y_test_raw = load_fragments(TEST_MAPPING, TEST_DIR)
    X_test = X_test_raw / 255.0
    y_test = label_enc.transform(y_test_raw)

    print(f"📊 Train: {X_train.shape[0]}, Test: {X_test.shape[0]}, Classes: {list(class_names)}")

    # Convert to PyTorch tensors
    X_train_t = torch.FloatTensor(X_train).unsqueeze(1)
    X_test_t = torch.FloatTensor(X_test).unsqueeze(1)
    y_train_t = torch.LongTensor(y_train)
    y_test_t = torch.LongTensor(y_test)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=BATCH_SIZE)

    # Setup model
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🖥️  Using device: {device}")

    model = FragmentCNN(input_size, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # Training
    print(f"\n🚀 Training CNN on {X_train.shape[0]} samples...")
    start_time = time.time()

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"  Epoch {epoch+1}/{EPOCHS}, Loss: {running_loss/len(train_loader):.4f}")

    train_time = time.time() - start_time

    # Evaluation
    model.eval()
    all_preds = []
    inference_start = time.time()
    with torch.no_grad():
        for batch_X, _ in test_loader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
    inference_time = time.time() - inference_start

    y_pred = np.array(all_preds)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))
    print(f"✅ Accuracy: {accuracy:.4f}")
    print(f"⏱️  Training time: {train_time:.2f}s")
    print(f"⏱️  Inference time: {inference_time:.4f}s")

    # Save model
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"💾 Model saved to: {MODEL_SAVE_PATH}")

    # Save results
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    results = {
        "model": "CNN",
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "train_samples": int(X_train.shape[0]),
        "test_samples": int(X_test.shape[0]),
        "train_time_seconds": float(train_time),
        "inference_time_seconds": float(inference_time),
        "classes": list(class_names),
    }
    with open(RESULTS_PATH, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"📄 Results saved to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
