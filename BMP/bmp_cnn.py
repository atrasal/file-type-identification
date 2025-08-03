import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Load dataset
data = np.load('forensics/BMP/bmp_dataset_split.npz')
X_train, y_train = data['X_train'], data['y_train']
X_test, y_test = data['X_test'], data['y_test']

# Convert to torch tensors
def to_tensor(X, y):
    X = torch.tensor(X, dtype=torch.float32).unsqueeze(1) / 255.0
    y = torch.tensor(y, dtype=torch.long)
    return X, y

X_train, y_train = to_tensor(X_train, y_train)
X_test, y_test = to_tensor(X_test, y_test)

# Class distribution info
num_headered = int((y_train == 1).sum().item())
num_headerless = int((y_train == 0).sum().item())
print(f"Train Class Distribution -> Headered: {num_headered}, Headerless: {num_headerless}")

# Create datasets
train_ds = TensorDataset(X_train, y_train)
test_ds = TensorDataset(X_test, y_test)

train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)
test_dl = DataLoader(test_ds, batch_size=64)

# Define 1D CNN model
class Simple1DCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 16, 5, padding=2)
        self.pool = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(16, 32, 5, padding=2)
        self.fc1 = nn.Linear(32 * 256, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize
model = Simple1DCNN()
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 10
for epoch in range(epochs):
    model.train()
    for xb, yb in train_dl:
        optimizer.zero_grad()
        outputs = model(xb).squeeze()
        loss = criterion(outputs, yb.float())
        loss.backward()
        optimizer.step()

# Evaluation
def evaluate_metrics(model, dataloader, y_true):
    model.eval()
    y_pred = []
    with torch.no_grad():
        for xb, _ in dataloader:
            outputs = torch.sigmoid(model(xb).squeeze())
            preds = (outputs > 0.5).long()
            y_pred.extend(preds.cpu().numpy())
    y_pred = np.array(y_pred)
    y_true = y_true.numpy()

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    return acc, prec, rec, f1, y_pred, y_true

# Run evaluation
acc, prec, rec, f1, y_pred, y_true = evaluate_metrics(model, test_dl, y_test)

# === Results ===
print("\n=== Test Set Results ===")
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1 Score : {f1:.4f}\n")

print("=== Classification Report ===")
print(classification_report(y_true, y_pred, target_names=['Headerless', 'Headered']))

# Per-class metrics (corrected)
for label, name in zip([0, 1], ['Headerless', 'Headered']):
    class_mask = (y_true == label)
    class_true = y_true[class_mask]
    class_pred = y_pred[class_mask]
    acc_cls = accuracy_score(class_true, class_pred)
    prec_cls = precision_score(class_true, class_pred, average='binary', pos_label=label, zero_division=0)
    rec_cls = recall_score(class_true, class_pred, average='binary', pos_label=label, zero_division=0)
    f1_cls = f1_score(class_true, class_pred, average='binary', pos_label=label, zero_division=0)

    print(f"\n{name} Fragments:")
    print(f"Accuracy : {acc_cls:.4f}")
    print(f"Precision: {prec_cls:.4f}")
    print(f"Recall   : {rec_cls:.4f}")
    print(f"F1 Score : {f1_cls:.4f}")
