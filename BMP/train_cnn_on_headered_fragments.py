import os
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score

# ---- CONFIG ----
# Use script's directory as base
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TRAIN_DIR = os.path.join(BASE_DIR, 'fragments_with_header_footer', 'train_fragments')
TEST_DIR = os.path.join(BASE_DIR, 'fragments_with_header_footer', 'test_fragments')

FRAGMENT_SIZE = 1024  # bytes
LABEL_BMP = 1
LABEL_NONBMP = 0  # If you add non-BMPs

# ---- DATA LOADING ----
def load_fragments_from_folder(folder, label):
    X = []
    y = []
    for fname in os.listdir(folder):
        if not fname.endswith('.bin'):
            continue
        fpath = os.path.join(folder, fname)
        with open(fpath, 'rb') as f:
            data = f.read()
        if len(data) != FRAGMENT_SIZE:
            continue
        arr = np.frombuffer(data, dtype=np.uint8).astype(np.float32) / 255.0
        X.append(arr)
        y.append(label)
    return np.array(X), np.array(y)

# Load BMP fragments
X_train, y_train = load_fragments_from_folder(TRAIN_DIR, LABEL_BMP)
X_test, y_test = load_fragments_from_folder(TEST_DIR, LABEL_BMP)

# If you have non-BMP fragments, load and append them here:
# X_train_nonbmp, y_train_nonbmp = load_fragments_from_folder('path/to/nonbmp/train', LABEL_NONBMP)
# X_test_nonbmp, y_test_nonbmp = load_fragments_from_folder('path/to/nonbmp/test', LABEL_NONBMP)
# X_train = np.concatenate([X_train, X_train_nonbmp])
# y_train = np.concatenate([y_train, y_train_nonbmp])
# X_test = np.concatenate([X_test, X_test_nonbmp])
# y_test = np.concatenate([y_test, y_test_nonbmp])

# Reshape for CNN input: (samples, length, channels)
X_train = X_train[..., np.newaxis]
X_test = X_test[..., np.newaxis]

print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")

# ---- MODEL ----
model = keras.Sequential([
    layers.Conv1D(32, 7, activation='relu', input_shape=(FRAGMENT_SIZE, 1)),
    layers.MaxPooling1D(2),
    layers.Conv1D(64, 5, activation='relu'),
    layers.MaxPooling1D(2),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# ---- TRAIN ----
history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=32,
    validation_data=(X_test, y_test)
)

# ---- EVALUATE ----
y_pred_prob = model.predict(X_test).flatten()
y_pred = (y_pred_prob > 0.5).astype(int)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, zero_division=0))
print("Recall:", recall_score(y_test, y_pred, zero_division=0))
print("F1-score:", f1_score(y_test, y_pred, zero_division=0))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
try:
    print("ROC-AUC:", roc_auc_score(y_test, y_pred_prob))
except Exception as e:
    print("ROC-AUC could not be computed:", e)
