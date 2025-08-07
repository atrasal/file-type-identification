import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# Constants
FRAGMENT_SIZE = 1024  # Size of each fragment in bytes

# Use script's directory as base
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', 'fragments_headerless_footerless', 'train_fragments'))

# Function to load fragments and labels
def load_fragments(directory):
    X = []
    y = []
    print(f"[INFO] Loading fragments from: {directory}")
    if not os.path.exists(directory):
        print(f"[ERROR] Directory does not exist: {directory}")
        return np.array([]), np.array([])
    files = [f for f in os.listdir(directory) if f.endswith('.bin')]
    print(f"[INFO] Found {len(files)} .bin files.")
    if len(files) == 0:
        print(f"[WARNING] No .bin files found in {directory}.")
    for filename in files:
        fpath = os.path.join(directory, filename)
        try:
            with open(fpath, 'rb') as f:
                fragment = f.read(FRAGMENT_SIZE)
            if len(fragment) == FRAGMENT_SIZE:
                X.append(np.frombuffer(fragment, dtype=np.uint8))
                y.append(1)  # All are BMP, so label = 1
            else:
                print(f"[WARNING] Skipping {filename}: fragment size {len(fragment)} != {FRAGMENT_SIZE}")
        except Exception as e:
            print(f"[ERROR] Failed to read {filename}: {e}")
    X = np.array(X)
    y = np.array(y)
    print(f"[INFO] Loaded {X.shape[0]} valid fragments.")
    return X, y

# Function to create CNN model
def create_model():
    model = Sequential()
    model.add(Conv1D(32, kernel_size=7, activation='relu', input_shape=(FRAGMENT_SIZE, 1)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.25))
    model.add(Conv1D(64, kernel_size=5, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))  # Binary classification: BMP vs non-BMP (future)
    return model

# Load and preprocess training data
X_train, y_train = load_fragments(TRAIN_DIR)
if X_train.size == 0:
    print("[ERROR] No training data loaded. Exiting.")
    exit(1)
X_train = X_train.astype('float32') / 255.0  # Normalize
X_train = np.expand_dims(X_train, axis=-1)   # Add channel dimension
print(f"[INFO] Training data shape: {X_train.shape}")

# Create and compile model
print("[INFO] Creating model...")
model = create_model()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Train model
print("[INFO] Starting training...")
history = model.fit(X_train, y_train, epochs=10, batch_size=32)
print("[INFO] Training complete.")

# Save the trained model
model_save_path = os.path.join(BASE_DIR, "bmp_headerless_footerless_fragment_classifier.h5")
model.save(model_save_path)
print(f"[INFO] Model saved as {model_save_path}")
