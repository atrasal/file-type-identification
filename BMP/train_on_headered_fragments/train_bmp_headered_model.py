import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Constants
FRAGMENT_SIZE = 1024  # Size of each fragment in bytes

# Use script's directory as base
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', 'fragments_with_header_footer', 'train_fragments'))

# Function to load fragments and labels
def load_fragments(directory):
    X = []
    y = []
    print(f"[INFO] Loading fragments from: {directory}")
    
    # Load BMP fragments (label = 1)
    bmp_dir = os.path.join(directory, 'bmp_fragments')
    if os.path.exists(bmp_dir):
        print(f"[INFO] Loading BMP fragments from: {bmp_dir}")
        files = [f for f in os.listdir(bmp_dir) if f.endswith('.bin')]
        print(f"[INFO] Found {len(files)} BMP .bin files.")
        for filename in files:
            fpath = os.path.join(bmp_dir, filename)
            try:
                with open(fpath, 'rb') as f:
                    fragment = f.read(FRAGMENT_SIZE)
                if len(fragment) == FRAGMENT_SIZE:
                    X.append(np.frombuffer(fragment, dtype=np.uint8))
                    y.append(1)  # BMP = label 1
                else:
                    print(f"[WARNING] Skipping {filename}: fragment size {len(fragment)} != {FRAGMENT_SIZE}")
            except Exception as e:
                print(f"[ERROR] Failed to read {filename}: {e}")
    else:
        print(f"[WARNING] BMP directory does not exist: {bmp_dir}")
    
    # Load non-BMP fragments (label = 0)
    non_bmp_dir = os.path.join(directory, 'non_bmp_fragments')
    if os.path.exists(non_bmp_dir):
        print(f"[INFO] Loading non-BMP fragments from: {non_bmp_dir}")
        files = [f for f in os.listdir(non_bmp_dir) if f.endswith('.bin')]
        print(f"[INFO] Found {len(files)} non-BMP .bin files.")
        for filename in files:
            fpath = os.path.join(non_bmp_dir, filename)
            try:
                with open(fpath, 'rb') as f:
                    fragment = f.read(FRAGMENT_SIZE)
                if len(fragment) == FRAGMENT_SIZE:
                    X.append(np.frombuffer(fragment, dtype=np.uint8))
                    y.append(0)  # non-BMP = label 0
                else:
                    print(f"[WARNING] Skipping {filename}: fragment size {len(fragment)} != {FRAGMENT_SIZE}")
            except Exception as e:
                print(f"[ERROR] Failed to read {filename}: {e}")
    else:
        print(f"[WARNING] Non-BMP directory does not exist: {non_bmp_dir}")
    
    X = np.array(X)
    y = np.array(y)
    
    # Count samples per class
    bmp_count = np.sum(y == 1)
    non_bmp_count = np.sum(y == 0)
    print(f"[INFO] Loaded {X.shape[0]} total fragments:")
    print(f"[INFO]   - BMP fragments: {bmp_count}")
    print(f"[INFO]   - Non-BMP fragments: {non_bmp_count}")
    
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
    model.add(Dense(1, activation='sigmoid'))  # Binary classification: BMP vs non-BMP
    return model

# Load and preprocess training data
X_train, y_train = load_fragments(TRAIN_DIR)
if X_train.size == 0:
    print("[ERROR] No training data loaded. Exiting.")
    exit(1)

# Split data into training and validation sets
X_train_split, X_val, y_train_split, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)

print(f"[INFO] Training set: {X_train_split.shape[0]} samples")
print(f"[INFO] Validation set: {X_val.shape[0]} samples")

# Preprocess data
X_train_split = X_train_split.astype('float32') / 255.0  # Normalize
X_train_split = np.expand_dims(X_train_split, axis=-1)   # Add channel dimension
X_val = X_val.astype('float32') / 255.0  # Normalize
X_val = np.expand_dims(X_val, axis=-1)   # Add channel dimension

print(f"[INFO] Training data shape: {X_train_split.shape}")
print(f"[INFO] Validation data shape: {X_val.shape}")

# Create and compile model
print("[INFO] Creating model...")
model = create_model()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Train model
print("[INFO] Starting training...")
history = model.fit(
    X_train_split, y_train_split, 
    epochs=10, 
    batch_size=32,
    validation_data=(X_val, y_val),
    verbose=1
)
print("[INFO] Training complete.")

# Save the trained model
model_save_path = os.path.join(BASE_DIR, "bmp_headered_fragment_classifier.h5")
model.save(model_save_path)
print(f"[INFO] Model saved as {model_save_path}")

# Print final validation metrics
val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
print(f"[INFO] Final validation accuracy: {val_accuracy:.4f}")
print(f"[INFO] Final validation loss: {val_loss:.4f}")