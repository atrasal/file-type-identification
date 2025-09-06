import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import tensorflow as tf
from tensorflow.keras import layers, models

# ========== CONFIG ==========
CHUNK_SIZE = 1024
TRAIN_DIR = os.path.join("CNN", "Training_fragments")
TEST_DIR = os.path.join("CNN", "Testing_fragments")
TRAIN_MAPPING_FILE = os.path.join(TRAIN_DIR, "fragment_mapping.csv")
TEST_MAPPING_FILE = os.path.join(TEST_DIR, "fragment_mapping.csv")

# ========== Load Fragment Function ==========
def load_hex_fragment(path):
    with open(path, 'r') as f:
        hex_str = f.read().strip()
        if len(hex_str) != CHUNK_SIZE * 2:
            return None
        return [int(hex_str[i:i+2], 16) for i in range(0, len(hex_str), 2)]

# ========== Load Dataset ==========
def load_fragments(mapping_file, base_dir):
    X, y = [], []
    mapping_df = pd.read_csv(mapping_file)
    print(f"[DEBUG] Loaded {len(mapping_df)} rows from {mapping_file}")
    missing_files = 0
    for _, row in mapping_df.iterrows():
        frag_id = row['fragment_id']
        file_type = row['file_type']
        frag_path = os.path.join(base_dir, f"{frag_id}.hex")
        if os.path.exists(frag_path):
            data = load_hex_fragment(frag_path)
            if data and len(data) == CHUNK_SIZE:
                X.append(data)
                y.append(file_type)
        else:
            missing_files += 1
    print(f"[DEBUG] Found {len(X)} valid fragments, {missing_files} missing .hex files.")
    return np.array(X), np.array(y)

# ========== Build CNN Model ==========
def build_model(num_classes):
    model = models.Sequential([
        layers.Input(shape=(CHUNK_SIZE, 1)),
        layers.Conv1D(64, kernel_size=5, activation='relu'),
        layers.MaxPooling1D(pool_size=2),
        layers.Conv1D(128, kernel_size=3, activation='relu'),
        layers.MaxPooling1D(pool_size=2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# ========== Evaluate on Unseen Data ==========
def test_on_unseen_data(model, label_encoder, test_dir):
    mapping_path = os.path.join(test_dir, "fragment_mapping.csv")
    if not os.path.exists(mapping_path):
        print(f"❌ Mapping file not found at: {mapping_path}")
        return

    mapping_df = pd.read_csv(mapping_path)
    X_test, y_true = [], []

    for _, row in mapping_df.iterrows():
        frag_id = row['fragment_id']
        true_label = row['file_type']
        frag_path = os.path.join(test_dir, f"{frag_id}.hex")
        if os.path.exists(frag_path):
            data = load_hex_fragment(frag_path)
            if data and len(data) == CHUNK_SIZE:
                X_test.append(data)
                y_true.append(true_label)
        else:
            print(f"⚠️ Missing file: {frag_path}")

    if not X_test:
        print("⚠️ No valid fragments found in test set.")
        return

    X_test = np.array(X_test) / 255.0
    X_test = X_test[..., np.newaxis]
    y_true_encoded = label_encoder.transform(y_true)

    y_pred_probs = model.predict(X_test)
    y_pred_encoded = np.argmax(y_pred_probs, axis=1)

    print("\n📊 Classification Report on Unseen Test Data:")
    print(classification_report(y_true_encoded, y_pred_encoded, target_names=label_encoder.classes_))
    print(f"✅ Accuracy on Unseen Test Set: {accuracy_score(y_true_encoded, y_pred_encoded):.4f}")

# ========== Main ==========
def main():
    # Load training data
    X, y = load_fragments(TRAIN_MAPPING_FILE, TRAIN_DIR)
    if len(X) == 0:
        print("[ERROR] No training data loaded. Check if fragments and mapping CSV exist and are valid.")
        return
    X = X / 255.0
    X = X[..., np.newaxis]

    label_enc = LabelEncoder()
    y_encoded = label_enc.fit_transform(y)
    class_names = label_enc.classes_

    X_train, X_val, y_train, y_val = train_test_split(
        X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
    )

    print(f"📦 Training on {X_train.shape[0]} samples with {len(class_names)} classes.")

    # Train model
    model = build_model(num_classes=len(class_names))
    model.fit(X_train, y_train, epochs=15, batch_size=64, validation_split=0.1, verbose=1)

    # Evaluate on internal validation split
    y_pred = model.predict(X_val)
    y_pred_labels = np.argmax(y_pred, axis=1)

    print("\n📊 Classification Report (Internal Validation):")
    print(classification_report(y_val, y_pred_labels, target_names=class_names))

    # Evaluate on unseen dataset
    test_on_unseen_data(model, label_enc, TEST_DIR)

if __name__ == "__main__":
    main()
