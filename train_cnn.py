import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow.keras import layers, models

DATASET_DIR = "fragments_dataset"
CHUNK_SIZE = 1024
MAPPING_FILE = os.path.join(DATASET_DIR, "fragment_mapping.csv")

def load_hex_fragment(path):
    with open(path, 'r') as f:
        hex_str = f.read().strip()
        if len(hex_str) != CHUNK_SIZE * 2:
            return None
        return [int(hex_str[i:i+2], 16) for i in range(0, len(hex_str), 2)]

def load_fragments():
    X, y = [], []
    mapping_df = pd.read_csv(MAPPING_FILE)
    for _, row in mapping_df.iterrows():
        frag_id = row['fragment_id']
        file_type = row['file_type']
        frag_path = os.path.join(DATASET_DIR, f"{frag_id}.hex")
        if os.path.exists(frag_path):
            data = load_hex_fragment(frag_path)
            if data and len(data) == CHUNK_SIZE:
                X.append(data)
                y.append(file_type)
    return np.array(X), np.array(y)

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

def main():
    X, y = load_fragments()
    X = X / 255.0  # Normalize
    X = X[..., np.newaxis]  # Add channel dim for Conv1D

    label_enc = LabelEncoder()
    y_encoded = label_enc.fit_transform(y)
    class_names = label_enc.classes_

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
    )

    print(f"Training on {X_train.shape[0]} samples with {len(class_names)} classes.")

    model = build_model(num_classes=len(class_names))

    model.fit(X_train, y_train, epochs=15, batch_size=64, validation_split=0.1, verbose=1)

    y_pred = model.predict(X_test)
    y_pred_labels = np.argmax(y_pred, axis=1)

    print("\n📊 Classification Report (CNN):")
    print(classification_report(y_test, y_pred_labels, target_names=class_names))

if __name__ == "__main__":
    main()
