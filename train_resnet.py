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
TRAIN_DIR = "fragments_dataset"
TEST_DIR = "test_fragments_dataset"
TRAIN_MAPPING_FILE = os.path.join(TRAIN_DIR, "fragment_mapping.csv")
TEST_MAPPING_FILE = os.path.join(TEST_DIR, "fragment_mapping.csv")

# ========== Load Hex Fragment ==========
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
    for _, row in mapping_df.iterrows():
        frag_id = row['fragment_id']
        file_type = row['file_type']
        frag_path = os.path.join(base_dir, f"{frag_id}.hex")
        if os.path.exists(frag_path):
            data = load_hex_fragment(frag_path)
            if data and len(data) == CHUNK_SIZE:
                X.append(data)
                y.append(file_type)
    return np.array(X), np.array(y)

# ========== 1D ResNet Block ==========
def resnet_block(x, filters, kernel_size=3):
    shortcut = x
    x = layers.Conv1D(filters, kernel_size, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(filters, kernel_size, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([shortcut, x])
    x = layers.Activation('relu')(x)
    return x

# ========== Build ResNet Model ==========
def build_resnet_model(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv1D(64, kernel_size=7, padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)

    # Residual Blocks
    for _ in range(3):
        x = resnet_block(x, 64)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs=inputs, outputs=outputs)
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
    X = X / 255.0
    X = X[..., np.newaxis]

    label_enc = LabelEncoder()
    y_encoded = label_enc.fit_transform(y)
    class_names = label_enc.classes_

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
    )

    print(f"📦 Training on {X_train.shape[0]} samples with {len(class_names)} classes.")

    # Train model
    model = build_resnet_model(input_shape=(CHUNK_SIZE, 1), num_classes=len(class_names))
    model.fit(X_train, y_train, epochs=20, batch_size=64, validation_split=0.1, verbose=1)

    # Internal validation
    y_pred = model.predict(X_test)
    y_pred_labels = np.argmax(y_pred, axis=1)

    print("\n📊 Classification Report (Internal Validation):")
    print(classification_report(y_test, y_pred_labels, target_names=class_names))

    # Evaluate on unseen dataset
    test_on_unseen_data(model, label_enc, TEST_DIR)

    # 🔁 Prediction loop
    while True:
        custom_hex_path = input("\n🔍 Enter path to a .hex file to classify (or type 'exit' to quit): ").strip()
        if custom_hex_path.lower() == 'exit':
            print("👋 Exiting prediction loop.")
            break

        if os.path.exists(custom_hex_path):
            custom_data = load_hex_fragment(custom_hex_path)
            if custom_data and len(custom_data) == CHUNK_SIZE:
                custom_arr = np.array(custom_data) / 255.0
                custom_arr = custom_arr.reshape(1, CHUNK_SIZE, 1)
                pred = model.predict(custom_arr)
                pred_label = np.argmax(pred, axis=1)[0]
                print(f"✅ Predicted file type: {class_names[pred_label]}")
            else:
                print(f"⚠️ Invalid hex data in '{custom_hex_path}'. Must be exactly {CHUNK_SIZE*2} hex characters.")
        else:
            print(f"❌ File '{custom_hex_path}' not found.")

if __name__ == "__main__":
    main()
