import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier

# ========== CONFIG ==========
CHUNK_SIZE = 1024
TRAIN_DIR = "fragments_dataset"
TEST_DIR = "test_fragments_dataset"
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
    y_true_encoded = label_encoder.transform(y_true)

    y_pred = model.predict(X_test)

    print("\n📊 Classification Report on Unseen Test Data:")
    print(classification_report(y_true_encoded, y_pred, target_names=label_encoder.classes_))
    print(f"✅ Accuracy on Unseen Test Set: {accuracy_score(y_true_encoded, y_pred):.4f}")

# ========== Main ==========
def main():
    # Load training data
    X, y = load_fragments(TRAIN_MAPPING_FILE, TRAIN_DIR)
    X = X / 255.0  # Normalize
    label_enc = LabelEncoder()
    y_encoded = label_enc.fit_transform(y)
    class_names = label_enc.classes_

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
    )

    print(f"📦 Training on {X_train.shape[0]} samples with {len(class_names)} classes.")

    # ========== XGBoost Model ==========
    model = XGBClassifier(
        objective='multi:softmax',
        num_class=len(class_names),
        max_depth=8,
        learning_rate=0.1,
        n_estimators=150,
        use_label_encoder=False,
        eval_metric='mlogloss'
    )

    model.fit(X_train, y_train)

    # Internal validation
    y_pred = model.predict(X_test)

    print("\n📊 Classification Report (Internal Validation):")
    print(classification_report(y_test, y_pred, target_names=class_names))

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
                custom_arr = custom_arr.reshape(1, -1)
                pred_label = model.predict(custom_arr)[0]
                print(f"✅ Predicted file type: {class_names[pred_label]}")
            else:
                print(f"⚠️ Invalid hex data in '{custom_hex_path}'. Must be exactly {CHUNK_SIZE*2} hex characters.")
        else:
            print(f"❌ File '{custom_hex_path}' not found.")

if __name__ == "__main__":
    main()
