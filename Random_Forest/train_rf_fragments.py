import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.decomposition import PCA

# ========== CONFIG ==========
CHUNK_SIZE = 1024
TRAIN_DIR = os.path.join("Random_Forest", "Training_fragments")
TEST_DIR = os.path.join("Random_Forest", "Testing_fragments")
TRAIN_MAPPING_FILE = os.path.join(TRAIN_DIR, "fragment_mapping.csv")
TEST_MAPPING_FILE = os.path.join(TEST_DIR, "fragment_mapping.csv")

# ========== Load Fragment Function ==========
def load_hex_fragment(path):
    """Load a hex fragment file and convert it to a list of integers (bytes)."""
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
    missing_files = []

    for _, row in mapping_df.iterrows():
        frag_id = row['fragment_id']
        file_type = row['file_type']
        frag_path = os.path.join(base_dir, f"{frag_id}.hex")

        if os.path.exists(frag_path):
            data = load_hex_fragment(frag_path)
            if data is not None and len(data) == CHUNK_SIZE:
                X.append(data)
                y.append(file_type)
        else:
            missing_files.append(frag_path)

    print(f"[DEBUG] Found {len(X)} valid fragments, {len(missing_files)} missing files.")
    if missing_files:
        print("[WARN] Example missing files:", missing_files[:10], "...")

    return np.array(X), np.array(y)

# ========== Evaluate on Unseen Data ==========
def test_on_unseen_data(model, label_encoder, test_dir, pca=None):
    mapping_path = os.path.join(test_dir, "fragment_mapping.csv")
    if not os.path.exists(mapping_path):
        print(f"❌ Mapping file not found at: {mapping_path}")
        return

    mapping_df = pd.read_csv(mapping_path)
    X_test, y_true = [], []
    missing_files = []

    for _, row in mapping_df.iterrows():
        frag_id = row['fragment_id']
        true_label = row['file_type']
        frag_path = os.path.join(test_dir, f"{frag_id}.hex")

        if os.path.exists(frag_path):
            data = load_hex_fragment(frag_path)
            if data is not None and len(data) == CHUNK_SIZE:
                X_test.append(data)
                y_true.append(true_label)
        else:
            missing_files.append(frag_path)

    print(f"[DEBUG] Test set: {len(X_test)} valid fragments, {len(missing_files)} missing files.")

    if not X_test:
        print("⚠️ No valid fragments found in test set.")
        return

    X_test = np.array(X_test)
    if pca:
        X_test = pca.transform(X_test)

    y_true_encoded = label_encoder.transform(y_true)
    y_pred_encoded = model.predict(X_test)

    print("\n📊 Classification Report on Unseen Test Data:")
    print(classification_report(y_true_encoded, y_pred_encoded, target_names=label_encoder.classes_))
    print(f"✅ Accuracy on Unseen Test Set: {accuracy_score(y_true_encoded, y_pred_encoded):.4f}")

# ========== Main ==========
def main():
    X, y = load_fragments(TRAIN_MAPPING_FILE, TRAIN_DIR)
    if len(X) == 0:
        print("[ERROR] No training data loaded. Check fragments and CSV.")
        return

    label_enc = LabelEncoder()
    y_encoded = label_enc.fit_transform(y)
    class_names = label_enc.classes_

    X_train, X_val, y_train, y_val = train_test_split(
        X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
    )

    print(f"📦 Training on {X_train.shape[0]} samples with {len(class_names)} classes.")

    # Dimensionality reduction
    pca = PCA(n_components=256, random_state=42)
    X_train_pca = pca.fit_transform(X_train)
    X_val_pca = pca.transform(X_val)

    # Random Forest with more trees + class balancing
    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_pca, y_train)

    y_pred = model.predict(X_val_pca)

    print("\n📊 Classification Report (Internal Validation):")
    print(classification_report(y_val, y_pred, target_names=class_names))
    print(f"✅ Accuracy on Internal Validation Set: {accuracy_score(y_val, y_pred):.4f}")

    # Test on unseen data
    test_on_unseen_data(model, label_enc, TEST_DIR, pca)

if __name__ == "__main__":
    main()
