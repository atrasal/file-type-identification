import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ====== CONFIG ======
DATASET_DIR = "fragments_dataset"  # Where your .npy fragments are stored
CHUNK_SIZE = 1024                  # Must match your fragment script
TEST_SIZE = 0.2                    # 80% train, 20% test
RANDOM_STATE = 42

# ====== LOAD DATA ======
def load_fragments(dataset_dir):
    X = []
    y = []
    labels = []

    for label in os.listdir(dataset_dir):
        label_path = os.path.join(dataset_dir, label)
        if not os.path.isdir(label_path):
            continue
        labels.append(label)
        for file in os.listdir(label_path):
            if file.endswith(".npy"):
                path = os.path.join(label_path, file)
                data = np.load(path)
                if data.shape[0] == CHUNK_SIZE:
                    X.append(data)
                    y.append(label)

    print(f"Loaded {len(X)} fragments from {len(labels)} classes.")
    return np.array(X), np.array(y), labels

# ====== MAIN TRAINING ======
def main():
    X, y, labels = load_fragments(DATASET_DIR)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    print(f"Training on {len(X_train)} samples, testing on {len(X_test)}.")

    clf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=labels))

if __name__ == "__main__":
    main()
