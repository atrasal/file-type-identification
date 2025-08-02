import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

DATASET_DIR = "fragments_dataset"
MAPPING_FILE = os.path.join(DATASET_DIR, "fragment_mapping.csv")
CHUNK_SIZE = 1024
TEST_SIZE = 0.2
RANDOM_STATE = 42

def load_hex_fragment(path):
    with open(path, 'r') as f:
        hex_str = f.read().strip()
        if len(hex_str) != CHUNK_SIZE * 2:
            return None
        return [int(hex_str[i:i+2], 16) for i in range(0, len(hex_str), 2)]

def load_fragments(noisy=True):
    X, y = [], []
    mapping_df = pd.read_csv(MAPPING_FILE)
    for _, row in mapping_df.iterrows():
        frag_id = row['fragment_id']
        file_type = row['file_type']
        frag_path = os.path.join(DATASET_DIR, f"{frag_id}.hex")

        if os.path.exists(frag_path):
            fragment = load_hex_fragment(frag_path)
            if fragment:
                X.append(fragment)
                y.append(file_type)

    print(f"Loaded {len(X)} fragments from {mapping_df['file_type'].nunique()} classes.")
    return X, y

def train_and_eval(label, X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(f"\n📊 {label} Model Report:")
    print(classification_report(y_test, y_pred))

def main():
    print("🧼 Clean Dataset:")
    X_clean, y_clean = load_fragments()
    train_and_eval("Clean", X_clean, y_clean)

if __name__ == "__main__":
    main()
