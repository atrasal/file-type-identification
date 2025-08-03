import os
import numpy as np
from sklearn.model_selection import train_test_split

fragment_dir = 'forensics/BMP/bmp_fragments/'
headerless_dir = 'forensics/BMP/bmp_headerless/'

def load_fragments(folder, label):
    X, y = [], []
    for fname in os.listdir(folder):
        if fname.endswith('.bin'):
            with open(os.path.join(folder, fname), 'rb') as f:
                data = f.read()
                if len(data) == 1024:  # Only use full fragments
                    arr = np.frombuffer(data, dtype=np.uint8)
                    X.append(arr)
                    y.append(label)
    return np.stack(X), np.array(y)

# Load headered (label 1) and headerless (label 0)
X_header, y_header = load_fragments(fragment_dir, 1)
X_headerless, y_headerless = load_fragments(headerless_dir, 0)

# Combine both
X_all = np.concatenate([X_header, X_headerless], axis=0)
y_all = np.concatenate([y_header, y_headerless], axis=0)

# Train-test split (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
)

# Save dataset
np.savez(
    'forensics/BMP/bmp_dataset_split.npz',
    X_train=X_train, y_train=y_train,
    X_test=X_test, y_test=y_test
)

print("✅ Dataset prepared and saved with both headered (1) and headerless (0) fragments.")
