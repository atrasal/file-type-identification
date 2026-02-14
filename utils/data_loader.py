"""
Shared data loading utilities for fragment-based file type classification.
All model training scripts import from this module.

Supports both:
  - Raw binary .bin fragments (4096 bytes)
  - Hex-encoded .hex text fragments (1024 bytes)
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


# ========== Constants ==========
CHUNK_SIZE = 4096  # Default for binary .bin fragments


# ========== Load a single fragment ==========
def load_fragment(path, chunk_size=CHUNK_SIZE):
    """
    Read a fragment file and return a list of byte values (0-255).
    Auto-detects hex-encoded (.hex) vs raw binary (.bin) format.
    """
    ext = os.path.splitext(path)[1].lower()

    if ext == '.hex':
        # Hex-encoded text
        with open(path, 'r') as f:
            hex_str = f.read().strip()
            if len(hex_str) != chunk_size * 2:
                return None
            return [int(hex_str[i:i+2], 16) for i in range(0, len(hex_str), 2)]
    else:
        # Raw binary (.bin or any other)
        with open(path, 'rb') as f:
            data = f.read()
            if len(data) != chunk_size:
                return None
            return list(data)


# Backward compatibility alias
load_hex_fragment = load_fragment


# ========== Load fragments from a mapping CSV ==========
def load_fragments(mapping_file, base_dir, chunk_size=CHUNK_SIZE, max_samples=None):
    """
    Load fragments referenced by a mapping CSV.

    Supports CSVs with columns:
      - fragment_name, label (new format from split_dataset.py)
      - fragment_id, file_type (old format)

    Args:
        mapping_file: Path to fragment_mapping.csv
        base_dir: Directory containing fragment files
        chunk_size: Expected size of each fragment in bytes
        max_samples: Optional limit on number of samples to load

    Returns:
        X: numpy array of shape (n_samples, chunk_size)
        y: numpy array of string labels
    """
    X, y = [], []
    mapping_df = pd.read_csv(mapping_file)

    if max_samples:
        mapping_df = mapping_df.sample(
            n=min(max_samples, len(mapping_df)), random_state=42
        )

    # Detect CSV format
    if 'fragment_name' in mapping_df.columns:
        name_col, label_col = 'fragment_name', 'label'
    else:
        name_col, label_col = 'fragment_id', 'file_type'

    print(f"📂 Loading {len(mapping_df)} fragments from {base_dir}...")
    for _, row in mapping_df.iterrows():
        frag_name = str(row[name_col])
        label = row[label_col]

        # Try the file directly, or with .hex extension for old format
        frag_path = os.path.join(base_dir, frag_name)
        if not os.path.exists(frag_path):
            frag_path = os.path.join(base_dir, f"{frag_name}.hex")
        if not os.path.exists(frag_path):
            continue

        data = load_fragment(frag_path, chunk_size)
        if data and len(data) == chunk_size:
            X.append(data)
            y.append(label)

    return np.array(X), np.array(y)


# ========== Encode labels ==========
def encode_labels(y):
    """
    Encode string labels to integers.

    Returns:
        y_encoded: numpy array of integer labels
        label_encoder: fitted LabelEncoder instance
        class_names: array of class name strings
    """
    label_enc = LabelEncoder()
    y_encoded = label_enc.fit_transform(y)
    class_names = label_enc.classes_
    return y_encoded, label_enc, class_names


# ========== Convenience: load + normalize + encode ==========
def prepare_dataset(mapping_file, base_dir, chunk_size=CHUNK_SIZE, max_samples=None):
    """
    Load fragments, normalize to [0,1], and encode labels.

    Returns:
        X: normalized numpy array (n_samples, chunk_size)
        y_encoded: integer labels
        label_encoder: fitted LabelEncoder
        class_names: array of class name strings
    """
    X, y = load_fragments(mapping_file, base_dir, chunk_size, max_samples)
    X = X / 255.0  # Normalize byte values to [0, 1]
    y_encoded, label_enc, class_names = encode_labels(y)
    print(f"📊 Dataset: {X.shape[0]} samples, {len(class_names)} classes: {list(class_names)}")
    return X, y_encoded, label_enc, class_names
