"""
Random Forest training script for file-type fragment classification.

Usage:
  python -m models.random_forest.train
  python models/random_forest/train.py
"""

import os
import sys
import time
import json
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from utils.data_loader import prepare_dataset, load_fragments

# ========== CONFIG ==========
TRAIN_DIR = "datasets/train"
VAL_DIR = "datasets/val"
TEST_DIR = "datasets/test"
TRAIN_MAPPING = os.path.join(TRAIN_DIR, "fragment_mapping.csv")
VAL_MAPPING = os.path.join(VAL_DIR, "fragment_mapping.csv")
TEST_MAPPING = os.path.join(TEST_DIR, "fragment_mapping.csv")

MODEL_SAVE_PATH = "saved_models/random_forest/rf_model.joblib"
RESULTS_PATH = "results/random_forest_results.json"


def main():
    print("📦 Loading training data...")
    X_train, y_train, label_enc, class_names = prepare_dataset(
        TRAIN_MAPPING, TRAIN_DIR
    )

    print("📦 Loading validation data...")
    X_val_raw, y_val_raw = load_fragments(VAL_MAPPING, VAL_DIR)
    X_val = X_val_raw / 255.0
    y_val = label_enc.transform(y_val_raw)

    print("📦 Loading test data...")
    X_test_raw, y_test_raw = load_fragments(TEST_MAPPING, TEST_DIR)
    X_test = X_test_raw / 255.0
    y_test = label_enc.transform(y_test_raw)

    print(f"📊 Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}, Classes: {list(class_names)}")

    # Train model
    print(f"\n🚀 Training Random Forest on {X_train.shape[0]} samples...")
    start_time = time.time()
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    train_time = time.time() - start_time

    # Evaluate on validation set
    y_val_pred = model.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    val_precision = precision_score(y_val, y_val_pred, average='macro', zero_division=0)
    val_recall = recall_score(y_val, y_val_pred, average='macro', zero_division=0)
    val_f1 = f1_score(y_val, y_val_pred, average='macro', zero_division=0)

    # Evaluate on test set
    inference_start = time.time()
    y_pred = model.predict(X_test)
    inference_time = time.time() - inference_start

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

    print("\n📊 Test Classification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))
    print(f"✅ Val  Accuracy: {val_accuracy:.4f}, F1: {val_f1:.4f}")
    print(f"✅ Test Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
    print(f"⏱️  Training time: {train_time:.2f}s")
    print(f"⏱️  Inference time: {inference_time:.4f}s")

    # Save model
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    joblib.dump({'model': model, 'label_encoder': label_enc}, MODEL_SAVE_PATH)
    print(f"💾 Model saved to: {MODEL_SAVE_PATH}")

    # Save results
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    results = {
        "model": "Random Forest",
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "val_accuracy": float(val_accuracy),
        "val_precision": float(val_precision),
        "val_recall": float(val_recall),
        "val_f1_score": float(val_f1),
        "train_samples": int(X_train.shape[0]),
        "val_samples": int(X_val.shape[0]),
        "test_samples": int(X_test.shape[0]),
        "train_time_seconds": float(train_time),
        "inference_time_seconds": float(inference_time),
        "classes": list(class_names),
    }
    with open(RESULTS_PATH, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"📄 Results saved to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
