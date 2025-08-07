import os
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings

# Suppress warnings about undefined metrics for single-class
warnings.filterwarnings("ignore")

# Constants
FRAGMENT_SIZE = 1024
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HEADERED_MODEL_PATH = os.path.join(BASE_DIR, 'train_on_headered_fragments', 'bmp_headered_fragment_classifier.h5')
HEADERLESS_MODEL_PATH = os.path.join(BASE_DIR, 'train_on_headerless_footerless_fragments', 'bmp_headerless_footerless_fragment_classifier.h5')
HEADERED_TEST_DIR = os.path.join(BASE_DIR, 'fragments_with_header_footer', 'test_fragments')
HEADERLESS_TEST_DIR = os.path.join(BASE_DIR, 'fragments_headerless_footerless', 'test_fragments')

# Load test fragments
def load_fragments(directory):
    X, y = [], []
    print(f"[INFO] Loading fragments from: {directory}")
    if not os.path.exists(directory):
        print(f"[ERROR] Directory does not exist: {directory}")
        return np.array([]), np.array([])
    files = [f for f in os.listdir(directory) if f.endswith('.bin')]
    print(f"[INFO] Found {len(files)} .bin files.")
    for filename in files:
        fpath = os.path.join(directory, filename)
        try:
            with open(fpath, 'rb') as f:
                fragment = f.read(FRAGMENT_SIZE)
            if len(fragment) == FRAGMENT_SIZE:
                X.append(np.frombuffer(fragment, dtype=np.uint8))
                y.append(1)  # All BMP = label 1
        except Exception as e:
            print(f"[ERROR] Failed to read {filename}: {e}")
    return np.array(X), np.array(y)

# Evaluate a model
def evaluate_model(model, X, y, model_name, test_label):
    X = X.astype('float32') / 255.0
    X = np.expand_dims(X, axis=-1)
    y_pred_prob = model.predict(X).flatten()
    y_pred = (y_pred_prob > 0.5).astype(int)

    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred, zero_division=0)
    rec = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)

    print(f"\nClassification Report on {test_label}:\n")
    print(f"{'':15} {'precision':>10} {'recall':>10} {'f1-score':>10} {'support':>10}")
    print(f"{'bmp':15} {prec:10.2f} {rec:10.2f} {f1:10.2f} {len(y):10}")
    print(f"{'accuracy':15} {'':10} {'':10} {acc:10.2f} {len(y):10}")
    print(f"{'macro avg':15} {prec:10.2f} {rec:10.2f} {f1:10.2f} {len(y):10}")
    print(f"{'weighted avg':15} {prec:10.2f} {rec:10.2f} {f1:10.2f} {len(y):10}")

    return {
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1-score': f1
    }

# Main
if __name__ == '__main__':
    print("[INFO] Loading models...")
    headered_model = load_model(HEADERED_MODEL_PATH)
    headerless_model = load_model(HEADERLESS_MODEL_PATH)

    print("[INFO] Loading test data...")
    X_test_headered, y_test_headered = load_fragments(HEADERED_TEST_DIR)
    X_test_headerless, y_test_headerless = load_fragments(HEADERLESS_TEST_DIR)

    if X_test_headered.size == 0 or y_test_headered.size == 0:
        print("[ERROR] No headered test data found. Exiting.")
        exit(1)
    if X_test_headerless.size == 0 or y_test_headerless.size == 0:
        print("[ERROR] No headerless test data found. Exiting.")
        exit(1)

    print("[INFO] Evaluating headered model...")
    results_headered = evaluate_model(headered_model, X_test_headered, y_test_headered, "Headered Model", "headered_test")

    print("[INFO] Evaluating headerless model...")
    results_headerless = evaluate_model(headerless_model, X_test_headerless, y_test_headerless, "Headerless Model", "headerless_footerless_test")

    # Side-by-side comparison
    print("\n=== Side-by-side Comparison ===")
    print(f"{'Metric':15} {'Headered':>10}   {'Headerless':>10}")
    print(f"{'-'*38}")
    for metric in ['Accuracy', 'Precision', 'Recall', 'F1-score']:
        h = results_headered[metric]
        l = results_headerless[metric]
        h_str = f"{h:.4f}" if isinstance(h, float) and h == h else "N/A"
        l_str = f"{l:.4f}" if isinstance(l, float) and l == l else "N/A"
        print(f"{metric:<15} {h_str:>10}   {l_str:>10}")
