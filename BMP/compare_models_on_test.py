import os
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import warnings

# Suppress warnings about undefined metrics for single-class
warnings.filterwarnings("ignore")

# Constants
FRAGMENT_SIZE = 1024
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HEADERED_MODEL_PATH = os.path.join(BASE_DIR, 'train_on_headered_fragments', 'bmp_headered_fragment_classifier.h5')
HEADERLESS_MODEL_PATH = os.path.join(BASE_DIR, 'train_on_headerless_footerless_fragments', 'bmp_headerless_footerless_fragment_classifier.h5')
HEADERED_TEST_DIR = os.path.join(BASE_DIR, 'fragments_with_header_footer', 'test_fragments', 'merged_fragments')
HEADERLESS_TEST_DIR = os.path.join(BASE_DIR, 'fragments_headerless_footerless', 'test_fragments', 'merged_fragments')

# Load test fragments
def load_fragments(directory):
    X, y = [], []
    print(f"[INFO] Loading fragments from: {directory}")
    if not os.path.exists(directory):
        print(f"[ERROR] Directory does not exist: {directory}")
        return np.array([]), np.array([])
    
    # Load BMP fragments (label = 1)
    bmp_dir = os.path.join(directory, 'bmp_fragments')
    if os.path.exists(bmp_dir):
        print(f"[INFO] Loading BMP fragments from: {bmp_dir}")
        files = [f for f in os.listdir(bmp_dir) if f.endswith('.bin')]
        print(f"[INFO] Found {len(files)} BMP .bin files.")
        for filename in files:
            fpath = os.path.join(bmp_dir, filename)
            try:
                with open(fpath, 'rb') as f:
                    fragment = f.read(FRAGMENT_SIZE)
                if len(fragment) == FRAGMENT_SIZE:
                    X.append(np.frombuffer(fragment, dtype=np.uint8))
                    y.append(1)  # BMP = label 1
            except Exception as e:
                print(f"[ERROR] Failed to read {filename}: {e}")
    else:
        print(f"[WARNING] BMP directory does not exist: {bmp_dir}")
    
    # Load non-BMP fragments (label = 0)
    non_bmp_dir = os.path.join(directory, 'non_bmp_fragments')
    if os.path.exists(non_bmp_dir):
        print(f"[INFO] Loading non-BMP fragments from: {non_bmp_dir}")
        files = [f for f in os.listdir(non_bmp_dir) if f.endswith('.bin')]
        print(f"[INFO] Found {len(files)} non-BMP .bin files.")
        for filename in files:
            fpath = os.path.join(non_bmp_dir, filename)
            try:
                with open(fpath, 'rb') as f:
                    fragment = f.read(FRAGMENT_SIZE)
                if len(fragment) == FRAGMENT_SIZE:
                    X.append(np.frombuffer(fragment, dtype=np.uint8))
                    y.append(0)  # non-BMP = label 0
            except Exception as e:
                print(f"[ERROR] Failed to read {filename}: {e}")
    else:
        print(f"[WARNING] Non-BMP directory does not exist: {non_bmp_dir}")
    
    X = np.array(X)
    y = np.array(y)
    
    # Count samples per class
    bmp_count = np.sum(y == 1)
    non_bmp_count = np.sum(y == 0)
    print(f"[INFO] Loaded {X.shape[0]} total fragments:")
    print(f"[INFO]   - BMP fragments: {bmp_count}")
    print(f"[INFO]   - Non-BMP fragments: {non_bmp_count}")
    
    return X, y

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

    print(f"\nClassification Report on {test_label}:")
    print(f"Model: {model_name}")
    print(f"{'':15} {'precision':>10} {'recall':>10} {'f1-score':>10} {'support':>10}")
    print(f"{'non-bmp':15} {precision_score(y, y_pred, pos_label=0, zero_division=0):10.2f} {recall_score(y, y_pred, pos_label=0, zero_division=0):10.2f} {f1_score(y, y_pred, pos_label=0, zero_division=0):10.2f} {np.sum(y == 0):10}")
    print(f"{'bmp':15} {prec:10.2f} {rec:10.2f} {f1:10.2f} {np.sum(y == 1):10}")
    print(f"{'accuracy':15} {'':10} {'':10} {acc:10.2f} {len(y):10}")
    print(f"{'macro avg':15} {precision_score(y, y_pred, average='macro', zero_division=0):10.2f} {recall_score(y, y_pred, average='macro', zero_division=0):10.2f} {f1_score(y, y_pred, average='macro', zero_division=0):10.2f} {len(y):10}")
    print(f"{'weighted avg':15} {precision_score(y, y_pred, average='weighted', zero_division=0):10.2f} {recall_score(y, y_pred, average='weighted', zero_division=0):10.2f} {f1_score(y, y_pred, average='weighted', zero_division=0):10.2f} {len(y):10}")

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

    print("\n" + "="*80)
    print("COMPREHENSIVE MODEL EVALUATION")
    print("="*80)

    # Test Headered Model on Headered Data
    print("\n[TEST 1] Headered Model on Headered Data")
    print("-" * 50)
    results_headered_on_headered = evaluate_model(
        headered_model, X_test_headered, y_test_headered, 
        "Headered Model", "headered_test_data"
    )

    # Test Headerless Model on Headerless Data
    print("\n[TEST 2] Headerless Model on Headerless Data")
    print("-" * 50)
    results_headerless_on_headerless = evaluate_model(
        headerless_model, X_test_headerless, y_test_headerless, 
        "Headerless Model", "headerless_test_data"
    )

    # Test Headered Model on Headerless Data (Cross-test)
    print("\n[TEST 3] Headered Model on Headerless Data (Cross-test)")
    print("-" * 50)
    results_headered_on_headerless = evaluate_model(
        headered_model, X_test_headerless, y_test_headerless, 
        "Headered Model", "headerless_test_data"
    )

    # Test Headerless Model on Headered Data (Cross-test)
    print("\n[TEST 4] Headerless Model on Headered Data (Cross-test)")
    print("-" * 50)
    results_headerless_on_headered = evaluate_model(
        headerless_model, X_test_headered, y_test_headered, 
        "Headerless Model", "headered_test_data"
    )

    # Comprehensive Comparison Table
    print("\n" + "="*80)
    print("COMPREHENSIVE COMPARISON TABLE")
    print("="*80)
    
    print(f"\n{'Test Case':<35} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1-Score':>10}")
    print("-" * 85)
    
    # Same-type tests
    print(f"{'Headered Model on Headered Data':<35} {results_headered_on_headered['Accuracy']:>10.4f} {results_headered_on_headered['Precision']:>10.4f} {results_headered_on_headered['Recall']:>10.4f} {results_headered_on_headered['F1-score']:>10.4f}")
    print(f"{'Headerless Model on Headerless Data':<35} {results_headerless_on_headerless['Accuracy']:>10.4f} {results_headerless_on_headerless['Precision']:>10.4f} {results_headerless_on_headerless['Recall']:>10.4f} {results_headerless_on_headerless['F1-score']:>10.4f}")
    
    # Cross-tests
    print(f"{'Headered Model on Headerless Data':<35} {results_headered_on_headerless['Accuracy']:>10.4f} {results_headered_on_headerless['Precision']:>10.4f} {results_headered_on_headerless['Recall']:>10.4f} {results_headered_on_headerless['F1-score']:>10.4f}")
    print(f"{'Headerless Model on Headered Data':<35} {results_headerless_on_headered['Accuracy']:>10.4f} {results_headerless_on_headered['Precision']:>10.4f} {results_headerless_on_headered['Recall']:>10.4f} {results_headerless_on_headered['F1-score']:>10.4f}")

    # Summary Analysis
    print("\n" + "="*80)
    print("SUMMARY ANALYSIS")
    print("="*80)
    
    # Best performing model on its own data type
    headered_own = results_headered_on_headered['Accuracy']
    headerless_own = results_headerless_on_headerless['Accuracy']
    
    if headered_own > headerless_own:
        print(f"✓ Headered model performs better on its own data type: {headered_own:.4f} vs {headerless_own:.4f}")
    else:
        print(f"✓ Headerless model performs better on its own data type: {headerless_own:.4f} vs {headered_own:.4f}")
    
    # Cross-test performance
    headered_cross = results_headered_on_headerless['Accuracy']
    headerless_cross = results_headerless_on_headered['Accuracy']
    
    print(f"Cross-test performance:")
    print(f"  - Headered model on headerless data: {headered_cross:.4f}")
    print(f"  - Headerless model on headered data: {headerless_cross:.4f}")
    
    # Generalization capability
    headered_generalization = (headered_own + headered_cross) / 2
    headerless_generalization = (headerless_own + headerless_cross) / 2
    
    print(f"\nGeneralization capability (average of own + cross performance):")
    print(f"  - Headered model: {headered_generalization:.4f}")
    print(f"  - Headerless model: {headerless_generalization:.4f}")
    
    if headered_generalization > headerless_generalization:
        print(f"✓ Headered model shows better generalization capability")
    else:
        print(f"✓ Headerless model shows better generalization capability")
