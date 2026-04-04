import json
import numpy as np
from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import joblib
import zlib
from scipy import stats
from collections import Counter
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# ==================== FILE SIGNATURE DATABASE ====================
FILE_SIGNATURES = {
    # Images
    'jpg':  {'header': b'\xff\xd8\xff',          'footer': b'\xff\xd9',       'header_len': 20,  'footer_len': 2},
    'jpeg': {'header': b'\xff\xd8\xff',          'footer': b'\xff\xd9',       'header_len': 20,  'footer_len': 2},
    'png':  {'header': b'\x89PNG\r\n\x1a\n',     'footer': b'IEND',           'header_len': 33,  'footer_len': 12},
    'gif':  {'header': b'GIF8',                  'footer': b'\x00\x3b',       'header_len': 13,  'footer_len': 2},
    'bmp':  {'header': b'BM',                    'footer': None,              'header_len': 54,  'footer_len': 0},
    'tiff': {'header': b'II\x2a\x00',            'footer': None,              'header_len': 8,   'footer_len': 0},

    # Documents
    'pdf':  {'header': b'%PDF',                  'footer': b'%%EOF',          'header_len': 15,  'footer_len': 6},
    'docx': {'header': b'PK\x03\x04',            'footer': b'PK\x05\x06',    'header_len': 30,  'footer_len': 22},
    'xlsx': {'header': b'PK\x03\x04',            'footer': b'PK\x05\x06',    'header_len': 30,  'footer_len': 22},
    'pptx': {'header': b'PK\x03\x04',            'footer': b'PK\x05\x06',    'header_len': 30,  'footer_len': 22},
    'doc':  {'header': b'\xd0\xcf\x11\xe0',      'footer': None,              'header_len': 512, 'footer_len': 0},

    # Audio
    'mp3':  {'header': b'\xff\xfb',              'footer': None,              'header_len': 4,   'footer_len': 0},
    'wav':  {'header': b'RIFF',                  'footer': None,              'header_len': 44,  'footer_len': 0},
    'flac': {'header': b'fLaC',                  'footer': None,              'header_len': 4,   'footer_len': 0},

    # Video
    'mp4':  {'header': b'ftyp',                  'footer': None,              'header_len': 8,   'footer_len': 0},
    'avi':  {'header': b'RIFF',                  'footer': None,              'header_len': 12,  'footer_len': 0},
    'mkv':  {'header': b'\x1a\x45\xdf\xa3',      'footer': None,              'header_len': 4,   'footer_len': 0},

    # Archives
    'zip':  {'header': b'PK\x03\x04',            'footer': b'PK\x05\x06',    'header_len': 30,  'footer_len': 22},
    'rar':  {'header': b'Rar!\x1a\x07',          'footer': None,              'header_len': 7,   'footer_len': 0},
    'gz':   {'header': b'\x1f\x8b',              'footer': None,              'header_len': 10,  'footer_len': 0},

    # Executables
    'exe':  {'header': b'MZ',                    'footer': None,              'header_len': 64,  'footer_len': 0},
    'elf':  {'header': b'\x7fELF',               'footer': None,              'header_len': 64,  'footer_len': 0},
}

# ==================== HEADER/FOOTER DETECTION & REMOVAL ====================

def detect_header_footer(data, file_ext):
    """
    Detect whether the file data contains known header/footer signatures.
    Returns dict with detection results.
    """
    ext = file_ext.lower().lstrip('.')
    result = {
        'has_header': False,
        'has_footer': False,
        'header_len': 0,
        'footer_len': 0,
        'file_type': ext,
        'detected_type': 'Unknown'
    }

    sig = FILE_SIGNATURES.get(ext)
    if not sig:
        # Try to match by content even if extension unknown
        for file_type, sig_info in FILE_SIGNATURES.items():
            if sig_info['header'] and data[:len(sig_info['header'])] == sig_info['header']:
                result['file_type'] = file_type
                result['detected_type'] = file_type.upper()
                result['has_header'] = True
                result['header_len'] = sig_info['header_len']
                if sig_info['footer'] and sig_info['footer'] in data[-64:]:
                    result['has_footer'] = True
                    result['footer_len'] = sig_info['footer_len']
                return result
        return result

    # Check header
    if sig['header'] and data[:len(sig['header'])] == sig['header']:
        result['has_header'] = True
        result['header_len'] = sig['header_len']
        result['detected_type'] = ext.upper()

    # Check footer
    if sig['footer'] and sig['footer'] in data[-64:]:
        result['has_footer'] = True
        result['footer_len'] = sig['footer_len']

    if result['has_header'] or result['has_footer']:
        result['detected_type'] = ext.upper()

    return result


def clean_file_data(file_bytes, file_ext=''):
    """
    Clean file by removing detected headers/footers.
    Returns cleaned bytes, original stats, and cleaned stats.
    """
    original_size = len(file_bytes)
    detection = detect_header_footer(file_bytes, file_ext)
    
    # Strip header and footer
    header_len = detection['header_len'] if detection['has_header'] else 0
    footer_len = detection['footer_len'] if detection['has_footer'] else 0
    
    if footer_len > 0:
        cleaned_data = file_bytes[header_len:-footer_len]
    else:
        cleaned_data = file_bytes[header_len:]
    
    cleaned_size = len(cleaned_data)
    
    return cleaned_data, detection, {
        'original_size': original_size,
        'cleaned_size': cleaned_size,
        'header_removed': header_len,
        'footer_removed': footer_len,
        'bytes_removed': header_len + footer_len,
        'removal_percentage': (header_len + footer_len) / original_size * 100 if original_size > 0 else 0
    }


def create_fragments(data, chunk_size=4096, num_fragments=5):
    """
    Create multiple fragments from cleaned data.
    Samples from beginning, middle, and end for better diversity.
    """
    if len(data) < chunk_size:
        # Pad if too small
        padded = np.pad(np.frombuffer(data, dtype=np.uint8), 
                       (0, chunk_size - len(data)), mode='constant')
        return [padded]
    
    fragments = []
    
    # Get fragments from different parts of the file
    positions = np.linspace(0, max(0, len(data) - chunk_size), num_fragments, dtype=int)
    
    for pos in positions:
        fragment = np.frombuffer(data[pos:pos + chunk_size], dtype=np.uint8)
        if len(fragment) < chunk_size:
            fragment = np.pad(fragment, (0, chunk_size - len(fragment)), mode='constant')
        fragments.append(fragment)
    
    return fragments


# Page configuration
st.set_page_config(
    page_title="File Type Identification - Model Comparison Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .section-header {
        border-bottom: 3px solid #1f77b4;
        padding-bottom: 10px;
        margin-top: 20px;
    }
    .model-name {
        font-size: 24px;
        font-weight: bold;
        color: #1f77b4;
    }
    </style>
    """, unsafe_allow_html=True)

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"
SAVED_MODELS_DIR = BASE_DIR / "saved_models"

# ==================== FEATURE EXTRACTION FUNCTIONS ====================

def byte_frequency_histogram(fragment):
    """Count frequency of each byte value (0-255)."""
    hist = np.bincount(fragment.astype(int), minlength=256)
    return hist / len(fragment)

def shannon_entropy(fragment):
    """Calculate Shannon entropy of byte distribution."""
    hist = np.bincount(fragment.astype(int), minlength=256)
    probs = hist / len(fragment)
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs)) if len(probs) > 0 else 0

def bigram_frequencies(fragment, top_n=20):
    """Count frequency of byte pairs (bigrams)."""
    frag_int = fragment.astype(int)
    bigrams = frag_int[:-1] * 256 + frag_int[1:]
    hist = np.bincount(bigrams, minlength=65536)
    top_freqs = np.sort(hist)[::-1][:top_n]
    result = top_freqs / len(bigrams) if len(bigrams) > 0 else np.zeros(top_n)
    if len(result) < top_n:
        result = np.pad(result, (0, top_n - len(result)))
    return result

def statistical_features(fragment):
    """Extract basic statistical features."""
    frag = fragment.astype(float)
    features = [
        np.mean(frag),
        np.std(frag),
        float(stats.skew(frag)),
        float(stats.kurtosis(frag)),
        np.median(frag),
        float(np.percentile(frag, 25)),
        float(np.percentile(frag, 75)),
        float(np.max(frag) - np.min(frag)),
        float(np.sum(frag == 0)) / len(frag),
        float(np.sum((frag >= 32) & (frag <= 126))) / len(frag),
    ]
    return np.nan_to_num(np.array(features), nan=0.0)

def block_entropy(fragment, n_blocks=16):
    """Calculate Shannon entropy over sub-blocks."""
    block_size = max(1, len(fragment) // n_blocks)
    entropies = []
    for i in range(n_blocks):
        block = fragment[i * block_size:(i + 1) * block_size]
        if len(block) == 0:
            entropies.append(0)
            continue
        hist = np.bincount(block.astype(int), minlength=256)
        probs = hist / len(block)
        probs = probs[probs > 0]
        entropies.append(-np.sum(probs * np.log2(probs)) if len(probs) > 0 else 0)
    return np.array(entropies)

def longest_runs(fragment):
    """Find longest consecutive run of same byte and zero bytes."""
    frag = fragment.astype(int)
    max_run = 1
    max_zero_run = 0
    current_run = 1
    current_zero_run = 1 if frag[0] == 0 else 0
    for i in range(1, len(frag)):
        if frag[i] == frag[i - 1]:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 1
        if frag[i] == 0:
            if frag[i - 1] == 0:
                current_zero_run += 1
            else:
                current_zero_run = 1
            max_zero_run = max(max_zero_run, current_zero_run)
        else:
            current_zero_run = 0
    return np.array([max_run / len(frag), max_zero_run / len(frag)])

def chi_squared_test(fragment):
    """Chi-squared test for byte distribution uniformity."""
    observed = np.bincount(fragment.astype(int), minlength=256).astype(float)
    expected = np.full(256, len(fragment) / 256.0)
    chi2 = np.sum((observed - expected) ** 2 / expected)
    return chi2 / len(fragment)

def compression_ratio(fragment):
    """Estimate compressibility using zlib."""
    raw_bytes = bytes(fragment.astype(np.uint8))
    compressed = zlib.compress(raw_bytes, level=1)
    return len(compressed) / len(raw_bytes)

def trigram_frequencies(fragment, top_n=10):
    """Count frequency of byte trigrams (3-grams)."""
    frag_int = fragment.astype(int)
    if len(frag_int) < 3:
        return np.zeros(top_n)
    trigrams = frag_int[:-2] * 65536 + frag_int[1:-1] * 256 + frag_int[2:]
    _, counts = np.unique(trigrams, return_counts=True)
    top_counts = np.sort(counts)[::-1][:top_n]
    if len(top_counts) < top_n:
        top_counts = np.pad(top_counts, (0, top_n - len(top_counts)))
    return top_counts / len(trigrams) if len(trigrams) > 0 else np.zeros(top_n)

def extract_features(file_bytes):
    """Extract all engineered features from file bytes."""
    fragment = np.frombuffer(file_bytes, dtype=np.uint8)
    if len(fragment) == 0:
        return np.zeros(317)
    
    # Pad or truncate to 4096 bytes
    if len(fragment) < 4096:
        fragment = np.pad(fragment, (0, 4096 - len(fragment)), mode='constant')
    else:
        fragment = fragment[:4096]
    
    features = np.concatenate([
        byte_frequency_histogram(fragment),
        [shannon_entropy(fragment)],
        bigram_frequencies(fragment, top_n=20),
        statistical_features(fragment),
        block_entropy(fragment),
        longest_runs(fragment),
        [chi_squared_test(fragment)],
        [compression_ratio(fragment)],
        trigram_frequencies(fragment, top_n=10),
    ])
    
    return features.astype(np.float32)

# ==================== MODEL LOADING & PREDICTION ====================

@st.cache_resource
def load_models():
    """Load all trained models."""
    models = {}
    labels = None
    
    # Load per_class_metrics from results to get labels
    try:
        cnn_results = json.loads((RESULTS_DIR / "cnn_results.json").read_text())
        labels = sorted(cnn_results.get("per_class_metrics", {}).keys())
    except:
        pass
    
    if not labels:
        labels = []
    
    num_classes = len(labels) if labels else 22
    chunk_size = 4096
    
    # ==================== CNN Model ====================
    try:
        cnn_model_path = SAVED_MODELS_DIR / "cnn" / "cnn_model.pth"
        if cnn_model_path.exists():
            class FragmentCNN(torch.nn.Module):
                def __init__(self, input_size, num_classes):
                    super().__init__()
                    self.conv1 = torch.nn.Conv1d(1, 64, kernel_size=5, padding=2)
                    self.pool1 = torch.nn.MaxPool1d(2)
                    self.conv2 = torch.nn.Conv1d(64, 128, kernel_size=3, padding=1)
                    self.pool2 = torch.nn.MaxPool1d(2)
                    self.flatten = torch.nn.Flatten()
                    self.fc1 = torch.nn.Linear(128 * (input_size // 4), 128)
                    self.dropout = torch.nn.Dropout(0.3)
                    self.fc2 = torch.nn.Linear(128, num_classes)
                
                def forward(self, x):
                    x = torch.relu(self.conv1(x))
                    x = self.pool1(x)
                    x = torch.relu(self.conv2(x))
                    x = self.pool2(x)
                    x = self.flatten(x)
                    x = torch.relu(self.fc1(x))
                    x = self.dropout(x)
                    x = self.fc2(x)
                    return x
            
            model = FragmentCNN(chunk_size, num_classes)
            checkpoint = torch.load(cnn_model_path, map_location="cpu")
            model.load_state_dict(checkpoint, strict=False)
            model.eval()
            models['cnn'] = {'model': model, 'type': 'cnn', 'input_shape': (1, chunk_size)}
    except Exception as e:
        st.warning(f"Could not load CNN: {e}")
    
    # ==================== LeNet Model ====================
    try:
        lenet_path = SAVED_MODELS_DIR / "lenet" / "lenet_model.pth"
        if lenet_path.exists():
            class LeNet1D(torch.nn.Module):
                def __init__(self, input_size, num_classes):
                    super().__init__()
                    self.conv1 = torch.nn.Conv1d(1, 6, kernel_size=5, padding=2)
                    self.pool = torch.nn.AvgPool1d(kernel_size=2, stride=2)
                    self.conv2 = torch.nn.Conv1d(6, 16, kernel_size=5, padding=2)
                    fc_input = 16 * (input_size // 4)
                    self.fc1 = torch.nn.Linear(fc_input, 120)
                    self.fc2 = torch.nn.Linear(120, 84)
                    self.fc3 = torch.nn.Linear(84, num_classes)
                
                def forward(self, x):
                    x = torch.relu(self.conv1(x))
                    x = self.pool(x)
                    x = torch.relu(self.conv2(x))
                    x = self.pool(x)
                    x = x.view(x.size(0), -1)
                    x = torch.relu(self.fc1(x))
                    x = torch.relu(self.fc2(x))
                    x = self.fc3(x)
                    return x
            
            model = LeNet1D(chunk_size, num_classes)
            checkpoint = torch.load(lenet_path, map_location="cpu")
            model.load_state_dict(checkpoint, strict=False)
            model.eval()
            models['lenet'] = {'model': model, 'type': 'lenet', 'input_shape': (1, chunk_size)}
    except Exception as e:
        st.warning(f"Could not load LeNet: {e}")
    
    # ==================== LSTM Model ====================
    try:
        lstm_path = SAVED_MODELS_DIR / "lstm" / "lstm_model.pth"
        if lstm_path.exists():
            seq_step = 16
            
            class FragmentLSTM(torch.nn.Module):
                def __init__(self, input_size, num_classes, hidden_size=128, num_layers=2):
                    super().__init__()
                    self.seq_len = input_size // seq_step
                    self.feature_dim = seq_step
                    self.lstm = torch.nn.LSTM(
                        self.feature_dim, hidden_size, num_layers,
                        batch_first=True, bidirectional=True, dropout=0.3 if num_layers > 1 else 0
                    )
                    self.fc = torch.nn.Linear(hidden_size * 2, num_classes)
                
                def forward(self, x):
                    batch_size = x.size(0)
                    x = x.view(batch_size, self.seq_len, self.feature_dim)
                    out, (h, c) = self.lstm(x)
                    out = out[:, -1, :]
                    out = self.fc(out)
                    return out
            
            model = FragmentLSTM(chunk_size, num_classes)
            checkpoint = torch.load(lstm_path, map_location="cpu")
            model.load_state_dict(checkpoint, strict=False)
            model.eval()
            models['lstm'] = {'model': model, 'type': 'lstm', 'input_shape': (1, chunk_size)}
    except Exception as e:
        st.warning(f"Could not load LSTM: {e}")
    
    # ==================== ResNet Model ====================
    try:
        resnet_path = SAVED_MODELS_DIR / "resnet" / "resnet_model.pth"
        if resnet_path.exists():
            class ResidualBlock1D(torch.nn.Module):
                def __init__(self, in_channels, out_channels, stride=1):
                    super().__init__()
                    self.conv1 = torch.nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
                    self.bn1 = torch.nn.BatchNorm1d(out_channels)
                    self.conv2 = torch.nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
                    self.bn2 = torch.nn.BatchNorm1d(out_channels)
                    self.shortcut = torch.nn.Sequential()
                    if stride != 1 or in_channels != out_channels:
                        self.shortcut = torch.nn.Sequential(
                            torch.nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                            torch.nn.BatchNorm1d(out_channels),
                        )
                
                def forward(self, x):
                    out = torch.relu(self.bn1(self.conv1(x)))
                    out = self.bn2(self.conv2(out))
                    out += self.shortcut(x)
                    return torch.relu(out)
            
            class ResNet1D(torch.nn.Module):
                def __init__(self, num_classes):
                    super().__init__()
                    self.conv1 = torch.nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3)
                    self.bn1 = torch.nn.BatchNorm1d(64)
                    self.pool = torch.nn.MaxPool1d(3, stride=2, padding=1)
                    self.layer1 = self._make_layer(64, 64, 2, stride=1)
                    self.layer2 = self._make_layer(64, 128, 2, stride=2)
                    self.layer3 = self._make_layer(128, 256, 2, stride=2)
                    self.avgpool = torch.nn.AdaptiveAvgPool1d(1)
                    self.fc = torch.nn.Linear(256, num_classes)
                
                def _make_layer(self, in_channels, out_channels, blocks, stride):
                    layers = []
                    layers.append(ResidualBlock1D(in_channels, out_channels, stride))
                    for _ in range(1, blocks):
                        layers.append(ResidualBlock1D(out_channels, out_channels, stride=1))
                    return torch.nn.Sequential(*layers)
                
                def forward(self, x):
                    x = torch.relu(self.bn1(self.conv1(x)))
                    x = self.pool(x)
                    x = self.layer1(x)
                    x = self.layer2(x)
                    x = self.layer3(x)
                    x = self.avgpool(x)
                    x = x.view(x.size(0), -1)
                    x = self.fc(x)
                    return x
            
            model = ResNet1D(num_classes)
            checkpoint = torch.load(resnet_path, map_location="cpu")
            model.load_state_dict(checkpoint, strict=False)
            model.eval()
            models['resnet'] = {'model': model, 'type': 'resnet', 'input_shape': (1, chunk_size)}
    except Exception as e:
        st.warning(f"Could not load ResNet: {e}")
    
    # ==================== Random Forest (Sklearn) ====================
    # Note: Random Forest model may not exist - skipping if missing
    try:
        rf_path = SAVED_MODELS_DIR / "random_forest" / "random_forest_model.joblib"
        if rf_path.exists():
            loaded = joblib.load(rf_path)
            model = loaded.get('model', loaded) if isinstance(loaded, dict) else loaded
            models['random_forest'] = {'model': model, 'type': 'sklearn'}
    except Exception as e:
        pass  # Silently skip if not found
    
    # ==================== SVM (Sklearn) ====================
    try:
        svm_path = SAVED_MODELS_DIR / "svm" / "svm_model.joblib"
        if svm_path.exists():
            loaded = joblib.load(svm_path)
            # Extract model from potential wrapper
            if isinstance(loaded, dict):
                model = loaded.get('model', loaded.get('svm', loaded))
            else:
                model = loaded
            
            # Validate and store
            if hasattr(model, 'predict') or hasattr(model, 'predict_proba'):
                models['svm'] = {'model': model, 'type': 'sklearn'}
    except Exception as e:
        pass  # Silently skip if loading fails
    
    # ==================== MLP (PyTorch) ====================
    try:
        mlp_path = SAVED_MODELS_DIR / "mlp" / "mlp_model.pth"
        if mlp_path.exists():
            class MLPClassifier(torch.nn.Module):
                def __init__(self, input_size, num_classes):
                    super().__init__()
                    self.fc1 = torch.nn.Linear(input_size, 512)
                    self.dropout1 = torch.nn.Dropout(0.3)
                    self.fc2 = torch.nn.Linear(512, 256)
                    self.dropout2 = torch.nn.Dropout(0.3)
                    self.fc3 = torch.nn.Linear(256, 128)
                    self.dropout3 = torch.nn.Dropout(0.3)
                    self.fc4 = torch.nn.Linear(128, num_classes)
                
                def forward(self, x):
                    # Expects (batch, 317) feature vector
                    if len(x.shape) > 2:
                        x = x.view(x.shape[0], -1)
                    x = torch.relu(self.fc1(x))
                    x = self.dropout1(x)
                    x = torch.relu(self.fc2(x))
                    x = self.dropout2(x)
                    x = torch.relu(self.fc3(x))
                    x = self.dropout3(x)
                    x = self.fc4(x)
                    return x
            
            # Try to load as PyTorch model
            model = MLPClassifier(317, num_classes)
            checkpoint = torch.load(mlp_path, map_location="cpu")
            
            # If checkpoint is a state dict
            if isinstance(checkpoint, dict) and 'fc1.weight' in checkpoint:
                model.load_state_dict(checkpoint, strict=False)
            else:
                # If checkpoint is an actual model, extract weights
                model = checkpoint
            
            model.eval()
            models['mlp'] = {'model': model, 'type': 'mlp_pytorch', 'input_shape': (317,)}
    except Exception as e:
        st.warning(f"Could not load MLP: {e}")
    
    # ==================== XGBoost ====================
    try:
        xgb_path = SAVED_MODELS_DIR / "xgboost" / "xgb_model.joblib"
        if xgb_path.exists():
            loaded = joblib.load(xgb_path)
            # Extract model from potential wrapper
            if isinstance(loaded, dict):
                model = loaded.get('model', loaded.get('xgboost', loaded.get('xgb', loaded)))
            else:
                model = loaded
            
            # Validate and store
            if hasattr(model, 'predict') or hasattr(model, 'predict_proba'):
                models['xgboost'] = {'model': model, 'type': 'sklearn'}
    except Exception as e:
        pass  # Silently skip if loading fails
    
    return models, labels

def predict_file(file_bytes, models, labels, cleaned_data=None):
    """
    Make predictions using all loaded models.
    Optionally uses cleaned_data (header/footer removed) for better accuracy.
    """
    if len(file_bytes) == 0:
        return None
    
    # Use cleaned data if provided (more accurate)
    data_to_process = cleaned_data if cleaned_data is not None else file_bytes
    
    predictions = {}
    
    try:
        # Get raw bytes for PyTorch models (Conv1d and LSTM models)
        raw_bytes = np.frombuffer(data_to_process[:4096], dtype=np.uint8)
        if len(raw_bytes) < 4096:
            raw_bytes = np.pad(raw_bytes, (0, 4096 - len(raw_bytes)), mode='constant')
        
        # Extract features for sklearn and MLP models
        features = extract_features(data_to_process)
        
        num_classes = len(labels) if labels else 22
        
        for model_name, model_info in models.items():
            try:
                model = model_info['model']
                model_type = model_info['type']
                
                probs = None
                
                if model_type in ['cnn', 'lenet', 'lstm', 'resnet']:
                    # PyTorch Conv1d and LSTM models: take raw bytes input (batch, channels, length)
                    x = torch.FloatTensor(raw_bytes).unsqueeze(0).unsqueeze(0)  # (1, 1, 4096)
                    
                    with torch.no_grad():
                        output = model(x)
                    
                    probs = torch.softmax(output, dim=1)[0].numpy()
                
                elif model_type == 'mlp_pytorch':
                    # MLP PyTorch model: takes engineered features as input (batch, 317)
                    x = torch.FloatTensor(features).unsqueeze(0)  # (1, 317)
                    
                    with torch.no_grad():
                        output = model(x)
                    
                    probs = torch.softmax(output, dim=1)[0].numpy()
                
                elif model_type == 'sklearn':
                    # Sklearn models: take engineered features
                    try:
                        probs = model.predict_proba(features.reshape(1, -1))[0]
                    except (AttributeError, TypeError):
                        # Fallback: use predict and convert to probabilities
                        try:
                            pred = model.predict(features.reshape(1, -1))[0]
                            probs = np.zeros(num_classes)
                            if pred < num_classes:
                                probs[int(pred)] = 1.0
                            else:
                                probs = np.ones(num_classes) / num_classes
                        except:
                            probs = np.ones(num_classes) / num_classes
                
                else:
                    continue
                
                if probs is None:
                    continue
                
                # Ensure probs has correct length and valid values
                if len(probs) < num_classes:
                    probs = np.pad(probs, (0, num_classes - len(probs)), mode='constant')
                elif len(probs) > num_classes:
                    probs = probs[:num_classes]
                
                # Check for NaN or Inf values
                if np.any(np.isnan(probs)) or np.any(np.isinf(probs)):
                    probs = np.ones(num_classes) / num_classes
                
                # Normalize probabilities
                prob_sum = np.sum(probs)
                if prob_sum > 0:
                    probs = probs / prob_sum
                else:
                    probs = np.ones(num_classes) / num_classes
                
                # Get prediction
                pred_idx = np.argmax(probs)
                pred_label = labels[pred_idx] if labels and pred_idx < len(labels) else f"Class {pred_idx}"
                confidence = float(probs[pred_idx])
                
                # Debug: Log top 3 predictions
                top_3_indices = np.argsort(probs)[-3:][::-1]
                top_3_preds = [(labels[i], probs[i]) for i in top_3_indices if i < len(labels)]
                
                predictions[model_name] = {
                    'predicted_class': pred_label,
                    'confidence': confidence,
                    'probabilities': probs,
                    'top_3': top_3_preds,
                    'all_probs_dict': {labels[i]: float(probs[i]) for i in range(len(labels))}
                }
            
            except Exception as e:
                st.warning(f"Error in {model_name}: {str(e)}")
                continue
        
        return predictions if predictions else None
    
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        import traceback
        st.error(traceback.format_exc())
        return None


# Title and description
st.title("🎯 File Type Identification - Model Comparison Dashboard")
st.markdown("Compare performance metrics, visualizations, and training histories of different ML models")

# Sidebar
st.sidebar.header("📊 Dashboard Controls")
show_section = st.sidebar.radio(
    "Select Section:",
    ["📈 Model Comparison", "🔍 Individual Model Analysis", "📁 File Upload & Predict"],
    index=0
)


def load_all_results():
    """Load all model results from JSON files"""
    models_data = {}
    json_files = sorted(RESULTS_DIR.glob("*_results.json"))
    
    for json_file in json_files:
        try:
            model_key = json_file.stem.replace("_results", "")
            data = json.loads(json_file.read_text(encoding="utf-8"))
            models_data[model_key] = data
        except Exception as e:
            st.warning(f"Could not load {json_file.name}: {e}")
    
    return models_data


def format_percentage(value):
    """Format decimal as percentage"""
    if value is None:
        return "N/A"
    return f"{value * 100:.2f}%"


def create_comparison_dataframe(models_data):
    """Create comparison DataFrame for all models"""
    rows = []
    
    for model_key, data in models_data.items():
        rows.append({
            "Model": model_key.replace("_", " ").upper(),
            "Accuracy": data.get("accuracy", 0),
            "Precision": data.get("precision", 0),
            "Recall": data.get("recall", 0),
            "F1 Score": data.get("f1_score", 0),
            "Val Accuracy": data.get("val_accuracy", 0),
            "Val F1 Score": data.get("val_f1_score", 0),
        })
    
    return pd.DataFrame(rows)


def plot_comparison_bars(df):
    """Create comparison bar charts"""
    metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]
    
    fig = go.Figure()
    
    for metric in metrics:
        fig.add_trace(go.Bar(
            x=df["Model"],
            y=df[metric],
            name=metric,
            text=[f"{v*100:.1f}%" for v in df[metric]],
            textposition="auto",
        ))
    
    fig.update_layout(
        title="Model Performance Comparison",
        xaxis_title="Model",
        yaxis_title="Score",
        barmode='group',
        height=500,
        hovermode='x unified',
        template='plotly_white'
    )
    
    return fig


def plot_radar_comparison(df):
    """Create radar chart for model comparison"""
    fig = go.Figure()
    
    metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]
    
    for idx, row in df.iterrows():
        fig.add_trace(go.Scatterpolar(
            r=[row[m] for m in metrics],
            theta=metrics,
            fill='toself',
            name=row["Model"],
            opacity=0.6
        ))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        height=600,
        title="Model Performance - Radar Comparison"
    )
    
    return fig


def plot_training_history(model_data, model_name):
    """Plot training and validation loss curves"""
    history = model_data.get("training_history", {})
    
    if not history:
        st.warning(f"No training history available for {model_name}")
        return None
    
    epochs = range(1, len(history.get("train_loss", [])) + 1)
    
    fig = go.Figure()
    
    # Add training loss
    if "train_loss" in history:
        fig.add_trace(go.Scatter(
            x=list(epochs),
            y=history["train_loss"],
            mode='lines+markers',
            name='Train Loss',
            line=dict(color='#1f77b4', width=2),
        ))
    
    # Add validation loss
    if "val_loss" in history:
        fig.add_trace(go.Scatter(
            x=list(epochs),
            y=history["val_loss"],
            mode='lines+markers',
            name='Validation Loss',
            line=dict(color='#ff7f0e', width=2),
        ))
    
    fig.update_layout(
        title=f"{model_name} - Training History",
        xaxis_title="Epoch",
        yaxis_title="Loss",
        hovermode='x unified',
        height=400,
        template='plotly_white'
    )
    
    return fig


def plot_accuracy_history(model_data, model_name):
    """Plot validation accuracy over epochs"""
    history = model_data.get("training_history", {})
    
    if "val_accuracy" not in history:
        return None
    
    epochs = range(1, len(history["val_accuracy"]) + 1)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=list(epochs),
        y=history["val_accuracy"],
        mode='lines+markers',
        name='Validation Accuracy',
        fill='tozeroy',
        line=dict(color='#2ca02c', width=2),
    ))
    
    fig.update_layout(
        title=f"{model_name} - Validation Accuracy Over Epochs",
        xaxis_title="Epoch",
        yaxis_title="Accuracy",
        hovermode='x',
        height=400,
        template='plotly_white'
    )
    
    return fig


def plot_confusion_matrix(model_data, model_name):
    """Plot confusion matrix as heatmap"""
    confusion_mat = model_data.get("confusion_matrix")
    
    if not confusion_mat:
        st.warning(f"No confusion matrix available for {model_name}")
        return None
    
    # Convert to numpy array
    cm = np.array(confusion_mat)
    
    # Get class labels from per_class_metrics
    per_class = model_data.get("per_class_metrics", {})
    labels = sorted(per_class.keys())
    
    # Create figure
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=labels,
        y=labels,
        colorscale='Blues',
        text=cm,
        texttemplate='%{text}',
        textfont={"size": 8},
        hoverongaps=False,
    ))
    
    fig.update_layout(
        title=f"{model_name} - Confusion Matrix",
        xaxis_title="Predicted",
        yaxis_title="Actual",
        height=600,
        width=800,
    )
    
    return fig


def plot_per_class_metrics(model_data, model_name):
    """Plot per-class precision, recall, and F1 score"""
    per_class = model_data.get("per_class_metrics", {})
    
    if not per_class:
        st.warning(f"No per-class metrics available for {model_name}")
        return None
    
    rows = []
    for cls, metrics in per_class.items():
        rows.append({
            "Class": cls,
            "Precision": metrics.get("precision", 0),
            "Recall": metrics.get("recall", 0),
            "F1": metrics.get("f1", 0),
            "Support": metrics.get("support", 0)
        })
    
    df = pd.DataFrame(rows).sort_values("F1", ascending=False)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(x=df["Class"], y=df["Precision"], name="Precision"))
    fig.add_trace(go.Bar(x=df["Class"], y=df["Recall"], name="Recall"))
    fig.add_trace(go.Bar(x=df["Class"], y=df["F1"], name="F1 Score"))
    
    fig.update_layout(
        title=f"{model_name} - Per-Class Metrics",
        xaxis_title="File Type",
        yaxis_title="Score",
        barmode='group',
        height=400,
        hovermode='x unified',
        template='plotly_white',
        xaxis_tickangle=-45
    )
    
    return fig


def display_model_metrics(model_data, model_name):
    """Display key metrics as formatted cards"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Accuracy",
            format_percentage(model_data.get("accuracy")),
            delta=format_percentage(model_data.get("val_accuracy", model_data.get("accuracy")))
        )
    
    with col2:
        st.metric(
            "Precision",
            format_percentage(model_data.get("precision")),
            delta=format_percentage(model_data.get("val_precision", model_data.get("precision")))
        )
    
    with col3:
        st.metric(
            "Recall",
            format_percentage(model_data.get("recall")),
            delta=format_percentage(model_data.get("val_recall", model_data.get("recall")))
        )
    
    with col4:
        st.metric(
            "F1 Score",
            format_percentage(model_data.get("f1_score")),
            delta=format_percentage(model_data.get("val_f1_score", model_data.get("f1_score")))
        )


# Load all model data
models_data = load_all_results()

if not models_data:
    st.error("No model results found in the results directory!")
    st.stop()


# ==================== SECTION 1: MODEL COMPARISON ====================
if show_section == "📈 Model Comparison":
    st.markdown("### 📊 Overall Model Performance Comparison")
    
    df_comparison = create_comparison_dataframe(models_data)
    
    # Display comparison table
    st.dataframe(
        df_comparison.style.format({
            "Accuracy": "{:.2%}",
            "Precision": "{:.2%}",
            "Recall": "{:.2%}",
            "F1 Score": "{:.2%}",
            "Val Accuracy": "{:.2%}",
            "Val F1 Score": "{:.2%}"
        }).highlight_max(subset=["Accuracy", "F1 Score"], color='yellow'),
        use_container_width=True
    )
    
    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Bar Charts", "🎯 Radar Chart", "🏆 Best Models", "📈 Rankings"])
    
    with tab1:
        st.plotly_chart(plot_comparison_bars(df_comparison), use_container_width=True, key='comparison_bars')
    
    with tab2:
        st.plotly_chart(plot_radar_comparison(df_comparison), use_container_width=True, key='radar_comparison')
    
    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            best_accuracy = df_comparison.loc[df_comparison["Accuracy"].idxmax()]
            st.markdown("#### 🥇 Best Accuracy")
            st.success(f"**{best_accuracy['Model']}**: {best_accuracy['Accuracy']*100:.2f}%")
        
        with col2:
            best_f1 = df_comparison.loc[df_comparison["F1 Score"].idxmax()]
            st.markdown("#### 🥇 Best F1 Score")
            st.success(f"**{best_f1['Model']}**: {best_f1['F1 Score']*100:.2f}%")
    
    with tab4:
        st.markdown("#### Model Rankings by Metric")
        
        for metric in ["Accuracy", "F1 Score", "Precision", "Recall"]:
            ranking_df = df_comparison[["Model", metric]].sort_values(metric, ascending=False).reset_index(drop=True)
            ranking_df.index = ranking_df.index + 1
            st.markdown(f"**{metric}**")
            st.dataframe(ranking_df.style.format({metric: "{:.2%}"}))


# ==================== SECTION 2: INDIVIDUAL MODEL ANALYSIS ====================
elif show_section == "🔍 Individual Model Analysis":
    st.markdown("### 🔍 Detailed Model Analysis")
    
    selected_model = st.selectbox(
        "Select a Model to Analyze:",
        sorted(models_data.keys()),
        format_func=lambda x: x.replace("_", " ").upper()
    )
    
    model_data = models_data[selected_model]
    model_display_name = selected_model.replace("_", " ").upper()
    
    st.markdown(f"### {model_display_name}")
    
    # Display metrics
    st.markdown("#### 📊 Performance Metrics")
    display_model_metrics(model_data, model_display_name)
    
    # Model parameters
    if "parameters" in model_data:
        st.markdown("#### 🔧 Model Parameters")
        params = model_data["parameters"]
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Parameters", f"{params.get('total_params', 0):,}")
        with col2:
            st.metric("Trainable Parameters", f"{params.get('trainable_params', 0):,}")
        with col3:
            st.metric("Model Size", f"{params.get('model_size_mb', 0):.2f} MB")
    
    # Training history
    st.markdown("#### 📈 Training History")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_loss = plot_training_history(model_data, model_display_name)
        if fig_loss:
            st.plotly_chart(fig_loss, use_container_width=True, key='training_loss')
    
    with col2:
        fig_acc = plot_accuracy_history(model_data, model_display_name)
        if fig_acc:
            st.plotly_chart(fig_acc, use_container_width=True, key='accuracy_history')
    
    # Confusion matrix
    st.markdown("#### 🔲 Confusion Matrix Heatmap")
    fig_cm = plot_confusion_matrix(model_data, model_display_name)
    if fig_cm:
        st.plotly_chart(fig_cm, use_container_width=True, key='confusion_matrix')
    
    # Per-class metrics
    st.markdown("#### 📊 Per-Class Performance Metrics")
    fig_per_class = plot_per_class_metrics(model_data, model_display_name)
    if fig_per_class:
        st.plotly_chart(fig_per_class, use_container_width=True, key='per_class_metrics')
    
    # Detailed per-class table
    if "per_class_metrics" in model_data:
        st.markdown("#### 📋 Detailed Per-Class Metrics Table")
        per_class_df = pd.DataFrame([
            {
                "File Type": cls,
                "Precision": metrics.get("precision", 0),
                "Recall": metrics.get("recall", 0),
                "F1 Score": metrics.get("f1", 0),
                "Support": metrics.get("support", 0)
            }
            for cls, metrics in model_data["per_class_metrics"].items()
        ]).sort_values("F1 Score", ascending=False)
        
        st.dataframe(
            per_class_df.style.format({
                "Precision": "{:.2%}",
                "Recall": "{:.2%}",
                "F1 Score": "{:.2%}"
            }).highlight_max(subset=["F1 Score"], color='lightgreen'),
            use_container_width=True
        )


# ==================== SECTION 3: FILE UPLOAD & PREDICT ====================
elif show_section == "📁 File Upload & Predict":
    st.markdown("### 📁 File Upload & Prediction")
    
    # Load models
    with st.spinner("🔄 Loading models..."):
        loaded_models, class_labels = load_models()
    
    if not loaded_models:
        st.warning("⚠️ No models could be loaded. Please ensure models are saved in `saved_models/` directory.")
    
    st.info(
        "📤 Upload any file to predict its type using 8 different machine learning models.\n\n"
        "The dashboard will extract features from the file and show predictions from each model."
    )
    
    uploaded_file = st.file_uploader(
        "📁 Upload a file to predict its type:",
        type=None,
        help="Upload any file type to predict its classification"
    )
    
    if uploaded_file is not None:
        file_bytes = uploaded_file.read()
        
        st.success(f"✅ File uploaded: **{uploaded_file.name}**")
        
        # Clean file (remove headers/footers)
        st.markdown("### 🔧 File Preprocessing")
        
        with st.spinner("🔍 Detecting file type and cleaning headers/footers..."):
            cleaned_data, detection, clean_stats = clean_file_data(file_bytes, uploaded_file.name)
        
        # Display preprocessing info
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📄 Original Size", f"{clean_stats['original_size']:,} bytes")
        with col2:
            st.metric("🧹 Cleaned Size", f"{clean_stats['cleaned_size']:,} bytes")
        with col3:
            st.metric("🗑️ Bytes Removed", f"{clean_stats['bytes_removed']:,} bytes")
        with col4:
            st.metric("📊 Removal %", f"{clean_stats['removal_percentage']:.1f}%")
        
        # Show detection details
        with st.expander("📋 Preprocessing Details", expanded=True):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Type Detection:**")
                if detection['detected_type'] != 'Unknown':
                    st.success(f"✅ Detected as: **{detection['detected_type']}**")
                else:
                    st.info("ℹ️ Type: Unknown (will use raw bytes)")
            
            with col2:
                st.markdown("**Header Information:**")
                if detection['has_header']:
                    st.success(f"✅ Header found & removed: **{detection['header_len']} bytes**")
                else:
                    st.info("ℹ️ No recognizable header")
            
            with col3:
                st.markdown("**Footer Information:**")
                if detection['has_footer']:
                    st.success(f"✅ Footer found & removed: **{detection['footer_len']} bytes**")
                else:
                    st.info("ℹ️ No recognizable footer")
            
            # Show original file info
            st.markdown("---")
            st.markdown("**Original File Info:**")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📄 File Name", uploaded_file.name.split('/')[-1][:40])
            with col2:
                st.metric("💾 Total Size", f"{len(file_bytes) / 1024:.2f} KB")
            with col3:
                file_ext = uploaded_file.name.split('.')[-1].upper() if '.' in uploaded_file.name else "UNKNOWN"
                st.metric("🏷️ Extension", file_ext)
        
        st.markdown("---")
        
        if loaded_models and class_labels:
            # Show loaded models status
            with st.expander("📦 Model Loading Status", expanded=False):
                model_status = []
                for model_name in loaded_models.keys():
                    model_status.append({
                        'Model': model_name.upper(),
                        'Status': '✅ Loaded',
                        'Type': loaded_models[model_name]['type']
                    })
                
                st.dataframe(
                    pd.DataFrame(model_status),
                    use_container_width=True,
                    hide_index=True
                )
            
            st.markdown("### 🔮 Predictions from All Models")
            
            # Show model accuracy information
            st.warning(
                "⚠️ **Model Accuracy Notice:** These models were trained on the available dataset and achieve 40-60% accuracy. "
                "This means predictions may not always be correct. Cross-reference multiple model predictions for better reliability."
            )
            
            with st.spinner("🤖 Running predictions on cleaned file fragments..."):
                # Use cleaned data for predictions
                predictions = predict_file(file_bytes, loaded_models, class_labels, cleaned_data=cleaned_data)
            
            if predictions:
                # Create summary dataframe
                prediction_data = []
                for model_name, pred_info in predictions.items():
                    prediction_data.append({
                        'Model': model_name.replace('_', ' ').upper(),
                        'Predicted File Type': pred_info['predicted_class'].upper(),
                        'Confidence': f"{pred_info['confidence']*100:.2f}%",
                        'Confidence Score': pred_info['confidence']
                    })
                
                df_predictions = pd.DataFrame(prediction_data)
                
                # Display predictions table
                st.markdown("#### 📊 Model Predictions Summary")
                st.dataframe(
                    df_predictions.style.format({}).highlight_max(
                        subset=['Confidence Score'], 
                        color='lightgreen'
                    ),
                    use_container_width=True,
                    hide_index=True
                )
                
                # Add debugging section to show why PPTX keeps winning
                with st.expander("🔍 Debug: Full Prediction Probabilities (Top 5 Classes)", expanded=False):
                    for model_name, pred_info in predictions.items():
                        st.markdown(f"**{model_name.upper()}** - Top 5 Predictions:")
                        
                        # Get top 5 predictions
                        top_5_indices = np.argsort(pred_info['probabilities'])[-5:][::-1]
                        
                        top_5_data = []
                        for rank, idx in enumerate(top_5_indices, 1):
                            if idx < len(class_labels):
                                top_5_data.append({
                                    'Rank': rank,
                                    'File Type': class_labels[idx].upper(),
                                    'Probability': f"{pred_info['probabilities'][idx]*100:.4f}%",
                                    'Score': pred_info['probabilities'][idx]
                                })
                        
                        if top_5_data:
                            df_top5 = pd.DataFrame(top_5_data)
                            st.dataframe(df_top5.style.format({'Score': '{:.6f}'}), use_container_width=True, hide_index=True)
                        st.divider()
                
                st.markdown("---")
                
                # Get most confident prediction
                best_pred = max(predictions.items(), key=lambda x: x[1]['confidence'])
                
                st.markdown("### 🎯 Most Confident Prediction")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"""
                    **Model:** {best_pred[0].replace('_', ' ').upper()}
                    
                    **Predicted Type:** `{best_pred[1]['predicted_class'].upper()}`
                    
                    **Confidence:** {best_pred[1]['confidence']*100:.2f}%
                    """)
                    
                    # Color based on confidence
                    if best_pred[1]['confidence'] > 0.8:
                        st.success("✅ High confidence prediction!")
                    elif best_pred[1]['confidence'] > 0.6:
                        st.info("ℹ️ Moderate confidence prediction")
                    else:
                        st.warning("⚠️ Low confidence - consider results with caution")
                
                with col2:
                    # Confidence gauge chart
                    confidence_score = best_pred[1]['confidence']
                    fig_gauge = go.Figure(go.Indicator(
                        mode="gauge+number+delta",
                        value=confidence_score * 100,
                        title={'text': "Confidence %"},
                        domain={'x': [0, 1], 'y': [0, 1]},
                        gauge={
                            'axis': {'range': [0, 100]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 30], 'color': "lightgray"},
                                {'range': [30, 70], 'color': "gray"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 60
                            }
                        }
                    ))
                    fig_gauge.update_layout(height=300, width=300)
                    st.plotly_chart(fig_gauge, use_container_width=False, key='confidence_gauge')
                
                st.markdown("---")
                
                # Show all predictions with visualization
                st.markdown("#### 📈 Confidence Comparison Across Models")
                
                # Bar chart of predictions
                fig_confidence = go.Figure()
                
                fig_confidence.add_trace(go.Bar(
                    x=df_predictions['Model'],
                    y=df_predictions['Confidence Score'],
                    text=df_predictions['Confidence'],
                    textposition='auto',
                    marker=dict(
                        color=df_predictions['Confidence Score'],
                        colorscale='RdYlGn',
                        showscale=True,
                        colorbar=dict(title="Confidence")
                    ),
                    hovertemplate='<b>%{x}</b><br>Confidence: %{text}<extra></extra>'
                ))
                
                fig_confidence.update_layout(
                    title="Prediction Confidence by Model",
                    xaxis_title="Model",
                    yaxis_title="Confidence Score",
                    height=400,
                    template='plotly_white',
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig_confidence, use_container_width=True, key='confidence_comparison')
                
                # Detailed predictions for each model
                st.markdown("#### 🔬 Detailed Prediction Results")
                
                tabs = st.tabs([model_name.upper() for model_name in predictions.keys()])
                
                for idx, (model_name, pred_info) in enumerate(predictions.items()):
                    with tabs[idx]:
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            st.markdown(f"**Predicted File Type:** `{pred_info['predicted_class'].upper()}`")
                            st.markdown(f"**Confidence Score:** {pred_info['confidence']:.4f} ({pred_info['confidence']*100:.2f}%)")
                            
                            # Get top 5 predictions
                            top_indices = np.argsort(pred_info['probabilities'])[-5:][::-1]
                            
                            st.markdown("**Top 5 Predictions:**")
                            top_pred_data = []
                            for rank, idx in enumerate(top_indices, 1):
                                if idx < len(class_labels):
                                    top_pred_data.append({
                                        'Rank': rank,
                                        'File Type': class_labels[idx].upper(),
                                        'Probability': f"{pred_info['probabilities'][idx]*100:.2f}%",
                                        'Score': pred_info['probabilities'][idx]
                                    })
                            
                            if top_pred_data:
                                df_top = pd.DataFrame(top_pred_data)
                                st.dataframe(
                                    df_top.style.format({}),
                                    use_container_width=True,
                                    hide_index=True
                                )
                        
                        with col2:
                            # Mini pie chart for top predictions
                            if top_pred_data:
                                labels_top = [row['File Type'] for row in top_pred_data]
                                scores_top = [row['Score'] for row in top_pred_data]
                                
                                fig_pie = go.Figure(data=[go.Pie(
                                    labels=labels_top,
                                    values=scores_top,
                                    textposition='inside',
                                    textinfo='label+percent'
                                )])
                                
                                fig_pie.update_layout(height=300, width=300)
                                st.plotly_chart(fig_pie, use_container_width=False, key=f"pie_{model_name}")
                
                st.markdown("---")
                
                # Voting summary
                st.markdown("#### 🗳️ Model Voting Summary")
                
                # Get consensus (majority vote)
                predictions_list = [pred_info['predicted_class'] for pred_info in predictions.values()]
                from collections import Counter
                vote_counts = Counter(predictions_list)
                
                consensus_label = vote_counts.most_common(1)[0][0]
                consensus_votes = vote_counts.most_common(1)[0][1]
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("🗳️ Consensus Prediction", consensus_label.upper())
                with col2:
                    st.metric("✅ Models in Agreement", f"{consensus_votes}/{len(predictions)}")
                with col3:
                    agreement_pct = (consensus_votes / len(predictions)) * 100
                    st.metric("📊 Agreement %", f"{agreement_pct:.1f}%")
                
                # Show all predictions side by side
                st.markdown("**All Model Predictions:**")
                
                predictions_display = []
                for model_name, pred_info in predictions.items():
                    predictions_display.append({
                        'Model': model_name.replace('_', ' ').upper(),
                        'Prediction': pred_info['predicted_class'].upper(),
                        'Confidence': f"{pred_info['confidence']*100:.1f}%"
                    })
                
                st.dataframe(pd.DataFrame(predictions_display), use_container_width=True, hide_index=True)
            
            else:
                st.error("❌ Could not generate predictions. Please check if file is valid.")
        
        else:
            st.error("❌ Models not properly loaded. Cannot make predictions.")
    
    else:
        st.info("👆 Upload a file above to get started with predictions!")
        
        # Show available models
        st.markdown("#### 📦 Available Models for Prediction")
        col1, col2, col3, col4 = st.columns(4)
        cols = [col1, col2, col3, col4]
        
        model_info = []
        for model_key in sorted(models_data.keys()):
            model_data = models_data[model_key]
            accuracy = model_data.get("accuracy", 0)
            f1 = model_data.get("f1_score", 0)
            model_info.append({
                'Model': model_key.upper(),
                'Accuracy': f"{accuracy*100:.2f}%",
                'F1 Score': f"{f1*100:.2f}%"
            })
        
        st.dataframe(pd.DataFrame(model_info), use_container_width=True, hide_index=True)


st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray; font-size: 12px; margin-top: 30px;'>
    📊 File Type Identification - Model Comparison Dashboard | Built with Streamlit & Plotly
    </div>
    """,
    unsafe_allow_html=True
)
