"""
Model loading and prediction functions.
"""
import json
import numpy as np
from pathlib import Path
import streamlit as st
import torch
import joblib

from utils import extract_features


BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"
SAVED_MODELS_DIR = BASE_DIR / "saved_models"


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
    try:
        rf_path = SAVED_MODELS_DIR / "random_forest" / "random_forest_model.joblib"
        if rf_path.exists():
            loaded = joblib.load(rf_path)
            model = loaded.get('model', loaded) if isinstance(loaded, dict) else loaded
            models['random_forest'] = {'model': model, 'type': 'sklearn'}
    except Exception as e:
        pass
    
    # ==================== SVM (Sklearn) ====================
    try:
        svm_path = SAVED_MODELS_DIR / "svm" / "svm_model.joblib"
        if svm_path.exists():
            loaded = joblib.load(svm_path)
            if isinstance(loaded, dict):
                model = loaded.get('model', loaded.get('svm', loaded))
            else:
                model = loaded
            
            if hasattr(model, 'predict') or hasattr(model, 'predict_proba'):
                models['svm'] = {'model': model, 'type': 'sklearn'}
    except Exception as e:
        pass
    
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
            
            model = MLPClassifier(317, num_classes)
            checkpoint = torch.load(mlp_path, map_location="cpu")
            
            if isinstance(checkpoint, dict) and 'fc1.weight' in checkpoint:
                model.load_state_dict(checkpoint, strict=False)
            else:
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
            if isinstance(loaded, dict):
                model = loaded.get('model', loaded.get('xgboost', loaded.get('xgb', loaded)))
            else:
                model = loaded
            
            if hasattr(model, 'predict') or hasattr(model, 'predict_proba'):
                models['xgboost'] = {'model': model, 'type': 'sklearn'}
    except Exception as e:
        pass
    
    return models, labels


def predict_file(file_bytes, models, labels, cleaned_data=None):
    """
    Make predictions using all loaded models.
    Optionally uses cleaned_data (header/footer removed) for better accuracy.
    """
    if len(file_bytes) == 0:
        return None
    
    data_to_process = cleaned_data if cleaned_data is not None else file_bytes
    predictions = {}
    
    try:
        # Get raw bytes for PyTorch models
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
                    x = torch.FloatTensor(raw_bytes).unsqueeze(0).unsqueeze(0)
                    with torch.no_grad():
                        output = model(x)
                    probs = torch.softmax(output, dim=1)[0].numpy()
                
                elif model_type == 'mlp_pytorch':
                    x = torch.FloatTensor(features).unsqueeze(0)
                    with torch.no_grad():
                        output = model(x)
                    probs = torch.softmax(output, dim=1)[0].numpy()
                
                elif model_type == 'sklearn':
                    try:
                        probs = model.predict_proba(features.reshape(1, -1))[0]
                    except (AttributeError, TypeError):
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
                
                # Ensure correct length
                if len(probs) < num_classes:
                    probs = np.pad(probs, (0, num_classes - len(probs)), mode='constant')
                elif len(probs) > num_classes:
                    probs = probs[:num_classes]
                
                # Check for NaN or Inf
                if np.any(np.isnan(probs)) or np.any(np.isinf(probs)):
                    probs = np.ones(num_classes) / num_classes
                
                # Normalize
                prob_sum = np.sum(probs)
                if prob_sum > 0:
                    probs = probs / prob_sum
                else:
                    probs = np.ones(num_classes) / num_classes
                
                # Get prediction
                pred_idx = np.argmax(probs)
                pred_label = labels[pred_idx] if labels and pred_idx < len(labels) else f"Class {pred_idx}"
                confidence = float(probs[pred_idx])
                
                predictions[model_name] = {
                    'predicted_class': pred_label,
                    'confidence': confidence,
                    'probabilities': probs,
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
