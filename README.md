# File Type Identification Using Machine Learning

A machine learning system that identifies file types from raw binary fragments — without relying on file headers, footers, or extensions. Useful for digital forensics, data recovery, and malware analysis.

## Supported File Types (22)

| Category | Types |
|---|---|
| Archives | 7zip, APK, GZIP, TAR |
| Documents | PDF, RTF, PPTX, EPS |
| Audio/Video | MP3, MP4, SWF |
| Images | TIF, BMP, GIF |
| Code | CSS, HTML, JavaScript, JSON |
| Executables | ELF, BIN, EXE, DLL |

## Project Structure

```
forensics/
├── datasets/
│   ├── fragments/              # Source fragments (by file type)
│   ├── train/ val/ test/       # 70/15/15 split
│   └── scripts/
│       ├── fragmenter.py       # Convert raw files → fragments
│       ├── split_dataset.py    # Stratified train/val/test split
│       └── clean_headers_footers.py  # Remove fragments with signatures
├── models/
│   ├── random_forest/train.py  # Ensemble (sklearn)
│   ├── xgboost/train.py        # Gradient boosting (xgboost)
│   ├── svm/train.py            # Support Vector Machine (sklearn)
│   ├── cnn/train.py            # 1D CNN (PyTorch)
│   ├── resnet/train.py         # 1D ResNet (PyTorch)
│   ├── mlp/train.py            # Multi-Layer Perceptron (PyTorch)
│   ├── lenet/train.py          # LeNet-1D (PyTorch)
│   └── lstm/train.py           # Bidirectional LSTM (PyTorch)
├── utils/
│   ├── data_loader.py          # Shared data loading utilities
│   └── feature_engineering.py  # 317 engineered features
├── saved_models/               # Trained model checkpoints
├── results/                    # Training metrics (JSON) + graphs
├── predict_input/              # Drop .bin files here for prediction
├── predict.py                  # Inference script
├── generate_graphs.py          # Comparison charts
└── requirements.txt
```

## Getting Started

### 1. Clone and Install

```bash
git clone <your-repo-url>
cd forensics

# Create virtual environment
python3 -m venv venv
source venv/bin/activate   # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Data

Place your raw files in `datasets/raw/` organized by file type:

```
datasets/raw/
├── pdf/
│   ├── file1.pdf
│   └── file2.pdf
├── mp3/
│   ├── song1.mp3
│   └── song2.mp3
└── ...
```

**If you already have fragments**, place them in `datasets/fragments/<type>Fragments/` with a CSV label file in each subfolder.

### 3. Fragment and Split

```bash
# Fragment raw files (auto-detects and strips headers/footers)
python datasets/scripts/fragmenter.py --input datasets/raw --output datasets/fragments

# Clean any remaining headers/footers
python datasets/scripts/clean_headers_footers.py --input datasets/fragments

# Split into train/val/test (70/15/15)
python datasets/scripts/split_dataset.py
```

### 4. Train Models

```bash
# Traditional ML (use batch feature extraction + 317 engineered features)
python models/random_forest/train.py
python models/xgboost/train.py
python models/svm/train.py

# Deep Learning (class-weighted loss + LR scheduling, 30 epochs)
python models/mlp/train.py
python models/lenet/train.py
python models/cnn/train.py
python models/resnet/train.py
python models/lstm/train.py
```

> **Note:** Run models one at a time to avoid out-of-memory issues on 16 GB machines.

Models are saved to `saved_models/` and metrics to `results/`.

### 5. Generate Comparison Graphs

```bash
python generate_graphs.py
```

### 6. Predict

```bash
# Put .bin fragment files in predict_input/ folder, then:
python predict.py predict_input/ --model rf

# Or predict a single file:
python predict.py path/to/fragment.bin --model rf

# Compare all models:
python predict.py predict_input/ --model all

# Save results to CSV:
python predict.py predict_input/ --model rf --save-csv
```

**Available models:** `rf`, `xgboost`, `svm`, `cnn`, `resnet`, `mlp`, `lenet`, `lstm`, `all`

## Models

| Model | Type | Input | Class Balancing | Framework |
|---|---|---|---|---|
| Random Forest | Ensemble (100 trees) | 317 engineered features | `class_weight='balanced'` | scikit-learn |
| XGBoost | Gradient Boosting | 317 engineered features | Inverse-frequency sample weights | xgboost |
| SVM | Linear SVC + Calibration | 317 engineered features | — | scikit-learn |
| MLP | Fully Connected (512→256→128) | Raw bytes (4096) | Weighted CrossEntropyLoss | PyTorch |
| LeNet-1D | Classic CNN (2 conv + 3 FC) | Raw bytes (4096) | Weighted CrossEntropyLoss | PyTorch |
| CNN | 1D Convolutional (2 conv layers) | Raw bytes (4096) | Weighted CrossEntropyLoss | PyTorch |
| ResNet | 1D Residual Network (6 conv layers) | Raw bytes (4096) | Weighted CrossEntropyLoss | PyTorch |
| LSTM | Bidirectional 2-layer LSTM | Raw bytes (256×16 sequence) | Weighted CrossEntropyLoss | PyTorch |

### Training Enhancements

- **Class-weighted loss**: All models account for class imbalance (22 classes with up to 222:1 sample ratio)
- **LR scheduling**: DL models use `ReduceLROnPlateau` (halves LR after 2 stale epochs)
- **Extended training**: 30 epochs with patience 5 (up from 15/3) for DL models
- **317 engineered features**: Byte histograms, entropy, bigrams, trigrams, block entropy, compression ratio, chi-squared test, and more

## Results

Trained on **22 file types** with **~1M total fragments** (712K train / 153K val / 153K test).

| Model | Accuracy | Precision | Recall | F1 (macro) | Train Time |
|---|---|---|---|---|---|
| **Random Forest** | **62.0%** | **81.1%** | 76.2% | **77.1%** | 3 min |
| **XGBoost** | 59.6% | 72.7% | **79.1%** | 74.8% | 5 min |
| **ResNet** | 58.5% | 74.9% | 77.8% | 75.4% | 20.7 hrs |
| CNN | 42.8% | 52.5% | 59.6% | 54.0% | 8.0 hrs |
| LeNet | 42.8% | 49.0% | 58.6% | 51.9% | 2.8 hrs |
| SVM | 42.2% | 61.8% | 47.6% | 46.6% | 1.2 hrs |
| LSTM | 40.9% | 42.0% | 57.4% | 42.9% | 19.3 hrs |
| MLP | 3.0% | 14.9% | 10.5% | 6.1% | 1.1 hrs |

> **Best model:** Random Forest with 317 engineered features achieves 77.1% macro F1. Text-based formats (HTML, CSS, JSON) are classified near-perfectly (F1 > 0.96), while compressed archives (gzip, tar, swf) remain challenging (F1 < 0.43) due to near-identical byte distributions.

## Data Pipeline

```
Raw Files → Fragmenter → Cleaner → Splitter → Training → Prediction
              ↓              ↓          ↓
         4096-byte      Remove      70% train
         chunks        headers/     15% val
                       footers      15% test
```

- **Fragment size**: 4096 bytes (configurable)
- **Format**: Raw binary `.bin` files
- **Split**: Stratified random split by file type to ensure balanced representation
- **Memory optimized**: Uses float32, batch processing, and lazy disk loading for large datasets

## Requirements

- Python 3.10+
- macOS / Linux
- ~4 GB RAM for training (uses float32 and incremental memory management for large datasets)

## License

MIT

