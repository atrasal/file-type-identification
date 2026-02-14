# File Type Identification Using Machine Learning

A machine learning system that identifies file types from raw binary fragments — without relying on file headers, footers, or extensions. Useful for digital forensics, data recovery, and malware analysis.

## Supported File Types

| Category | Types |
|---|---|
| Archives | 7zip, APK |
| Documents | PDF, RTF, XLSX |
| Audio/Video | MP3, MP4 |
| Images | TIF |
| Code | CSS, HTML, JavaScript, JSON |
| Executables | ELF, BIN |

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
│   ├── random_forest/train.py
│   ├── cnn/train.py
│   ├── xgboost/train.py
│   └── resnet/train.py
├── utils/
│   └── data_loader.py          # Shared data loading utilities
├── saved_models/               # Trained model checkpoints
├── results/                    # Training metrics (JSON)
├── predict_input/              # Drop .bin files here for prediction
├── predict.py                  # Inference script
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
# Train any or all models
python models/random_forest/train.py
python models/xgboost/train.py
python models/cnn/train.py
python models/resnet/train.py
```

Models are saved to `saved_models/` and metrics to `results/`.

### 5. Predict

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

**Available models:** `rf`, `xgboost`, `cnn`, `resnet`, `all`

## Models

| Model | Type | Framework |
|---|---|---|
| Random Forest | Ensemble | scikit-learn |
| XGBoost | Gradient Boosting | xgboost |
| CNN | 1D Convolutional Neural Network | PyTorch |
| ResNet | 1D Residual Network | PyTorch |

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
- **Split**: Stratified by file type to ensure balanced representation

## Requirements

- Python 3.10+
- macOS / Linux
- ~4GB RAM for training (more for full dataset)

## License

MIT
