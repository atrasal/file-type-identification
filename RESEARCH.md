# File Type Identification from Raw Binary Fragments Using Machine Learning

> A comprehensive technical documentation of the methodology, workflow, and design decisions behind our file fragment classification system for digital forensics.

---

## 1. Problem Statement

### 1.1 Background

In digital forensics, investigators frequently encounter storage media where file system metadata has been destroyed — whether through intentional tampering, hardware damage, or data corruption. Traditional file identification relies on:

- **File extensions** (`.pdf`, `.mp3`) — trivially spoofed or missing
- **Magic bytes / File signatures** — headers (`%PDF`, `PK\x03\x04`) and footers (`%%EOF`) that identify file types
- **File system metadata** — inode tables, FATs, MFTs that map files to sectors

When these are unavailable, the only remaining data is **raw binary content** scattered across disk sectors, with no indication of which file type each sector belongs to, or even which sectors belong to the same file.

### 1.2 Research Objective

**Can machine learning models accurately classify file types from raw binary fragments alone — without access to file extensions, headers, footers, or any metadata?**

This has direct applications in:
- **Data recovery** — identifying file content on corrupted drives
- **Forensic analysis** — reconstructing evidence from destroyed or wiped media
- **Malware analysis** — identifying disguised or embedded file types
- **Network forensics** — classifying intercepted data streams

### 1.3 Scope

We classify **4096-byte binary fragments** into **22 file types** across 6 categories:

| Category      | File Types               |
|---------------|---------------------------|
| Archives      | 7zip, APK, GZIP, TAR     |
| Documents     | PDF, RTF, PPTX, EPS      |
| Audio/Video   | MP3, MP4, SWF            |
| Images        | TIF, BMP, GIF            |
| Code/Text     | CSS, HTML, JavaScript, JSON |
| Executables   | ELF, BIN, EXE, DLL      |

---

## 2. Data Pipeline

### 2.1 Overview

```
Raw Files → Fragmenter → Header/Footer Cleaner → Stratified Splitter → Training Data
```

Each stage is designed to simulate the forensic scenario: the model should never see any structural clue that wouldn't exist in a random disk sector.

### 2.2 Stage 1: Fragmentation

**Script:** `datasets/scripts/fragmenter.py`

Raw files are split into fixed-size **4096-byte chunks** (matching the standard disk sector/page size). The fragmenter:

1. **Detects file signatures** using a database of known headers/footers for 20+ file types
2. **Strips headers and footers** before fragmenting, ensuring fragments contain only body content
3. **Discards undersized fragments** — the last chunk of a file is dropped if smaller than 4096 bytes to maintain uniform input size

**Why 4096 bytes?**
- Standard filesystem block size (ext4, NTFS, APFS)
- Represents the minimal unit a forensic tool would recover from a disk
- Large enough to contain statistical patterns, small enough for efficient training

**Why strip headers/footers?**
- Headers contain magic bytes (e.g., `%PDF`, `\x89PNG`) that trivially identify file types
- If the model learns to match magic bytes, it provides no value over simple signature matching
- Our goal is to classify from **content patterns** alone

### 2.3 Stage 2: Header/Footer Cleaning

**Script:** `datasets/scripts/clean_headers_footers.py`

A second pass that scans all generated fragments and removes any that still contain recognizable file signatures. This catches:

- Fragments that happen to start at a file boundary
- Embedded file headers within container formats
- Text-based structural markers (e.g., `<html>`, `<!DOCTYPE>`, `<?xml>`)

**Signature detection checks:**
- First 32 bytes of each fragment against 30+ known header signatures
- Last 32 bytes against 7+ known footer signatures
- Text-based markers for HTML, JavaScript, JSON, CSS

This ensures the training data is **completely free of trivially identifiable patterns**.

### 2.4 Stage 3: Dataset Splitting

**Script:** `datasets/scripts/split_dataset.py`

Fragments are split into three sets using **stratified sampling** to ensure each file type is proportionally represented:

| Set       | Ratio | Purpose                                           |
|-----------|-------|---------------------------------------------------|
| Training  | 70%   | Model training                                    |
| Validation| 15%   | Hyperparameter tuning, early stopping decisions    |
| Test      | 15%   | Final unbiased performance evaluation              |

**Stratification** ensures that if a file type has fewer samples (e.g., TIF), it gets proportional representation in all three sets rather than being accidentally excluded from test.

---

## 3. Feature Representation

### 3.1 Raw Byte Input (Deep Learning Models)

For CNN, ResNet, MLP, LeNet, and LSTM, each fragment is represented as:
- A **1D vector of 4096 floating-point values**
- Each byte value (0–255) is **normalized to [0, 1]** by dividing by 255
- Fed as a **single-channel 1D signal** (shape: `[1, 4096]`)

**Rationale:** Deep learning models can automatically discover relevant patterns through learned convolutional filters, eliminating the need for manual feature engineering.

### 3.2 Engineered Features (Tree-Based Models)

For Random Forest, XGBoost, and SVM, raw byte sequences are transformed into **317 hand-crafted features**:

| Feature Group               | Count | Description                                                 |
|-----------------------------|-------|-------------------------------------------------------------|
| **Byte frequency histogram**| 256   | Normalized frequency of each byte value (0–255)             |
| **Shannon entropy**         | 1     | Information-theoretic measure of byte randomness            |
| **Bigram frequencies**      | 20    | Top-20 most frequent consecutive byte-pair patterns         |
| **Statistical features**    | 10    | Mean, std, skewness, kurtosis, median, Q1, Q3, range, zero-byte ratio, ASCII printable ratio |
| **Block entropy**           | 16    | Shannon entropy computed over 16 equal sub-blocks of the fragment |
| **Longest runs**            | 2     | Longest consecutive same-byte run and longest zero-byte run (normalized) |
| **Chi-squared test**        | 1     | Byte distribution uniformity test — low = compressed/random, high = structured |
| **Compression ratio**       | 1     | zlib compressibility estimate — already-compressed data won't compress further |
| **Trigram frequencies**     | 10    | Top-10 most frequent consecutive 3-byte patterns            |

**Module:** `utils/feature_engineering.py`

**Why different representations?**

Tree-based models (Random Forest, XGBoost) make axis-aligned splits on individual features — they cannot learn spatial relationships between adjacent bytes. For example:

- A CNN can learn that byte pattern `[0x3C, 0x68, 0x74, 0x6D, 0x6C]` (`<html`) appearing anywhere in a fragment suggests HTML
- A Random Forest can only ask "Is byte at position 42 > 120?" — which is much less informative

By providing pre-computed features like byte histograms and entropy, we give tree-based models access to the same aggregate information that CNNs learn to compute through their filters.

#### 3.2.1 Byte Frequency Histogram (256 features)

Counts the normalized frequency of each possible byte value (0x00 through 0xFF). This captures the statistical fingerprint of a file type:

- **Text files** (HTML, CSS, JSON, JS): High concentration in the ASCII printable range (0x20–0x7E)
- **Compressed/encrypted files** (7zip, APK): Near-uniform distribution across all 256 byte values
- **PDF body content**: Mix of compressed streams and structural keywords
- **Audio data** (MP3): Characteristic distribution from audio frame encoding

#### 3.2.2 Shannon Entropy (1 feature)

$$H = -\sum_{i=0}^{255} p_i \log_2(p_i)$$

Where $p_i$ is the probability of byte value $i$ in the fragment.

| Entropy Range | Interpretation         | File Types           |
|---------------|------------------------|----------------------|
| 0–3 bits      | Highly structured      | Text: CSS, HTML, JSON |
| 3–6 bits      | Moderately random      | PDF, RTF, ELF        |
| 6–8 bits      | Near-random/compressed | 7zip, APK, MP3, MP4  |

Entropy is one of the single most discriminative features for file type classification.

#### 3.2.3 Byte-Pair (Bigram) Frequencies (20 features)

Captures the top-20 most frequent two-byte sequences. This reveals structural patterns:

- HTML fragments will have high `<d`, `iv`, `cl` bigrams
- JavaScript will have `fu`, `nc`, `ti`, `on` bigrams
- Compressed data will have no dominant bigrams (uniform distribution)

#### 3.2.4 Statistical Features (10 features)

| Feature                 | What It Captures                                          |
|-------------------------|-----------------------------------------------------------|
| Mean byte value         | Central tendency — text data centers around ASCII range    |
| Standard deviation      | Spread — compressed data has high std, text has low        |
| Skewness                | Asymmetry of byte distribution                            |
| Kurtosis                | Tail heaviness — detects unusual byte concentrations       |
| Median                  | Robust center measure                                     |
| Q1 (25th percentile)    | Lower distribution boundary                               |
| Q3 (75th percentile)    | Upper distribution boundary                               |
| Range (max - min)       | Full spread (usually 255 for most types)                  |
| Zero-byte ratio         | Proportion of 0x00 bytes — high in executables/padding    |
| ASCII printable ratio   | Proportion of bytes in 0x20–0x7E — high in text formats   |

> **Critical constraint:** All 317 features are derived **exclusively from raw byte content**. No metadata, file system information, headers, or footers are used. This maintains the core forensic constraint of the research.

#### 3.2.5 Block Entropy (16 features)

The fragment is divided into 16 equal sub-blocks (256 bytes each), and Shannon entropy is calculated for each. This reveals **internal structure variation**:

- **Compressed files**: Uniformly high entropy across all blocks (~7.5–8.0 bits)
- **Documents with mixed content**: Varying entropy — text sections have low entropy, embedded images have high
- **Executables**: Low entropy in code sections, high entropy in data sections

Block entropy helps distinguish file types that have similar global entropy but different internal structure.

#### 3.2.6 Longest Runs (2 features)

- **Longest same-byte run**: The longest consecutive sequence of the same byte value, normalized by fragment length
- **Longest zero-byte run**: The longest consecutive sequence of `0x00` bytes, normalized by fragment length

These detect **padding regions** (common in executables and archives) and **sparse data** patterns.

#### 3.2.7 Chi-Squared Randomness Test (1 feature)

Measures how far the byte distribution deviates from a uniform distribution:

$$\chi^2 = \sum_{i=0}^{255} \frac{(O_i - E_i)^2}{E_i}$$

- **Low χ²**: Near-uniform distribution → compressed/encrypted data
- **High χ²**: Concentrated distribution → structured text or code

This is particularly useful for distinguishing **compressed archives** (7zip, gzip, tar) from each other and from encrypted data.

#### 3.2.8 Compression Ratio (1 feature)

Attempts to compress the fragment using zlib (level 1, fast) and measures the ratio:

$$\text{ratio} = \frac{\text{compressed size}}{\text{original size}}$$

- **Ratio ≈ 1.0**: Already compressed (7zip, gzip, MP3) — cannot be compressed further
- **Ratio << 1.0**: Highly structured/repetitive (text, padding) — compresses well

#### 3.2.9 Trigram Frequencies (10 features)

Extends bigram analysis to 3-byte patterns. Returns the top-10 most frequent trigrams, capturing:
- Multi-byte structural keywords (e.g., HTML tags, variable declarations)
- Encoding patterns in media formats
- Instruction sequences in executables

---

## 4. Model Architectures

### 4.1 Overview

We evaluate **eight model architectures** spanning three paradigms:

| Paradigm              | Models                         | Input                          |
|-----------------------|--------------------------------|--------------------------------|
| **Traditional ML**    | Random Forest, XGBoost, SVM    | Engineered features (317 dims) |
| **Deep Learning (CNN)** | LeNet-1D, CNN, ResNet        | Raw bytes (4096 dims)          |
| **Deep Learning (RNN)** | LSTM                         | Raw bytes (256×16 sequence)    |
| **Deep Learning (FC)** | MLP                           | Raw bytes (4096 dims)          |

All models are adapted for **1D input** since binary fragments are sequential data, not images.

### 4.2 MLP (Multi-Layer Perceptron)

**Architecture:** `models/mlp/train.py`

```
Input (4096)
  → Linear(4096 → 512) → ReLU → Dropout(0.3)
  → Linear(512 → 256)  → ReLU → Dropout(0.3)
  → Linear(256 → 128)  → ReLU → Dropout(0.2)
  → Linear(128 → num_classes)
```

**Design rationale:**
- Simplest deep learning baseline — no convolutions, no recurrence
- Tests whether fully-connected layers can learn useful byte-level patterns
- Serves as a lower bound for deep learning performance — if CNN/ResNet don't beat MLP, spatial feature learning isn't helping

### 4.3 LeNet-1D

**Architecture:** `models/lenet/train.py`

```
Input (1, 4096)
  → Conv1d(1→6, kernel=5, pad=2) → ReLU → AvgPool1d(2)
  → Conv1d(6→16, kernel=5, pad=2) → ReLU → AvgPool1d(2)
  → Flatten()
  → Linear(16×1024 → 120) → ReLU
  → Linear(120 → 84) → ReLU
  → Linear(84 → num_classes)
```

**Design rationale:**
- Adapted from the classic LeNet-5 (LeCun et al., 1998) for 1D byte sequences
- Uses **average pooling** (original LeNet design) instead of max pooling
- Only 6 and 16 filters — a lightweight CNN baseline to compare with the deeper CNN and ResNet
- Shows whether a minimal CNN architecture is sufficient for byte-level classification

### 4.4 1D Convolutional Neural Network (CNN)

**Architecture:** `models/cnn/train.py`

```
Input (1, 4096)
  → Conv1d(1→64, kernel=5, pad=2) → ReLU → MaxPool1d(2)
  → Conv1d(64→128, kernel=3, pad=1) → ReLU → MaxPool1d(2)
  → Flatten()
  → Linear(128×1024 → 128) → ReLU → Dropout(0.3)
  → Linear(128 → num_classes)
```

**Design rationale:**
- **Kernel size 5** in the first layer captures byte-level 5-grams (common for detecting text patterns like HTML tags)
- **Kernel size 3** in the second layer captures higher-level pattern combinations
- **MaxPool1d(2)** reduces dimensionality by half at each stage, from 4096 → 2048 → 1024
- **Dropout(0.3)** prevents overfitting on the large training set

**What the filters learn:**
- First conv layer: Low-level byte patterns (2-5 byte sequences), similar to n-gram matching
- Second conv layer: Combinations of low-level patterns, analogous to word-level features

### 4.5 1D Residual Network (ResNet)

**Architecture:** `models/resnet/train.py`

```
Input (1, 4096)
  → Conv1d(1→64, kernel=7, stride=2, pad=3) → BatchNorm → ReLU → MaxPool1d(3, stride=2)
  → ResBlock(64→64, 2 blocks)
  → ResBlock(64→128, 2 blocks, stride=2)
  → ResBlock(128→256, 2 blocks, stride=2)
  → AdaptiveAvgPool1d(1)
  → Linear(256 → num_classes)
```

Each Residual Block:
```
x → Conv1d → BatchNorm → ReLU → Conv1d → BatchNorm → (+x) → ReLU
     └──────────── shortcut connection ────────────────┘
```

**Design rationale:**
- **Residual connections** allow training much deeper networks without vanishing gradients
- **Batch normalization** stabilizes training and allows higher learning rates
- **AdaptiveAvgPool1d(1)** produces a fixed-size output regardless of input length
- **Three residual stages** progressively increase channel depth (64→128→256) while reducing spatial dimensions

### 4.6 Bidirectional LSTM

**Architecture:** `models/lstm/train.py`

```
Input (1, 4096) → reshape to (256, 16)
  → Bidirectional LSTM(input=16, hidden=128, layers=2, dropout=0.3)
  → Concatenate forward + backward final hidden states
  → Dropout(0.3)
  → Linear(256 → num_classes)
```

**Design rationale:**
- Treats the 4096-byte fragment as a **sequence of 256 time steps**, each with 16-dimensional input
- **Bidirectional** processing captures both forward and backward byte dependencies
- **2 LSTM layers** allow the model to learn hierarchical temporal features
- Tests whether sequential (recurrent) modeling captures different patterns than spatial (convolutional) approaches
- LSTM can theoretically capture long-range dependencies across the entire 4096-byte fragment

### 4.7 Random Forest

**Architecture:** `models/random_forest/train.py`

- **100 decision trees** trained on random subsets of the 317 engineered features
- Each tree votes for a class, and the majority vote is the prediction
- **Class balancing**: `class_weight='balanced'` — automatically upweights minority classes
- **Feature importance** is intrinsically computed — reveals which features matter most

**Design rationale:**
- Provides interpretable baseline — feature importances show which byte patterns are most discriminative
- Class weighting compensates for severe imbalance (222:1 ratio between largest and smallest classes)
- Fast training and inference compared to deep learning
- Cannot learn spatial patterns from raw bytes — hence requires engineered features

### 4.8 XGBoost

**Architecture:** `models/xgboost/train.py`

- **Gradient-boosted ensemble** of 100 decision trees
- Max depth: 6, learning rate: 0.1
- **Early stopping** on validation set (patience: 10 rounds)
- **Class balancing**: Inverse-frequency sample weights computed from training distribution
- Trained on the same 317 engineered features as Random Forest

**Design rationale:**
- Gradient boosting focuses on correcting errors from previous trees — iteratively improves on hard-to-classify samples
- Early stopping prevents overfitting while maximizing performance
- Generally outperforms Random Forest on structured/tabular data

### 4.9 SVM (Support Vector Machine)

**Architecture:** `models/svm/train.py`

- **LinearSVC** with `CalibratedClassifierCV` for probability estimates
- Max iterations: 2000
- 3-fold cross-validation for calibration
- Trained on the same 317 engineered features

**Design rationale:**
- SVMs find the maximum-margin decision boundary — effective when features are well-engineered
- **Linear kernel** is computationally efficient for high-dimensional feature spaces (287 features)
- Calibration wrapper enables probability outputs for confidence scoring

---

## 5. Training Methodology

### 5.1 Training Configuration

| Parameter        | MLP / LeNet / CNN / ResNet / LSTM | RF          | XGBoost     | SVM           |
|------------------|----------------------------------|-------------|-------------|---------------|
| Epochs/Rounds    | 30                               | N/A         | 100         | N/A           |
| Batch Size       | 64                               | Full        | Full        | Full          |
| Learning Rate    | 0.001 (adaptive)                 | N/A         | 0.1         | N/A           |
| LR Scheduler     | ReduceLROnPlateau (patience=2, factor=0.5) | N/A | N/A | N/A |
| Optimizer        | Adam                             | N/A         | Gradient Boost | N/A         |
| Early Stopping   | ✅ (patience=5)                  | ❌          | ✅ (patience=10) | ❌        |
| Loss Function    | Weighted CrossEntropy            | Gini (balanced) | Multi-class LogLoss | Hinge |
| Class Balancing  | Inverse-frequency class weights  | `class_weight='balanced'` | Sample weights | — |
| Device           | MPS/CPU                          | CPU         | CPU         | CPU           |

### 5.2 Memory-Efficient Training

With 1M+ fragments (each 4096 bytes), naive loading requires ~11 GB RAM. We use three strategies:

- **float32** instead of float64 (halves memory)
- **Batch feature extraction** (RF, XGBoost, SVM): loads 50K fragments at a time, extracts features, frees raw data
- **Lazy disk loading** (MLP, LeNet, CNN, ResNet, LSTM): `LazyFragmentDataset` reads fragments from disk on-the-fly via PyTorch `DataLoader`

### 5.3 Class Imbalance Handling

The dataset has severe class imbalance (up to 222:1 ratio between pptx at 100K samples and html at 316 samples). Without mitigation, models learn to predict majority classes and ignore minority ones.

**Approach for DL models:** Class weights are computed as inverse class frequency and passed to `CrossEntropyLoss`. This makes misclassifying a rare class (e.g., CSS, HTML) proportionally more costly than misclassifying a common one (e.g., pptx, tar):

$$w_c = \frac{N}{C \times n_c}$$

Where $N$ = total samples, $C$ = number of classes, $n_c$ = samples in class $c$.

**Approach for RF:** `class_weight='balanced'` applies the same inverse-frequency weighting internally.

**Approach for XGBoost:** Per-sample weights derived from inverse class frequency are passed to `model.fit()`.

### 5.4 Learning Rate Scheduling

All DL models use `ReduceLROnPlateau` with `patience=2` and `factor=0.5`. If validation loss does not improve for 2 consecutive epochs, the learning rate is halved. This allows:
- Aggressive initial learning (LR = 0.001)
- Fine-grained convergence as the model plateaus
- Better final performance compared to a fixed learning rate

### 5.5 Early Stopping Strategy

For all deep learning models (MLP, LeNet, CNN, ResNet, LSTM), training is monitored against **validation loss**. If validation loss fails to improve for 5 consecutive epochs, training stops and the model weights are reverted to the best checkpoint.

This prevents **overfitting** — the model stops learning general patterns and starts memorizing training-specific noise.

### 5.4 Evaluation Metrics

All models are evaluated on the held-out **test set** using:

| Metric     | Computation               | What It Measures                                     |
|------------|--------------------------|------------------------------------------------------|
| Accuracy   | Correct / Total           | Overall correctness                                  |
| Precision  | TP / (TP + FP), macro avg | How often the model is right when it predicts a type |
| Recall     | TP / (TP + FN), macro avg | How often the model finds all instances of a type    |
| F1 Score   | 2 × (P × R) / (P + R)    | Harmonic mean of precision and recall                |

**Macro averaging** is used to give equal weight to each file type, regardless of how many samples it has. This prevents high-frequency classes from dominating the metrics.

### 5.5 Per-Class Analysis

Each model records:
- **Per-class precision, recall, F1, and support** (sample count) — revealing which file types are easy/hard to classify
- **Confusion matrix** — showing exactly which types get confused with each other

---

## 6. Results and Analysis

### 6.1 Model Performance Summary

Results from training on **22 file types** with **~1M total fragments** (712K train / 153K val / 153K test):

| Model | Accuracy | Precision | Recall | F1 (macro) | Val F1 | Train Time |
|---|---|---|---|---|---|---|
| **Random Forest** | **62.0%** | **81.1%** | 76.2% | **77.1%** | 77.3% | 3 min |
| **ResNet** | 58.5% | 74.9% | 77.8% | 75.4% | 75.4% | 20.7 hrs |
| **XGBoost** | 59.6% | 72.7% | **79.1%** | 74.8% | 74.7% | 5 min |
| CNN | 42.8% | 52.5% | 59.6% | 54.0% | 53.7% | 8.0 hrs |
| LeNet | 42.8% | 49.0% | 58.6% | 51.9% | 52.6% | 2.8 hrs |
| SVM | 42.2% | 61.8% | 47.6% | 46.6% | 46.6% | 1.2 hrs |
| LSTM | 40.9% | 42.0% | 57.4% | 42.9% | 43.3% | 19.3 hrs |
| MLP | 3.0% | 14.9% | 10.5% | 6.1% | 6.3% | 1.1 hrs |

### 6.2 Per-Class Performance (Random Forest — Best Model)

| File Type | F1 Score | Difficulty | Why |
|---|---|---|---|
| RTF | 0.998 | Easy | Distinctive control words and ASCII structure |
| JSON | 0.996 | Easy | Bracket/quote-heavy ASCII pattern |
| JavaScript | 0.993 | Easy | Keyword-rich ASCII with function/var patterns |
| EPS | 0.990 | Easy | PostScript operators and numeric data |
| BMP | 0.975 | Easy | Characteristic pixel data with fixed-range byte values |
| CSS | 0.964 | Easy | Selector/property ASCII patterns |
| HTML | 0.944 | Easy | Tag-heavy ASCII content |
| APK | 0.644 | Hard | ZIP-compressed — similar to PPTX, GZIP |
| PPTX | 0.619 | Hard | ZIP-compressed XML — confused with APK, 7ZIP |
| 7ZIP | 0.600 | Hard | Compressed — confused with GZIP, TAR |
| SWF | 0.429 | Very Hard | Compressed body — looks like other archives |
| TAR | 0.414 | Very Hard | Uncompressed but variable content from archived files |
| EXE | 0.393 | Very Hard | PE binary — confused with DLL, compressed sections |
| GZIP | 0.370 | Very Hard | Compressed — near-uniform bytes, indistinguishable from other archives |

### 6.3 Training Details (Deep Learning Models)

| Model | Epochs Run | Early Stopped | Best Val Loss |
|---|---|---|---|
| ResNet | 23 / 30 | ✅ (patience 5) | 1.067 |
| CNN | 20 / 30 | ✅ (patience 5) | 1.617 |
| LeNet | 19 / 30 | ✅ (patience 5) | 1.592 |
| MLP | 8 / 30 | ✅ (patience 5) | 2.888 |
| LSTM | 30 / 30 | ❌ (ran all epochs) | 1.633 |

### 6.4 Key Findings

1. **Feature engineering dominates** — Random Forest with 317 engineered features achieves the best F1 (77.1%) while training in only 3 minutes, outperforming all DL models that train for hours
2. **ResNet is the best DL model** — residual connections and batch normalization enable it to reach 75.4% F1, close to RF but requiring 20+ hours of training
3. **Class weighting improves minority class recall** — weighted loss functions increase recall on rare classes (CSS, HTML, ELF) at the cost of some raw accuracy on majority classes
4. **Compressed formats are fundamentally hard** — GZIP, TAR, SWF, and 7ZIP have F1 < 0.43 because their byte distributions are nearly identical (high entropy, near-uniform)
5. **Text formats are trivially separable** — HTML, CSS, JSON, JavaScript, and RTF achieve F1 > 0.94 due to distinctive ASCII patterns
6. **MLP fails on this task** — fully connected layers without spatial induction bias cannot learn useful byte-level patterns from raw 4096-dimensional input

### 6.5 Analysis

The fundamental insight is that **spatial locality matters in binary data**. A CNN can detect that bytes `[0x25, 0x50, 0x44, 0x46]` form the sequence `%PDF` even in the middle of a document — it learns position-invariant patterns through its sliding convolutional windows.

Tree-based models split on individual feature values independently and cannot capture multi-dimensional spatial patterns without explicit feature engineering. However, with well-crafted features (entropy, byte histograms, bigrams, block entropy, compression ratio), they achieve the strongest macro F1 scores.

The accuracy-vs-F1 gap (62% accuracy but 77% F1 for RF) reflects the class imbalance: macro F1 gives equal weight to all 22 classes, while accuracy is dominated by the majority classes. The model performs excellently on 14 of 22 classes but struggles on the 8 compressed/archive types that are statistically indistinguishable at the byte level.

---

## 7. Prediction Pipeline

**Script:** `predict.py`

The inference pipeline mirrors the training setup:

```
Input .bin file
  → Read raw bytes → Normalize (÷ 255)
  → If sklearn model: check if engineered features needed → extract 317 features
  → If PyTorch model: reshape to [1, 4096] tensor
  → Forward pass → Softmax → Top-K predictions with confidence
```

Supports single-file prediction, batch prediction, CSV output, and cross-model comparison.

---

## 8. Visualization and Analysis Tools

### Graph Generation

**Script:** `generate_graphs.py`

Produces 8 comparison charts:

| Graph                     | Purpose                                                  |
|---------------------------|----------------------------------------------------------|
| Model Comparison          | Bar chart of accuracy, precision, recall, F1             |
| Training Loss Curves      | Convergence visualization for CNN/ResNet                 |
| Confusion Matrices        | Normalized heatmaps showing class-level errors           |
| Per-Class F1              | Grouped bars revealing strong/weak file types per model  |
| Model Size vs Accuracy    | Trade-off between model complexity and performance       |
| Training Time vs Accuracy | Computational cost vs benefit                            |
| Dataset Distribution      | Class balance visualization                              |
| Summary Table             | Formatted metrics overview                               |

---

## 9. Technical Design Decisions

### 9.1 Why 1D Convolutions Instead of 2D?

Some research converts byte sequences into 2D images (e.g., 64×64 grayscale). We use **1D convolutions** because:

- Binary data is inherently **sequential**, not spatial — adjacent bytes have temporal but not spatial relationships
- 1D convolutions are more computationally efficient for 4096-length sequences
- 1D preserves the natural byte ordering without introducing artificial 2D structure

### 9.2 Why Not Use Pre-trained Models?

Pre-trained models (ImageNet, NLP transformers) are trained on natural images or text — domains with fundamentally different statistical properties from raw binary data. Transfer learning would provide no benefit here, and the input dimensionality doesn't match.

### 9.3 Why Adam Optimizer?

Adam combines momentum and adaptive learning rates, making it robust to:
- Varying gradient magnitudes across different byte patterns
- The diverse loss landscapes of multi-class classification
- Requires minimal hyperparameter tuning compared to SGD

### 9.4 Why Macro-Averaged Metrics?

With imbalanced class sizes, accuracy alone can be misleading — a model could achieve high accuracy by always predicting the majority class. Macro averaging computes metrics per-class first, then averages them equally, ensuring each file type contributes equally to the final score.

---

## 10. Future Work

### 10.1 Additional Models
- Vision Transformer (ViT) adapted for 1D byte sequences
- EfficientNet for accuracy/parameter trade-off
- Attention-based models for interpretable byte-level importance

### 10.2 Same-File Fragment Clustering
Beyond classifying file types, cluster fragments from the same original file using:
- Byte-level similarity matching (n-gram, cosine, Jaccard)
- Embedding-based clustering with contrastive learning

### 10.3 Dataset Expansion
- Additional file types (DOCX, PNG, WAV, etc.)
- Cross-dataset evaluation for robustness testing
- Adversarial testing with intentionally confusing fragments

---

## 11. Reproducibility

### Environment
- Python 3.10+
- macOS / Linux
- Hardware: MPS (Apple Silicon) or CUDA GPU recommended

### Dependencies
```
numpy, pandas, scikit-learn, torch, xgboost, joblib, matplotlib, seaborn, scipy
```

### Full Pipeline
```bash
# 1. Fragment raw files
python datasets/scripts/fragmenter.py --input datasets/raw --output datasets/fragments

# 2. Clean headers/footers
python datasets/scripts/clean_headers_footers.py --input datasets/fragments

# 3. Split dataset
python datasets/scripts/split_dataset.py

# 4. Train all models (one at a time to avoid OOM)
python models/random_forest/train.py
python models/xgboost/train.py
python models/svm/train.py
python models/mlp/train.py
python models/lenet/train.py
python models/cnn/train.py
python models/resnet/train.py
python models/lstm/train.py

# 5. Generate comparison graphs
python generate_graphs.py

# 6. Predict
python predict.py predict_input/ --model all
```

All results are reproducible with `random_state=42` / `seed=42` across all stochastic operations.
