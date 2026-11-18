# BabyJoke video classification: comparative spatiotemporal architectures

## Table of Contents
1. [Overview](#overview)
2. [Dataset Characteristics](#dataset-characteristics)
3. [Data Preprocessing Pipeline](#data-preprocessing-pipeline)
4. [Model Architectures](#model-architectures)
5. [Feature Extraction Methods](#feature-extraction-methods)
6. [Training Framework](#training-framework)
7. [Evaluation Methodology](#evaluation-methodology)
8. [Performance Optimization](#performance-optimization)
9. [Workflow Diagram](#workflow-diagram)
10. [Implementation Details](#implementation-details)

---

## Overview

This repository compares four architectures for **five-class joke-type video classification** on BabyJoke clips: `CNN-LSTM`, `3D-CNN`, `Temporal Attention Network (TAN)`, and a multi-cue motion–face–gesture model. After dropping 13 duplicate `VideoID` rows, the evaluation cohort is **1,404 clips from 88 infants**.

Conclusions are limited to joke-type recognition under infant-held-out CV. Responsive-caregiving measurement is out of scope unless separately validated (see `SUBMISSION_DECLARATIONS.md`).

### Key contributions
- Comparative evaluation of four spatiotemporal / multi-cue architectures
- Multi-cue fusion with cross-modal attention, gating, and cue/fusion ablations
- Nested **ChildID-grouped** 10-fold CV (`config/shared_fold_indices.json`, seed 3443)
- Committed reference fold artifacts in `analysis/reference_results/` (same schema as live `outputs/`)

---

## Dataset Characteristics (BabyJoke)

### Video Dataset
- **Cohort after VideoID dedup**: 1,404 clips / 88 infants (13 duplicate VideoIDs dropped; see `config/cohort/`)
- **Frame sampling (implemented)**: 20 frames per video via uniform `np.linspace` indices
- **Resolution**: 224×224
- **Formats**: MP4 under `data/videos/`

### Joke Type Classes (5 Classes) and Description

| Class ID | Joke Type | Description |
|----------|-----------|-------------|
| 0 | NomNomNom | Food-related humor |
| 1 | Peekaboo | Hide-and-seek humor |
| 2 | TearingPaper | Paper-tearing actions |
| 3 | ThatsNotACat | Misidentification humor |
| 4 | ThatsNotAHat | Object confusion humor |

### Metadata Structure
- **VideoID**: Unique identifier for each video
- **JokeType**: Ground truth categorical label
- **Total Seconds**: Video duration in seconds
- **Additional Fields**: Filtered and sorted metadata

---

## Data Preprocessing Pipeline

### 1. Video Frame Extraction
```python
# Uniform temporal sampling
def sample_frames(video_path, max_frames=20, target_size=(224, 224)):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)
    # Extract and resize frames
```

### 2. Augmentation (config defaults; main runners currently pass `augmented=False`)
Config still lists brightness/contrast/flip options in `config/config.py`. The main `scripts/run_*.py` paths extract features with **augmentation off** unless you enable it explicitly.

### 3. Normalization
- Pixel values scaled to [0, 1] where RGB volumes are used
- Sequence length fixed at 20 frames

---

## Model Architectures

### 1. CNN-LSTM Architecture

#### Input Processing
- **Input Shape**: `(batch_size, 20, 2048)`
- **Data Source**: RGB frames → ResNet50 feature extraction → temporal sequences
- **Preprocessing**: ResNet50 backbone (pre-trained on ImageNet)

#### Architecture Components

**A. Multi-Scale CNN Feature Extraction**
```python
# Multi-scale convolutions
conv1 = Conv1D(64, 3, activation='relu', padding='same')(inputs)
conv2 = Conv1D(128, 5, activation='relu', padding='same')(inputs)
# Concatenation of multi-scale features
x = Concatenate()([conv1, conv2])
x = MaxPooling1D(2)(x)
```

**B. Temporal Encoder (Bidirectional LSTM)**
```python
# LSTM layers with bidirectional processing
x = Bidirectional(LSTM(512, return_sequences=True))(x)
x = Bidirectional(LSTM(256, return_sequences=True))(x)
x = Bidirectional(LSTM(128, return_sequences=True))(x)
x = LayerNormalization()(x)
```

**C. Self-Attention Mechanism**
```python 
# Multi-head self-attention
attention = MultiHeadAttention(
    num_heads=8, 
    key_dim=64
)(x, x)
x = Add()([attention, x])  # Residual connection
x = LayerNormalization()(x)
```

**D. Classification Head**
```python
# Dense layers with residual connections
x = Dense(256, activation='relu')(x)
x = Dense(128, activation='relu')(x)
x = Dense(64, activation='relu')(x)
outputs = Dense(5, activation='softmax')(x)
```

#### Key Features
- Multi-scale temporal feature extraction
- Bidirectional temporal modeling
- Self-attention for temporal dependencies
- Residual connections for gradient flow
- Layer normalization for training stability

---

### 2. 3D-CNN Architecture

#### Input Processing
- **Input Shape**: `(batch_size, 20, 224, 224, 3)`
- **Data Source**: Raw video frames (no pre-extraction or use of Pre-trained model)
- **Channels**: RGB (3 channels)

#### Architecture Components

**A. 3D CNN Backbone**
```python
# 3D convolutional layers
x = Conv3D(64, (3, 7, 7), strides=(1, 2, 2), padding='same')(inputs)
x = BatchNormalization()(x)
x = SpatialDropout3D(0.1)(x)

# Residual blocks
for filters in [128, 256, 512]:
    x = self._residual_block_3d(x, filters)
```

**B. Residual Connections**
```python
def _residual_block_3d(self, x, filters):
    residual = x
    x = Conv3D(filters, (3, 3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv3D(filters, (3, 3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Add()([x, residual])  # Skip connection
    return Activation('relu')(x)
```

**C. Spatial-Temporal Attention**
```python
# Global average pooling for spatial dimensions
x = GlobalAveragePooling3D()(x)
# Multi-head attention for temporal modeling
attention = MultiHeadAttention(num_heads=8, key_dim=64)(x, x)
x = Add()([attention, x])
```

**D. Classification Head**
```python
# Dense layers
x = Dense(1024, activation='relu')(x)
x = Dense(512, activation='relu')(x)
x = Dense(256, activation='relu')(x)
x = Dense(128, activation='relu')(x)
outputs = Dense(5, activation='softmax')(x)
```

#### Key Features
- Direct 3D spatiotemporal processing
- Residual connections for deep learning
- Spatial-temporal attention mechanism
- End-to-end learning from raw pixels
- Batch normalization for training stability

---

### 3. Temporal Attention Network (TAN) Architecture

#### Input Processing
- **Input Shape**: `(batch_size, 20, 2048)`
- **Data Source**: RGB frames → ResNet50 feature extraction → temporal sequences
- **Positional Encoding**: Sinusoidal encoding for temporal awareness

#### Architecture Components

**A. Positional Encoding**
```python
def _add_positional_encoding(self, inputs):
    seq_len = 20  # Fixed sequence length
    feature_dim = 2048  # Feature dimension
    # Precomputed sinusoidal positional encoding
    pos_encoding = self._get_positional_encoding(seq_len, feature_dim)
    return Add()([inputs, pos_encoding])
```

**B. Multi-Layer Transformer Blocks (4 blocks)**
```python
for _ in range(4):  # 4 transformer blocks
    # Multi-head self-attention
    attention = MultiHeadAttention(
        num_heads=8, 
        key_dim=64
    )(x, x)
    x = Add()([attention, x])  # Residual connection
    x = LayerNormalization()(x)
    
    # Feed-forward network
    ffn = Dense(512, activation='relu')(x)
    ffn = Dense(2048)(ffn)
    x = Add()([ffn, x])  # Residual connection
    x = LayerNormalization()(x)
```

**C. Global Attention Pooling**
```python
def _build_global_attention_pooling(self, x):
    # Compute attention weights
    attention_weights = Dense(1, activation='softmax')(x)
    # Weighted temporal aggregation
    pooled = Multiply()([x, attention_weights])
    return GlobalAveragePooling1D()(pooled)
```

**D. Classification Head**
```python
# Dense layers with residual connections
x = Dense(512, activation='relu')(x)
x = Dense(256, activation='relu')(x)
x = Dense(128, activation='relu')(x)
outputs = Dense(5, activation='softmax')(x)
```

#### Key Features
- Transformer-based architecture
- Positional encoding for temporal awareness
- Multi-head attention for complex dependencies
- Global attention pooling for sequence summarization
- Precomputed constants to avoid symbolic tensor operations

---

### 4. Multimodal Architecture (Behavioral Features)

#### Input Processing
- **Motion Input**: `(batch_size, 20, 8)` - Optical flow features
- **Facial Input**: `(batch_size, 20, 15)` - MediaPipe Face Mesh features
- **Gesture Input**: `(batch_size, 20, 45)` - MediaPipe Hands features

#### Architecture Components

**A. Modality-Specific Encoders**

**Motion Encoder:**
```python
# Dense layers for motion features
x = Dense(64, activation='relu')(motion_input)
x = BatchNormalization()(x)
x = Dropout(0.2)(x)
x = Dense(128, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.2)(x)
# LSTM for temporal modeling
x = Bidirectional(LSTM(64, return_sequences=True))(x)
x = LayerNormalization()(x)
```

**Facial Encoder:**
```python
# Similar structure to motion encoder
x = Dense(64, activation='relu')(facial_input)
x = BatchNormalization()(x)
x = Dropout(0.2)(x)
x = Dense(128, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.2)(x)
x = Bidirectional(LSTM(64, return_sequences=True))(x)
x = LayerNormalization()(x)
```

**Gesture Encoder:**
```python
# Larger capacity for gesture features
x = Dense(128, activation='relu')(gesture_input)
x = BatchNormalization()(x)
x = Dropout(0.2)(x)
x = Dense(256, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.2)(x)
x = Bidirectional(LSTM(128, return_sequences=True))(x)
x = LayerNormalization()(x)
```

**B. Cross-modal fusion (default = full)**
Pairwise multi-head cross-attention among motion/face/hands, temporal pooling, concatenation, sigmoid gating, then projection. Ablation switches (`concat`, `attention_only`, `gating_only`, cue-alone) are selected via `--ablation` / `ablation_variant`.

**C. Classification Head**
```python
# Dense layers with residual connections
x = Dense(256, activation='relu')(fused_features)
x = Dense(128, activation='relu')(x)
x = Dense(64, activation='relu')(x)
outputs = Dense(5, activation='softmax')(x)
```

#### Key Features
- Behavioral feature extraction (motion, facial, gesture)
- Modality-specific temporal modeling
- Cross-modal fusion with gating mechanism
- No RGB visual features (pure behavioral focus)
- MediaPipe integration for robust feature extraction

---

## Quickstart

Setup and run the full pipeline:
```
make setup
make all
```

Run individual models:
```
python scripts/run_cnn_lstm.py
python scripts/run_3d_cnn.py
python scripts/run_tan.py
python scripts/run_multimodal.py
python scripts/run_multimodal_ablation.py
```

Outputs under `outputs/` (gitignored) follow the analysis schema:
- `shared_fold_indices.json` (installed from `config/shared_fold_indices.json`)
- `all_models_performance.csv`
- `multi-cue|3d_cnn|tan|cnn_lstm/fold_*_predictions.npz`

Note: `data/` is git-ignored. Place videos in `data/videos/` and set the metadata path in `config/config.py`.

---

## Repository Structure

```
config/                  # Central configuration (StandardConfig)
models/                  # CNN_LSTM, 3D_CNN, TAN, Multimodal
src/                     # Cross-validation, data processing, results 

management

scripts/                 # Entry points to run each model
analysis/                # Notebook + figures/ and outputs/ (ignored by git)
outputs/                 # Per-model artifacts + global CSV (ignored by git)
data/                    # Your dataset (ignored by git; includes .gitkeep)
Makefile                 # Convenience targets (setup, all, analysis, clean)
README.md                # This document
```

---

## Reproducibility

- Cross-validation: nested ChildID `StratifiedGroupKFold` (outer test untouched; inner monitor for early stopping / LR schedule). Frozen splits: `config/shared_fold_indices.json`.
- Metrics: macro-averaged precision/recall/F1 are primary; weighted counterparts are also stored.
- Seeds: numpy=44342, tensorflow=4254, python=4422, sklearn=42, cv_random_state=3443 (initialized in each `scripts/run_*.py`).
- Optimizer: Adam(lr=5e-4, β1=0.9, β2=0.999, ε=1e-7, clipnorm=1.0); label smoothing ε=0.05 where configured.
- Scheduler: ReduceLROnPlateau(factor=0.5, patience=8, min_lr=1e-7) on **inner** val loss; EarlyStopping patience=15 on **inner** val accuracy.

---

## Running Analyses

Re-run the statistical analysis notebook and save outputs:
```
make analysis
```
Artifacts are written to `analysis/figures/` and `analysis/outputs/`.

---

## Troubleshooting

- MediaPipe on macOS ARM: Use `pip install mediapipe` inside your venv; Conda may not have ARM builds. If install fails, try a recent version like `pip install mediapipe==0.10.14`.
- TensorFlow MPS: Ensure an up-to-date TensorFlow. Mixed precision is enabled by default; the 3D CNN uses float32 where needed for MaxPool3D kernels.
- Cache cleanup: Run `make clean` to remove `data/cache/` and temporary artifacts.

---

## License & Citation

This repository is intended for academic research and reproducibility. 
Please cite the associated paper if you use this work. 

## Feature Extraction Methods
### 1. RGB Feature Extraction (CNN-LSTM, 3D-CNN, TAN)