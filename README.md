# A Multimodal Deep Learning Approach for Automated Analysis of Caregive-Child Interactions: A Comparative Study of Spatiotemporal Architectures.

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

This document presents a comprehensive deep learning framework for automated Baby joke classification in a short video, comparing four different neural network architectures `(Spatiotemporal)`: `CNN-LSTM`, `3D-CNN`, `Temporal Attention Network (TAN)`, and a `novel Multimodal` approach focusing on behavioral features. The framework processes 1,417 video samples across 5 joke types using a robust 10-fold cross-validation strategy.


### Key Contributions
- **Comparative Analysis**: Systematic evaluation of four different deep learning architectures
- **Behavioral Focus**: Novel multimodal approach using motion, facial, and gesture features
- **Robust Evaluation**: 10-fold stratified cross-validation with comprehensive metrics
- **Production-Ready**: Optimized for MacBook M3 MPS with efficient memory management

---

## Dataset Characteristics (BabyJoke)

### Video Dataset
- **Total Valid Videos**: 1,417 samples
- **Frame Sampling**: 20 frames per video (uniform temporal sampling)
- **Resolution**: 224×224 pixels (standardized and based on the back born architecture)
- **Duration**: Variable (uniformly sampled across temporal dimension)
- **Formats**: MP4
- **Quality**: High-definition video content

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

### 2. Data Augmentation
- **Brightness Range**: $\pm$0.2
- **Contrast Range**: (0.8, 1.2)
- **Saturation Range**: (0.8, 1.2)
- **Hue Range**: $\pm$0.1
- **Horizontal Flip**: Enabled (50% probability)
- **Rotation Range**: $\pm$15 degrees
- **Augmentation Probability**: 0.5

### 3. Normalization
- **Pixel Values**: Normalized to [0, 1] range
- **Feature Scaling**: Standard normalization for extracted features
- **Temporal Alignment**: Uniform sequence length (20 frames)

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

**B. Cross-Modal Fusion**
```python
def _build_cross_modal_fusion(self, motion_features, facial_features, gesture_features):
    # Temporal pooling to fixed-size vectors
    motion_vec = GlobalAveragePooling1D()(motion_features)
    facial_vec = GlobalAveragePooling1D()(facial_features)
    gesture_vec = GlobalAveragePooling1D()(gesture_features)
    
    # Concatenate pooled vectors
    fused_vec = Concatenate()([motion_vec, facial_vec, gesture_vec])
    
    # Gating mechanism
    gate = Dense(int(fused_vec.shape[-1]), activation='sigmoid')(fused_vec)
    gated = Multiply()([fused_vec, gate])
    
    # Fusion projection
    fused_features = Dense(256, activation='relu')(gated)
    fused_features = BatchNormalization()(fused_features)
    fused_features = Dropout(0.3)(fused_features)
    
    return tf.expand_dims(fused_features, axis=1)
```

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
```

Outputs are saved to `outputs/`:
- Per-model folders: `CNN_LSTM/`, `3D_CNN/`, `TAN/`, `Multimodal/`
- Global CSV: `outputs/all_models_performance.csv` (40 rows across 4 models × 10 folds)

Note: `data/` is git-ignored. Place your videos in `data/videos/` and ensure metadata path in `config/config.py`.

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

- Cross-validation: Stratified 10-fold; predictions aggregated across folds produce a full-dataset confusion matrix (each sample is predicted exactly once as part of its validation fold).
- Seeds: numpy=44342, tensorflow=4254, python=4422, sklearn=42, cv_random_state=3443.
- Label smoothing: ε = 0.05; Optimizer: Adam(lr=5e-4, β1=0.9, β2=0.999, ε=1e-7, clipnorm=1.0).
- Scheduler: ReduceLROnPlateau(factor=0.5, patience=8, min_lr=1e-7); Early stopping: patience=25.

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