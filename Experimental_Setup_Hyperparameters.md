# Experimental Setup and Hyperparameter Configuration

## Training Configuration and Hyperparameters

This section provides a comprehensive overview of the experimental setup, including all hyperparameters, training configurations, and reproducibility measures employed across all four deep learning architectures.

### Global Training Parameters

All models were trained using identical global parameters to ensure fair comparison:

**Dataset Configuration:**
- **Frame Sampling**: 20 frames per video (uniform temporal sampling)
- **Frame Resolution**: 224×224 pixels (standardized)
- **Number of Classes**: 5 joke categories (NomNomNom, Peekaboo, TearingPaper, ThatsNotACat, ThatsNotAHat)
- **Cross-Validation**: 10-fold stratified cross-validation
- **CV Random State**: 3443 (for reproducible splits)

**Training Hyperparameters:**
- **Batch Size**: 16 (optimized for M3 MPS performance)
- **Number of Epochs**: 150
- **Initial Learning Rate**: 0.0005
- **Optimizer**: Adam with β₁=0.9, β₂=0.999, ε=1e-7
- **Gradient Clipping**: 1.0 (L2 norm)
- **Weight Decay**: 1e-5 (L2 regularization)

**Learning Rate Scheduling:**
- **Reduction Factor**: 0.5
- **Reduction Patience**: 8 epochs
- **Minimum Learning Rate**: 1e-7
- **Early Stopping Patience**: 25 epochs (monitoring validation accuracy)

**Regularization Techniques:**
- **Label Smoothing**: ε = 0.05 (applied to categorical cross-entropy loss)
- **Dropout Rates**: Variable across layers (0.1-0.4)
- **Batch Normalization**: Applied after each dense/convolutional layer
- **Mixed Precision Training**: Enabled (float16) for memory efficiency

### Model-Specific Architectures and Hyperparameters

#### CNN-LSTM Architecture

**CNN Feature Extraction:**
- **Multi-scale Convolutions**: 3 parallel Conv1D layers (kernel sizes: 3, 5, 7)
- **Filters per Scale**: 64 filters each
- **Residual Blocks**: 3 blocks with 128 filters
- **Pooling**: MaxPooling1D with stride=2
- **Regularization**: L1=1e-5, L2=1e-4

**LSTM Configuration:**
- **Bidirectional LSTM**: 256 units (first layer), 128 units (second layer)
- **Dropout**: 0.3 (recurrent and input)
- **Return Sequences**: True (first layer), False (second layer)

**Attention Mechanism:**
- **Multi-head Attention**: 8 heads, key dimension=64
- **Feed-forward Dimension**: 512
- **Attention Dropout**: 0.2

**Classification Head:**
- **Dense Layers**: [512, 512, 128] units
- **Dropout Rates**: [0.4, 0.3, 0.2]
- **Residual Connections**: Between first two dense layers

#### Temporal Attention Network (TAN)

**Positional Encoding:**
- **Sequence Length**: 20 frames
- **Feature Dimension**: 2048
- **Encoding Type**: Sinusoidal positional encoding

**Transformer Blocks:**
- **Number of Blocks**: 4
- **Attention Heads**: 8 per block
- **Key Dimension**: 64
- **Feed-forward Dimension**: 512
- **Dropout**: 0.2 (attention and feed-forward)

**Global Attention Pooling:**
- **Pooling Methods**: Attention-weighted, Global Average, Global Max
- **Concatenation**: All three pooling outputs combined

**Classification Head:**
- **Dense Layers**: [512, 512, 128] units
- **Dropout Rates**: [0.4, 0.3, 0.2]
- **Residual Connections**: Between first two dense layers

#### 3D CNN Architecture

**3D Convolutional Backbone:**
- **Initial Conv3D**: 64 filters, kernel=(3,7,7), stride=(1,2,2)
- **Residual Blocks**: 4 blocks with [64, 128, 256, 512] filters
- **Kernel Size**: (3,3,3) for all residual convolutions
- **Pooling**: MaxPooling3D with stride=(1,2,2) (spatial only)

**Spatial-Temporal Attention:**
- **Attention Heads**: 8
- **Key Dimension**: 64
- **Temporal Pooling**: TimeDistributed GlobalAveragePooling2D

**Classification Head:**
- **Temporal Pooling**: Global Average + Global Max concatenation
- **Dense Layers**: [512, 512, 128] units
- **Dropout Rates**: [0.4, 0.3, 0.2]
- **Batch Size**: 8 (reduced for memory constraints)

#### Multimodal Architecture

**Modality-Specific Encoders:**

*Motion Encoder:*
- **Input Shape**: (20, 8) - optical flow features
- **Dense Layers**: [64, 128] units
- **LSTM**: Bidirectional, 64 units
- **Dropout**: 0.2 (dense), 0.3 (LSTM)

*Facial Encoder:*
- **Input Shape**: (20, 15) - MediaPipe Face Mesh features
- **Dense Layers**: [64, 128] units
- **LSTM**: Bidirectional, 64 units
- **Dropout**: 0.2 (dense), 0.3 (LSTM)

*Gesture Encoder:*
- **Input Shape**: (20, 45) - MediaPipe Hands features
- **Dense Layers**: [128, 256] units
- **LSTM**: Bidirectional, 128 units
- **Dropout**: 0.2 (dense), 0.3 (LSTM)

**Cross-Modal Fusion:**
- **Fusion Method**: Concatenation with gating mechanism
- **Fusion Dimension**: 256
- **Gating**: Sigmoid activation for feature weighting

**Classification Head:**
- **Pooling**: Global Average + Global Max concatenation
- **Dense Layers**: [512, 512, 128] units
- **Dropout Rates**: [0.4, 0.3, 0.2]
- **Residual Connections**: Between first two dense layers

### Reproducibility and Random Seeds

To ensure reproducible results across all experiments, the following random seeds were fixed:

```python
RANDOM_SEEDS = {
    'numpy': 44342,
    'tensorflow': 4254,
    'python': 4422,
    'sklearn': 42,
    'cv_random_state': 3443
}
```

These seeds control:
- **NumPy random operations**: 44342
- **TensorFlow/Keras initialization**: 4254
- **Python random module**: 4422
- **Scikit-learn operations**: 42
- **Cross-validation splits**: 3443

### Hardware and Software Configuration

**Hardware Specifications:**
- **Processor**: Apple M3 chip with Metal Performance Shaders (MPS)
- **Memory**: Unified memory architecture
- **Mixed Precision**: Enabled for memory efficiency

**Software Environment:**
- **TensorFlow**: Optimized for M3 MPS acceleration
- **Mixed Precision Policy**: float16 (except 3D CNN: float32)
- **XLA Compilation**: Enabled for performance optimization
- **Memory Growth**: Dynamic GPU memory allocation

**TensorFlow Optimizations:**
- **JIT Compilation**: Enabled
- **Threading**: Automatic optimization (0 threads for inter/intra-op)
- **Memory Management**: Dynamic growth enabled
- **Mixed Precision**: Global policy set to 'mixed_float16'

### Evaluation Metrics

**Primary Metrics:**
- **Accuracy**: Overall classification accuracy
- **Top-3 Accuracy**: Top-3 categorical accuracy
- **Precision**: Weighted average precision
- **Recall**: Weighted average recall
- **F1-Score**: Weighted average F1-score

**Additional Metrics:**
- **Cohen's Kappa**: Inter-rater agreement coefficient
- **Training Time**: Total training duration per fold
- **Inference Time**: Average prediction time per sample

### Data Augmentation (Optional)

While data augmentation was configured but not actively used in the final experiments, the following parameters were available:

**Augmentation Parameters:**
- **Probability**: 0.5 (50% chance of applying augmentation)
- **Brightness Range**: $\PM$0.2
- **Contrast Range**: [0.8, 1.2]
- **Saturation Range**: [0.8, 1.2]
- **Hue Range**: $\PM$0.1
- **Horizontal Flip**: Enabled
- **Rotation Range**: $\PM$15 degrees

### Class Weighting and Imbalance Handling

**Class Weighting:**
- **Method**: Computed using sklearn.utils.class_weight.compute_class_weight
- **Strategy**: 'balanced' (inversely proportional to class frequencies)
- **Application**: Applied during training via class_weight parameter

**Label Smoothing:**
- **Smoothing Factor**: ε = 0.05
- **Implementation**: CategoricalCrossentropy(label_smoothing=0.05)
- **Purpose**: Improve model calibration and generalization

### Callback Configuration

**Early Stopping:**
- **Monitor**: validation_accuracy
- **Patience**: 25 epochs
- **Restore Best Weights**: True
- **Min Delta**: 0 (default)

**Learning Rate Reduction:**
- **Monitor**: validation_loss
- **Factor**: 0.5
- **Patience**: 8 epochs
- **Min Learning Rate**: 1e-7
- **Cooldown**: 0 (default)

**Model Checkpointing:**
- **Monitor**: validation_accuracy
- **Save Best Only**: True
- **File Format**: .keras (TensorFlow SavedModel format)

This comprehensive hyperparameter configuration ensures consistent, reproducible, and optimized training across all four deep learning architectures, enabling fair comparison and robust evaluation of their performance on the joke classification task.
