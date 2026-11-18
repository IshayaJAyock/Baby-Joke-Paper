"""
Standardized Configuration for Video Classification Models
========================================================

This module provides a unified configuration system for all video classification models
to ensure consistent hyperparameters, data processing, and evaluation across:
- CNN-LSTM
- 3D-CNN  
- Temporal Attention Network (TAN)
- Multimodal
"""

import os
import yaml
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any
import logging

logger = logging.getLogger(__name__)

@dataclass
class StandardConfig:
    """Standardized configuration for all video classification models"""
    
    num_frames: int = 20                    # 20 frames per video for optimal temporal representation
    frame_size: Tuple[int, int] = (224, 224)  # Standard input size
    num_classes: int = 5                    # Number of joke types
    batch_size: int = 16                    # Optimized for M3 MPS and performance
    num_epochs: int = 150                   # Training epochs
    learning_rate: float = 0.0005           # Lower learning rate for stability
    validation_split: float = 0.2           # Validation split ratio
    test_split: float = 0.2                 # Test split ratio
    
    
    cv_folds: int = 10                      # 10-fold cross-validation
    cv_random_state: int = 3443               # Reproducible CV splits
    stratified_cv: bool = True              # Use stratified CV for balanced folds
    

    enable_augmentation: bool = False       # Main runners leave augmentation off
    augmentation_probability: float = 0.5   # Probability of applying augmentation
    brightness_range: float = 0.2           # Brightness variation range
    contrast_range: Tuple[float, float] = (0.8, 1.2)  # Contrast variation range
    saturation_range: Tuple[float, float] = (0.8, 1.2)  # Saturation variation range
    hue_range: float = 0.1                  # Hue variation range
    horizontal_flip: bool = True            # Enable horizontal flipping
    rotation_range: int = 15                # Random rotation range in degrees
    
    
    cnn_lstm: Dict[str, Any] = None
    cnn_3d: Dict[str, Any] = None
    temporal_attention: Dict[str, Any] = None
    multimodal: Dict[str, Any] = None
    

    early_stopping_patience: int = 25       # Inner-monitor EarlyStopping patience (matches result schema)
    early_stopping_start_epoch: int = 15    # Ignore early epochs before ES monitoring
    use_group_cv: bool = True               # StratifiedGroupKFold on ChildID (infant hold-out)
    reduce_lr_patience: int = 8             # Learning rate reduction patience
    reduce_lr_factor: float = 0.5           # Learning rate reduction factor
    min_learning_rate: float = 1e-7         # Minimum learning rate
    weight_decay: float = 1e-5              # Weight decay for regularization
    
    
    data_dir: str = "data/main_data"
    video_dir: str = "data/videos"
    metadata_path: str = "data/excel2_modified_filtered_sorted_by_Total Seconds.xlsx"
    results_dir: str = "outputs"
    models_dir: str = "models"
    logs_dir: str = "logs"
    
    
    enable_m3_optimization: bool = True     # Enable M3 MPS optimizations
    mixed_precision: bool = True            # Enable mixed precision training
    xla_compilation: bool = True            # Enable XLA compilation
    num_workers: int = 4                    # Number of parallel workers
    
    
    save_predictions: bool = True           # Save model predictions
    save_confusion_matrix: bool = True      # Save confusion matrices
    save_training_plots: bool = True        # Save training history plots
    generate_excel_report: bool = True      # Generate Excel comparison report
    
    def __post_init__(self):
        """Initialize model-specific parameters after dataclass creation"""
        if self.cnn_lstm is None:
            self.cnn_lstm = {
                'cnn_backbone': 'resnet50',
                'lstm_units': [512, 256, 128],
                'dropout_rate': 0.2,
                'dense_units': [256, 128, 64],
                'use_bidirectional': True,
                'return_sequences': True,
                'attention_heads': 8,
                'key_dim': 64
            }
        
        if self.cnn_3d is None:
            self.cnn_3d = {
                'conv3d_filters': [64, 128, 256, 512],
                'kernel_size': (3, 3, 3),
                'pool_size': (2, 2, 2),
                'use_residual': True,
                'use_batch_norm': True,
                'spatial_dropout': 0.1,
                'dropout_rate': 0.3,
                'dense_units': [1024, 512, 256, 128]
            }
        
        if self.temporal_attention is None:
            self.temporal_attention = {
                'num_attention_heads': 8,
                'attention_dim': 64,
                'lstm_units': [512, 256],
                'dense_units': [512, 256, 128],
                'dropout_rate': 0.2,
                'use_bidirectional': True,
                'use_layer_norm': True,
                'num_transformer_blocks': 4
            }
        
        if self.multimodal is None:
            self.multimodal = {
                # Implemented streams in scripts/run_multimodal.py (not rgb/audio).
                # fusion_method 'gated_cross_attention' maps to ablation 'full'
                # in models/multimodal.py. Ablation table variants:
                # cue_motion, cue_face, cue_hands, concat, attention_only,
                # gating_only, full — see scripts/run_multimodal_ablation.py.
                'fusion_method': 'gated_cross_attention',
                'ablation_variant': 'full',
                'modalities': ['motion', 'face', 'hands'],
                'motion_dim': 8,
                'facial_dim': 15,
                'gesture_dim': 45,
                'fusion_units': 256,
                'attention_heads': 8,
                'dropout_rate': 0.3
            }

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return asdict(self)
    
    def save_yaml(self, filepath: str) -> None:
        """Save configuration to YAML file"""
        config_dict = self.to_dict()
        with open(filepath, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        logger.info(f"Configuration saved to {filepath}")
    
    @classmethod
    def from_yaml(cls, filepath: str) -> 'StandardConfig':
        """Load configuration from YAML file"""
        with open(filepath, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
    
    def validate(self) -> bool:
        """Validate configuration parameters"""
        errors = []
        
        # Validate frame parameters
        if self.num_frames <= 0:
            errors.append("num_frames must be positive")
        if self.frame_size[0] <= 0 or self.frame_size[1] <= 0:
            errors.append("frame_size must have positive dimensions")
        
        # Validate training parameters
        if self.batch_size <= 0:
            errors.append("batch_size must be positive")
        if self.num_epochs <= 0:
            errors.append("num_epochs must be positive")
        if not 0 < self.learning_rate < 1:
            errors.append("learning_rate must be between 0 and 1")
        
        # Validate split ratios
        if not 0 < self.validation_split < 1:
            errors.append("validation_split must be between 0 and 1")
        if not 0 < self.test_split < 1:
            errors.append("test_split must be between 0 and 1")
        if self.validation_split + self.test_split >= 1:
            errors.append("validation_split + test_split must be less than 1")
        
        # Validate cross-validation
        if self.cv_folds < 2:
            errors.append("cv_folds must be at least 2")
        
        if errors:
            for error in errors:
                logger.error(f"Configuration validation error: {error}")
            return False
        
        logger.info("Configuration validation passed")
        return True


# PREDEFINED CONFIGURATIONS
def get_standard_config() -> StandardConfig:
    """Get the standard configuration for all models"""
    return StandardConfig()

def get_fast_config() -> StandardConfig:
    """Get a fast configuration for quick testing"""
    config = StandardConfig()
    config.num_frames = 8
    config.num_epochs = 10
    config.cv_folds = 3
    config.batch_size = 16
    return config

def get_high_accuracy_config() -> StandardConfig:
    """Get a configuration optimized for high accuracy"""
    config = StandardConfig()
    config.num_frames = 32
    config.num_epochs = 200
    config.learning_rate = 0.0005
    config.batch_size = 4
    config.early_stopping_patience = 30
    return config


# CONFIGURATION MANAGEMENT

class ConfigManager:
    """Manager for handling multiple configurations"""
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
    
    def save_config(self, config: StandardConfig, name: str) -> None:
        """Save configuration with given name"""
        filepath = self.config_dir / f"{name}.yaml"
        config.save_yaml(str(filepath))
    
    def load_config(self, name: str) -> StandardConfig:
        """Load configuration by name"""
        filepath = self.config_dir / f"{name}.yaml"
        return StandardConfig.from_yaml(str(filepath))
    
    def list_configs(self) -> List[str]:
        """List available configurations"""
        return [f.stem for f in self.config_dir.glob("*.yaml")]


# JOKE TYPE MAPPING
JOKE_TYPE_MAPPING = {
    'NomNomNom': 0,
    'Peekaboo': 1,
    'TearingPaper': 2,
    'ThatsNotACat': 3,
    'ThatsNotAHat': 4
}

JOKE_TYPE_NAMES = [
    'NomNomNom',
    'Peekaboo', 
    'TearingPaper',
    'ThatsNotACat',
    'ThatsNotAHat'
]


# RANDOM SEEDS FOR REPRODUCIBILITY

RANDOM_SEEDS = {
    'numpy': 44342,
    'tensorflow': 4254,
    'python': 4422,
    'sklearn': 42
}

def set_random_seeds(seeds: Dict[str, int] = None) -> None:
    """Set random seeds for reproducibility"""
    if seeds is None:
        seeds = RANDOM_SEEDS
    
    import numpy as np
    import tensorflow as tf
    import random
    
    np.random.seed(seeds['numpy'])
    tf.random.set_seed(seeds['tensorflow'])
    random.seed(seeds['python'])
    
    logger.info(f"Random seeds set: {seeds}")

if __name__ == "__main__":
    # Example usage
    config = get_standard_config()
    print("Standard Configuration:")
    print(f"Frames: {config.num_frames}")
    print(f"Frame size: {config.frame_size}")
    print(f"Batch size: {config.batch_size}")
    print(f"CV folds: {config.cv_folds}")
    
    # Validate configuration
    if config.validate():
        print("Configuration is valid!")
    else:
        print("Configuration has errors!")
