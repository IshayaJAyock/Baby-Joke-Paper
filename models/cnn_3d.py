"""
Advanced 3D CNN Model for Video Classification
=============================================

This module implements a highly optimized 3D CNN model with advanced techniques:
- 3D ResNet architecture with residual connections
- Multi-scale feature extraction
- Spatial and temporal attention mechanisms
- Advanced regularization techniques
- Optimized for high performance

Author: Ishaya, Jeremiah Ayock
Date: 2025
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, GlobalAveragePooling3D, 
    LayerNormalization, Reshape, Conv3D, MaxPooling3D,
    BatchNormalization, Activation, Add, Concatenate,
    MultiHeadAttention, GlobalMaxPooling3D, AveragePooling3D,
    GlobalAveragePooling1D, GlobalMaxPooling1D, TimeDistributed
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, 
    LearningRateScheduler, TensorBoard
)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import TopKCategoricalAccuracy
from tensorflow.keras.regularizers import l1_l2
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import warnings
import time




# Suppress all warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

logger = logging.getLogger(__name__)

class CNN3D:
    """Advanced 3D CNN model with performance optimizations"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.num_classes = config.get('num_classes', 5)
        self.sequence_length = config.get('max_frames', 16)
        self.frame_height = config.get('frame_height', 224)
        self.frame_width = config.get('frame_width', 224)
        self.channels = 3
        self.dropout_rate = config.get('dropout_rate', 0.3)
        self.learning_rate = config.get('learning_rate', 0.001)
        
        # Advanced configuration
        self.attention_heads = 8
        self.key_dim = 64
        self.ff_dim = 512
        
        # Configure TensorFlow for M3 MPS
        self._configure_tf()
        
        # Create model
        self.model = self._build_model()
        
    def _configure_tf(self):
        """Configure TensorFlow for optimal M3 MPS performance"""
        try:
            # Configure TensorFlow for M3 MPS
            tf.config.optimizer.set_jit(True)
            
            # Only configure threading if not already initialized
            try:
                tf.config.threading.set_inter_op_parallelism_threads(0)
                tf.config.threading.set_intra_op_parallelism_threads(0)
            except RuntimeError:
                # Already initialized, skip threading config
                pass
            
            # Configure memory growth
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                try:
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    logger.info("GPU memory growth enabled")
                except RuntimeError as e:
                    logger.warning(f"GPU memory growth failed: {e}")
            
            # Use float32 to ensure CPU-supported 3D pooling kernels
            tf.keras.mixed_precision.set_global_policy('float32')
            logger.info("TensorFlow configured (float32 policy for 3D ops)")
            
        except Exception as e:
            logger.warning(f"TensorFlow configuration failed: {e}")
    
    def _build_model(self) -> Model:
        """Build the advanced 3D CNN model with performance optimizations"""
        try:
            # Input layer
            inputs = Input(shape=(self.sequence_length, self.frame_height, self.frame_width, self.channels), 
                          name='video_input')
            
            # 3D CNN backbone with residual connections
            cnn_features = self._build_3d_cnn_backbone(inputs)
            
            # Spatial-temporal attention (safe sequence attention)
            attention_features = self._build_spatial_temporal_attention(cnn_features)
            
            # Global pooling and classification
            outputs = self._build_classification_head(attention_features)
            
            # Create model
            model = Model(inputs=inputs, outputs=outputs, name='CNN3D')
            
            # Optimizer
            optimizer = Adam(
                learning_rate=self.learning_rate,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-7,
                clipnorm=1.0
            )
            
            model.compile(
                optimizer=optimizer,
                loss=CategoricalCrossentropy(label_smoothing=0.05),
                metrics=['accuracy', TopKCategoricalAccuracy(k=3, name='top_3_accuracy')]
            )
            
            logger.info("3D CNN model built and compiled")
            return model
            
        except Exception as e:
            logger.error(f"Model building failed: {e}")
            raise
    
    def _build_3d_cnn_backbone(self, inputs):
        """Build 3D CNN backbone with residual connections"""
        # Initial convolution
        x = Conv3D(64, (3, 7, 7), strides=(1, 2, 2), padding='same',
                  kernel_regularizer=l1_l2(1e-5, 1e-4))(inputs)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling3D((1, 3, 3), strides=(1, 2, 2), padding='same')(x)
        
        # Residual blocks
        x = self._residual_block_3d(x, 64, 1, 'res1')
        x = self._residual_block_3d(x, 128, 2, 'res2')
        x = self._residual_block_3d(x, 256, 2, 'res3')
        x = self._residual_block_3d(x, 512, 2, 'res4')
        
        return x
    
    def _residual_block_3d(self, x, filters, stride, name):
        """3D Residual block"""
        shortcut = x
        
        # First convolution
        x = Conv3D(filters, (3, 3, 3), strides=(1, stride, stride), padding='same',
                  kernel_regularizer=l1_l2(1e-5, 1e-4), name=f'{name}_conv1')(x)
        x = BatchNormalization(name=f'{name}_bn1')(x)
        x = Activation('relu')(x)
        
        # Second convolution
        x = Conv3D(filters, (3, 3, 3), padding='same',
                  kernel_regularizer=l1_l2(1e-5, 1e-4), name=f'{name}_conv2')(x)
        x = BatchNormalization(name=f'{name}_bn2')(x)
        
        # Shortcut connection
        if stride > 1 or shortcut.shape[-1] != filters:
            shortcut = Conv3D(filters, (1, 1, 1), strides=(1, stride, stride),
                            kernel_regularizer=l1_l2(1e-5, 1e-4), name=f'{name}_shortcut')(shortcut)
            shortcut = BatchNormalization(name=f'{name}_shortcut_bn')(shortcut)
        
        # Add shortcut
        x = Add(name=f'{name}_add')([x, shortcut])
        x = Activation('relu', name=f'{name}_relu')(x)
        
        return x
    
    def _build_spatial_temporal_attention(self, x):
        """Temporal attention over per-frame spatially pooled features."""
        # Pool spatial dimensions per time step -> (batch, time, channels)
        seq_features = tf.keras.layers.TimeDistributed(
            tf.keras.layers.GlobalAveragePooling2D()
        )(x)
        
        # Self-attention across time
        attn = MultiHeadAttention(
            num_heads=self.attention_heads,
            key_dim=self.key_dim,
            dropout=0.2
        )(seq_features, seq_features)
        attn = LayerNormalization()(attn + seq_features)
        return attn
    
    def _build_classification_head(self, x):
        """Build classification head with temporal pooling"""
        # x is (batch, time, channels)
        avg_pool = GlobalAveragePooling1D()(x)
        max_pool = GlobalMaxPooling1D()(x)
        x = Concatenate()([avg_pool, max_pool])
        
        # Dense layers with residual connections
        x1 = Dense(512, activation='relu', kernel_regularizer=l1_l2(1e-5, 1e-4))(x)
        x1 = BatchNormalization()(x1)
        x1 = Dropout(0.4)(x1)
        
        x2 = Dense(512, activation='relu', kernel_regularizer=l1_l2(1e-5, 1e-4))(x1)
        x2 = BatchNormalization()(x2)
        x2 = Dropout(0.3)(x2)
        
        # Skip connection
        x = Add()([x1, x2])
        x = Activation('relu')(x)
        
        # Final dense layers
        x = Dense(128, activation='relu', kernel_regularizer=l1_l2(1e-5, 1e-4))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        
        # Output layer
        outputs = Dense(self.num_classes, activation='softmax', name='predictions')(x)
        
        return outputs
    
    def _create_callbacks(self, model_dir: Optional[Path]) -> List[tf.keras.callbacks.Callback]:
        """Create training callbacks"""
        try:
            callbacks = []
            
            if model_dir:
                # Early stopping with patience
                early_stopping = EarlyStopping(
                    monitor='val_accuracy',
                    patience=15,
                    restore_best_weights=True,
                    verbose=1
                )
                callbacks.append(early_stopping)
                
                # Model checkpoint
                checkpoint = ModelCheckpoint(
                    filepath=model_dir / 'best_model.keras',
                    monitor='val_accuracy',
                    save_best_only=True,
                    verbose=1
                )
                callbacks.append(checkpoint)
            
            # Learning rate reduction
            lr_scheduler = ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=8,
                min_lr=1e-7,
                verbose=1
            )
            callbacks.append(lr_scheduler)
            
            
            return callbacks
            
        except Exception as e:
            logger.warning(f"Callback creation failed: {e}")
            return []
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: np.ndarray, y_val: np.ndarray, 
              model_dir: Optional[Path], epochs: int = 100) -> Dict[str, Any]:
        """Train the model with advanced techniques"""
        try:
            logger.info("Starting 3D CNN training...")
            
            # Convert labels to categorical
            y_train_cat = to_categorical(y_train, num_classes=self.num_classes)
            y_val_cat = to_categorical(y_val, num_classes=self.num_classes)
            
            # Create callbacks
            callbacks = self._create_callbacks(model_dir)
            
            # Train model
            start_time = time.time()
            history = self.model.fit(
                X_train, y_train_cat,
                validation_data=(X_val, y_val_cat),
                epochs=epochs,
                batch_size=self.config.get('batch_size', 8),  # Smaller batch size for 3D CNN
                callbacks=callbacks,
                class_weight=self.config.get('class_weight'),
                verbose=1,
                use_multiprocessing=False,
                workers=1
            )
            
            training_time = time.time() - start_time
            
            # Save final model (only if model_dir is provided)
            final_model_path = None
            if model_dir:
                final_model_path = model_dir / 'final_model.keras'
                self.model.save(final_model_path)
            
            logger.info(f"3D CNN training completed in {training_time:.2f}s")
            
            return {
                'history': history.history,
                'training_time': training_time,
                'best_epoch': len(history.history['loss']),
                'final_model_path': str(final_model_path) if final_model_path else None
            }
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Evaluate the model"""
        try:
            y_test_cat = to_categorical(y_test, num_classes=self.num_classes)
            
            # Evaluate model
            test_loss, test_accuracy, test_top3_accuracy = self.model.evaluate(
                X_test, y_test_cat, verbose=0
            )
            
            # Get predictions
            y_pred = self.model.predict(X_test, verbose=0)
            y_pred_classes = np.argmax(y_pred, axis=1)
            
            # Calculate additional metrics
            from sklearn.metrics import precision_score, recall_score, f1_score
            
            precision = precision_score(y_test, y_pred_classes, average='weighted')
            recall = recall_score(y_test, y_pred_classes, average='weighted')
            f1 = f1_score(y_test, y_pred_classes, average='weighted')
            
            return {
                'accuracy': test_accuracy,
                'top3_accuracy': test_top3_accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'y_pred': y_pred_classes,
                'y_true': y_test
            }
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            raise
    
    def clear_session(self):
        """Clear TensorFlow session"""
        try:
            tf.keras.backend.clear_session()
            logger.info("TensorFlow session cleared")
        except Exception as e:
            logger.warning(f"Session clearing failed: {e}")

