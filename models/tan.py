"""
Advanced Temporal Attention Network (TAN) for Video Classification
================================================================

This module implements a highly optimized TAN model with advanced techniques:
- Multi-head self-attention mechanisms
- Positional encoding for temporal sequences
- Residual connections and layer normalization
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
    Input, Dense, LSTM, Dropout, GlobalAveragePooling1D, 
    LayerNormalization, Reshape, Conv1D, MaxPooling1D,
    BatchNormalization, Activation, Add, Concatenate,
    MultiHeadAttention, GlobalMaxPooling1D, Embedding, Softmax, Multiply
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

class TAN:
    """Advanced Temporal Attention Network with performance optimizations"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.num_classes = config.get('num_classes', 5)
        self.sequence_length = config.get('max_frames', 16)
        self.feature_dim = config.get('feature_dim', 2048)
        self.attention_heads = config.get('attention_heads', 8)
        self.attention_dim = config.get('attention_dim', 64)
        self.dropout_rate = config.get('dropout_rate', 0.3)
        self.learning_rate = config.get('learning_rate', 0.001)
        
        # Advanced configuration
        self.num_transformer_blocks = 4
        self.ff_dim = 512
        self.key_dim = 64
        
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
            
            # Set mixed precision policy
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
            logger.info("TensorFlow configured")
            
        except Exception as e:
            logger.warning(f"TensorFlow configuration failed: {e}")
    
    def _build_model(self) -> Model:
        """Build the advanced TAN model with performance optimizations"""
        try:
            # Input layer
            inputs = Input(shape=(self.sequence_length, self.feature_dim), name='sequence_input')
            
            # Positional encoding
            x = self._add_positional_encoding(inputs)
            
            # Multi-layer transformer blocks
            x = self._build_transformer_blocks(x)
            
            # Global attention pooling
            x = self._build_global_attention_pooling(x)
            
            # Classification head
            outputs = self._build_classification_head(x)
            
            # Create model
            model = Model(inputs=inputs, outputs=outputs, name='TAN')
            
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
            
            logger.info("TAN model built and compiled")
            return model
            
        except Exception as e:
            logger.error(f"Model building failed: {e}")
            raise
    
    def _add_positional_encoding(self, inputs):
        """Add positional encoding to input features using precomputed constants."""
        # Use known static shapes to avoid TF ops on symbolic tensors
        seq_len = int(self.sequence_length)
        d_model = int(inputs.shape[-1])
        
        # Precompute sinusoidal positional encoding with numpy
        pos = np.arange(seq_len)[:, np.newaxis]
        i = np.arange(d_model)[np.newaxis, :]
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        angle_rads = pos * angle_rates
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        pos_encoding = angle_rads[np.newaxis, ...]  # shape (1, seq_len, d_model)
        
        # Convert to constant of correct dtype and add
        pos_encoding_const = tf.constant(pos_encoding, dtype=inputs.dtype)
        x = inputs + pos_encoding_const
        x = LayerNormalization()(x)
        return x
    
    def _get_positional_encoding(self, seq_len, d_model):
        """Deprecated; not used in graph to avoid TF ops on symbolic tensors."""
        pos = np.arange(int(seq_len))[:, np.newaxis]
        i = np.arange(int(d_model))[np.newaxis, :]
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        angle_rads = pos * angle_rates
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        pos_encoding = angle_rads[np.newaxis, ...]
        return tf.constant(pos_encoding, dtype=tf.float32)
    
    def _build_transformer_blocks(self, x):
        """Build multiple transformer blocks"""
        for i in range(self.num_transformer_blocks):
            x = self._transformer_block(x, f"transformer_{i}")
        
        return x
    
    def _transformer_block(self, x, name):
        """Single transformer block with self-attention and feed-forward"""
        # Multi-head self-attention
        attention_output = MultiHeadAttention(
            num_heads=self.attention_heads,
            key_dim=self.key_dim,
            dropout=0.2,
            name=f"{name}_attention"
        )(x, x)
        
        # Add & Norm
        attention_output = Dropout(0.2)(attention_output)
        x = LayerNormalization(name=f"{name}_norm1")(x + attention_output)
        
        # Feed-forward network
        ffn = Dense(self.ff_dim, activation='relu', 
                   kernel_regularizer=l1_l2(1e-5, 1e-4))(x)
        ffn = Dropout(0.2)(ffn)
        ffn = Dense(x.shape[-1], kernel_regularizer=l1_l2(1e-5, 1e-4))(ffn)
        
        # Add & Norm
        ffn = Dropout(0.2)(ffn)
        x = LayerNormalization(name=f"{name}_norm2")(x + ffn)
        
        return x
    
    def _build_global_attention_pooling(self, x):
        """Build global attention pooling mechanism with softmax over time axis."""
        # Compute unnormalized attention scores per timestep
        attention_scores = Dense(1)(x)  # shape (batch, time, 1)
        attention_weights = Softmax(axis=1)(attention_scores)
        attention_output = Multiply()([x, attention_weights])
        global_pooled = tf.keras.layers.Lambda(lambda t: tf.reduce_sum(t, axis=1))(attention_output)
        
        # Also use standard pooling
        avg_pool = GlobalAveragePooling1D()(x)
        max_pool = GlobalMaxPooling1D()(x)
        
        # Concatenate all pooling methods
        x = Concatenate()([global_pooled, avg_pool, max_pool])
        return x
    
    def _build_classification_head(self, x):
        """Build advanced classification head"""
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
        """Create training callbacks (early stopping always on; monitors INNER val only)."""
        try:
            callbacks = []
            patience = int(self.config.get('early_stopping_patience', 15))

            early_stopping = EarlyStopping(
                monitor='val_accuracy',
                patience=patience,
                restore_best_weights=True,
                verbose=1,
            )
            callbacks.append(early_stopping)

            if model_dir:
                Path(model_dir).mkdir(parents=True, exist_ok=True)
                checkpoint = ModelCheckpoint(
                    filepath=str(Path(model_dir) / 'best_model.keras'),
                    monitor='val_accuracy',
                    save_best_only=True,
                    verbose=1,
                )
                callbacks.append(checkpoint)

            lr_scheduler = ReduceLROnPlateau(
                monitor='val_loss',
                factor=float(self.config.get('reduce_lr_factor', 0.5)),
                patience=int(self.config.get('reduce_lr_patience', 8)),
                min_lr=float(self.config.get('min_learning_rate', 1e-7)),
                verbose=1,
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
            logger.info("Starting TAN training...")
            
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
                batch_size=self.config.get('batch_size', 16),
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
            
            logger.info(f"TAN training completed in {training_time:.2f}s")
            
            return {
                'history': history.history,
                'training_time': training_time,
                'best_epoch': int(np.argmax(history.history.get('val_accuracy', history.history['loss'])) + 1) if history.history.get('val_accuracy') else len(history.history['loss']),
                'epochs_ran': len(history.history['loss']),
                'train_loss': float(history.history['loss'][-1]),
                'validation_loss': float(history.history['val_loss'][-1]) if 'val_loss' in history.history else None,
                'final_model_path': str(final_model_path) if final_model_path else None
            }
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Evaluate on an untouched outer test fold; report macro + weighted metrics."""
        try:
            from src.eval_metrics import classification_metrics

            y_test_cat = to_categorical(y_test, num_classes=self.num_classes)
            test_loss, test_accuracy, test_top3_accuracy = self.model.evaluate(
                X_test, y_test_cat, verbose=0
            )
            y_proba = self.model.predict(X_test, verbose=0)
            y_pred_classes = np.argmax(y_proba, axis=1)
            metrics = classification_metrics(y_test, y_pred_classes, y_proba)
            return {
                'accuracy': float(test_accuracy),
                'top3_accuracy': float(test_top3_accuracy),
                'validation_loss': float(test_loss),
                **metrics,
                'y_pred': y_pred_classes,
                'y_true': y_test,
                'y_proba': y_proba,
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

