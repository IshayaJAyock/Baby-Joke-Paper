"""
Advanced CNN-LSTM Model for Video Classification
===============================================

This module implements a highly optimized CNN-LSTM model with advanced techniques:
- Multi-scale CNN feature extraction
- Bidirectional LSTM with attention mechanisms
- Residual connections and skip connections
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
    Input, Dense, LSTM, Dropout, GlobalAveragePooling2D, 
    LayerNormalization, Reshape, Conv1D, MaxPooling1D,
    BatchNormalization, Activation, Add, Concatenate,
    MultiHeadAttention, GlobalMaxPooling1D, GlobalAveragePooling1D
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, 
    LearningRateScheduler, TensorBoard
)
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.metrics import TopKCategoricalAccuracy
from tensorflow.keras.losses import CategoricalCrossentropy
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import warnings
import time

# Suppress all warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

logger = logging.getLogger(__name__)

class CNNLSTM:
    """Advanced CNN-LSTM model with performance optimizations"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.num_classes = config.get('num_classes', 5)
        self.sequence_length = config.get('max_frames', 16)
        self.feature_dim = config.get('feature_dim', 2048)
        self.lstm_units = config.get('lstm_units', 256)
        self.dropout_rate = config.get('dropout_rate', 0.3)
        self.learning_rate = config.get('learning_rate', 0.001)
        
        # Advanced configuration
        self.attention_heads = 8
        self.key_dim = 64
        self.ff_dim = 512
        self.num_transformer_blocks = 2
        
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
        """Build the advanced CNN-LSTM model with performance optimizations"""
        try:
            # Input layer
            inputs = Input(shape=(self.sequence_length, self.feature_dim), name='sequence_input')
            
            # Multi-scale CNN feature extraction
            cnn_features = self._build_multi_scale_cnn(inputs)
            
            # Advanced temporal modeling with attention
            temporal_features = self._build_temporal_encoder(cnn_features)
            
            # Feature fusion and classification
            outputs = self._build_classification_head(temporal_features)
            
            # Create model
            model = Model(inputs=inputs, outputs=outputs, name='CNN_LSTM')
            
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
            
            logger.info("CNN-LSTM model built and compiled")
            return model
            
        except Exception as e:
            logger.error(f"Model building failed: {e}")
            raise
    
    def _build_multi_scale_cnn(self, inputs):
        """Build multi-scale CNN feature extractor"""
        # Multi-scale convolutions
        conv1 = Conv1D(64, 3, activation='relu', padding='same', 
                      kernel_regularizer=l1_l2(1e-5, 1e-4))(inputs)
        conv1 = BatchNormalization()(conv1)
        conv1 = Dropout(0.1)(conv1)
        
        conv2 = Conv1D(64, 5, activation='relu', padding='same',
                      kernel_regularizer=l1_l2(1e-5, 1e-4))(inputs)
        conv2 = BatchNormalization()(conv2)
        conv2 = Dropout(0.1)(conv2)
        
        conv3 = Conv1D(64, 7, activation='relu', padding='same',
                      kernel_regularizer=l1_l2(1e-5, 1e-4))(inputs)
        conv3 = BatchNormalization()(conv3)
        conv3 = Dropout(0.1)(conv3)
        
        # Concatenate multi-scale features
        x = Concatenate()([conv1, conv2, conv3])
        x = MaxPooling1D(2)(x)
        
        # Residual blocks
        for i in range(3):
            residual = x
            x = Conv1D(128, 3, activation='relu', padding='same',
                      kernel_regularizer=l1_l2(1e-5, 1e-4))(x)
            x = BatchNormalization()(x)
            x = Dropout(0.2)(x)
            x = Conv1D(128, 3, activation='relu', padding='same',
                      kernel_regularizer=l1_l2(1e-5, 1e-4))(x)
            x = BatchNormalization()(x)
            # Project residual to match channels if needed
            if residual.shape[-1] != x.shape[-1]:
                residual = Conv1D(int(x.shape[-1]), 1, padding='same')(residual)
                residual = BatchNormalization()(residual)
            x = Add()([x, residual])
            x = Activation('relu')(x)
            x = MaxPooling1D(2)(x)
        
        return x
    
    def _build_temporal_encoder(self, x):
        """Build temporal encoder with attention mechanisms"""
        # Bidirectional LSTM with attention
        lstm_out = tf.keras.layers.Bidirectional(
            LSTM(256, return_sequences=True, dropout=0.3, recurrent_dropout=0.3,
                 kernel_regularizer=l1_l2(1e-5, 1e-4))
        )(x)
        lstm_out = LayerNormalization()(lstm_out)
        
        # Multi-head self-attention
        attention = MultiHeadAttention(
            num_heads=self.attention_heads, 
            key_dim=self.key_dim, 
            dropout=0.2
        )(lstm_out, lstm_out)
        attention = LayerNormalization()(attention + lstm_out)
        
        # Feed-forward network (match attention channels for residual add)
        ffn = Dense(self.ff_dim, activation='relu')(attention)
        ffn = Dropout(0.2)(ffn)
        ffn = Dense(self.attention_heads * self.key_dim)(ffn)  # 512 to match attention
        ffn = LayerNormalization()(ffn + attention)
        
        # Second LSTM layer
        lstm_out2 = tf.keras.layers.Bidirectional(
            LSTM(128, return_sequences=False, dropout=0.3, recurrent_dropout=0.3,
                 kernel_regularizer=l1_l2(1e-5, 1e-4))
        )(ffn)
        lstm_out2 = LayerNormalization()(lstm_out2)
        
        return lstm_out2
    
    def _build_classification_head(self, x):
        """Build advanced classification head"""
        # x is already a flat feature vector (no temporal dim), so no 1D pooling here
        
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
            logger.info("Starting model training...")
            
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
            
            logger.info(f"Training completed in {training_time:.2f}s")
            
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

