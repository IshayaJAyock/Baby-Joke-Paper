"""
Advanced Multimodal Model for Video Classification
=================================================

This module implements a highly optimized multimodal model with advanced techniques:
- Cross-modal attention mechanisms
- Feature fusion with gating mechanisms
- Advanced regularization techniques
- Multi-scale feature extraction
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
    MultiHeadAttention, GlobalMaxPooling1D, Multiply
)  # MultiHeadAttention used for C_MF / C_FM / C_GM cross-modal blocks
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, 
    LearningRateScheduler, TensorBoard
)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.metrics import TopKCategoricalAccuracy
from tensorflow.keras.losses import CategoricalCrossentropy
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

# Ablation variants for the multi-cue fusion model (video pipeline).
# (i) cues alone → cue_motion / cue_face / cue_hands
# (ii) concat → pool+concat+project (no attention, no gate)
# (iii) attention_only → cross-attn + concat + project (no gate)
# (iv) gating_only → pool+concat+gate+project (no cross-attn)
# (v) full → cross-attn + concat + gate + project
ABLATION_VARIANTS = (
    'cue_motion',
    'cue_face',
    'cue_hands',
    'concat',
    'attention_only',
    'gating_only',
    'full',
)


class Multimodal:
    """Multi-cue fusion model with configurable ablation variants."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.num_classes = config.get('num_classes', 5)
        self.sequence_length = config.get('max_frames', 20)
        self.motion_dim = config.get('motion_dim', 8)
        self.facial_dim = config.get('facial_dim', 15)
        self.gesture_dim = config.get('gesture_dim', 45)
        self.dropout_rate = config.get('dropout_rate', 0.3)
        self.learning_rate = config.get('learning_rate', 0.001)
        
        # Advanced configuration
        self.attention_heads = int(config.get('attention_heads', 8))
        self.key_dim = 64
        self.ff_dim = 512
        self.fusion_dim = int(config.get('fusion_units', config.get('fusion_dim', 256)))
        self.ablation_variant = str(
            config.get('ablation_variant', config.get('fusion_method', 'full'))
        ).lower()
        if self.ablation_variant in ('gated_cross_attention', 'attention'):
            self.ablation_variant = 'full'
        if self.ablation_variant not in ABLATION_VARIANTS:
            raise ValueError(
                f"Unknown ablation_variant={self.ablation_variant!r}. "
                f"Expected one of {ABLATION_VARIANTS}"
            )
        
        # Configure TensorFlow for M3 MPS
        self._configure_tf()
        
        # Create model
        self.model = self._build_model()
        logger.info(f"Multimodal built with ablation_variant={self.ablation_variant}")
        
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
        """Build multi-cue model for the selected ablation variant."""
        try:
            variant = self.ablation_variant

            # --- (i) cues alone: single-modality encoder + head ---
            if variant == 'cue_motion':
                inp = Input(shape=(self.sequence_length, self.motion_dim), name='motion_input')
                feat = self._build_motion_encoder(inp)
                pooled = GlobalAveragePooling1D()(feat)
                fused = self._project_vector(pooled, name_prefix='cue_motion')
                outputs = self._build_classification_head_from_vector(fused)
                inputs = inp
            elif variant == 'cue_face':
                inp = Input(shape=(self.sequence_length, self.facial_dim), name='facial_input')
                feat = self._build_facial_encoder(inp)
                pooled = GlobalAveragePooling1D()(feat)
                fused = self._project_vector(pooled, name_prefix='cue_face')
                outputs = self._build_classification_head_from_vector(fused)
                inputs = inp
            elif variant == 'cue_hands':
                inp = Input(shape=(self.sequence_length, self.gesture_dim), name='gesture_input')
                feat = self._build_gesture_encoder(inp)
                pooled = GlobalAveragePooling1D()(feat)
                fused = self._project_vector(pooled, name_prefix='cue_hands')
                outputs = self._build_classification_head_from_vector(fused)
                inputs = inp
            else:
                # Multi-input fusion variants: concat / attention_only / gating_only / full
                motion_input = Input(shape=(self.sequence_length, self.motion_dim), name='motion_input')
                facial_input = Input(shape=(self.sequence_length, self.facial_dim), name='facial_input')
                gesture_input = Input(shape=(self.sequence_length, self.gesture_dim), name='gesture_input')

                motion_features = self._build_motion_encoder(motion_input)
                facial_features = self._build_facial_encoder(facial_input)
                gesture_features = self._build_gesture_encoder(gesture_input)

                fused_features = self._build_cross_modal_fusion(
                    motion_features, facial_features, gesture_features, variant=variant
                )
                outputs = self._build_classification_head(fused_features)
                inputs = [motion_input, facial_input, gesture_input]

            model = Model(inputs=inputs, outputs=outputs, name=f'Multimodal_{variant}')

            optimizer = Adam(
                learning_rate=float(self.config.get('learning_rate', getattr(self, 'learning_rate', 0.0005))),
                beta_1=float(self.config.get('adam_beta_1', 0.9)),
                beta_2=float(self.config.get('adam_beta_2', 0.999)),
                epsilon=float(self.config.get('adam_epsilon', 1e-7)),
                clipnorm=float(self.config.get('clipnorm', 1.0)),
            )
            model.compile(
                optimizer=optimizer,
                loss=CategoricalCrossentropy(label_smoothing=float(self.config.get('label_smoothing', 0.05))),
                metrics=['accuracy', TopKCategoricalAccuracy(k=3, name='top_3_accuracy')]
            )
            logger.info("Multimodal model built and compiled")
            return model

        except Exception as e:
            logger.error(f"Model building failed: {e}")
            raise

    def _project_vector(self, vec, name_prefix: str):
        """Shared projection used by single-cue ablations."""
        x = Dense(
            self.fusion_dim, activation='relu',
            kernel_regularizer=l1_l2(1e-5, 1e-4), name=f'{name_prefix}_proj'
        )(vec)
        x = BatchNormalization(name=f'{name_prefix}_bn')(x)
        x = Dropout(0.3, name=f'{name_prefix}_do')(x)
        return x

    def _build_classification_head_from_vector(self, x):
        """Classification head when input is already a pooled vector (no time axis)."""
        x1 = Dense(512, activation='relu', kernel_regularizer=l1_l2(1e-5, 1e-4))(x)
        x1 = BatchNormalization()(x1)
        x1 = Dropout(0.4)(x1)

        x2 = Dense(512, activation='relu', kernel_regularizer=l1_l2(1e-5, 1e-4))(x1)
        x2 = BatchNormalization()(x2)
        x2 = Dropout(0.3)(x2)

        x = Add()([x1, x2])
        x = Activation('relu')(x)
        x = Dense(128, activation='relu', kernel_regularizer=l1_l2(1e-5, 1e-4))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        return Dense(self.num_classes, activation='softmax', name='predictions')(x)
    
    
    def _build_motion_encoder(self, inputs):
        """Build motion feature encoder"""
        # Dense layers for motion features
        x = Dense(64, activation='relu', kernel_regularizer=l1_l2(1e-5, 1e-4))(inputs)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        
        x = Dense(128, activation='relu', kernel_regularizer=l1_l2(1e-5, 1e-4))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        
        # LSTM for temporal modeling
        x = tf.keras.layers.Bidirectional(
            LSTM(64, return_sequences=True, dropout=0.3, recurrent_dropout=0.3)
        )(x)
        x = LayerNormalization()(x)
        
        return x
    
    def _build_facial_encoder(self, inputs):
        """Build facial feature encoder"""
        # Dense layers for facial features
        x = Dense(64, activation='relu', kernel_regularizer=l1_l2(1e-5, 1e-4))(inputs)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        
        x = Dense(128, activation='relu', kernel_regularizer=l1_l2(1e-5, 1e-4))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        
        # LSTM for temporal modeling
        x = tf.keras.layers.Bidirectional(
            LSTM(64, return_sequences=True, dropout=0.3, recurrent_dropout=0.3)
        )(x)
        x = LayerNormalization()(x)
        
        return x
    
    def _build_gesture_encoder(self, inputs):
        """Build gesture feature encoder"""
        # Dense layers for gesture features
        x = Dense(128, activation='relu', kernel_regularizer=l1_l2(1e-5, 1e-4))(inputs)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        
        x = Dense(256, activation='relu', kernel_regularizer=l1_l2(1e-5, 1e-4))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        
        # LSTM for temporal modeling
        x = tf.keras.layers.Bidirectional(
            LSTM(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.3)
        )(x)
        x = LayerNormalization()(x)
        
        return x
    
    def _build_cross_modal_fusion(
        self, motion_features, facial_features, gesture_features, variant: str = 'full'
    ):
        """Fusion block with ablation switches.

        Variants
        --------
        concat : GAP each encoder → concat → project (no attention, no gate)
        attention_only : pairwise MHA → GAP → concat → project (no gate)
        gating_only : GAP → concat → sigmoid gate → project (no MHA)
        full : pairwise MHA → GAP → concat → gate → project
        """
        d = int(self.fusion_dim)
        heads = int(self.attention_heads)
        key_dim = max(8, d // max(heads, 1))
        use_attention = variant in ('attention_only', 'full')
        use_gate = variant in ('gating_only', 'full')

        M = Dense(d, activation='relu', name='proj_motion')(motion_features)
        F = Dense(d, activation='relu', name='proj_face')(facial_features)
        G = Dense(d, activation='relu', name='proj_gesture')(gesture_features)

        if use_attention:
            C_MF = MultiHeadAttention(
                num_heads=heads, key_dim=key_dim, dropout=0.2, name='attn_MF'
            )(query=M, value=F, key=F)
            C_MF = LayerNormalization(name='ln_MF')(C_MF + M)

            C_FM = MultiHeadAttention(
                num_heads=heads, key_dim=key_dim, dropout=0.2, name='attn_FM'
            )(query=F, value=M, key=M)
            C_FM = LayerNormalization(name='ln_FM')(C_FM + F)

            C_GM = MultiHeadAttention(
                num_heads=heads, key_dim=key_dim, dropout=0.2, name='attn_GM'
            )(query=G, value=M, key=M)
            C_GM = LayerNormalization(name='ln_GM')(C_GM + G)

            motion_seq, face_seq, gest_seq = C_MF, C_FM, C_GM
        else:
            motion_seq, face_seq, gest_seq = M, F, G

        motion_vec = GlobalAveragePooling1D(name='gap_motion')(motion_seq)
        facial_vec = GlobalAveragePooling1D(name='gap_face')(face_seq)
        gesture_vec = GlobalAveragePooling1D(name='gap_gesture')(gest_seq)

        fused_vec = Concatenate(name='concat_MF_FM_GM')(
            [motion_vec, facial_vec, gesture_vec]
        )

        if use_gate:
            gate = Dense(
                int(fused_vec.shape[-1]), activation='sigmoid', name='fusion_gate'
            )(fused_vec)
            fused_vec = Multiply(name='gated_fusion')([fused_vec, gate])

        fused_features = Dense(
            self.fusion_dim, activation='relu',
            kernel_regularizer=l1_l2(1e-5, 1e-4), name='fusion_proj'
        )(fused_vec)
        fused_features = BatchNormalization(name='fusion_bn')(fused_features)
        fused_features = Dropout(0.3, name='fusion_do')(fused_features)
        fused_features = tf.keras.layers.Lambda(
            lambda t: tf.expand_dims(t, axis=1), name='fusion_expand'
        )(fused_features)
        return fused_features
    
    def _build_classification_head(self, x):
        """Build classification head"""
        # Global pooling
        avg_pool = GlobalAveragePooling1D()(x)
        max_pool = GlobalMaxPooling1D()(x)
        
        # Concatenate pooling methods
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
        """Create training callbacks (early stopping always on; monitors INNER val only)."""
        try:
            callbacks = []
            patience = int(self.config.get('early_stopping_patience', 25))
            es_kwargs = dict(
                monitor='val_accuracy',
                patience=patience,
                restore_best_weights=True,
                verbose=1,
            )
            start_from = int(self.config.get('early_stopping_start_epoch', 15))
            if start_from > 0:
                es_kwargs['start_from_epoch'] = start_from
            early_stopping = EarlyStopping(**es_kwargs)
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

    
    @staticmethod
    def select_inputs_for_variant(X: Any, variant: str) -> Any:
        """Map a 3-modality list ``[motion, face, hands]`` to model inputs."""
        if variant == 'cue_motion':
            return X[0] if isinstance(X, (list, tuple)) else X
        if variant == 'cue_face':
            return X[1] if isinstance(X, (list, tuple)) else X
        if variant == 'cue_hands':
            return X[2] if isinstance(X, (list, tuple)) else X
        return X

    def train(self, X_train: List[np.ndarray], y_train: np.ndarray, 
              X_val: List[np.ndarray], y_val: np.ndarray, 
              model_dir: Optional[Path], epochs: int = 100) -> Dict[str, Any]:
        """Train the model with advanced techniques"""
        try:
            logger.info(f"Starting multimodal training ({self.ablation_variant})...")
            X_train = self.select_inputs_for_variant(X_train, self.ablation_variant)
            X_val = self.select_inputs_for_variant(X_val, self.ablation_variant)
            
            # Convert labels to categorical
            y_train_cat = to_categorical(y_train, num_classes=self.num_classes)
            y_val_cat = to_categorical(y_val, num_classes=self.num_classes)
            
            # Create callbacks (disabled for outer-train refit phase)
            enable_callbacks = bool(self.config.get('enable_callbacks', True))
            if self.config.get('training_phase') == 'refit':
                enable_callbacks = False
            callbacks = self._create_callbacks(model_dir) if enable_callbacks else []
            
            # Train model
            start_time = time.time()
            history = self.model.fit(
                X_train, y_train_cat,
                validation_data=(X_val, y_val_cat) if enable_callbacks else None,
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
            
            logger.info(f"Multimodal training completed in {training_time:.2f}s")
            
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
    
    def evaluate(self, X_test: List[np.ndarray], y_test: np.ndarray) -> Dict[str, Any]:
        """Evaluate on an untouched outer test fold; report macro + weighted metrics."""
        try:
            from src.eval_metrics import classification_metrics

            X_test = self.select_inputs_for_variant(X_test, self.ablation_variant)
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