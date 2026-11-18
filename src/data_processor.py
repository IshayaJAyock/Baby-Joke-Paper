"""
Pure TensorFlow/Keras Data Processing Pipeline for Video Classification
====================================================================

This module provides a pure TensorFlow/Keras data processing pipeline optimized for MacBook M3 MPS.
No multiprocessing, no threading - just pure TensorFlow operations.

Author: Ishaya, Jeremiah Ayock
Date: 2025
"""

import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import pickle
import hashlib
import random
import time
import warnings

# It is important to suppress all warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

logger = logging.getLogger(__name__)

@dataclass
class VideoMetadata:
    """Metadata for a processed video"""
    video_id: str
    label: int
    num_frames: int
    frame_size: Tuple[int, int]
    duration: float
    fps: float
    augmented: bool
    processing_time: float

class PureTFDataProcessor:
    """Pure TensorFlow/Keras data processor for video classification"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.frame_size = (config.get('frame_height', 224), config.get('frame_width', 224))
        self.max_frames = config.get('max_frames', 20)  # 20 frames for better temporal representation
        self.cache_dir = Path(config.get('cache_dir', 'data/cache'))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure TensorFlow for M3 MPS
        self._configure_tf_m3()
        
        # Create feature extractor
        self.feature_extractor = self._create_feature_extractor()
        
    def _configure_tf_m3(self):
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
            logger.info("TensorFlow configured for M3 MPS")
            
        except Exception as e:
            logger.warning(f"TensorFlow configuration failed: {e}")
    
    def _create_feature_extractor(self):
        """Create ResNet50 feature extractor"""
        try:
            base_model = tf.keras.applications.ResNet50V2(
                weights='imagenet',
                include_top=False,
                input_shape=(*self.frame_size, 3),
                pooling='avg'  # This already returns 2D features (batch_size, features)
            )
            
            # Freeze base model
            base_model.trainable = False
            
            # Create feature extractor
            inputs = tf.keras.Input(shape=(*self.frame_size, 3))
            x = base_model(inputs, training=False)
            # No need for GlobalAveragePooling2D since pooling='avg' already does this
            x = tf.keras.layers.Dropout(0.2)(x)
            
            feature_extractor = tf.keras.Model(inputs, x, name='feature_extractor')
            
            logger.info("Feature extractor created")
            return feature_extractor
            
        except Exception as e:
            logger.error(f"Failed to create feature extractor: {e}")
            raise
    
    def _extract_frames_tf(self, video_path: str) -> np.ndarray:
        """Extract frames using pure TensorFlow operations"""
        try:
            # Read video using OpenCV
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Cannot open video: {video_path}")
            
            frames = []
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            # Calculate frame indices for uniform sampling
            if total_frames <= self.max_frames:
                frame_indices = list(range(total_frames))
            else:
                frame_indices = np.linspace(0, total_frames - 1, self.max_frames, dtype=int)
            
            # Extract frames
            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if ret:
                    # Resize frame
                    frame = cv2.resize(frame, self.frame_size)
                    # Convert BGR to RGB
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame)
            
            cap.release()
            
            if not frames:
                raise ValueError(f"No frames extracted from: {video_path}")
            
            # Pad or truncate to max_frames
            while len(frames) < self.max_frames:
                frames.append(frames[-1])  # Repeat last frame
            
            frames = frames[:self.max_frames]
            
            return np.array(frames, dtype=np.float32) / 255.0
            
        except Exception as e:
            logger.error(f"Frame extraction failed for {video_path}: {e}")
            raise
    
    def _augment_frames_tf(self, frames: np.ndarray) -> np.ndarray:
        """Augment frames using TensorFlow operations"""
        try:
            # Convert to tensor
            frames_tensor = tf.convert_to_tensor(frames, dtype=tf.float32)
            
            # Random horizontal flip
            if random.random() > 0.5:
                frames_tensor = tf.image.flip_left_right(frames_tensor)
            
            # Random brightness adjustment
            if random.random() > 0.5:
                frames_tensor = tf.image.random_brightness(frames_tensor, max_delta=0.2)
            
            # Random contrast adjustment
            if random.random() > 0.5:
                frames_tensor = tf.image.random_contrast(frames_tensor, lower=0.8, upper=1.2)
            
            # Geometric rotation is intentionally omitted: tf.image.rot90 only
            # supports 90° steps and a prior placeholder used k=0 (no-op). Flip /
            # brightness / contrast above are the active augmentations.
            
            return frames_tensor.numpy()
            
        except Exception as e:
            logger.warning(f"Augmentation failed: {e}")
            return frames
    
    def _extract_features_tf(self, frames: np.ndarray) -> np.ndarray:
        """Extract features using TensorFlow"""
        try:
            # Ensure frames are in correct format
            if len(frames.shape) == 4:  # (frames, height, width, channels)
                # Process each frame
                features = []
                for frame in frames:
                    frame_batch = np.expand_dims(frame, axis=0)
                    frame_features = self.feature_extractor(frame_batch, training=False)
                    # Squeeze to remove batch dimension: (1, feature_dim) -> (feature_dim,)
                    features.append(frame_features.numpy().squeeze(0))
                
                return np.array(features)
            else:
                raise ValueError(f"Invalid frames shape: {frames.shape}")
                
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            raise
    
    def _get_cache_path(self, video_path: str, augmented: bool = False) -> Path:
        """Get cache file path"""
        video_name = Path(video_path).stem
        cache_key = f"{video_name}_{augmented}_{self.max_frames}_{self.frame_size}"
        cache_hash = hashlib.md5(cache_key.encode()).hexdigest()
        return self.cache_dir / f"{cache_hash}.pkl"
    
    def _load_from_cache(self, cache_path: Path) -> Optional[Tuple[np.ndarray, VideoMetadata]]:
        """Load data from cache"""
        try:
            if cache_path.exists():
                with open(cache_path, 'rb') as f:
                    cached_data = pickle.load(f)
                
                if self._validate_cached_data(cached_data):
                    logger.info(f"Loaded from cache: {cache_path}")
                    return cached_data['features'], cached_data['metadata']
                else:
                    logger.warning(f"Invalid cache data: {cache_path}")
                    cache_path.unlink()
            
            return None
            
        except Exception as e:
            logger.warning(f"Cache load failed: {e}")
            if cache_path.exists():
                cache_path.unlink()
            return None
    
    def _save_to_cache(self, cache_path: Path, features: np.ndarray, metadata: VideoMetadata):
        """Save data to cache"""
        try:
            cache_data = {
                'features': features,
                'metadata': metadata
            }
            
            # Write to temporary file first
            temp_path = cache_path.with_suffix('.tmp')
            with open(temp_path, 'wb') as f:
                pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Atomic rename
            temp_path.rename(cache_path)
            logger.info(f"Saved to cache: {cache_path}")
            
        except Exception as e:
            logger.warning(f"Cache save failed: {e}")
    
    def _validate_cached_data(self, cached_data: dict) -> bool:
        """Validate cached data"""
        try:
            if not isinstance(cached_data, dict):
                return False
            
            if 'features' not in cached_data or 'metadata' not in cached_data:
                return False
            
            features = cached_data['features']
            metadata = cached_data['metadata']
            
            if not isinstance(features, np.ndarray) or not isinstance(metadata, VideoMetadata):
                return False
            
            if features.shape[0] != self.max_frames:
                return False
            
            return True
            
        except Exception:
            return False
    
    def process_video(self, video_path: str, label: int, augmented: bool = False) -> Tuple[np.ndarray, VideoMetadata]:
        """Process a single video"""
        try:
            start_time = time.time()
            
            # Check cache first
            cache_path = self._get_cache_path(video_path, augmented)
            cached_result = self._load_from_cache(cache_path)
            if cached_result:
                return cached_result
            
            # Extract frames
            frames = self._extract_frames_tf(video_path)
            
            # Augment if requested
            if augmented:
                frames = self._augment_frames_tf(frames)
            
            # Extract features
            features = self._extract_features_tf(frames)
            
            # Create metadata
            metadata = VideoMetadata(
                video_id=Path(video_path).stem,
                label=label,
                num_frames=len(frames),
                frame_size=self.frame_size,
                duration=len(frames) / 30.0,  # Approximate
                fps=30.0,
                augmented=augmented,
                processing_time=time.time() - start_time
            )
            
            # Save to cache
            self._save_to_cache(cache_path, features, metadata)
            
            logger.info(f"Processed video: {video_path}")
            return features, metadata
            
        except Exception as e:
            logger.error(f"Video processing failed: {video_path}: {e}")
            raise
    
    def process_dataset(self, video_paths: List[str], labels: List[int], 
                       augmented: bool = False, cache_results: bool = True) -> Tuple[np.ndarray, np.ndarray, List[VideoMetadata]]:
        """Process entire dataset"""
        try:
            logger.info(f"Processing {len(video_paths)} videos...")
            
            all_features = []
            all_labels = []
            all_metadata = []
            
            # Process videos sequentially (no multiprocessing)
            for i, (video_path, label) in enumerate(zip(video_paths, labels)):
                try:
                    features, metadata = self.process_video(video_path, label, augmented)
                    all_features.append(features)
                    all_labels.append(label)
                    all_metadata.append(metadata)
                    
                    if (i + 1) % 10 == 0:
                        logger.info(f"Processed {i + 1}/{len(video_paths)} videos")
                        
                except Exception as e:
                    logger.warning(f"Skipping video {video_path}: {e}")
                    continue
            
            if not all_features:
                raise ValueError("No videos processed successfully")
            
            # Convert to numpy arrays
            X = np.array(all_features)
            y = np.array(all_labels)
            
            logger.info(f"Dataset processed: {X.shape}, {y.shape}")
            return X, y, all_metadata
            
        except Exception as e:
            logger.error(f"Dataset processing failed: {e}")
            raise
    
    def clear_corrupted_cache(self):
        """Clear corrupted cache files"""
        try:
            corrupted_count = 0
            
            for cache_file in self.cache_dir.glob("*.pkl"):
                try:
                    with open(cache_file, 'rb') as f:
                        pickle.load(f)
                except Exception:
                    logger.info(f"Removing corrupted cache: {cache_file}")
                    cache_file.unlink()
                    corrupted_count += 1
            
            for temp_file in self.cache_dir.glob("*.tmp"):
                temp_file.unlink()
                corrupted_count += 1
            
            if corrupted_count > 0:
                logger.info(f"Cleaned {corrupted_count} corrupted cache files")
            else:
                logger.info("No corrupted cache files found")
                
        except Exception as e:
            logger.warning(f"Cache cleanup failed: {e}")
    
    def create_tf_dataset(self, X: np.ndarray, y: np.ndarray, batch_size: int = 32, 
                         shuffle: bool = True, prefetch: bool = True) -> tf.data.Dataset:
        """Create TensorFlow dataset for training"""
        try:
            # Create dataset
            dataset = tf.data.Dataset.from_tensor_slices((X, y))
            
            # Shuffle if requested
            if shuffle:
                dataset = dataset.shuffle(buffer_size=len(X))
            
            # Batch
            dataset = dataset.batch(batch_size)
            
            # Prefetch for performance
            if prefetch:
                dataset = dataset.prefetch(tf.data.AUTOTUNE)
            
            logger.info(f"Created TF dataset: {dataset}")
            return dataset
            
        except Exception as e:
            logger.error(f"TF dataset creation failed: {e}")
            raise
