"""
TensorFlow/Keras Multimodal Model Training and Evaluation Script
===================================================================

This script trains and evaluates the Multimodal model with 10-fold cross-validation using pure TensorFlow/Keras.
Optimized for MacBook M3 MPS with no multiprocessing or threading.

Features:
- TensorFlow/Keras implementation
- Optimized for M3 MPS
- No multiprocessing or threading
- Efficient memory usage
- Comprehensive evaluation

Author: Ishaya, Jeremiah Ayock
Date: 2025
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import warnings
import atexit
import signal
import tensorflow as tf
import cv2
import mediapipe as mp
from scipy.spatial import distance
from dataclasses import dataclass

# Suppress all warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['SKLEARN_THREADING_LAYER'] = 'threading'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMBA_NUM_THREADS'] = '1'

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global cleanup function
def cleanup_resources():
    """Clean up resources to prevent any issues"""
    try:
        # Clear TensorFlow session
        tf.keras.backend.clear_session()
        tf.compat.v1.reset_default_graph()
        
        # Force garbage collection
        import gc
        gc.collect()
        
        logger.info("Resources cleaned up")
        
    except Exception as e:
        logger.warning(f"Cleanup failed: {e}")

# Register cleanup function
atexit.register(cleanup_resources)

# Signal handlers
def signal_handler(signum, frame):
    cleanup_resources()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def configure_tf_m3():
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

def load_data(config: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load and prepare data"""
    try:
        logger.info("Loading data...")
        
        # Load metadata
        metadata_path = Path(config['metadata_path'])
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        
        # Try to read as Excel first, then CSV
        try:
            df = pd.read_excel(metadata_path)
        except Exception:
            df = pd.read_csv(metadata_path)
        logger.info(f"Loaded {len(df)} samples")
        
        # Create video paths and labels from Excel data
        df['video_path'] = 'data/videos/' + df['VideoID']
        
        # Map joke types to numeric labels
        joke_type_mapping = {
            'NomNomNom': 0,
            'Peekaboo': 1,
            'TearingPaper': 2,
            'ThatsNotACat': 3,
            'ThatsNotAHat': 4
        }
        df['label'] = df['JokeType'].map(joke_type_mapping)
        
        config['class_names'] = ['NomNomNom', 'Peekaboo', 'TearingPaper', 'ThatsNotACat', 'ThatsNotAHat']

        # Drop duplicate VideoIDs (keep first), matching the audited cohort (1,404 clips)
        before = len(df)
        df = df.drop_duplicates(subset=['VideoID'], keep='first').reset_index(drop=True)
        if len(df) != before:
            logger.info(f"Dropped {before - len(df)} duplicate VideoID rows -> {len(df)} clips")

        if 'ChildID' not in df.columns:
            raise KeyError('ChildID column required for StratifiedGroupKFold')
        groups = df['ChildID'].astype(str).to_numpy()

        video_paths = np.array(df['video_path'].tolist())
        labels = np.array(df['label'].tolist())

        logger.info(
            f"Data prepared: {len(video_paths)} videos, {len(set(labels))} classes, "
            f"{len(np.unique(groups))} ChildID groups"
        )
        return video_paths, labels, groups
        
    except Exception as e:
        logger.error(f"Data loading failed: {e}")
        raise

# Feature Extractors from multimodal-ok_1nd_April.py
@dataclass
class FeatureMetadata:
    """Metadata for extracted features"""
    name: str
    shape: tuple
    description: str 

class MotionFeatureExtractor:
    """
    Extracts motion features using optical flow.
    Output shape: (num_frames, 8)
    Features: [mean_magnitude, std_magnitude, max_magnitude, mean_angle, 
              std_angle, motion_direction_x, motion_direction_y, motion_intensity]
    """
    def __init__(self): 
        self.flow_params = dict(
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )

    def extract(self, frames: np.ndarray) -> Optional[np.ndarray]:
        try:
            motion_features = []
            prev_frame = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
            
            for frame in frames[1:]:
                curr_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Calculate optical flow
                flow = cv2.calcOpticalFlowFarneback(
                    prev_frame, curr_frame, None, **self.flow_params
                )
                
                # Calculate magnitude and angle
                magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                
                # Extract motion statistics
                mean_mag = np.mean(magnitude)
                std_mag = np.std(magnitude)
                max_mag = np.max(magnitude)
                mean_angle = np.mean(angle)
                std_angle = np.std(angle)
                
                # Calculate motion direction
                motion_dir_x = np.mean(flow[..., 0])
                motion_dir_y = np.mean(flow[..., 1])
                
                # Calculate motion intensity
                motion_intensity = np.mean(np.sqrt(flow[..., 0]**2 + flow[..., 1]**2))
                
                features = [
                    mean_mag, std_mag, max_mag, mean_angle, std_angle,
                    motion_dir_x, motion_dir_y, motion_intensity
                ]
                motion_features.append(features)
                prev_frame = curr_frame
            
            # Handle first frame (no motion)
            motion_features.insert(0, [0] * 8)
            
            features_array = np.array(motion_features)
            logger.info(f"Extracted motion features with shape {features_array.shape}")
            return features_array

        except Exception as e:
            logger.error(f"Error in motion feature extraction: {str(e)}")
            return None

class FacialFeatureExtractor:
    """
    Extracts facial features using MediaPipe Face Mesh.
    Output shape: (num_frames, 15)
    Features: [eye_aspect_ratio_left, eye_aspect_ratio_right, mouth_aspect_ratio,
              eye_distance, eyebrow_height, mouth_height, face_width, face_height,
              nose_tip_position_x, nose_tip_position_y, face_tilt,
              left_eye_openness, right_eye_openness, mouth_openness, smile_ratio]
    """
    
    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=2,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )

    def _calculate_eye_aspect_ratio(self, eye_landmarks: np.ndarray) -> float:
        """Calculate the eye aspect ratio"""
        vertical_dist = (
            distance.euclidean(eye_landmarks[1], eye_landmarks[5]) +
            distance.euclidean(eye_landmarks[2], eye_landmarks[4])
        ) / 2
        horizontal_dist = distance.euclidean(eye_landmarks[0], eye_landmarks[3])
        return vertical_dist / horizontal_dist if horizontal_dist > 0 else 0

    def extract(self, frames: np.ndarray) -> Optional[np.ndarray]:
        try:
            facial_features = []
            
            for frame in frames:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.face_mesh.process(frame_rgb)
                
                if results.multi_face_landmarks:
                    landmarks = np.array([[lm.x, lm.y, lm.z] 
                                        for lm in results.multi_face_landmarks[0].landmark])
                    
                    # Extract facial measurements
                    left_eye = landmarks[33:38]
                    right_eye = landmarks[133:138]
                    mouth_points = landmarks[0:17]
                    
                    features = [
                        self._calculate_eye_aspect_ratio(left_eye),
                        self._calculate_eye_aspect_ratio(right_eye),
                        distance.euclidean(mouth_points[0], mouth_points[6]),  # mouth ratio
                        distance.euclidean(landmarks[33], landmarks[133]),     # eye distance
                        np.mean([landmarks[66][1], landmarks[105][1]]),       # eyebrow height
                        distance.euclidean(landmarks[0], landmarks[17]),      # mouth height
                        distance.euclidean(landmarks[234], landmarks[454]),   # face width
                        distance.euclidean(landmarks[10], landmarks[152]),    # face height
                        landmarks[4][0],                                      # nose tip x
                        landmarks[4][1],                                      # nose tip y
                        np.arctan2(landmarks[33][1] - landmarks[133][1],     # face tilt
                                 landmarks[33][0] - landmarks[133][0]),
                        np.mean([landmarks[159][2], landmarks[145][2]]),     # left eye openness
                        np.mean([landmarks[386][2], landmarks[374][2]]),     # right eye openness
                        np.mean([landmarks[13][2], landmarks[14][2]]),       # mouth openness
                        distance.euclidean(landmarks[61], landmarks[291])     # smile ratio
                    ]
                else:
                    features = [0] * 15  # No face detected
                    
                facial_features.append(features)
            
            features_array = np.array(facial_features)
            logger.info(f"Extracted facial features with shape {features_array.shape}")
            return features_array

        except Exception as e:
            logger.error(f"Error in facial feature extraction: {str(e)}")
            return None

class GestureFeatureExtractor:
    """
    Extracts gesture features using MediaPipe Hands.
    Output shape: (num_frames, 45) - 21 landmarks x 2 coordinates (x, y) + 3 metrics
    """
    def __init__(self):
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=4,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
    def extract(self, frames: np.ndarray) -> Optional[np.ndarray]:
        try:
            gesture_features = []
            
            for frame in frames:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.hands.process(frame_rgb)
                
                frame_features = []
                if results.multi_hand_landmarks:
                    # Get the first hand's landmarks
                    hand_landmarks = results.multi_hand_landmarks[0]
                    
                    # Extract x, y coordinates for each landmark
                    for landmark in hand_landmarks.landmark:
                        frame_features.extend([landmark.x, landmark.y])
                        
                    # Calculate additional gesture metrics
                    # Distance between thumb tip and index finger tip (pinch)
                    thumb_tip = np.array([hand_landmarks.landmark[4].x, hand_landmarks.landmark[4].y])
                    index_tip = np.array([hand_landmarks.landmark[8].x, hand_landmarks.landmark[8].y])
                    pinch_distance = np.linalg.norm(thumb_tip - index_tip)
                    
                    # Overall hand spread (distance between thumb tip and pinky tip)
                    pinky_tip = np.array([hand_landmarks.landmark[20].x, hand_landmarks.landmark[20].y])
                    hand_spread = np.linalg.norm(thumb_tip - pinky_tip)
                    
                    # Hand orientation (angle between middle finger and palm)
                    wrist = np.array([hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y])
                    middle_tip = np.array([hand_landmarks.landmark[12].x, hand_landmarks.landmark[12].y])
                    orientation = np.arctan2(middle_tip[1] - wrist[1], middle_tip[0] - wrist[0])
                    
                    frame_features.extend([pinch_distance, hand_spread, orientation])
                else:
                    # If no hand detected, fill with zeros
                    frame_features = [0.0] * 45  # 21 landmarks * 2 + 3 metrics
                
                gesture_features.append(frame_features)
            
            features_array = np.array(gesture_features)
            logger.info(f"Extracted gesture features with shape {features_array.shape}")
            return features_array

        except Exception as e:
            logger.error(f"Error in gesture feature extraction: {str(e)}")
            return None

def sample_frames(video_path: str, max_frames: int, target_size: tuple = (224, 224)) -> Optional[np.ndarray]:
    """
    Sample frames uniformly from a video file.
    """
    try:
        logger.info(f"Opening video file: {video_path}")
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            logger.error(f"Could not open video: {video_path}")
            return None

        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        logger.info(f"Video properties - Frames: {total_frames}, FPS: {fps}, Size: {width}x{height}")

        if total_frames == 0:
            logger.error(f"Video has no frames: {video_path}")
            return None
        
        if total_frames < max_frames:
            logger.warning(f"Video has fewer frames ({total_frames}) than requested ({max_frames})")

        # Calculate sampling indices
        indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)
        frames = []

        for idx in indices:
            logger.debug(f"Reading frame {idx}")
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.resize(frame, target_size)
                frames.append(frame)
            else:
                logger.warning(f"Failed to read frame {idx} from {video_path}")

        cap.release()

        if len(frames) == 0:
            logger.error(f"No frames were successfully read from {video_path}")
            return None

        if len(frames) < max_frames:
            logger.warning(f"Only extracted {len(frames)} frames from {video_path}")
            last_frame = frames[-1] if frames else np.zeros(
                (target_size[0], target_size[1], 3), dtype=np.uint8
            )
            frames.extend([last_frame] * (max_frames - len(frames)))

        frames_array = np.array(frames)
        logger.info(f"Successfully sampled {len(frames)} frames from {video_path}")
        return frames_array

    except Exception as e:
        logger.error(f"Error sampling frames from {video_path}: {str(e)}")
        return None

def process_multimodal_data(
    video_paths: List[str],
    labels: List[int],
    config: Dict[str, Any],
) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray]:
    """Extract real motion/face/hands features from videos.

    Failed decodes / extractors are **skipped** (no synthetic zero-filled clips).
    Returns ``(X, y_kept, keep_indices)`` so callers can filter ChildID groups.
    """
    try:
        logger.info("Processing multimodal data with motion, facial, and gesture features...")

        labels_arr = np.asarray(labels, dtype=np.int32)
        motion_extractor = MotionFeatureExtractor()
        facial_extractor = FacialFeatureExtractor()
        gesture_extractor = GestureFeatureExtractor()

        X_motion: List[np.ndarray] = []
        X_facial: List[np.ndarray] = []
        X_gesture: List[np.ndarray] = []
        keep_indices: List[int] = []

        logger.info(f"Processing {len(video_paths)} videos for multimodal features...")

        for i, video_path in enumerate(video_paths):
            if i % 100 == 0:
                logger.info(f"Processing video {i+1}/{len(video_paths)}")

            try:
                frames = sample_frames(video_path, config['max_frames'], (224, 224))
                if frames is None:
                    logger.warning(f"Skipping (no frames): {video_path}")
                    continue

                motion_features = motion_extractor.extract(frames)
                facial_features = facial_extractor.extract(frames)
                gesture_features = gesture_extractor.extract(frames)
                if motion_features is None or facial_features is None or gesture_features is None:
                    logger.warning(f"Skipping (cue extract failed): {video_path}")
                    continue

                X_motion.append(motion_features)
                X_facial.append(facial_features)
                X_gesture.append(gesture_features)
                keep_indices.append(i)

            except Exception as e:
                logger.warning(f"Skipping (error) {video_path}: {e}")

        if not keep_indices:
            raise RuntimeError("No videos yielded usable multimodal features")

        keep = np.asarray(keep_indices, dtype=np.int64)
        y = labels_arr[keep]
        X_motion_a = np.asarray(X_motion, dtype=np.float32)
        X_facial_a = np.asarray(X_facial, dtype=np.float32)
        X_gesture_a = np.asarray(X_gesture, dtype=np.float32)

        n_skip = len(video_paths) - len(keep)
        logger.info("Multimodal data processed:")
        logger.info(f"  Kept {len(keep)} / {len(video_paths)} videos (skipped {n_skip})")
        logger.info(f"  Motion features: {X_motion_a.shape}")
        logger.info(f"  Facial features: {X_facial_a.shape}")
        logger.info(f"  Gesture features: {X_gesture_a.shape}")
        logger.info(f"  Labels: {y.shape}")

        return [X_motion_a, X_facial_a, X_gesture_a], y, keep

    except Exception as e:
        logger.error(f"Multimodal data processing failed: {e}")
        raise

def train_single_fold(X_train: List[np.ndarray], y_train: np.ndarray, 
                     X_val: List[np.ndarray], y_val: np.ndarray, 
                     fold_dir: Path, config: Dict[str, Any]) -> Dict[str, Any]:
    """Train model for a single fold"""
    try:
        from models.multimodal import Multimodal
        
        # Create model
        model = Multimodal(config)
        
        # Train model
        training_results = model.train(
            X_train, y_train, X_val, y_val, fold_dir, 
            epochs=config.get('num_epochs', 100)
        )
        
        # Evaluate model
        eval_results = model.evaluate(X_val, y_val)
        
        # Combine results
        results = {
            **training_results,
            **eval_results
        }
        
        # Clear model
        model.clear_session()
        
        return results
        
    except Exception as e:
        logger.error(f"Single fold training failed: {e}")
        raise

def _parse_cli_args():
    import argparse
    from models.multimodal import ABLATION_VARIANTS
    p = argparse.ArgumentParser(description="Multi-cue fusion CV on real videos")
    p.add_argument(
        "--ablation",
        type=str,
        default=None,
        help=(
            "Ablation variant: cue_motion|cue_face|cue_hands|concat|"
            "attention_only|gating_only|full. Default = full (gated cross-attn)."
        ),
    )
    p.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Override results directory (default: config.results_dir or "
             "outputs/ablation_multicue/<variant> when --ablation is set)",
    )
    p.add_argument(
        "--list-ablations",
        action="store_true",
        help="Print supported ablation variants and exit",
    )
    args = p.parse_args()
    if args.list_ablations:
        print("Supported ablation variants:")
        for v in ABLATION_VARIANTS:
            print(f"  - {v}")
        raise SystemExit(0)
    if args.ablation is not None and args.ablation not in ABLATION_VARIANTS:
        raise SystemExit(
            f"Unknown --ablation={args.ablation!r}. Choose from {ABLATION_VARIANTS}"
        )
    return args


def main():
    """Main function"""
    try:
        cli = _parse_cli_args()
        logger.info("Starting Pure TensorFlow/Keras Multimodal Training")
        
        # Configure TensorFlow
        configure_tf_m3()
        
        # Load configuration from config file
        from config.config import StandardConfig
        from models.multimodal import ABLATION_VARIANTS  # noqa: F401
        std_config = StandardConfig()
        ablation = cli.ablation or "full"
        
        # Configuration
        results_dir = cli.results_dir
        if results_dir is None:
            if cli.ablation is not None:
                results_dir = str(Path("outputs/ablation_multicue") / ablation)
            else:
                results_dir = std_config.results_dir

        config = {
            'metadata_path': std_config.metadata_path,
            'cache_dir': 'data/cache',
            'results_dir': results_dir,
            'max_frames': std_config.num_frames,
            'frame_height': std_config.frame_size[0],
            'frame_width': std_config.frame_size[1],
            'motion_dim': 8,  # Motion features (optical flow)
            'facial_dim': 15,  # Facial features (MediaPipe Face Mesh)
            'gesture_dim': 45,  # Gesture features (MediaPipe Hands)
            'fusion_method': ablation,
            'ablation_variant': ablation,
            'attention_heads': std_config.multimodal['attention_heads'],
            'fusion_units': std_config.multimodal['fusion_units'],
            'dropout_rate': std_config.multimodal['dropout_rate'],
            'learning_rate': std_config.learning_rate,
            'batch_size': std_config.batch_size,
            'num_epochs': std_config.num_epochs,
            'n_splits': std_config.cv_folds,
            'random_state': std_config.cv_random_state,
            'use_group_cv': getattr(std_config, 'use_group_cv', True),
            'num_classes': std_config.num_classes
        }
        Path(config['results_dir']).mkdir(parents=True, exist_ok=True)
        logger.info(f"Ablation variant: {ablation}")
        
        # Load data
        video_paths, labels, groups = load_data(config)

        # Process multimodal data (always extract all 3 cues; model selects inputs)
        X, y, keep = process_multimodal_data(video_paths.tolist(), labels.tolist(), config)
        groups = np.asarray(groups)[keep]

        logger.info(f"Dataset processed: {len(X)} modalities, {y.shape}, groups={len(groups)}")

        # Create cross-validation framework
        from src.cross_validation import PureTFCrossValidation
        cv_framework = PureTFCrossValidation(config)

        # Run cross-validation
        logger.info("Running cross-validation...")
        results = cv_framework.run_cross_validation(
            model_name=f"Multimodal_{ablation}",
            model_trainer=train_single_fold,
            X=X, y=y, config=config, groups=groups
        )
        
        # Create summary report
        summary_report = cv_framework.create_summary_report(results)
        logger.info(f"\n{summary_report}")
        
        # Save summary report
        summary_file = Path(config['results_dir']) / 'Multimodal_summary.txt'
        with open(summary_file, 'w') as f:
            f.write(summary_report)
        
        logger.info(f"Summary report saved to: {summary_file}")
        
        # Cleanup
        logger.info("Cleaning up resources...")
        cleanup_resources()
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        cleanup_resources()
        raise

if __name__ == "__main__":
    main()
