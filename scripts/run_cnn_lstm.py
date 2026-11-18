
"""
CNN-LSTM Model Training and Evaluation Script
=================================================================

This script trains and evaluates the CNN-LSTM model with 10-fold cross-validation using pure TensorFlow/Keras.
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
        
        # Store class names for later use
        config['class_names'] = ['NomNomNom', 'Peekaboo', 'TearingPaper', 'ThatsNotACat', 'ThatsNotAHat']

        # Drop duplicate VideoIDs (keep first), matching the audited cohort (1,404 clips)
        before = len(df)
        df = df.drop_duplicates(subset=['VideoID'], keep='first').reset_index(drop=True)
        if len(df) != before:
            logger.info(f"Dropped {before - len(df)} duplicate VideoID rows -> {len(df)} clips")

        # ChildID == dyad in this workbook (1:1 with ParentID); used for group CV
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

def train_single_fold(X_train, y_train, X_val, y_val, fold_dir, config, X_test=None, y_test=None):
    """Train on fit set, monitor INNER val for callbacks, score OUTER test when provided."""
    try:
        from models.cnn_lstm import CNNLSTM
        model = CNNLSTM(config)
        training_results = model.train(
            X_train, y_train, X_val, y_val, fold_dir,
            epochs=config.get('num_epochs', 100),
        )
        eval_X = X_test if X_test is not None else X_val
        eval_y = y_test if y_test is not None else y_val
        eval_results = model.evaluate(eval_X, eval_y)
        results = {**training_results, **eval_results}
        model.clear_session()
        return results
    except Exception as e:
        logger.error(f"Single fold training failed: {e}")
        raise


def main():
    """Main function"""
    try:
        logger.info("Starting Pure TensorFlow/Keras CNN-LSTM Training")
        
        # Configure TensorFlow
        configure_tf_m3()
        
        # Load configuration from config file
        from config.config import StandardConfig, set_random_seeds
        set_random_seeds()
        std_config = StandardConfig()
        
        # Configuration
        config = {
            'metadata_path': std_config.metadata_path,
            'cache_dir': 'data/cache',
            'results_dir': std_config.results_dir,
            'max_frames': std_config.num_frames,
            'frame_height': std_config.frame_size[0],
            'frame_width': std_config.frame_size[1],
            'feature_dim': 2048,
            'lstm_units': std_config.cnn_lstm['lstm_units'][0],
            'dropout_rate': std_config.cnn_lstm['dropout_rate'],
            'learning_rate': std_config.learning_rate,
            'batch_size': std_config.batch_size,
            'num_epochs': std_config.num_epochs,
            'n_splits': std_config.cv_folds,
            'random_state': std_config.cv_random_state,
            'use_group_cv': getattr(std_config, 'use_group_cv', True),
            'num_classes': std_config.num_classes
        }
        
        # Load data
        video_paths, labels, groups = load_data(config)
        
        # Create data processor
        from src.data_processor import PureTFDataProcessor
        data_processor = PureTFDataProcessor(config)
        
        # Clear corrupted cache
        logger.info("Clearing corrupted cache...")
        data_processor.clear_corrupted_cache()
        
        # Process dataset
        logger.info("Processing dataset...")
        X, y, metadata = data_processor.process_dataset(
            video_paths.tolist(), labels.tolist(), 
            augmented=False, cache_results=True
        )
        
        logger.info(f"Dataset processed: {X.shape}, {y.shape}")
        
        # Create cross-validation framework
        from src.cross_validation import PureTFCrossValidation
        cv_framework = PureTFCrossValidation(config)
        
        # Run cross-validation
        logger.info("Running cross-validation...")
        results = cv_framework.run_cross_validation(
            model_name="CNN_LSTM",
            model_trainer=train_single_fold,
            X=X, y=y, config=config, groups=groups
        )
        
        # Update class names in results manager
        if hasattr(cv_framework, 'results_manager'):
            cv_framework.results_manager.class_names = config.get('class_names', 
                ['NomNomNom', 'Peekaboo', 'TearingPaper', 'ThatsNotACat', 'ThatsNotAHat'])
        
        # Create summary report
        summary_report = cv_framework.create_summary_report(results)
        logger.info(f"\n{summary_report}")
        
        # Save summary report
        summary_file = Path(config['results_dir']) / 'CNN_LSTM_summary.txt'
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
