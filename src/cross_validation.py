"""
Pure TensorFlow/Keras Cross-Validation Framework
==============================================

This module provides a pure TensorFlow/Keras cross-validation framework optimized for MacBook M3 MPS.
No multiprocessing, no threading - just pure TensorFlow operations.

Author: Ishaya, Jeremiah Ayock
Date: 2025
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Callable
import warnings
import time
from .results_manager import ResultsManager
import json

# Suppress all warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

logger = logging.getLogger(__name__)

class PureTFCrossValidation:
    """Pure TensorFlow/Keras cross-validation framework"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.n_splits = config.get('n_splits', 10)
        self.random_state = config.get('random_state', 42)
        self.results_dir = Path(config.get('results_dir', 'results'))
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize results manager
        self.results_manager = ResultsManager(str(self.results_dir))
        
        # Configure TensorFlow for M3 MPS
        self._configure_tf_m3()
        
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
    
    def run_cross_validation(self, model_name: str, model_trainer: Callable, 
                           X: Any, y: np.ndarray, 
                           config: Dict[str, Any],
                           groups: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Run cross-validation.

        Infant / dyad hold-out (default when ``groups`` is provided or
        ``config['use_group_cv']`` is True): ``StratifiedGroupKFold`` on
        ``ChildID``. Clip-wise ``StratifiedKFold`` is retained only if
        ``use_group_cv`` is explicitly False *and* groups are omitted.
        """
        try:
            logger.info(f"Starting {self.n_splits}-fold cross-validation for {model_name}")
            
            # Support multimodal inputs: X can be a list/tuple of modality arrays
            is_multimodal = isinstance(X, (list, tuple))
            if is_multimodal:
                if len(X) == 0:
                    raise ValueError("Received empty modalities list for X")
                # Validate that all modalities have the same number of samples and match y
                num_samples_per_modality = [len(modality) for modality in X]
                if len(set(num_samples_per_modality)) != 1:
                    raise ValueError(f"All modalities must have the same number of samples, got: {num_samples_per_modality}")
                n_samples = num_samples_per_modality[0]
            else:
                # Assume X is a single array-like with length equal to y
                try:
                    n_samples = len(X)
                except Exception:
                    n_samples = len(y)
            
            if n_samples != len(y):
                raise ValueError(f"Number of samples in X ({n_samples}) and y ({len(y)}) must match")

            use_group_cv = bool(config.get('use_group_cv', True))
            if groups is not None:
                groups = np.asarray(groups)
                if len(groups) != n_samples:
                    raise ValueError(
                        f"groups length ({len(groups)}) must match n_samples ({n_samples})"
                    )

            if use_group_cv:
                if groups is None:
                    raise ValueError(
                        "use_group_cv=True requires `groups` (e.g. ChildID) so that "
                        "no infant appears in both train and validation."
                    )
                kfold = StratifiedGroupKFold(
                    n_splits=self.n_splits, shuffle=True, random_state=self.random_state
                )
                split_iter = kfold.split(np.zeros(n_samples), y, groups)
                logger.info(
                    "Using StratifiedGroupKFold on groups "
                    f"(n_groups={len(np.unique(groups))}, seed={self.random_state})"
                )
            else:
                kfold = StratifiedKFold(
                    n_splits=self.n_splits, shuffle=True, random_state=self.random_state
                )
                split_iter = kfold.split(np.zeros(n_samples), y)
                logger.warning(
                    "Using clip-wise StratifiedKFold (use_group_cv=False); "
                    "infant identity can leak between train and validation."
                )
            
            # Store results
            fold_results = []
            all_predictions = []
            all_true_labels = []
            
            # Run cross-validation
            for fold, (train_idx, val_idx) in enumerate(split_iter):
                try:
                    logger.info(f"Processing fold {fold + 1}/{self.n_splits}")
                    
                    # Split data
                    if is_multimodal:
                        X_train = [modality[train_idx] for modality in X]
                        X_val = [modality[val_idx] for modality in X]
                    else:
                        X_train, X_val = X[train_idx], X[val_idx]
                    y_train, y_val = y[train_idx], y[val_idx]
                    
                    # Train model (no fold directory creation)
                    fold_start_time = time.time()
                    # Compute per-fold class weights to tackle imbalance
                    try:
                        from sklearn.utils.class_weight import compute_class_weight
                        classes = np.unique(y_train)
                        class_weights_arr = compute_class_weight('balanced', classes=classes, y=y_train)
                        class_weight = {int(c): float(w) for c, w in zip(classes, class_weights_arr)}
                    except Exception:
                        class_weight = None
                    
                    fold_results_dict = model_trainer(
                        X_train, y_train, X_val, y_val, None, {**config, 'class_weight': class_weight}
                    )
                    fold_time = time.time() - fold_start_time
                    
                    # Add fold information
                    fold_results_dict['fold'] = fold + 1
                    fold_results_dict['fold_time'] = fold_time
                    # Record sizes (use first modality when multimodal)
                    fold_results_dict['train_size'] = len(X_train[0]) if is_multimodal else len(X_train)
                    fold_results_dict['val_size'] = len(X_val[0]) if is_multimodal else len(X_val)
                    
                    # Store predictions
                    if 'y_pred' in fold_results_dict:
                        all_predictions.extend(fold_results_dict['y_pred'])
                        all_true_labels.extend(y_val)
                    
                    # Add true labels and predictions to results for kappa calculation
                    fold_results_dict['y_true'] = y_val
                    if 'y_pred' not in fold_results_dict:
                        # If predictions not in results, add them
                        fold_results_dict['y_pred'] = fold_results_dict.get('y_pred', [])
                    
                    # Add to results manager
                    self.results_manager.add_fold_result(model_name, fold + 1, fold_results_dict)
                    
                    fold_results.append(fold_results_dict)
                    
                    logger.info(f"Fold {fold + 1} completed in {fold_time:.2f}s")
                    
                    # Clear TensorFlow session after each fold
                    tf.keras.backend.clear_session()
                    
                except Exception as e:
                    logger.error(f"Fold {fold + 1} failed: {e}")
                    continue
            
            # Calculate overall metrics
            overall_metrics = self._calculate_overall_metrics(fold_results, all_predictions, all_true_labels)
            
            # Save results
            results = {
                'model_name': model_name,
                'n_splits': self.n_splits,
                'fold_results': fold_results,
                'overall_metrics': overall_metrics,
                'config': config
            }
            
            # Save to file
            results_file = self.results_dir / f"{model_name}_cv_results.json"
            self._save_results(results, results_file)
            
            # Save model-specific results (use real class names; don't create per-model CSV copies)
            if all_predictions and all_true_labels:
                try:
                    from config.config import JOKE_TYPE_NAMES
                    class_names = JOKE_TYPE_NAMES
                except Exception:
                    class_names = [f"Class_{i}" for i in range(len(np.unique(all_true_labels)))]
                self.results_manager.save_model_classification_report(
                    model_name, np.array(all_true_labels), np.array(all_predictions), class_names
                )
                self.results_manager.save_model_confusion_matrix(
                    model_name, np.array(all_true_labels), np.array(all_predictions), class_names
                )
            
            logger.info(f"Cross-validation completed for {model_name}")
            return results
            
        except Exception as e:
            logger.error(f"Cross-validation failed: {e}")
            raise
    
    def _calculate_overall_metrics(self, fold_results: List[Dict], 
                                 all_predictions: List, all_true_labels: List) -> Dict[str, Any]:
        """Calculate overall metrics across all folds"""
        try:
            if not fold_results:
                return {}
            
            # Calculate mean and std for each metric
            metrics = ['accuracy', 'precision', 'recall', 'f1_score']
            overall_metrics = {}
            
            for metric in metrics:
                values = [fold.get(metric, 0) for fold in fold_results if metric in fold]
                if values:
                    overall_metrics[f'{metric}_mean'] = np.mean(values)
                    overall_metrics[f'{metric}_std'] = np.std(values)
                    overall_metrics[f'{metric}_min'] = np.min(values)
                    overall_metrics[f'{metric}_max'] = np.max(values)
            
            # Calculate overall confusion matrix
            if all_predictions and all_true_labels:
                cm = confusion_matrix(all_true_labels, all_predictions)
                overall_metrics['confusion_matrix'] = cm.tolist()
                
                # Calculate overall classification report
                report = classification_report(all_true_labels, all_predictions, output_dict=True)
                overall_metrics['classification_report'] = report
            
            # Calculate training times
            training_times = [fold.get('training_time', 0) for fold in fold_results if 'training_time' in fold]
            if training_times:
                overall_metrics['total_training_time'] = np.sum(training_times)
                overall_metrics['avg_training_time'] = np.mean(training_times)
                overall_metrics['std_training_time'] = np.std(training_times)
            
            return overall_metrics
            
        except Exception as e:
            logger.warning(f"Overall metrics calculation failed: {e}")
            return {}
    
    def _save_results(self, results: Dict[str, Any], results_file: Path):
        """Save results to file"""
        try:
            # Convert numpy arrays to lists for JSON serialization
            def convert_numpy(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, dict):
                    return {key: convert_numpy(value) for key, value in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy(item) for item in obj]
                else:
                    return obj
            
            # Convert results
            results_serializable = convert_numpy(results)
            
            # Save to JSON
            with open(results_file, 'w') as f:
                json.dump(results_serializable, f, indent=2)
            
            logger.info(f"Results saved to: {results_file}")
            
        except Exception as e:
            logger.warning(f"Results saving failed: {e}")
    
    def create_summary_report(self, results: Dict[str, Any]) -> str:
        """Create a summary report"""
        try:
            report = []
            report.append("=" * 60)
            report.append(f"CROSS-VALIDATION RESULTS: {results['model_name']}")
            report.append("=" * 60)
            report.append(f"Number of folds: {results['n_splits']}")
            report.append("")
            
            # Overall metrics
            overall_metrics = results.get('overall_metrics', {})
            if overall_metrics:
                report.append("OVERALL METRICS:")
                report.append("-" * 20)
                
                metrics = ['accuracy', 'precision', 'recall', 'f1_score']
                for metric in metrics:
                    mean_key = f'{metric}_mean'
                    std_key = f'{metric}_std'
                    if mean_key in overall_metrics and std_key in overall_metrics:
                        report.append(f"{metric.capitalize()}: {overall_metrics[mean_key]:.4f} ± {overall_metrics[std_key]:.4f}")
                
                # Training time
                if 'total_training_time' in overall_metrics:
                    report.append(f"Total training time: {overall_metrics['total_training_time']:.2f}s")
                    report.append(f"Average training time: {overall_metrics['avg_training_time']:.2f}s")
            
            report.append("")
            report.append("FOLD DETAILS:")
            report.append("-" * 20)
            
            # Individual fold results
            fold_results = results.get('fold_results', [])
            for i, fold in enumerate(fold_results):
                report.append(f"Fold {i + 1}:")
                report.append(f"  Accuracy: {fold.get('accuracy', 0):.4f}")
                report.append(f"  Precision: {fold.get('precision', 0):.4f}")
                report.append(f"  Recall: {fold.get('recall', 0):.4f}")
                report.append(f"  F1-Score: {fold.get('f1_score', 0):.4f}")
                report.append(f"  Training time: {fold.get('training_time', 0):.2f}s")
                report.append("")
            
            return "\n".join(report)
            
        except Exception as e:
            logger.warning(f"Summary report creation failed: {e}")
            return "Summary report not available"
    
    def clear_session(self):
        """Clear TensorFlow session"""
        try:
            tf.keras.backend.clear_session()
            logger.info("TensorFlow session cleared")
        except Exception as e:
            logger.warning(f"Session clearing failed: {e}")