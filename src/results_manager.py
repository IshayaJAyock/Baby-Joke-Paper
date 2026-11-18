"""
Results Manager for Model Training
=================================

This module handles the organization and saving of training results in a structured format
that matches the outputs/ structure exactly:
- Single global CSV file: all_models_performance.csv (contains all 4 models)
- Individual model subfolders with consistent naming: CNN_LSTM/, TAN/, Multimodal/, 3D_CNN/
- Each model folder contains: classification_report.txt, confusion_matrix.png, accuracy.png, loss.png
- Proper organization of results by model type
- Includes Kappa's score in all reports

Author: Ishaya, Jeremiah Ayock
Date: 2025
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score
import warnings

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class ResultsManager:
    """Manages the organization and saving of training results"""
    
    def __init__(self, results_dir: str = "results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # Initialize results storage
        self.all_results = []
        self.model_results = {}
        self.class_names = None
        
        # Initialize the main CSV file
        self.main_csv_path = self.results_dir / "all_models_performance.csv"
        self._initialize_main_csv()
        
        # Model name to folder mapping (matching outputs/ structure)
        self.model_folder_mapping = {
            'CNN_LSTM': 'CNN_LSTM',
            'TAN': 'TAN',  
            'Multimodal': 'Multimodal',
            '3D_CNN': '3D_CNN'
        }
    
    def _initialize_main_csv(self):
        """Initialize the main CSV file with headers"""
        try:
            if not self.main_csv_path.exists():
                # Create CSV with headers
                headers = [
                    'model_name', 'fold', 'accuracy', 'precision', 'recall', 
                    'f1_score', 'kappa_score', 'training_time'
                ]
                df = pd.DataFrame(columns=headers)
                df.to_csv(self.main_csv_path, index=False)
                logger.info(f"Initialized main CSV: {self.main_csv_path}")
        except Exception as e:
            logger.error(f"Failed to initialize main CSV: {e}")
    
    def _get_model_folder_name(self, model_name: str) -> str:
        """Get the correct folder name for a model based on outputs/ structure"""
        return self.model_folder_mapping.get(model_name, model_name)
    
    def _calculate_kappa_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Cohen's Kappa score"""
        try:
            return cohen_kappa_score(y_true, y_pred)
        except Exception as e:
            logger.warning(f"Kappa score calculation failed: {e}")
            return 0.0
        
    def add_fold_result(self, model_name: str, fold: int, results: Dict[str, Any]):
        """Add results from a single fold"""
        try:
            # Add model name and fold to results
            results['model_name'] = model_name
            results['fold'] = fold
            
            # Calculate kappa score if predictions are available
            if 'y_pred' in results and 'y_true' in results:
                kappa_score = self._calculate_kappa_score(results['y_true'], results['y_pred'])
                results['kappa_score'] = kappa_score
            else:
                results['kappa_score'] = 0.0
            
            # Store in all results
            self.all_results.append(results)
            
            # Store in model-specific results
            if model_name not in self.model_results:
                self.model_results[model_name] = []
            self.model_results[model_name].append(results)
            
            # Append to main CSV immediately
            self._append_to_main_csv(model_name, fold, results)
            
            logger.info(f"Added fold {fold} results for {model_name}")
            
        except Exception as e:
            logger.error(f"Failed to add fold result: {e}")
            raise
    
    def _append_to_main_csv(self, model_name: str, fold: int, results: Dict[str, Any]):
        """Append fold results to the main CSV file"""
        try:
            # Prepare row data
            row_data = {
                'model_name': model_name,
                'fold': fold,
                'accuracy': results.get('accuracy', 0.0),
                'precision': results.get('precision', 0.0),
                'recall': results.get('recall', 0.0),
                'f1_score': results.get('f1_score', 0.0),
                'kappa_score': results.get('kappa_score', 0.0),
                'training_time': results.get('training_time', 0.0)
            }
            
            # Append to CSV
            df = pd.DataFrame([row_data])
            df.to_csv(self.main_csv_path, mode='a', header=False, index=False)
            
        except Exception as e:
            logger.error(f"Failed to append to main CSV: {e}")
    
    def save_all_results_csv(self, filename: str = "all_models_results.csv"):
        """Save all results to a single CSV file"""
        try:
            if not self.all_results:
                logger.warning("No results to save")
                return
            
            # Convert to DataFrame
            df = pd.DataFrame(self.all_results)
            
            # Save to CSV
            csv_path = self.results_dir / filename
            df.to_csv(csv_path, index=False)
            
            logger.info(f"All results saved to: {csv_path}")
            
        except Exception as e:
            logger.error(f"Failed to save CSV results: {e}")
            raise
    
    def save_model_classification_report(self, model_name: str, y_true: np.ndarray, 
                                       y_pred: np.ndarray, class_names: List[str] = None):
        """Save classification report for a specific model"""
        try:
            # Create model subfolder with outputs/ naming convention
            model_folder_name = self._get_model_folder_name(model_name)
            model_dir = self.results_dir / model_folder_name
            model_dir.mkdir(exist_ok=True)
            
            # Generate classification report
            if class_names is None:
                class_names = [f"Class_{i}" for i in range(len(np.unique(y_true)))]
            
            # Ensure all classes appear in the report (even if missing in y_true)
            labels = list(range(len(class_names)))
            report = classification_report(
                y_true, y_pred,
                labels=labels,
                target_names=class_names,
                output_dict=True,
                zero_division=0
            )
            
            # Calculate kappa score
            kappa_score = self._calculate_kappa_score(y_true, y_pred)
            
            # Save as text file (matching outputs/ format exactly)
            txt_path = model_dir / "classification_report.txt"
            with open(txt_path, 'w') as f:
                f.write("Classification Report\n")
                f.write(classification_report(
                    y_true, y_pred,
                    labels=labels,
                    target_names=class_names,
                    zero_division=0
                ))
                f.write(f"\nCohen's Kappa Score: {kappa_score:.4f}\n")
            
            logger.info(f"Classification report saved to: {txt_path}")
            
        except Exception as e:
            logger.error(f"Failed to save classification report: {e}")
            raise
    
    def save_model_confusion_matrix(self, model_name: str, y_true: np.ndarray, 
                                  y_pred: np.ndarray, class_names: List[str] = None):
        """Save confusion matrix visualization for a specific model"""
        try:
            # Create model subfolder with outputs/ naming convention
            model_folder_name = self._get_model_folder_name(model_name)
            model_dir = self.results_dir / model_folder_name
            model_dir.mkdir(exist_ok=True)
            
            # Generate confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            
            # Counts are drawn large enough to fill the cell; the default
            # annotation size leaves a three-digit count sitting in a wide box.
            fig, ax = plt.subplots(figsize=(8.2, 7.2))
            sns.heatmap(
                cm, annot=True, fmt='d', cmap='Blues', square=True, ax=ax,
                xticklabels=class_names, yticklabels=class_names,
                linewidths=0.6, linecolor='white',
                annot_kws={'size': 28, 'weight': 'bold'},
            )
            ax.set_title(f'Confusion Matrix - {model_name}', pad=12)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            fig.tight_layout()
            
            # Save plot
            plot_path = model_dir / "confusion_matrix.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Confusion matrix saved to: {plot_path}")
            
        except Exception as e:
            logger.error(f"Failed to save confusion matrix: {e}")
            raise
    
    def save_model_training_history(self, model_name: str, history: Dict[str, List[float]]):
        """Save training history visualization for a specific model"""
        try:
            # Create model subfolder with outputs/ naming convention
            model_folder_name = self._get_model_folder_name(model_name)
            model_dir = self.results_dir / model_folder_name
            model_dir.mkdir(exist_ok=True)
            
            # Create separate accuracy and loss plots (matching outputs/ structure)
            
            # Plot accuracy
            plt.figure(figsize=(10, 6))
            plt.plot(history['accuracy'], label='Training Accuracy')
            if 'val_accuracy' in history:
                plt.plot(history['val_accuracy'], label='Validation Accuracy')
            plt.title(f'Model Accuracy - {model_name}')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.grid(True)
            
            # Save accuracy plot
            accuracy_path = model_dir / "accuracy.png"
            plt.savefig(accuracy_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Plot loss
            plt.figure(figsize=(10, 6))
            plt.plot(history['loss'], label='Training Loss')
            if 'val_loss' in history:
                plt.plot(history['val_loss'], label='Validation Loss')
            plt.title(f'Model Loss - {model_name}')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
            
            # Save loss plot
            loss_path = model_dir / "loss.png"
            plt.savefig(loss_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Training history saved to: {accuracy_path} and {loss_path}")
            
        except Exception as e:
            logger.error(f"Failed to save training history: {e}")
            raise
    
    def copy_main_csv_to_model_folder(self, model_name: str):
        """Deprecated: Do not duplicate the main CSV into model folders."""
        logger.info("Skipping copying main CSV into model folder to avoid duplicates.")
    
    def save_model_summary(self, model_name: str, summary: str):
        """Save model summary for a specific model"""
        try:
            # Create model subfolder with outputs/ naming convention
            model_folder_name = self._get_model_folder_name(model_name)
            model_dir = self.results_dir / model_folder_name
            model_dir.mkdir(exist_ok=True)
            
            # Save summary
            summary_path = model_dir / f"{model_name.lower()}_summary.txt"
            with open(summary_path, 'w') as f:
                f.write(f"Model Summary - {model_name}\n")
                f.write("=" * 50 + "\n\n")
                f.write(summary)
            
            logger.info(f"Model summary saved to: {summary_path}")
            
        except Exception as e:
            logger.error(f"Failed to save model summary: {e}")
            raise
    
    def get_model_summary_stats(self, model_name: str) -> Dict[str, Any]:
        """Get summary statistics for a specific model"""
        try:
            if model_name not in self.model_results:
                return {}
            
            model_data = self.model_results[model_name]
            
            # Calculate summary statistics
            metrics = ['accuracy', 'val_accuracy', 'loss', 'val_loss']
            summary = {}
            
            for metric in metrics:
                values = [fold[metric] for fold in model_data if metric in fold]
                if values:
                    summary[f"{metric}_mean"] = np.mean(values)
                    summary[f"{metric}_std"] = np.std(values)
                    summary[f"{metric}_min"] = np.min(values)
                    summary[f"{metric}_max"] = np.max(values)
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to get model summary stats: {e}")
            return {}
    
    def save_final_summary(self):
        """Save final summary of all models"""
        try:
            summary_path = self.results_dir / "final_summary.txt"
            
            with open(summary_path, 'w') as f:
                f.write("Final Training Summary\n")
                f.write("=" * 50 + "\n\n")
                
                for model_name in self.model_results.keys():
                    f.write(f"\n{model_name}:\n")
                    f.write("-" * 30 + "\n")
                    
                    stats = self.get_model_summary_stats(model_name)
                    for metric, value in stats.items():
                        f.write(f"{metric}: {value:.4f}\n")
            
            logger.info(f"Final summary saved to: {summary_path}")
            
        except Exception as e:
            logger.error(f"Failed to save final summary: {e}")
            raise
