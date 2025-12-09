"""
Evaluation and Explanation Module
Computes metrics, creates visualizations, analyzes errors
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score,
    confusion_matrix, classification_report, roc_auc_score
)
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)


class MetricsCalculator:
    """Compute comprehensive evaluation metrics."""
    
    def __init__(self, num_classes=3, class_names=None):
        self.num_classes = num_classes
        self.class_names = class_names or ['Low Risk', 'Moderate Risk', 'High Risk']
    
    def compute_metrics(self, y_true, y_pred, y_proba=None):
        """Compute all evaluation metrics."""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'macro_f1': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'weighted_f1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
            'macro_precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'macro_recall': recall_score(y_true, y_pred, average='macro', zero_division=0),
        }
        
        # Per-class metrics
        per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
        per_class_precision = precision_score(y_true, y_pred, average=None, zero_division=0)
        per_class_recall = recall_score(y_true, y_pred, average=None, zero_division=0)
        
        metrics['per_class_metrics'] = {
            'f1': per_class_f1,
            'precision': per_class_precision,
            'recall': per_class_recall
        }
        
        return metrics
    
    def print_metrics(self, metrics, model_name):
        """Pretty-print metrics."""
        logger.info(f"\n{'='*60}")
        logger.info(f"Metrics for {model_name}")
        logger.info(f"{'='*60}")
        logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"Macro F1: {metrics['macro_f1']:.4f}")
        logger.info(f"Weighted F1: {metrics['weighted_f1']:.4f}")
        logger.info(f"Macro Precision: {metrics['macro_precision']:.4f}")
        logger.info(f"Macro Recall: {metrics['macro_recall']:.4f}")
        
        logger.info(f"\nPer-Class Metrics:")
        for i, class_name in enumerate(self.class_names):
            logger.info(f" {class_name}:")
            logger.info(f"  F1: {metrics['per_class_metrics']['f1'][i]:.4f}")
            logger.info(f"  Precision: {metrics['per_class_metrics']['precision'][i]:.4f}")
            logger.info(f"  Recall: {metrics['per_class_metrics']['recall'][i]:.4f}")


class ResultsVisualizer:
    """Generate confusion matrices, metrics plots, etc."""
    
    def __init__(self, class_names=None):
        self.class_names = class_names or ['Low Risk', 'Moderate Risk', 'High Risk']
    
    def plot_confusion_matrix(self, y_true, y_pred, model_name, save_path=None):
        """Plot confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300)
            logger.info(f"Confusion matrix saved to {save_path}")
        
        plt.close()
    
    def plot_per_class_metrics(self, metrics_dict, save_path=None):
        """Plot per-class F1, precision, recall across models."""
        models = list(metrics_dict.keys())
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        for idx, metric_name in enumerate(['f1', 'precision', 'recall']):
            ax = axes[idx]
            x = np.arange(len(models))
            width = 0.25
            
            for class_idx, class_name in enumerate(self.class_names):
                values = [metrics_dict[model]['per_class_metrics'][metric_name][class_idx]
                         for model in models]
                ax.bar(x + class_idx * width, values, width, label=class_name)
            
            ax.set_xlabel('Model')
            ax.set_ylabel(metric_name.capitalize())
            ax.set_title(f'{metric_name.capitalize()} by Class')
            ax.set_xticks(x + width)
            ax.set_xticklabels(models, rotation=45, ha='right')
            ax.legend()
            ax.set_ylim([0, 1])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300)
            logger.info(f"Per-class metrics plot saved to {save_path}")
        
        plt.close()
    
    def plot_model_comparison(self, metrics_dict, save_path=None):
        """Compare macro metrics across models."""
        models = list(metrics_dict.keys())
        macro_f1s = [metrics_dict[m]['macro_f1'] for m in models]
        weighted_f1s = [metrics_dict[m]['weighted_f1'] for m in models]
        accuracies = [metrics_dict[m]['accuracy'] for m in models]
        
        x = np.arange(len(models))
        width = 0.25
        
        plt.figure(figsize=(10, 6))
        plt.bar(x - width, accuracies, width, label='Accuracy')
        plt.bar(x, macro_f1s, width, label='Macro F1')
        plt.bar(x + width, weighted_f1s, width, label='Weighted F1')
        
        plt.xlabel('Model')
        plt.ylabel('Score')
        plt.title('Model Comparison: Overall Metrics')
        plt.xticks(x, models, rotation=45, ha='right')
        plt.legend()
        plt.ylim([0, 1])
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300)
            logger.info(f"Model comparison plot saved to {save_path}")
        
        plt.close()


class ErrorAnalyzer:
    """Analyze and categorize model errors."""
    
    def __init__(self, class_names=None):
        self.class_names = class_names or ['Low Risk', 'Moderate Risk', 'High Risk']
    
    def categorize_errors(self, texts, y_true, y_pred):
        """Identify and categorize misclassifications."""
        errors = []
        
        for i, (text, true_label, pred_label) in enumerate(zip(texts, y_true, y_pred)):
            if true_label != pred_label:
                errors.append({
                    'text': text[:100] if len(str(text)) > 100 else str(text),
                    'true_label': self.class_names[true_label],
                    'pred_label': self.class_names[pred_label],
                    'error_type': f"{self.class_names[true_label]} → {self.class_names[pred_label]}"
                })
        
        error_df = pd.DataFrame(errors)
        
        logger.info(f"\nTotal errors: {len(error_df)} / {len(y_true)}")
        logger.info(f"Error rate: {len(error_df) / len(y_true) * 100:.2f}%")
        
        if len(error_df) > 0:
            logger.info(f"\nError type distribution:\n{error_df['error_type'].value_counts()}")
        
        return error_df
    
    def print_sample_errors(self, error_df, num_samples=5):
        """Print sample misclassifications."""
        if len(error_df) == 0:
            logger.info("No errors found!")
            return
        
        logger.info(f"\nSample misclassifications (first {num_samples}):")
        for idx, row in error_df.head(num_samples).iterrows():
            logger.info(f"\n Text: {row['text']}...")
            logger.info(f" True: {row['true_label']} | Pred: {row['pred_label']}")


def aggregate_results(results_dict):
    """Aggregate results across all models."""
    summary_df = pd.DataFrame({
        'Model': list(results_dict.keys()),
        'Accuracy': [results_dict[m]['accuracy'] for m in results_dict.keys()],
        'Macro F1': [results_dict[m]['macro_f1'] for m in results_dict.keys()],
        'Weighted F1': [results_dict[m]['weighted_f1'] for m in results_dict.keys()],
    })
    
    logger.info("\n" + "="*60)
    logger.info("FINAL RESULTS SUMMARY")
    logger.info("="*60)
    logger.info(f"\n{summary_df.to_string(index=False)}")
    
    return summary_df


if __name__ == "__main__":
    logger.info("Evaluation module loaded successfully.")
