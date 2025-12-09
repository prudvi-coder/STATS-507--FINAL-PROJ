"""
Main Pipeline: Orchestrate full clinical risk classification project.

Trains the baselines (LR, RF, XGBoost) and transformers (ClinicalBERT, BioBERT, PubMedBERT).

Computes the evaluation metrics, visualizations, and error analysis.
"""
from utils import set_seed
import os
import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import logging
from pathlib import Path

from data_prep import load_mimic_like_data, prepare_datasets, create_torch_datasets
from models import SklearnBaseline, TransformerTrainer
from evaluate_explain import (
    MetricsCalculator, ResultsVisualizer, ErrorAnalyzer,
    aggregate_results
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    'data': {
        'num_samples': 500,
        'test_size': 0.15,
        'val_size': 0.15,
        'balance': False,
    },
    'models': {
        'baseline': ['lr', 'rf', 'xgb'],
        'transformer': ['clinicalbert', 'biobert', 'pubmedbert'],
    },
    'training': {
        'batch_size': 16,
        'num_epochs': 5,
        'learning_rate': 1e-5,
        'max_length': 128,
    },
    'output_dir': './results',
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {DEVICE}")

# ============================================================================
# MAIN PIPELINE
# ============================================================================

class ClinicalRiskClassificationPipeline:
    """End-to-end pipeline for clinical risk classification."""
    
    def __init__(self, config=CONFIG):
        self.config = config
        self.results = {}
        self.all_test_texts = None
        self.all_test_labels = None
        
        # Create output directory
        Path(self.config['output_dir']).mkdir(exist_ok=True)
    
    def run(self):
        """Execute full pipeline."""
        logger.info("\n" + "="*80)
        logger.info("CLINICAL RISK CLASSIFICATION PIPELINE")
        logger.info("="*80)
        
        # Step 1: Load and prepare data
        logger.info("\n[STEP 1] Loading and preparing data...")
        df, split_data = self._load_data()
        
        # Store test set for error analysis
        self.all_test_texts = split_data['test_texts'].values
        self.all_test_labels = split_data['test_labels'].values
        
        # Step 2: Train baseline models
        logger.info("\n[STEP 2] Training baseline models (LR, RF, XGBoost)...")
        self._train_baselines(split_data)
        
        # Step 3: Train transformer models
        logger.info("\n[STEP 3] Training transformer models...")
        self._train_transformers(split_data)
        
        # Step 4: Evaluate all models
        logger.info("\n[STEP 4] Evaluating all models...")
        self._evaluate_all_models(split_data)
        
        # Step 5: Visualize results
        logger.info("\n[STEP 5] Generating visualizations...")
        self._visualize_results()
        
        # Step 6: Error analysis
        logger.info("\n[STEP 6] Performing error analysis...")
        self._error_analysis()
        
        logger.info("\n" + "="*80)
        logger.info("PIPELINE COMPLETE")
        logger.info("="*80)
    
    def _load_data(self):
        """Load and prepare dataset."""
        DATA_PATH = "healthrisk_1500.csv"
        df = load_mimic_like_data(
            filepath=DATA_PATH,
            num_samples=self.config['data']['num_samples']
        )
        split_data = prepare_datasets(
            df,
            test_size=self.config['data']['test_size'],
            val_size=self.config['data']['val_size'],
            balance=self.config['data']['balance']
        )
        return df, split_data
    
    def _train_baselines(self, split_data):
        """Train sklearn baseline models."""
        logger.info(f"Training baseline models: {self.config['models']['baseline']}")
        
        train_texts = split_data['train_texts'].values
        train_labels = split_data['train_labels'].values
        
        for model_type in self.config['models']['baseline']:
            logger.info(f"\nTraining {model_type.upper()}...")
            
            baseline = SklearnBaseline(model_type=model_type, num_classes=3)
            baseline.fit(train_texts, train_labels)
            
            self.results[model_type.upper()] = baseline
    
    def _train_transformers(self, split_data):
        """Train transformer models."""
        logger.info(f"Training transformer models: {self.config['models']['transformer']}")
        
        from transformers import AutoTokenizer
        
        for model_name in self.config['models']['transformer']:
            logger.info(f"\nTraining {model_name.upper()}...")
            
            try:
                if model_name == "pubmedbert":
                    custom_lr = 5e-6
                    logger.info(f"Using custom learning rate {custom_lr} for PubMedBERT")
                else:
                    custom_lr = self.config['training']['learning_rate']
        
                trainer = TransformerTrainer(
                    model_name=model_name,
                    num_classes=3,
                    batch_size=self.config['training']['batch_size'],
                    learning_rate=custom_lr,
                    num_epochs=self.config['training']['num_epochs']
                )
                
                # Create dataloaders
                tokenizer = trainer.tokenizer
                train_ds, val_ds, test_ds = create_torch_datasets(
                    split_data, tokenizer,
                    max_length=self.config['training']['max_length']
                )
                
                train_loader = DataLoader(
                    train_ds,
                    batch_size=self.config['training']['batch_size'],
                    shuffle=True
                )
                
                val_loader = DataLoader(
                    val_ds,
                    batch_size=self.config['training']['batch_size'],
                    shuffle=False
                )
                
                # Fine-tune
                trainer.fit(
                    train_loader,
                    val_loader,
                    num_epochs=self.config['training']['num_epochs']
                )
                
                self.results[model_name.upper()] = trainer
                
                # Save model
                model_path = os.path.join(self.config['output_dir'], f"{model_name}_finetuned")
                trainer.save_model(model_path)
                
            except Exception as e:
                logger.error(f"Error training {model_name}: {e}")
                continue
    
    def _evaluate_all_models(self, split_data):
        """Evaluate all trained models on test set."""
        test_texts = split_data['test_texts'].values
        test_labels = split_data['test_labels'].values
        
        metrics_calculator = MetricsCalculator()
        
        for model_name, model in list(self.results.items()):
            logger.info(f"\nEvaluating {model_name}...")
            
            if isinstance(model, SklearnBaseline):
                # Sklearn baseline
                y_pred = model.predict(test_texts)
                y_proba = model.predict_proba(test_texts)
            
            else:
                # Transformer
                tokenizer = model.tokenizer
                from data_prep import ClinicalNotesDataset
                
                test_ds = ClinicalNotesDataset(
                    test_texts, test_labels, tokenizer,
                    max_length=self.config['training']['max_length']
                )
                
                test_loader = DataLoader(
                    test_ds,
                    batch_size=self.config['training']['batch_size'],
                    shuffle=False
                )
                
                y_pred, y_proba = model.predict(test_loader)
            
            # Compute metrics
            metrics = metrics_calculator.compute_metrics(test_labels, y_pred, y_proba)
            metrics_calculator.print_metrics(metrics, model_name)
            
            # Store results
            self.results[f"{model_name}_metrics"] = metrics
            self.results[f"{model_name}_pred"] = y_pred
            self.results[f"{model_name}_proba"] = y_proba
    
    def _visualize_results(self):
        """Generate visualizations."""
        visualizer = ResultsVisualizer()
        
        # Extract all metrics
        metrics_dict = {
            k.replace('_metrics', ''): v
            for k, v in self.results.items() if '_metrics' in k
        }
        
        # Plot model comparison
        summary_df = aggregate_results(metrics_dict)
        
        visualizer.plot_model_comparison(
            metrics_dict,
            save_path=os.path.join(self.config['output_dir'], 'model_comparison.png')
        )
        
        # Plot per-class metrics
        visualizer.plot_per_class_metrics(
            metrics_dict,
            save_path=os.path.join(self.config['output_dir'], 'per_class_metrics.png')
        )
        
        # Confusion matrices for each model
        for model_name in list(self.results.keys()):
            if '_pred' in model_name:
                base_name = model_name.replace('_pred', '')
                y_pred = self.results[model_name]
                
                visualizer.plot_confusion_matrix(
                    self.all_test_labels, y_pred, base_name,
                    save_path=os.path.join(self.config['output_dir'], f'confusion_matrix_{base_name}.png')
                )
        
        # Save summary
        summary_df.to_csv(
            os.path.join(self.config['output_dir'], 'results_summary.csv'),
            index=False
        )
        
        logger.info(f"Results saved to {self.config['output_dir']}")
    
    def _error_analysis(self):
        """Analyze model errors."""
        error_analyzer = ErrorAnalyzer()
        
        for model_name in list(self.results.keys()):
            if '_pred' in model_name:
                base_name = model_name.replace('_pred', '')
                y_pred = self.results[model_name]
                
                logger.info(f"\n[Error Analysis] {base_name}")
                
                error_df = error_analyzer.categorize_errors(
                    self.all_test_texts, self.all_test_labels, y_pred
                )
                
                error_analyzer.print_sample_errors(error_df, num_samples=3)
                
                # Save error log
                error_df.to_csv(
                    os.path.join(self.config['output_dir'], f'errors_{base_name}.csv'),
                    index=False
                )


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    set_seed(42)
    pipeline = ClinicalRiskClassificationPipeline(config=CONFIG)
    pipeline.run()
    
    logger.info(f"\nAll outputs saved to: {CONFIG['output_dir']}")
