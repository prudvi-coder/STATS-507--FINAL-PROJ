"""
Models Module
Implements both baseline (sklearn) and transformer models

This module provides:
- SklearnBaseline: LR, RF, XGBoost with TF-IDF
- TransformerTrainer: Fine-tune ClinicalBERT, BioBERT, PubMedBERT
"""

import logging
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import warnings

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SklearnBaseline:
    """Wrapper for sklearn-based baseline models with TF-IDF."""
    
    def __init__(self, model_type='lr', num_classes=3, random_state=42):
        self.model_type = model_type
        self.num_classes = num_classes
        self.random_state = random_state
        self.vectorizer = None
        self.model = None
    
    def fit(self, X_train, y_train):
        """Train the model."""
        
        # Create TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )
        
        X_train_vectorized = self.vectorizer.fit_transform(X_train)
        
        # Initialize model
        if self.model_type == 'lr':
            self.model = LogisticRegression(
                C=1.0,
                penalty='l2',
                solver='lbfgs',
                multi_class='multinomial',
                class_weight='balanced',
                random_state=self.random_state,
                max_iter=1000
            )
        
        elif self.model_type == 'rf':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=self.random_state,
                class_weight='balanced',
                n_jobs=-1
            )
        
        elif self.model_type == 'xgb':
            self.model = XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=self.random_state,
                eval_metric='mlogloss'
            )
        
        self.model.fit(X_train_vectorized, y_train)
        logger.info(f"{self.model_type.upper()} model trained")
    
    def predict(self, X_test):
        """Get predictions."""
        X_test_vectorized = self.vectorizer.transform(X_test)
        return self.model.predict(X_test_vectorized)
    
    def predict_proba(self, X_test):
        """Get prediction probabilities."""
        X_test_vectorized = self.vectorizer.transform(X_test)
        return self.model.predict_proba(X_test_vectorized)


class TransformerTrainer:
    """Fine-tune transformer models from HuggingFace."""
    
    MODEL_MAPPING = {
        'clinicalbert': 'emilyalsentzer/clinicalbert',
        'biobert': 'dmis-lab/biobert-v1.1',
        'pubmedbert': 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext',
    }
    
    def __init__(self, model_name='clinicalbert', num_classes=3, 
                 batch_size=16, learning_rate=2e-5, num_epochs=3):
        
        self.model_name = model_name
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        model_id = self.MODEL_MAPPING[model_name]
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            num_labels=num_classes
        ).to(self.device)
        
        logger.info(f"Loaded {model_name}")
        self.history = {"train_loss": [], "val_loss": []}
    
    def fit(self, train_loader, val_loader, num_epochs=None):
        """Fine-tune the transformer."""
        
        if num_epochs is None:
            num_epochs = self.num_epochs
        
        optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)
        
        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0
            
            for batch in train_loader:
                optimizer.zero_grad()
                
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_loader)
            logger.info(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}")
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                val_loss = 0
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['label'].to(self.device)
                    
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    
                    val_loss += outputs.loss.item()
                
                avg_val_loss = val_loss / len(val_loader)
                logger.info(f"Validation Loss: {avg_val_loss:.4f}")
            self.history["train_loss"].append(avg_loss)
            self.history["val_loss"].append(avg_val_loss)
    
    def predict(self, test_loader):
        """Get predictions and probabilities."""
        
        self.model.eval()
        predictions = []
        probabilities = []
        
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                logits = outputs.logits
                preds = torch.argmax(logits, dim=-1).cpu().numpy()
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                
                predictions.extend(preds)
                probabilities.extend(probs)
        
        return np.array(predictions), np.array(probabilities)
    
    def save_model(self, path):
        """Save the fine-tuned model."""
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        logger.info(f"Model saved to {path}")
