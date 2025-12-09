"""
Data Preparation Module
Handles all data loading, preprocessing, and splitting

This module creates synthetic clinical notes for demonstration and prepares
the data for training machine learning models.
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import re
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ClinicalNotesDataset(Dataset):
    """Custom PyTorch Dataset for clinical notes."""
    
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'label': torch.tensor(label, dtype=torch.long)
        }


class DataPreprocessor:
    """Preprocess clinical notes: clean PHI, normalize, encode."""
    
    def __init__(self):
        self.phi_patterns = [
            r'\b\d{1,2}/\d{1,2}/\d{2,4}\b',  # dates
            r'\b\d{3}-\d{2}-\d{4}\b',        # SSN-like
            r'\[\*{2}\w+\*{2}\]',           # de-identified markers
        ]
    
    def clean_phi(self, text):
        """Remove or mask PHI patterns."""
        for pattern in self.phi_patterns:
            text = re.sub(pattern, '[MASKED]', text)
        return text
    
    def normalize_text(self, text):
        """Basic normalization: lowercase, remove extra whitespace."""
        text = text.lower()
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def preprocess(self, texts):
        """Apply all preprocessing steps."""
        cleaned = [self.clean_phi(t) for t in texts]
        normalized = [self.normalize_text(t) for t in cleaned]
        return normalized


def load_mimic_like_data(filepath=None, num_samples=500):
    """
    Load or generate MIMIC-like clinical notes dataset.
    
    If filepath is None, generates synthetic data for demonstration.
    """
    if filepath and isinstance(filepath, str):
        df = pd.read_csv(filepath)
    else:
        logger.warning("Generating synthetic clinical notes for demonstration.")
        
        # Synthetic data for local testing
        low_risk_notes = [
            "Patient presents with mild headache. Vitals stable. No acute findings.",
            "Follow-up visit for hypertension management. BP well-controlled on current meds.",
            "Routine physical exam. Patient in good health. No complaints.",
            "Cough and congestion. Likely viral URI. Symptomatic treatment recommended.",
            "Back pain related to muscle strain. Prescribed physical therapy.",
        ]
        
        moderate_risk_notes = [
            "Chest pain, troponin negative. EKG normal. Monitor closely.",
            "SOB on exertion. Mild hypoxia. Consider chest imaging.",
            "Palpitations and dizziness. HR elevated. Needs ECG and cardiology consult.",
            "Abdominal pain, fever. Possible appendicitis. Urgent imaging needed.",
            "Fever 101.5F, cough. R/O pneumonia. Start empiric antibiotics.",
        ]
        
        high_risk_notes = [
            "Acute CP radiating to L arm. Diaphoretic. Elevated troponin. STEMI alert.",
            "Severe SOB, acute hypoxia. SpO2 88%. Possible PE. Activate respiratory team.",
            "Altered mental status, BP 220/120. Possible stroke. Stat CT ordered.",
            "Hemorrhage GI bleed. Hemoglobin drop 8.2. Massive transfusion protocol.",
            "Septic shock. Lactate 4.5. Hypotensive. ICU admission stat.",
        ]
        
        # Create balanced dataset with 70% low, 25% moderate, 5% high
        texts = low_risk_notes * 70 + moderate_risk_notes * 25 + high_risk_notes * 5
        labels = [0] * (70 * len(low_risk_notes)) + [1] * (25 * len(moderate_risk_notes)) + [2] * (5 * len(high_risk_notes))
        
        # Trim to requested sample size
        texts = texts[:num_samples]
        labels = labels[:num_samples]
        
        df = pd.DataFrame({'text': texts, 'risk_level': labels})
    
    logger.info(f"Loaded {len(df)} clinical notes.")
    logger.info(f"Class distribution:\n{df['risk_level'].value_counts()}")
    
    return df


def prepare_datasets(df, test_size=0.15, val_size=0.15, balance=True):
    """
    Split data into train/val/test. Optionally balance training set.
    
    Risk levels: 0 = Low, 1 = Moderate, 2 = High
    """
    preprocessor = DataPreprocessor()
    df['text'] = preprocessor.preprocess(df['text'].values)
    
    # Stratified split: train, temp (val+test)
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        df['text'], df['risk_level'],
        test_size=test_size + val_size,
        stratify=df['risk_level'],
        random_state=42
    )
    
    # Split temp into val and test
    val_size_split = val_size / (test_size + val_size)
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts, temp_labels,
        test_size=val_size_split,
        stratify=temp_labels,
        random_state=42
    )
    
    # Optional: balance training set using oversampling
    if balance:
        ros = RandomOverSampler(random_state=42)
        X_train_balanced, y_train_balanced = ros.fit_resample(
            train_texts.values.reshape(-1, 1), train_labels.values
        )
        
        train_texts = pd.Series(X_train_balanced.flatten())
        train_labels = pd.Series(y_train_balanced)
        
        logger.info(f"After balancing, train set size: {len(train_texts)}")
    
    logger.info(f"Train: {len(train_texts)}, Val: {len(val_texts)}, Test: {len(test_texts)}")
    logger.info(f"Train label dist:\n{pd.Series(train_labels).value_counts()}")
    
    return {
        'train_texts': train_texts,
        'train_labels': train_labels,
        'val_texts': val_texts,
        'val_labels': val_labels,
        'test_texts': test_texts,
        'test_labels': test_labels,
    }


def create_torch_datasets(split_data, tokenizer, max_length=128):
    """Convert pandas Series to PyTorch datasets."""
    
    train_dataset = ClinicalNotesDataset(
        split_data['train_texts'].values,
        split_data['train_labels'].values,
        tokenizer,
        max_length
    )
    
    val_dataset = ClinicalNotesDataset(
        split_data['val_texts'].values,
        split_data['val_labels'].values,
        tokenizer,
        max_length
    )
    
    test_dataset = ClinicalNotesDataset(
        split_data['test_texts'].values,
        split_data['test_labels'].values,
        tokenizer,
        max_length
    )
    
    return train_dataset, val_dataset, test_dataset


if __name__ == "__main__":
    # Test data loading and preprocessing
    df = load_mimic_like_data(num_samples=500)
    split_data = prepare_datasets(df, balance=True)
    
    # Test tokenization
    tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract")
    train_ds, val_ds, test_ds = create_torch_datasets(split_data, tokenizer)
    
    sample = train_ds[0]
    print(f"Sample input_ids shape: {sample['input_ids'].shape}")
    print(f"Sample label: {sample['label']}")
