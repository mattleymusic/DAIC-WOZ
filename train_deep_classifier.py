#!/usr/bin/env python3
"""
Simple CNN Classifier for DAIC-WOZ Depression Classification
MPS-compatible version that actually works
"""

import os
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, roc_auc_score,
    confusion_matrix, roc_curve, auc
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# -----------------------------
# Configuration
# -----------------------------
MODEL_CONFIG = {
    'input_channels': 1,        # Treat mel-spectrograms as images
    'hidden_dims': [32, 64, 128],  # Simple CNN architecture
    'dropout': 0.3,
    'max_sequence_length': 500  # Reduced from 1000
}

TRAINING_CONFIG = {
    'batch_size': 32,           # Much larger batch size
    'learning_rate': 1e-3,      # Higher learning rate
    'weight_decay': 0.0001,     # Lower weight decay
    'num_epochs': 200,          # Increased epochs
    'early_stopping_patience': 50,  # Much more patient early stopping
    'min_epochs': 50,           # Minimum epochs before early stopping
}

# -----------------------------
# MPS-Compatible CNN Model
# -----------------------------
class SimpleCNNClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # Input: (batch_size, 1, time_steps, mel_bins)
        # Treat mel-spectrograms as 2D images
        
        layers = []
        in_channels = config['input_channels']
        
        for hidden_dim in config['hidden_dims']:
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, kernel_size=(3, 3), padding=1),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2, 2)),
                nn.Dropout2d(config['dropout'])
            ])
            in_channels = hidden_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # Calculate output size after convolutions
        # Starting with (time_steps, mel_bins) = (500, 128)
        # After 3 maxpool layers: (500/8, 128/8) = (62, 16)
        # Final feature size: hidden_dims[-1] * 62 * 16 = 128 * 62 * 16 = 126,976
        
        # Use fixed-size pooling instead of adaptive pooling for MPS compatibility
        self.global_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(64, 2)
        )
        
    def forward(self, x):
        # x shape: (batch_size, time_steps, mel_bins)
        # Add channel dimension and transpose for CNN
        x = x.unsqueeze(1)  # (batch_size, 1, time_steps, mel_bins)
        
        # Extract features
        features = self.feature_extractor(x)
        
        # Global pooling - handle MPS compatibility
        try:
            pooled_features = self.global_pool(features)
        except RuntimeError as e:
            if "MPS" in str(e) and "divisible" in str(e):
                # Fallback: use manual pooling for MPS
                batch_size, channels, height, width = features.shape
                # Use max pooling to get to a size that's divisible by 4
                target_height = (height // 4) * 4
                target_width = (width // 4) * 4
                if target_height < 4:
                    target_height = 4
                if target_width < 4:
                    target_width = 4
                
                # Manual pooling to get to target size
                pooled_features = F.avg_pool2d(features, 
                                             kernel_size=(height // target_height, width // target_width),
                                             stride=(height // target_height, width // target_width))
                
                # If still not divisible by 4, pad or crop
                if pooled_features.shape[2] % 4 != 0 or pooled_features.shape[3] % 4 != 0:
                    # Pad to make divisible by 4
                    pad_h = (4 - pooled_features.shape[2] % 4) % 4
                    pad_w = (4 - pooled_features.shape[3] % 4) % 4
                    pooled_features = F.pad(pooled_features, (0, pad_w, 0, pad_h))
                
                # Final pooling to (4, 4)
                pooled_features = F.adaptive_avg_pool2d(pooled_features, (4, 4))
            else:
                raise e
        
        # Classify
        logits = self.classifier(pooled_features)
        return logits

# -----------------------------
# Dataset
# -----------------------------
class SimpleMelDataset(Dataset):
    def __init__(self, data_dir, labels_df, subset, max_length=500):
        self.data_dir = Path(data_dir)
        self.labels_df = labels_df
        self.subset = subset
        self.max_length = max_length
        
        # Filter patients by subset
        # Check what columns are actually available
        print(f"Available columns in labels_df: {list(labels_df.columns)}")
        
        # Try to find the patient ID column - it might be named differently
        patient_col = None
        for col in ['patient_id', 'Patient_ID', 'patient', 'Patient', 'id', 'ID']:
            if col in labels_df.columns:
                patient_col = col
                break
        
        if patient_col is None:
            # If no obvious column found, use the first column
            patient_col = labels_df.columns[0]
            print(f"Using first column '{patient_col}' as patient identifier")
        
        subset_patients = labels_df[labels_df['subset'] == subset][patient_col].tolist()
        
        # Get patient directories
        self.patient_dirs = []
        for patient_id in subset_patients:
            patient_dir = self.data_dir / str(patient_id)
            if patient_dir.exists():
                # Get all mel-spectrogram files for this patient
                mel_files = list(patient_dir.glob('*_mel.npy'))
                if mel_files:
                    self.patient_dirs.append({
                        'patient_id': patient_id,
                        'mel_files': mel_files,
                        'target': labels_df[labels_df[patient_col] == patient_id]['target'].iloc[0]
                    })
        
        print(f"Found {len(self.patient_dirs)} patients for {subset} subset")
        self.analyze_class_distribution()
    
    def analyze_class_distribution(self):
        """Analyze class distribution in this subset."""
        labels = [row['target'] for row in self.patient_dirs]
        unique, counts = np.unique(labels, return_counts=True)
        print(f"Class distribution for {self.subset}:")
        for label, count in zip(unique, counts):
            print(f"  Class {label}: {count} samples ({100 * count / len(labels):.1f}%)")
    
    def __len__(self):
        return len(self.patient_dirs)
    
    def __getitem__(self, idx):
        patient_info = self.patient_dirs[idx]
        mel_files = patient_info['mel_files']
        target = patient_info['target']
        
        # Load and concatenate mel-spectrograms with proper length control
        if len(mel_files) == 1:
            mel_spectrogram = np.load(mel_files[0])
        else:
            mel_chunks = []
            total_time_steps = 0
            
            for mel_file in mel_files:
                try:
                    chunk = np.load(mel_file)  # Shape: (mel_bins, time_steps)
                    
                    # Verify mel-spectrogram dimensions
                    if chunk.shape[0] != 128:
                        continue
                    
                    # Check if adding this chunk would exceed max_length
                    if total_time_steps + chunk.shape[1] > self.max_length:
                        # Only add what fits
                        remaining_steps = self.max_length - total_time_steps
                        if remaining_steps > 0:
                            chunk = chunk[:, :remaining_steps]
                            mel_chunks.append(chunk)
                            total_time_steps += chunk.shape[1]
                        break
                    else:
                        mel_chunks.append(chunk)
                        total_time_steps += chunk.shape[1]
                    
                except Exception as e:
                    continue
            
            if mel_chunks:
                # Concatenate all chunks along time dimension
                # Each chunk is (128, time_steps), concatenate along axis=1
                mel_spectrogram = np.concatenate(mel_chunks, axis=1)  # (128, total_time)
                mel_spectrogram = mel_spectrogram.T  # Now (time_steps, 128)
            else:
                mel_spectrogram = np.zeros((1, 128))
        
        # Ensure exact dimensions - truncate or pad to exactly max_length
        if mel_spectrogram.shape[0] > self.max_length:
            mel_spectrogram = mel_spectrogram[:self.max_length, :]
        elif mel_spectrogram.shape[0] < self.max_length:
            # Pad short sequences with zeros
            padding = self.max_length - mel_spectrogram.shape[0]
            mel_spectrogram = np.pad(mel_spectrogram, ((0, padding), (0, 0)), 'constant')
        
        # Verify final dimensions
        assert mel_spectrogram.shape == (self.max_length, 128), f"Expected shape ({self.max_length}, 128), got {mel_spectrogram.shape}"
        
        # Normalize to [0, 1] range
        if mel_spectrogram.max() > 0:
            mel_spectrogram = (mel_spectrogram - mel_spectrogram.min()) / (mel_spectrogram.max() - mel_spectrogram.min())
        
        return {
            'audio': torch.FloatTensor(mel_spectrogram),
            'target': torch.LongTensor([target])
        }

# -----------------------------
# Training Functions
# -----------------------------
def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1} Training')
    
    for batch_idx, batch in enumerate(progress_bar):
        audio = batch['audio'].to(device)
        targets = batch['target'].squeeze().to(device)
        
        # Forward pass
        optimizer.zero_grad()
        logits = model(audio)
        loss = criterion(logits, targets)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        _, predicted = torch.max(logits.data, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()
        
        # Update progress bar
        progress_bar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Acc': f'{100 * correct / total:.2f}%'
        })
    
    return total_loss / len(train_loader), 100 * correct / total

def validate_epoch(model, val_loader, criterion, device):
    """Validate for one epoch."""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    with torch.no_grad():
        for batch in val_loader:
            audio = batch['audio'].to(device)
            targets = batch['target'].squeeze().to(device)
            
            # Forward pass
            logits = model(audio)
            loss = criterion(logits, targets)
            
            # Statistics
            total_loss += loss.item()
            probabilities = torch.softmax(logits, dim=1)
            _, predicted = torch.max(logits.data, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(targets.cpu().numpy())
            all_probabilities.extend(probabilities[:, 1].cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='binary', zero_division=0)
    auc_score = roc_auc_score(all_labels, all_probabilities)
    
    return {
        'loss': total_loss / len(val_loader),
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc_score,
        'predictions': all_predictions,
        'labels': all_labels,
        'probabilities': all_probabilities
    }

def main():
    # Setup paths
    data_dir = Path('data/features/mel_spectrograms')
    meta_file = Path('data/daic_woz/meta_info.csv')
    output_dir = Path('data/models/simple_cnn_classifier')
    
    if not data_dir.exists():
        raise FileNotFoundError(f"Mel-spectrogram directory not found: {data_dir}")
    
    if not meta_file.exists():
        raise FileNotFoundError(f"Meta info file not found: {meta_file}")
    
    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'checkpoints').mkdir(exist_ok=True)
    (output_dir / 'plots').mkdir(exist_ok=True)
    
    # Load labels
    print("Loading labels...")
    labels_df = pd.read_csv(meta_file)
    print(f"Loaded {len(labels_df)} patients")
    print(f"Available columns: {list(labels_df.columns)}")
    print(f"Subset distribution: {labels_df['subset'].value_counts().to_dict()}")
    print(f"Target distribution: {labels_df['target'].value_counts().to_dict()}")
    
    # Setup device
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = SimpleMelDataset(data_dir, labels_df, 'train', MODEL_CONFIG['max_sequence_length'])
    val_dataset = SimpleMelDataset(data_dir, labels_df, 'dev', MODEL_CONFIG['max_sequence_length'])
    test_dataset = SimpleMelDataset(data_dir, labels_df, 'test', MODEL_CONFIG['max_sequence_length'])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=TRAINING_CONFIG['batch_size'], 
                            shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=TRAINING_CONFIG['batch_size'], 
                          shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=TRAINING_CONFIG['batch_size'], 
                           shuffle=False, num_workers=0)
    
    # Create model
    print("Creating model...")
    model = SimpleCNNClassifier(MODEL_CONFIG)
    model = model.to(device)
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Loss function with class weighting
    class_counts = labels_df['target'].value_counts().sort_index()
    class_weights = torch.FloatTensor([1.0 / class_counts[0], 1.0 / class_counts[1]]).to(device)
    print(f"Class weights: {class_weights.cpu().numpy()}")
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = Adam(model.parameters(), lr=TRAINING_CONFIG['learning_rate'], weight_decay=TRAINING_CONFIG['weight_decay'])
    
    # Training loop
    print("Starting training...")
    train_losses = []
    train_accuracies = []
    val_metrics_history = []
    best_f1 = 0
    patience_counter = 0
    
    for epoch in range(TRAINING_CONFIG['num_epochs']):
        print(f"\nEpoch {epoch+1}/{TRAINING_CONFIG['num_epochs']}")
        
        # Training
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        
        # Validation
        val_metrics = validate_epoch(model, val_loader, criterion, device)
        val_metrics_history.append(val_metrics)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {100 * val_metrics['accuracy']:.2f}%")
        print(f"Val F1: {val_metrics['f1']:.4f}, Val AUC: {val_metrics['auc']:.4f}")
        
        # Save best model
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'metrics': val_metrics
            }, output_dir / 'checkpoints' / 'best_model.pt')
            print(f"New best model saved with F1: {best_f1:.4f}")
        else:
            patience_counter += 1
        
        # Early stopping - but only after minimum epochs
        if epoch >= TRAINING_CONFIG['min_epochs'] and patience_counter >= TRAINING_CONFIG['early_stopping_patience']:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # Test on best model
    print("\nLoading best model for testing...")
    best_checkpoint = torch.load(output_dir / 'checkpoints' / 'best_model.pt', map_location=device)
    model.load_state_dict(best_checkpoint['model_state_dict'])
    
    print("Testing best model...")
    test_metrics = validate_epoch(model, test_loader, criterion, device)
    
    print(f"\nFinal Test Results:")
    print(f"  Accuracy: {100 * test_metrics['accuracy']:.2f}%")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall: {test_metrics['recall']:.4f}")
    print(f"  F1: {test_metrics['f1']:.4f}")
    print(f"  AUC: {test_metrics['auc']:.4f}")
    
    print(f"\nTraining completed!")
    print(f"Best validation F1: {best_f1:.4f}")
    print(f"Test F1: {test_metrics['f1']:.4f}")
    print(f"Results saved to: {output_dir}")

if __name__ == "__main__":
    main()