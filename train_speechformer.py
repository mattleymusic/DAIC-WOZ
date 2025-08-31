#!/usr/bin/env python3
"""
SpeechFormer CTC-inspired Classifier for DAIC-WOZ Depression Classification
Based on the successful SpeechFormer CTC architecture
Optimized for MacBook Pro M4 Pro with 48GB unified memory
"""

import os
import json
import time
import math
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
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
    'mel_bins': 128,            # Fixed mel-frequency bins
    'frame_dim': 256,           # Frame-level feature dimension
    'segment_dim': 512,         # Segment-level feature dimension
    'utterance_dim': 768,       # Utterance-level feature dimension
    'num_heads': 8,             # Number of attention heads
    'num_layers': 4,            # Number of transformer layers
    'dropout': 0.1,             # Dropout rate
    'max_segments': 64,         # Maximum number of segments to process
    'max_time': 500,            # Maximum time steps per segment
    'segment_pool_size': 8,     # Pooling size for segment-level features
}

TRAINING_CONFIG = {
    'batch_size': 16,           # Smaller batch size for memory efficiency
    'learning_rate': 5e-5,      # Lower learning rate for transformers
    'weight_decay': 0.01,       # Weight decay for regularization
    'num_epochs': 100,          # Reasonable number of epochs
    'early_stopping_patience': 20,
    'min_epochs': 30,           # Minimum epochs before early stopping
    'warmup_epochs': 5,         # Warmup epochs
    'gradient_clip_norm': 1.0,  # Gradient clipping
}

# -----------------------------
# SpeechFormer CTC Architecture
# -----------------------------
class SegmentPooling(nn.Module):
    """Pool segment-level features to fixed size."""
    def __init__(self, input_dim, output_dim, pool_size):
        super().__init__()
        self.pool_size = pool_size
        self.projection = nn.Linear(input_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)
        
    def forward(self, x):
        # x: (batch_size, num_segments, input_dim)
        batch_size, num_segments, input_dim = x.shape
        
        # Adaptive pooling to fixed size
        if num_segments <= self.pool_size:
            # Pad if too few segments
            padding = self.pool_size - num_segments
            x = F.pad(x, (0, 0, 0, padding))
        else:
            # Pool if too many segments
            x = F.adaptive_avg_pool1d(x.transpose(1, 2), self.pool_size).transpose(1, 2)
        
        # Project and normalize
        x = self.projection(x)
        x = self.norm(x)
        return x

class MultiScaleAttention(nn.Module):
    """Multi-scale attention mechanism for different temporal scales."""
    def __init__(self, dim, num_heads, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Self-attention with optional mask
        attn_out, _ = self.attention(x, x, x, attn_mask=mask)
        x = self.norm(x + self.dropout(attn_out))
        return x

class SpeechFormerCTC(nn.Module):
    """SpeechFormer CTC-inspired architecture."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Frame-level processing (mel-spectrogram to frame features)
        self.frame_encoder = nn.Sequential(
            nn.Linear(config['mel_bins'], config['frame_dim']),
            nn.LayerNorm(config['frame_dim']),
            nn.ReLU(),
            nn.Dropout(config['dropout'])
        )
        
        # Segment-level processing
        self.segment_pooling = SegmentPooling(
            config['frame_dim'], 
            config['segment_dim'], 
            config['segment_pool_size']
        )
        
        # Multi-scale transformer layers
        self.transformer_layers = nn.ModuleList([
            MultiScaleAttention(config['segment_dim'], config['num_heads'], config['dropout'])
            for _ in range(config['num_layers'])
        ])
        
        # Utterance-level aggregation
        self.utterance_pooling = nn.AdaptiveAvgPool1d(1)
        self.utterance_projection = nn.Linear(config['segment_dim'], config['utterance_dim'])
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(config['dropout']),
            nn.Linear(config['utterance_dim'], config['utterance_dim'] // 2),
            nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(config['utterance_dim'] // 2, 2)  # Binary classification
        )
        
    def forward(self, mel_chunks, chunk_lengths):
        # mel_chunks: (batch_size, num_chunks, mel_bins, max_time)
        # chunk_lengths: (batch_size, num_chunks) - actual time steps per chunk
        
        batch_size, num_chunks, mel_bins, max_time = mel_chunks.shape
        
        # Process each chunk to frame-level features
        frame_features = []
        for i in range(num_chunks):
            chunk = mel_chunks[:, i, :, :]  # (batch_size, mel_bins, max_time)
            lengths = chunk_lengths[:, i]   # (batch_size,)
            
            # Process each time step
            chunk_features = []
            for j in range(max_time):
                # Extract mel-spectrogram frame
                frame = chunk[:, :, j]  # (batch_size, mel_bins)
                # Encode frame
                frame_feat = self.frame_encoder(frame)  # (batch_size, frame_dim)
                chunk_features.append(frame_feat)
            
            # Stack frames and average pool to segment-level
            chunk_features = torch.stack(chunk_features, dim=1)  # (batch_size, max_time, frame_dim)
            
            # Average pool across time dimension to get segment representation
            segment_feat = torch.mean(chunk_features, dim=1)  # (batch_size, frame_dim)
            frame_features.append(segment_feat)
        
        # Stack segment features
        segment_features = torch.stack(frame_features, dim=1)  # (batch_size, num_chunks, frame_dim)
        
        # Pool segments to fixed size
        segment_features = self.segment_pooling(segment_features)  # (batch_size, pool_size, segment_dim)
        
        # Apply transformer layers
        x = segment_features
        for transformer in self.transformer_layers:
            x = transformer(x)
        
        # Utterance-level aggregation
        x = x.transpose(1, 2)  # (batch_size, segment_dim, pool_size)
        x = self.utterance_pooling(x).squeeze(-1)  # (batch_size, segment_dim)
        x = self.utterance_projection(x)  # (batch_size, utterance_dim)
        
        # Classification
        logits = self.classifier(x)
        return logits

# -----------------------------
# Dataset
# -----------------------------
class SpeechFormerDataset(Dataset):
    """Dataset for SpeechFormer CTC approach."""
    def __init__(self, data_dir, labels_df, subset, max_chunks=64, max_time=500):
        self.data_dir = Path(data_dir)
        self.labels_df = labels_df
        self.subset = subset
        self.max_chunks = max_chunks
        self.max_time = max_time
        
        # Find patient ID column
        patient_col = None
        for col in ['participant', 'patient_id', 'Patient_ID', 'patient', 'Patient', 'id', 'ID']:
            if col in labels_df.columns:
                patient_col = col
                break
        
        if patient_col is None:
            patient_col = labels_df.columns[0]
            print(f"Using column '{patient_col}' as patient identifier")
        
        subset_patients = labels_df[labels_df['subset'] == subset][patient_col].tolist()
        
        # Get patient data
        self.patient_data = []
        for patient_id in subset_patients:
            patient_dir = self.data_dir / str(patient_id)
            if patient_dir.exists():
                mel_files = list(patient_dir.glob('*_mel.npy'))
                if mel_files:
                    # Sort files by part number for consistent ordering
                    mel_files.sort(key=lambda x: int(x.stem.split('_')[1]))
                    
                    self.patient_data.append({
                        'patient_id': patient_id,
                        'mel_files': mel_files,
                        'target': labels_df[labels_df[patient_col] == patient_id]['target'].iloc[0]
                    })
        
        print(f"Found {len(self.patient_data)} patients for {subset} subset")
        self.analyze_class_distribution()
    
    def analyze_class_distribution(self):
        """Analyze class distribution."""
        labels = [row['target'] for row in self.patient_data]
        unique, counts = np.unique(labels, return_counts=True)
        print(f"Class distribution for {self.subset}:")
        for label, count in zip(unique, counts):
            print(f"  Class {label}: {count} samples ({100 * count / len(labels):.1f}%)")
    
    def __len__(self):
        return len(self.patient_data)
    
    def __getitem__(self, idx):
        patient_info = self.patient_data[idx]
        mel_files = patient_info['mel_files']
        target = patient_info['target']
        
        # Load mel-spectrograms
        mel_chunks = []
        chunk_lengths = []
        
        for mel_file in mel_files[:self.max_chunks]:  # Limit number of chunks
            try:
                chunk = np.load(mel_file)  # Shape: (mel_bins, time_steps)
                
                # Verify dimensions
                if chunk.shape[0] != 128:
                    continue
                
                # Handle time dimension
                if chunk.shape[1] > self.max_time:
                    chunk = chunk[:, :self.max_time]
                elif chunk.shape[1] < self.max_time:
                    # Pad with zeros
                    padding = self.max_time - chunk.shape[1]
                    chunk = np.pad(chunk, ((0, 0), (0, padding)), 'constant')
                
                mel_chunks.append(chunk)
                chunk_lengths.append(min(chunk.shape[1], self.max_time))
                
            except Exception as e:
                continue
        
        # Ensure we have at least one chunk
        if not mel_chunks:
            mel_chunks = [np.zeros((128, self.max_time))]
            chunk_lengths = [self.max_time]
        
        # Pad to max_chunks if needed
        while len(mel_chunks) < self.max_chunks:
            mel_chunks.append(np.zeros((128, self.max_time)))
            chunk_lengths.append(0)
        
        # Stack chunks
        mel_chunks = np.stack(mel_chunks)  # (max_chunks, mel_bins, max_time)
        chunk_lengths = np.array(chunk_lengths)
        
        # Normalize to [0, 1] range (log-mel spectrograms are already normalized)
        # Just ensure they're in reasonable range
        if mel_chunks.max() > 0:
            mel_chunks = (mel_chunks - mel_chunks.min()) / (mel_chunks.max() - mel_chunks.min())
        
        return {
            'mel_chunks': torch.FloatTensor(mel_chunks),
            'chunk_lengths': torch.LongTensor(chunk_lengths),
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
        mel_chunks = batch['mel_chunks'].to(device)
        chunk_lengths = batch['chunk_lengths'].to(device)
        targets = batch['target'].squeeze().to(device)
        
        # Forward pass
        optimizer.zero_grad()
        logits = model(mel_chunks, chunk_lengths)
        loss = criterion(logits, targets)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), TRAINING_CONFIG['gradient_clip_norm'])
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
            mel_chunks = batch['mel_chunks'].to(device)
            chunk_lengths = batch['chunk_lengths'].to(device)
            targets = batch['target'].squeeze().to(device)
            
            # Forward pass
            logits = model(mel_chunks, chunk_lengths)
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
    output_dir = Path('data/models/speechformer_ctc')
    
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
    train_dataset = SpeechFormerDataset(data_dir, labels_df, 'train', 
                                      MODEL_CONFIG['max_segments'], MODEL_CONFIG['max_time'])
    val_dataset = SpeechFormerDataset(data_dir, labels_df, 'dev', 
                                    MODEL_CONFIG['max_segments'], MODEL_CONFIG['max_time'])
    test_dataset = SpeechFormerDataset(data_dir, labels_df, 'test', 
                                     MODEL_CONFIG['max_segments'], MODEL_CONFIG['max_time'])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=TRAINING_CONFIG['batch_size'], 
                            shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=TRAINING_CONFIG['batch_size'], 
                          shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=TRAINING_CONFIG['batch_size'], 
                           shuffle=False, num_workers=0)
    
    # Create model
    print("Creating SpeechFormer CTC model...")
    model = SpeechFormerCTC(MODEL_CONFIG)
    model = model.to(device)
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss function with class weighting
    class_counts = labels_df['target'].value_counts().sort_index()
    class_weights = torch.FloatTensor([1.0 / class_counts[0], 1.0 / class_counts[1]]).to(device)
    print(f"Class weights: {class_weights.cpu().numpy()}")
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = AdamW(model.parameters(), lr=TRAINING_CONFIG['learning_rate'], 
                     weight_decay=TRAINING_CONFIG['weight_decay'])
    
    # Learning rate scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=TRAINING_CONFIG['num_epochs'])
    
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
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'metrics': val_metrics,
                'model_config': MODEL_CONFIG,
                'training_config': TRAINING_CONFIG
            }, output_dir / 'checkpoints' / 'best_model.pt')
            print(f"New best model saved with F1: {best_f1:.4f}")
        else:
            patience_counter += 1
        
        # Learning rate scheduling
        scheduler.step()
        
        # Early stopping
        if epoch >= TRAINING_CONFIG['min_epochs'] and patience_counter >= TRAINING_CONFIG['early_stopping_patience']:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # Test on best model - Fix the checkpoint loading issue
    print("\nLoading best model for testing...")
    try:
        # Try loading with weights_only=False for PyTorch 2.6 compatibility
        best_checkpoint = torch.load(output_dir / 'checkpoints' / 'best_model.pt', 
                                   map_location=device, weights_only=False)
        model.load_state_dict(best_checkpoint['model_state_dict'])
        print("Successfully loaded checkpoint with weights_only=False")
    except Exception as e:
        print(f"Failed to load checkpoint: {e}")
        print("Testing with current model state...")
    
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