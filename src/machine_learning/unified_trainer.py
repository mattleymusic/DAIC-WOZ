#!/usr/bin/env python3
"""
Unified Machine Learning Trainer - Simplified ML Pipeline

This script provides a unified machine learning pipeline that trains a single
classifier with different feature types using a simplified, reusable approach.

Quick start:
1. Edit the configuration parameters at the top of this file (lines 68-138)
2. Choose your feature set: FEATURE_TYPE = "paper_fused_features"
3. Choose your single model: MODELS_TO_TRAIN = ['svm']
4. Run: python src/machine_learning/unified_trainer.py

Available feature sets:
- "paper_fused_features"     # Paper fused features (10 COVAREP + 16 HOSF) ⭐ NEW!
- "unified_paper_features"   # Unified paper features
- "egemaps"                  # eGeMAPS features
- "emotionmsp"               # EmotionMSP features
- "hubert"                   # HuBERT features
- "hosf_fusion"              # HOSF Fusion features
- "umap_egemaps"             # UMAP-transformed eGeMAPS
- "umap_emotionmsp"          # UMAP-transformed EmotionMSP
- "umap_hubert"              # UMAP-transformed HuBERT
- "umap_hosf_fusion"         # UMAP-transformed HOSF Fusion

Available models:
- "svm"    # Support Vector Machine (RBF kernel, C=0.7)
- "knn"    # K-Nearest Neighbors (k=2, distance weighting)
- "rf"     # Random Forest (400 trees, balanced subsample)
- "xgb"    # XGBoost (gradient boosting, optimized for class imbalance)
- "lgb"    # LightGBM (gradient boosting, fast training)
- "ensemble" # Voting ensemble of SVM + LightGBM (best performers)

Features:
- Supports multiple feature types with a single model
- Implements balanced sampling with oversampling option
- Handles chunk-level to patient-level aggregation
- Configurable parameters in __main__ section
- Automatic configuration validation

Usage:
    python src/machine_learning/unified_trainer.py

Author: Unified ML pipeline for depression classification
"""

import os
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import pickle
import glob
from pathlib import Path
from datetime import datetime
import warnings
import time
import sys

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Add the project root to the path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

# =============================================================================
# CONFIGURATION PARAMETERS - EDIT THESE TO CUSTOMIZE YOUR EXPERIMENT
# =============================================================================

# QUICK CONFIGURATION - Just change these 3 lines!
FEATURE_TYPE = "paper_hosf"  # Try: "paper_fused_features", "egemap", "hubert", "emotionmsp", "paper_covarep", "paper_hosf", "umap_egemaps", "umap_emotionmsp", "umap_hubert"
CHUNK_LENGTH = "4.0s"                  # Available: "3.0s", "5.0s", "10.0s", "20.0s", "30.0s", "4.0s"
OVERLAP = "0.0s"        # Available overlap: "0.0s", "1.5s", "2.5s", "5.0s", "10.0s", "15.0s"
MODELS_TO_TRAIN = ['svm']              # Single classifier: ['svm'], ['rf'], ['knn'], ['xgb'], ['lgb'], or ['ensemble'] - Testing LightGBM on UMAP!
DATA_ROOT = "data"
# PATIENT LEVEL Chunks
BALANCED_SAMPLING = True  # Enable balanced sampling (equal chunks per patient) - PATIENT LEVEL
OVERSAMPLE_MINORITY = True  # Enable oversampling - PATIENT LEVEL
# CLASS LEVEL
UNDERSAMPLE_MAJORITY_CLASS = False  # Global default; RF/XGB will auto-enable undersampling
RANDOM_SEED = 42         # Random seed for reproducible sampling
EVALUATE_TEST_SET = True  # Whether to evaluate on test set



def load_unified_features_and_labels(feature_type="unified_paper_features", chunk_length="4.0s", overlap="0.0s", data_root="data", balanced_sampling=True, random_seed=42, oversample_minority=True):
    """
    Load unified features and labels for machine learning.
    
    Args:
        feature_type (str): Type of features to load (unified_paper_features, egemaps, hubert, etc.)
        chunk_length (str): Chunk length (e.g., "4.0s")
        overlap (str): Overlap length (e.g., "0.0s")
        data_root (str): Root directory for data
        balanced_sampling (bool): Whether to apply balanced sampling (equal chunks per patient)
        random_seed (int): Random seed for reproducible sampling
        oversample_minority (bool): Whether to oversample patients with fewer chunks
    
    Returns:
        dict: Dictionary containing train, dev, and test data
    """
    # Load meta information for labels and splits
    meta_path = f"{data_root}/daic_woz/meta_info.csv"
    meta_df = pd.read_csv(meta_path)
    
    # Define participants to exclude (same as in paper reproduction)
    excluded_participants = [
        '324_P', '323_P', '322_P', '321_P',
        '300_P', '305_P', '318_P', '341_P', '362_P', '451_P', '458_P', '480_P'
    ]
    
    # Filter out excluded participants
    meta_df = meta_df[~meta_df['participant'].isin(excluded_participants)]
    
    print(f"Excluded {len(excluded_participants)} participants")
    print(f"Remaining participants: {len(meta_df)}")
    
    # Create mappings
    participant_to_target = dict(zip(meta_df['participant'], meta_df['target']))
    participant_to_subset = dict(zip(meta_df['participant'], meta_df['subset']))
    
    # Define feature directory based on feature type
    # All feature types follow the same pattern: data/features/{feature_type}/{chunk_length}_{overlap}_overlap
    feature_dir = f"{data_root}/features/{feature_type}/{chunk_length}_{overlap}_overlap"
    
    if not os.path.exists(feature_dir):
        raise ValueError(f"Feature directory not found: {feature_dir}")
    
    print(f"Loading {feature_type} features from: {feature_dir}")
    
    # Combine features and labels
    splits = {'train': [], 'dev': [], 'test': []}
    split_labels = {'train': [], 'dev': [], 'test': []}
    split_participants = {'train': [], 'dev': [], 'test': []}
    
    # Process each participant
    for participant in meta_df['participant']:
        target = participant_to_target[participant]
        subset = participant_to_subset[participant]
        
        # Load features for this participant
        participant_feature_dir = os.path.join(feature_dir, participant)
        if not os.path.exists(participant_feature_dir):
            print(f"Warning: No features found for participant {participant}")
            continue
        
        # Get all feature files for this participant
        # Handle different naming patterns for different feature types
        if feature_type == "paper_fused_features":
            # Paper fused features use pattern: chunk_[XXX]_paper_fused_features_features.csv
            feature_files = glob.glob(os.path.join(participant_feature_dir, f"*_{feature_type}_features.csv"))
        elif feature_type in ["paper_covarep", "paper_hosf"]:
            # Paper-specific features use pattern: chunk_[XXX]_paper_[TYPE]_features.csv
            feature_files = glob.glob(os.path.join(participant_feature_dir, f"*_{feature_type}_features.csv"))
        elif feature_type.startswith("umap_"):
            # UMAP features use pattern: chunk_[XXX]_[TYPE]_umap_features.csv
            # Extract the base feature type (e.g., "egemaps" from "umap_egemaps" -> "egemap")
            base_feature_type = feature_type.replace("umap_", "")
            # Handle special case where "egemaps" should become "egemap"
            if base_feature_type == "egemaps":
                base_feature_type = "egemap"
            feature_files = glob.glob(os.path.join(participant_feature_dir, f"*_{base_feature_type}_umap_features.csv"))
        else:
            # Standard feature types follow pattern: chunk_[XXX]_[FEATURE_TYPE]_features.csv
            feature_files = glob.glob(os.path.join(participant_feature_dir, f"*_{feature_type}_features.csv"))
        
        feature_files.sort()  # Ensure consistent ordering
        
        # Process each feature file
        for feature_file in feature_files:
            try:
                # Load features
                feature_df = pd.read_csv(feature_file)
                
                # For UMAP features, exclude metadata columns (original_participant, original_label, original_file)
                if feature_type.startswith('umap_'):
                    # Find numeric columns only (exclude metadata columns)
                    numeric_columns = feature_df.select_dtypes(include=[np.number]).columns
                    # Explicitly exclude original_label column to prevent data leakage
                    numeric_columns = [col for col in numeric_columns if col != 'original_label']
                    features = feature_df[numeric_columns].iloc[0].values
                else:
                    features = feature_df.iloc[0].values
                
                # Check for NaN or infinite values
                if np.any(np.isnan(features)) or np.any(np.isinf(features)):
                    print(f"Warning: NaN or infinite values in features for {os.path.basename(feature_file)}")
                    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
                
                # Add to appropriate split
                splits[subset].append(features)
                split_labels[subset].append(target)
                split_participants[subset].append(participant)
                
            except Exception as e:
                print(f"Error loading {feature_file}: {str(e)}")
                continue
    
    # Apply balanced sampling if requested (TRAIN SET ONLY)
    if balanced_sampling and splits['train']:
        print("\nApplying balanced sampling (train set only)...")
        np.random.seed(random_seed)

        subset = 'train'
        # Create DataFrame for easier manipulation
        subset_df = pd.DataFrame({
            'features': splits[subset],
            'labels': split_labels[subset],
            'participants': split_participants[subset]
        })

        # Count chunks per participant
        participant_counts = subset_df['participants'].value_counts()
        min_chunks = participant_counts.min()
        max_chunks = participant_counts.max()

        print(f"{subset.capitalize()} set - Chunks per participant:")
        print(f"  Min: {min_chunks}, Max: {max_chunks}, Mean: {participant_counts.mean():.1f}")

        if oversample_minority:
            print(f"  Using oversampling strategy: patients with <{max_chunks} chunks will be oversampled")
            balanced_samples = []
            for participant in subset_df['participants'].unique():
                participant_data = subset_df[subset_df['participants'] == participant]
                num_chunks = len(participant_data)
                if num_chunks == max_chunks:
                    balanced_samples.append(participant_data)
                else:
                    if num_chunks < max_chunks:
                        repeat_times = max_chunks // num_chunks
                        remainder = max_chunks % num_chunks
                        repeated_data = pd.concat([participant_data] * repeat_times, ignore_index=True)
                        if remainder > 0:
                            additional_samples = participant_data.sample(n=remainder, random_state=random_seed)
                            repeated_data = pd.concat([repeated_data, additional_samples], ignore_index=True)
                        balanced_samples.append(repeated_data)
                    else:
                        balanced_samples.append(participant_data)
        else:
            print(f"  Using undersampling strategy: all patients limited to {min_chunks} chunks")
            balanced_samples = []
            for participant in subset_df['participants'].unique():
                participant_data = subset_df[subset_df['participants'] == participant]
                if len(participant_data) >= min_chunks:
                    sampled_data = participant_data.sample(n=min_chunks, random_state=random_seed)
                    balanced_samples.append(sampled_data)
                else:
                    balanced_samples.append(participant_data)

        if balanced_samples:
            balanced_df = pd.concat(balanced_samples, ignore_index=True)
            splits['train'] = balanced_df['features'].tolist()
            split_labels['train'] = balanced_df['labels'].tolist()
            split_participants['train'] = balanced_df['participants'].tolist()

            final_counts = pd.Series(split_participants['train']).value_counts()
            print(f"  After balancing: Min: {final_counts.min()}, Max: {final_counts.max()}, Mean: {final_counts.mean():.1f}")
            print(f"  Total samples: {len(balanced_df)} (was {len(subset_df)})")
        else:
            print("  No data available for train set after balancing")
    
    # Convert to numpy arrays
    data = {}
    for subset in ['train', 'dev', 'test']:
        if splits[subset]:  # Only create arrays if data exists
            data[f'{subset}_features'] = np.array(splits[subset])
            data[f'{subset}_labels'] = np.array(split_labels[subset])
            data[f'{subset}_participants'] = split_participants[subset]
        else:
            data[f'{subset}_features'] = np.array([])
            data[f'{subset}_labels'] = np.array([])
            data[f'{subset}_participants'] = []
    
    print(f"\nLoaded {feature_type} features:")
    print(f"Train set: {len(data['train_features'])} samples")
    print(f"Dev set: {len(data['dev_features'])} samples")
    print(f"Test set: {len(data['test_features'])} samples")
    if len(data['train_features']) > 0:
        print(f"Feature dimension: {data['train_features'].shape[1]}")
    
    return data


def train_model(model_type, train_features, train_labels, dev_features, dev_labels, random_seed=42, undersample_majority_class=False):
    """
    Train a machine learning model.
    
    Args:
        model_type (str): Type of model ('svm', 'knn', 'rf')
        train_features (np.array): Training features
        train_labels (np.array): Training labels
        dev_features (np.array): Development features
        dev_labels (np.array): Development labels
        random_seed (int): Random seed for reproducibility
        undersample_majority_class (bool): Whether to undersample non-depressed patients to balance classes
    
    Returns:
        tuple: (trained_model, scaler, dev_predictions)
    """
    print(f"Training {model_type.upper()} model...")
    
    # Check for class imbalance
    class_counts = np.bincount(train_labels)
    print(f"Training class distribution: {class_counts}")
    
    # Apply undersampling if requested
    if undersample_majority_class:
        print("Applying undersampling to balance classes...")
        np.random.seed(random_seed)
        
        # Find indices for each class
        non_depressed_indices = np.where(train_labels == 0)[0]
        depressed_indices = np.where(train_labels == 1)[0]
        
        # Calculate how many non-depressed samples to keep (match depressed count)
        target_count = len(depressed_indices)
        
        # Randomly sample non-depressed indices
        sampled_non_depressed_indices = np.random.choice(
            non_depressed_indices, 
            size=target_count, 
            replace=False
        )
        
        # Combine indices
        balanced_indices = np.concatenate([sampled_non_depressed_indices, depressed_indices])
        
        # Shuffle the indices
        np.random.shuffle(balanced_indices)
        
        # Apply undersampling
        train_features = train_features[balanced_indices]
        train_labels = train_labels[balanced_indices]
        
        # Print results
        new_class_counts = np.bincount(train_labels)
        print(f"After undersampling class distribution: {new_class_counts}")
        print(f"Removed {len(non_depressed_indices) - target_count} non-depressed samples")
    
    # Scale features
    scaler = StandardScaler()
    train_features_scaled = scaler.fit_transform(train_features)
    dev_features_scaled = scaler.transform(dev_features)
    
    # Calculate class weights for balancing
    # Use balanced class counts if undersampling was applied
    if undersample_majority_class:
        balanced_class_counts = np.bincount(train_labels)
        n_classes = len(balanced_class_counts)
        class_weights = {}
        for i in range(n_classes):
            class_weights[i] = len(train_labels) / (n_classes * balanced_class_counts[i])
    else:
        n_classes = len(class_counts)
        class_weights = {}
        for i in range(n_classes):
            class_weights[i] = len(train_labels) / (n_classes * class_counts[i])
    
    print(f"Class weights: {class_weights}")
    
    # Initialize model based on type
    if model_type.lower() == 'svm':
        # Use class weighting only if not undersampling (since undersampling already balances classes)
        if undersample_majority_class:
            model = SVC(kernel='rbf', C=0.7, class_weight=None, random_state=random_seed)
            print("Using SVM without class weighting (data already balanced by undersampling)")
        else:
            model = SVC(kernel='rbf', C=0.7, class_weight='balanced', random_state=random_seed)
            print("Using SVM with balanced class weighting")
        # Fit the model
        model.fit(train_features_scaled, train_labels)
    elif model_type.lower() == 'knn':
        # KNN doesn't support class_weight directly, but we can use class balancing through sampling
        if undersample_majority_class:
            # Use SMOTE for KNN when undersampling is requested
            print("Applying SMOTE for KNN to balance classes...")
            model = KNeighborsClassifier(n_neighbors=2, weights='distance')
            smote = SMOTE(random_state=random_seed, k_neighbors=3)
            train_features_balanced, train_labels_balanced = smote.fit_resample(train_features_scaled, train_labels)
            
            # Print SMOTE results
            original_counts = np.bincount(train_labels)
            balanced_counts = np.bincount(train_labels_balanced)
            print(f"Original class distribution: {original_counts}")
            print(f"After SMOTE class distribution: {balanced_counts}")
            print(f"SMOTE generated {balanced_counts[1] - original_counts[1]} synthetic depressed samples")
            
            # Fit the model with balanced data
            model.fit(train_features_balanced, train_labels_balanced)
        else:
            # Use standard KNN without oversampling - rely on class weighting in evaluation
            print("Using KNN without SMOTE - relying on class weighting in evaluation")
            model = KNeighborsClassifier(n_neighbors=2, weights='distance')
            model.fit(train_features_scaled, train_labels)
    elif model_type.lower() == 'rf':
        # Use class weighting only if not undersampling (since undersampling already balances classes)
        if undersample_majority_class:
            model = RandomForestClassifier(
                n_estimators=500,  # More trees for better performance
                max_depth=10,      # Limit depth to prevent overfitting
                max_features='sqrt',  # Feature sampling for better generalization
                min_samples_split=10,  # Require more samples to split
                min_samples_leaf=5,    # Require more samples in leaves
                class_weight=None, 
                random_state=random_seed
            )
            print("Using Random Forest without class weighting (data already balanced by undersampling)")
        else:
            # Use built-in balanced class weighting
            class_weight = 'balanced'
            print("Using Random Forest with balanced class weighting")
            
            model = RandomForestClassifier(
                n_estimators=500,  # More trees for better performance
                max_depth=10,      # Limit depth to prevent overfitting
                max_features='sqrt',  # Feature sampling for better generalization
                min_samples_split=10,  # Require more samples to split
                min_samples_leaf=5,    # Require more samples in leaves
                class_weight=class_weight, 
                random_state=random_seed
            )
        # Fit the model
        model.fit(train_features_scaled, train_labels)
    elif model_type.lower() == 'xgb':
        # XGBoost with optimized parameters for depression classification
        if undersample_majority_class:
            # Calculate scale_pos_weight for balanced classes
            scale_pos_weight = len(train_labels[train_labels == 0]) / len(train_labels[train_labels == 1])
            model = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=scale_pos_weight,
                random_state=random_seed,
                eval_metric='logloss'
            )
            print(f"Using XGBoost with scale_pos_weight={scale_pos_weight:.2f} (data already balanced by undersampling)")
        else:
            # Use full inverse frequency ratio as scale_pos_weight
            if class_counts[1] > 0:
                scale_pos_weight = class_counts[0] / class_counts[1]
            else:
                scale_pos_weight = 1.0
            
            model = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=scale_pos_weight,
                random_state=random_seed,
                eval_metric='logloss'
            )
            print(f"Using XGBoost with scale_pos_weight={scale_pos_weight:.2f} for class balancing")
        # Fit the model
        model.fit(train_features_scaled, train_labels)
    elif model_type.lower() == 'lgb':
        # LightGBM with optimized parameters for depression classification
        if undersample_majority_class:
            # Calculate class_weight for balanced classes
            class_weight = {0: 1.0, 1: len(train_labels[train_labels == 0]) / len(train_labels[train_labels == 1])}
            model = lgb.LGBMClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                class_weight=class_weight,
                random_state=random_seed,
                verbose=-1
            )
            print(f"Using LightGBM with class_weight={class_weight} (data already balanced by undersampling)")
        else:
            # Calculate class_weight for original class distribution
            class_weight = {0: 1.0, 1: class_counts[0] / class_counts[1]} if class_counts[1] > 0 else {0: 1.0, 1: 1.0}
            model = lgb.LGBMClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                class_weight=class_weight,
                random_state=random_seed,
                verbose=-1
            )
            print(f"Using LightGBM with class_weight={class_weight} for class balancing")
        # Fit the model
        model.fit(train_features_scaled, train_labels)
    elif model_type.lower() == 'ensemble':
        # Voting ensemble of SVM + LightGBM (best performers)
        print("Creating voting ensemble of SVM + LightGBM...")
        
        # Create SVM component
        svm_component = SVC(kernel='rbf', C=0.7, class_weight='balanced', random_state=random_seed, probability=True)
        
        # Create LightGBM component
        class_weight = {0: 1.0, 1: class_counts[0] / class_counts[1]} if class_counts[1] > 0 else {0: 1.0, 1: 1.0}
        lgb_component = lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight=class_weight,
            random_state=random_seed,
            verbose=-1
        )
        
        # Create voting ensemble (soft voting for better performance)
        model = VotingClassifier(
            estimators=[
                ('svm', svm_component),
                ('lightgbm', lgb_component)
            ],
            voting='soft',  # Use probability voting for better performance
            weights=[1.0, 1.0]  # Equal weights for now, could be tuned
        )
        
        print(f"Using ensemble with soft voting: SVM + LightGBM")
        print(f"LightGBM class_weight: {class_weight}")
        
        # Fit the ensemble
        model.fit(train_features_scaled, train_labels)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Make predictions on dev set
    dev_predictions = model.predict(dev_features_scaled)
    
    return model, scaler, dev_predictions


def compute_patient_level_results(y_true, y_pred, participants, model_name="Model"):
    """
    Compute patient-level results using majority voting.
    
    Args:
        y_true (np.array): True labels for each chunk
        y_pred (np.array): Predicted labels for each chunk
        participants (list): Participant IDs for each chunk
        model_name (str): Name of the model for reporting
    
    Returns:
        tuple: (patient_accuracy, patient_f1_weighted, patient_f1_macro, patient_classification_report, patient_df)
    """
    # Create DataFrame for easier manipulation
    df = pd.DataFrame({
        'participant': participants,
        'true_label': y_true,
        'pred_label': y_pred
    })
    
    # Group by participant and compute majority vote
    patient_results = []
    for participant in df['participant'].unique():
        participant_data = df[df['participant'] == participant]
        
        # Majority vote for predictions
        pred_majority = participant_data['pred_label'].mode().iloc[0] if len(participant_data['pred_label'].mode()) > 0 else participant_data['pred_label'].iloc[0]
        true_label = participant_data['true_label'].iloc[0]  # All chunks from same participant have same label
        
        patient_results.append({
            'participant': participant,
            'true_label': true_label,
            'pred_label': pred_majority,
            'num_chunks': len(participant_data)
        })
    
    patient_df = pd.DataFrame(patient_results)
    
    # Calculate patient-level metrics
    patient_y_true = patient_df['true_label'].values
    patient_y_pred = patient_df['pred_label'].values
    
    patient_accuracy = accuracy_score(patient_y_true, patient_y_pred)
    patient_f1_weighted = f1_score(patient_y_true, patient_y_pred, average='weighted', zero_division=0)
    patient_f1_macro = f1_score(patient_y_true, patient_y_pred, average='macro', zero_division=0)
    
    # Generate classification report
    patient_classification_report = classification_report(
        patient_y_true, patient_y_pred, 
        target_names=['Non-Depressed', 'Depressed'],
        output_dict=True
    )
    
    return patient_accuracy, patient_f1_weighted, patient_f1_macro, patient_classification_report, patient_df


def evaluate_model(y_true, y_pred, model_name, split_name="dev", participants=None):
    """
    Evaluate model performance with both chunk-level and patient-level results.
    
    Args:
        y_true (np.array): True labels
        y_pred (np.array): Predicted labels
        model_name (str): Name of the model
        split_name (str): Name of the split being evaluated
        participants (list): Participant IDs for each chunk (optional)
    
    Returns:
        dict: Evaluation metrics including both chunk and patient level results
    """
    # Chunk-level metrics
    accuracy = accuracy_score(y_true, y_pred)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    # Generate chunk-level classification report
    chunk_classification_report = classification_report(
        y_true, y_pred, 
        target_names=['Non-Depressed', 'Depressed'],
        output_dict=True
    )
    
    # Calculate patient-level results if participants are provided
    patient_results = None
    if participants is not None:
        patient_accuracy, patient_f1_weighted, patient_f1_macro, patient_classification_report, patient_df = compute_patient_level_results(
            y_true, y_pred, participants, model_name
        )
        patient_results = {
            'accuracy': patient_accuracy,
            'f1_weighted': patient_f1_weighted,
            'f1_macro': patient_f1_macro,
            'classification_report': patient_classification_report,
            'patient_df': patient_df
        }
    
    # Print chunk-level results
    print(f"\n{model_name} Chunk-Level Results ({split_name} set):")
    print(f"Number of chunks: {len(y_true)}")
    print(f"Chunk-level Accuracy: {accuracy:.4f}")
    print(f"Chunk-level F1 Score (weighted): {f1_weighted:.4f}")
    print(f"Chunk-level F1 Score (macro): {f1_macro:.4f}")
    
    # Print patient-level results if available
    if patient_results is not None:
        print(f"\n{model_name} Patient-Level Results ({split_name} set):")
        print(f"Number of patients: {len(patient_results['patient_df'])}")
        print(f"Patient-level Accuracy: {patient_results['accuracy']:.4f}")
        print(f"Patient-level F1 Score (weighted): {patient_results['f1_weighted']:.4f}")
        print(f"Patient-level F1 Score (macro): {patient_results['f1_macro']:.4f}")
    
    return {
        'chunk_accuracy': accuracy,
        'chunk_f1_weighted': f1_weighted,
        'chunk_f1_macro': f1_macro,
        'chunk_classification_report': chunk_classification_report,
        'patient_results': patient_results,
        'model_name': model_name,
        'split': split_name,
        'num_chunks': len(y_true),
        'num_patients': len(patient_results['patient_df']) if patient_results else None
    }


def write_results_section(f, results, split_name):
    """
    Write a results section for a specific split (dev/test) to the file.
    
    Args:
        f: File handle
        results (dict): Evaluation results
        split_name (str): Name of the split (dev/test)
    """
    f.write(f"\n{'='*70}\n")
    f.write(f"{split_name.upper()} SET RESULTS\n")
    f.write(f"{'='*70}\n\n")
    f.write(f"Split: {results['split']}\n")
    f.write(f"Number of chunks: {results['num_chunks']}\n")
    if results['num_patients'] is not None:
        f.write(f"Number of patients: {results['num_patients']}\n")
    f.write("\n")
    
    f.write("CHUNK-LEVEL RESULTS:\n")
    f.write("-" * 30 + "\n")
    f.write(f"Chunk-level Accuracy: {results['chunk_accuracy']:.4f}\n")
    f.write(f"Chunk-level F1 Score (weighted): {results['chunk_f1_weighted']:.4f}\n")
    f.write(f"Chunk-level F1 Score (macro): {results['chunk_f1_macro']:.4f}\n\n")
    
    # Write chunk-level classification report
    f.write("Chunk-Level Classification Report:\n")
    f.write("-" * 40 + "\n")
    chunk_report = results['chunk_classification_report']
    f.write(f"{'':>20} {'precision':>10} {'recall':>10} {'f1-score':>10} {'support':>10}\n")
    f.write("\n")
    f.write(f"{'Non-Depressed':>20} {chunk_report['Non-Depressed']['precision']:>10.2f} {chunk_report['Non-Depressed']['recall']:>10.2f} {chunk_report['Non-Depressed']['f1-score']:>10.2f} {chunk_report['Non-Depressed']['support']:>10.0f}\n")
    f.write(f"{'Depressed':>20} {chunk_report['Depressed']['precision']:>10.2f} {chunk_report['Depressed']['recall']:>10.2f} {chunk_report['Depressed']['f1-score']:>10.2f} {chunk_report['Depressed']['support']:>10.0f}\n")
    f.write("\n")
    f.write(f"{'accuracy':>20} {'':>30} {chunk_report['accuracy']:>10.2f} {results['num_chunks']:>10.0f}\n")
    f.write(f"{'macro avg':>20} {chunk_report['macro avg']['precision']:>10.2f} {chunk_report['macro avg']['recall']:>10.2f} {chunk_report['macro avg']['f1-score']:>10.2f} {chunk_report['macro avg']['support']:>10.0f}\n")
    f.write(f"{'weighted avg':>20} {chunk_report['weighted avg']['precision']:>10.2f} {chunk_report['weighted avg']['recall']:>10.2f} {chunk_report['weighted avg']['f1-score']:>10.2f} {chunk_report['weighted avg']['support']:>10.0f}\n")
    
    # Write patient-level results if available
    if results['patient_results'] is not None:
        f.write("\n\nPATIENT-LEVEL RESULTS (Majority Vote):\n")
        f.write("-" * 40 + "\n")
        patient_results = results['patient_results']
        f.write(f"Patient-level Accuracy: {patient_results['accuracy']:.4f}\n")
        f.write(f"Patient-level F1 Score (weighted): {patient_results['f1_weighted']:.4f}\n")
        f.write(f"Patient-level F1 Score (macro): {patient_results['f1_macro']:.4f}\n\n")
        
        f.write("Patient-Level Classification Report:\n")
        f.write("-" * 40 + "\n")
        patient_report = patient_results['classification_report']
        f.write(f"{'':>20} {'precision':>10} {'recall':>10} {'f1-score':>10} {'support':>10}\n")
        f.write("\n")
        f.write(f"{'Non-Depressed':>20} {patient_report['Non-Depressed']['precision']:>10.2f} {patient_report['Non-Depressed']['recall']:>10.2f} {patient_report['Non-Depressed']['f1-score']:>10.2f} {patient_report['Non-Depressed']['support']:>10.0f}\n")
        f.write(f"{'Depressed':>20} {patient_report['Depressed']['precision']:>10.2f} {patient_report['Depressed']['recall']:>10.2f} {patient_report['Depressed']['f1-score']:>10.2f} {patient_report['Depressed']['support']:>10.0f}\n")
        f.write("\n")
        f.write(f"{'accuracy':>20} {'':>30} {patient_report['accuracy']:>10.2f} {results['num_patients']:>10.0f}\n")
        f.write(f"{'macro avg':>20} {patient_report['macro avg']['precision']:>10.2f} {patient_report['macro avg']['recall']:>10.2f} {patient_report['macro avg']['f1-score']:>10.2f} {patient_report['macro avg']['support']:>10.0f}\n")
        f.write(f"{'weighted avg':>20} {patient_report['weighted avg']['precision']:>10.2f} {patient_report['weighted avg']['recall']:>10.2f} {patient_report['weighted avg']['f1-score']:>10.2f} {patient_report['weighted avg']['support']:>10.0f}\n")


def save_combined_results(model, scaler, dev_results, test_results, model_name, feature_type, chunk_length, overlap, results_dir="data/results"):
    """
    Save the trained model, scaler, and combined dev+test results in a single file.
    
    Args:
        model: Trained model
        scaler: Fitted StandardScaler
        dev_results (dict): Development set evaluation results
        test_results (dict): Test set evaluation results (can be None)
        model_name (str): Name of the model
        feature_type (str): Type of features used
        chunk_length (str): Chunk length used
        overlap (str): Overlap used
        results_dir (str): Directory to save results
    """
    # Create feature-specific subdirectory
    feature_dir = os.path.join(results_dir, feature_type)
    os.makedirs(feature_dir, exist_ok=True)
    
    # Create timestamped directory for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_id = f"{feature_type}_{model_name}_{chunk_length}_{overlap}_overlap_{timestamp}"
    run_dir = os.path.join(feature_dir, experiment_id)
    os.makedirs(run_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(run_dir, "model.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    # Save scaler
    scaler_path = os.path.join(run_dir, "scaler.pkl")
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    # Save combined results
    results_path = os.path.join(run_dir, "results.txt")
    with open(results_path, 'w') as f:
        f.write(f"{feature_type.upper()} Results - {model_name} for {chunk_length}_{overlap}_overlap\n")
        f.write(f"Run timestamp: {timestamp}\n")
        f.write("=" * 70 + "\n")
        f.write("COMBINED DEV + TEST RESULTS\n")
        f.write("=" * 70 + "\n")
        
        # Write dev results
        write_results_section(f, dev_results, "development")
        
        # Write test results if available
        if test_results is not None:
            write_results_section(f, test_results, "test")
        else:
            f.write(f"\n{'='*70}\n")
            f.write("TEST SET RESULTS\n")
            f.write(f"{'='*70}\n\n")
            f.write("Test set evaluation was not performed.\n")
            f.write("Set EVALUATE_TEST_SET = True to enable test set evaluation.\n")
    
    print(f"Combined results saved to: {run_dir}")
    return run_dir


def validate_configuration(feature_type, chunk_length, overlap, models_to_train, data_root="data"):
    """
    Validate the configuration parameters and provide helpful error messages.
    
    Args:
        feature_type (str): Feature type to validate
        chunk_length (str): Chunk length to validate
        overlap (str): Overlap to validate
        models_to_train (list): Models to validate
        data_root (str): Data root directory
    
    Returns:
        bool: True if configuration is valid
    """
    # Valid feature types (matching actual directories in data/features/)
    valid_feature_types = [
        "paper_fused_features", "egemap", "emotionmsp", "hubert", 
        "paper_covarep", "paper_hosf", "umap_egemaps", "umap_egemaps_2D", "umap_egemaps_3D",
        "umap_emotionmsp", "umap_hubert", "relief_selected_paper_covarep"
    ]
    
    # Valid models
    valid_models = ["svm", "knn", "rf", "xgb", "lgb", "ensemble"]
    
    # Check feature type
    if feature_type not in valid_feature_types:
        print(f"Invalid feature type: {feature_type}")
        print(f"Valid options: {valid_feature_types}")
        return False
    
    # Check models
    invalid_models = [model for model in models_to_train if model not in valid_models]
    if invalid_models:
        print(f"Invalid models: {invalid_models}")
        print(f"Valid options: {valid_models}")
        return False
    
    # Check if feature directory exists
    feature_dir = f"{data_root}/features/{feature_type}/{chunk_length}_{overlap}_overlap"
    if not os.path.exists(feature_dir):
        print(f"Feature directory not found: {feature_dir}")
        print(f"Please ensure the features have been extracted for this configuration.")
        return False
    
    print(f"Configuration validated successfully")
    print(f"   Feature type: {feature_type}")
    print(f"   Chunk config: {chunk_length}_{overlap}_overlap")
    print(f"   Models: {models_to_train}")
    return True


def main():
    """
    Main function for unified machine learning training.
    """
    print("Unified Machine Learning Trainer")
    print("=" * 60)
    
    # Configuration parameters are now defined at the top of the script (lines 68-138)
    # Just edit the variables at the top of the file to customize your experiment!
    
    print(f"Feature type: {FEATURE_TYPE}")
    print(f"Chunk length: {CHUNK_LENGTH}")
    print(f"Overlap: {OVERLAP}")
    print(f"Balanced sampling: {BALANCED_SAMPLING}")
    print(f"Oversample minority: {OVERSAMPLE_MINORITY}")
    print(f"Random seed: {RANDOM_SEED}")
    print(f"Models to train: {MODELS_TO_TRAIN}")
    print(f"Evaluate test set: {EVALUATE_TEST_SET}")
    print("=" * 60)
    
    # Validate configuration before proceeding
    if not validate_configuration(FEATURE_TYPE, CHUNK_LENGTH, OVERLAP, MODELS_TO_TRAIN, DATA_ROOT):
        print("Configuration validation failed. Please fix the issues above.")
        return
    
    try:
        # Load features and labels
        print("Loading features and labels...")
        # If training SVM, do not apply train balancing/oversampling to keep it fast and use class weights only
        if MODELS_TO_TRAIN[0].lower() == 'svm':
            data = load_unified_features_and_labels(
                FEATURE_TYPE, CHUNK_LENGTH, OVERLAP, DATA_ROOT,
                False, RANDOM_SEED, False
            )
        else:
            data = load_unified_features_and_labels(
                FEATURE_TYPE, CHUNK_LENGTH, OVERLAP, DATA_ROOT, 
                BALANCED_SAMPLING, RANDOM_SEED, OVERSAMPLE_MINORITY
            )
        
        # Extract data for each split
        train_features = data['train_features']
        train_labels = data['train_labels']
        train_participants = data['train_participants']
        
        dev_features = data['dev_features']
        dev_labels = data['dev_labels']
        dev_participants = data['dev_participants']
        
        test_features = data['test_features']
        test_labels = data['test_labels']
        test_participants = data['test_participants']
        
        print(f"\nLoaded {FEATURE_TYPE} features:")
        print(f"Train set: {len(train_features)} samples from {len(set(train_participants))} patients")
        print(f"Dev set: {len(dev_features)} samples from {len(set(dev_participants))} patients")
        print(f"Test set: {len(test_features)} samples from {len(set(test_participants))} patients")
        
        # Check if we have enough data
        if len(train_features) == 0:
            print("Error: No training data available")
            return
        if len(dev_features) == 0:
            print("Error: No development data available")
            return
        
        # Train and evaluate single model
        trained_models = {}
        
        # Train the single model specified in MODELS_TO_TRAIN
        model_type = MODELS_TO_TRAIN[0]
        print("\n" + "="*60)
        print(f"TRAINING {model_type.upper()} MODEL")
        print("="*60)
        
        # Auto-enable undersampling for RF and XGB to improve minority recall
        auto_undersample = UNDERSAMPLE_MAJORITY_CLASS or (model_type.lower() in ["rf", "xgb"]) 
        model, scaler, dev_predictions = train_model(
            model_type, train_features, train_labels, dev_features, dev_labels, RANDOM_SEED, auto_undersample
        )
        
        dev_results = evaluate_model(dev_labels, dev_predictions, model_type.upper(), "dev", dev_participants)
        
        # Evaluate on test set if available
        test_results = None
        if EVALUATE_TEST_SET and len(test_features) > 0:
            print(f"\nEvaluating {model_type.upper()} on test set...")
            test_features_scaled = scaler.transform(test_features)
            test_predictions = model.predict(test_features_scaled)
            test_results = evaluate_model(test_labels, test_predictions, model_type.upper(), "test", test_participants)
        
        # Save combined dev + test results in a single file
        save_combined_results(model, scaler, dev_results, test_results, model_type.upper(), FEATURE_TYPE, CHUNK_LENGTH, OVERLAP)
        
        trained_models[model_type] = {'model': model, 'scaler': scaler}
        
        print("\n" + "="*60)
        print("UNIFIED TRAINING COMPLETED")
        print("="*60)
        print(f"Results saved to data/results/{FEATURE_TYPE}/")
        print(f"Trained model: {model_type.upper()}")
        
    except Exception as e:
        print(f"Error during unified training: {str(e)}")
        raise


if __name__ == "__main__":
    """
    QUICK START:
    1. Edit the configuration parameters at the top of this file (lines 68-138)
    2. Choose your feature set: FEATURE_TYPE = "paper_fused_features"
    3. Choose your single model: MODELS_TO_TRAIN = ['svm']
    4. Run: python src/machine_learning/unified_trainer.py
    """
    main()
