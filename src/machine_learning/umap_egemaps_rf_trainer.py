#!/usr/bin/env python3
"""
UMAP-eGeMAPS Chunk-Level Random Forest Trainer with Majority Voting for DAIC-WOZ Dataset

This script trains Random Forest models at the chunk level on UMAP-transformed eGeMAPS features for all available chunk lengths.
It performs majority voting to aggregate chunk predictions to patient-level predictions.
Handles class imbalance using balanced class weights.

Features:
- Trains on individual chunks (not averaged per patient)
- Performs majority voting per patient for final predictions
- Trains on all available chunk lengths automatically
- Uses predefined train/dev/test splits from meta_info.csv
- Handles class imbalance with balanced class weights
- Uses optimized Random Forest parameters for speed and performance
- Saves both chunk-level and patient-level results
- Reports performance at both chunk and patient levels

Usage:
    python src/machine_learning/umap_egemaps_rf_trainer.py

Author: DAIC-WOZ Machine Learning Pipeline
"""

import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import pickle
import glob
from pathlib import Path
from datetime import datetime
import warnings
from tqdm import tqdm
import time
warnings.filterwarnings("ignore")

def get_available_chunk_lengths(data_root="data"):
    """
    Get all available chunk length configurations from the created_data directory.

    Args:
        data_root (str): Root directory for data

    Returns:
        list: List of available chunk length configurations (e.g., ["3.0s", "5.0s", ...])
    """
    created_data_dir = f"{data_root}/created_data"
    if not os.path.exists(created_data_dir):
        raise ValueError(f"Created data directory not found: {created_data_dir}")

    available_configs = []
    for item in os.listdir(created_data_dir):
        if os.path.isdir(os.path.join(created_data_dir, item)) and item != "concatenated_diarisation":
            # Parse chunk_length from directory name (e.g., "3.0s_1.5s_overlap" -> "3.0s")
            if "_" in item:
                chunk_length = item.split("_")[0]
                available_configs.append(chunk_length)

    # Sort by chunk length
    available_configs.sort(key=lambda x: float(x.replace("s", "")))
    return available_configs

def load_umap_egemaps_chunk_features_and_labels(chunk_length, overlap, data_root="data"):
    """
    Load UMAP-transformed eGeMAPS features at chunk level with corresponding participant labels.
    Each chunk inherits the label of its participant.
    Uses the predefined train/dev/test splits from meta_info.csv.
    Filters out specified participants.

    Args:
        chunk_length (str): Chunk length (e.g., "5.0s")
        overlap (str): Overlap length (e.g., "2.5s")
        data_root (str): Root directory for data

    Returns:
        dict: Dictionary containing train, dev, and test data at chunk level
    """
    # Construct the feature directory path
    feature_dir = f"{data_root}/features/umap_egemaps/{chunk_length}_{overlap}_overlap"

    if not os.path.exists(feature_dir):
        raise ValueError(f"Feature directory not found: {feature_dir}")

    # Load meta information for labels and splits
    meta_path = f"{data_root}/daic_woz/meta_info.csv"
    meta_df = pd.read_csv(meta_path)

    # Define participants to exclude
    excluded_participants = [
        '324_P', '323_P', '322_P', '321_P',  # String format
        '300_P', '305_P', '318_P', '341_P', '362_P', '451_P', '458_P', '480_P'  # Add _P suffix
    ]

    # Filter out excluded participants
    meta_df = meta_df[~meta_df['participant'].isin(excluded_participants)]

    print(f"Excluded {len(excluded_participants)} participants")
    print(f"Remaining participants: {len(meta_df)}")

    # Create mappings
    participant_to_target = dict(zip(meta_df['participant'], meta_df['target']))
    participant_to_subset = dict(zip(meta_df['participant'], meta_df['subset']))

    # Get all available participant directories in features
    available_participants = [d for d in os.listdir(feature_dir)
                            if os.path.isdir(os.path.join(feature_dir, d))]

    print(f"Found {len(available_participants)} participant directories in features")
    print(f"Meta info contains {len(meta_df)} participants (after filtering)")

    # Find intersection of available features and meta_info participants
    meta_participants = set(meta_df['participant'])
    available_participants_set = set(available_participants)

    common_participants = meta_participants.intersection(available_participants_set)
    missing_from_features = meta_participants - available_participants_set
    missing_from_meta = available_participants_set - meta_participants

    print(f"Common participants: {len(common_participants)}")
    if missing_from_features:
        print(f"Participants in meta_info but missing features: {len(missing_from_features)}")
    if missing_from_meta:
        print(f"Participants with features but not in meta_info: {len(missing_from_meta)}")

    # Initialize data structures for each split
    splits = {'train': [], 'dev': [], 'test': []}
    split_labels = {'train': [], 'dev': [], 'test': []}
    split_participants = {'train': [], 'dev': [], 'test': []}
    split_chunk_info = {'train': [], 'dev': [], 'test': []}  # Store chunk metadata

    # Process only participants that exist in both features and meta_info
    total_chunks = 0
    print("Loading chunk features with progress bar...")

    # Create progress bar for participants
    print(f"Loading features for {len(common_participants)} participants...")
    with tqdm(total=len(common_participants), desc="Participants", unit="participant") as pbar:
        for participant in common_participants:
            participant_dir = os.path.join(feature_dir, participant)
            feature_files = glob.glob(os.path.join(participant_dir, "*_umap_features.csv"))

            if not feature_files:
                print(f"Warning: No feature files found for participant {participant}")
                pbar.update(1)
                continue

            # Get target label and subset for this participant
            target = participant_to_target[participant]
            subset = participant_to_subset[participant]

            # Load features from all chunks for this participant
            participant_chunks = 0
            for file_path in feature_files:
                features_df = pd.read_csv(file_path)
                # Extract UMAP features (columns starting with 'umap_')
                umap_columns = [col for col in features_df.columns if col.startswith('umap_')]
                chunk_features = features_df[umap_columns].iloc[0].values  # Take first row

                # Extract chunk information from filename
                filename = os.path.basename(file_path)
                chunk_info = {
                    'participant': participant,
                    'filename': filename,
                    'target': target,
                    'subset': subset
                }

                # Add to appropriate split
                splits[subset].append(chunk_features)
                split_labels[subset].append(target)
                split_participants[subset].append(participant)
                split_chunk_info[subset].append(chunk_info)
                total_chunks += 1
                participant_chunks += 1

            pbar.set_postfix({
                'participant': participant,
                'chunks_loaded': participant_chunks,
                'total_chunks': total_chunks
            })
            pbar.update(1)

    print(f"Total chunks processed: {total_chunks}")

    # Convert to numpy arrays
    data = {}
    for subset in ['train', 'dev', 'test']:
        if splits[subset]:  # Only create arrays if data exists
            data[f'{subset}_features'] = np.array(splits[subset])
            data[f'{subset}_labels'] = np.array(split_labels[subset])
            data[f'{subset}_participants'] = split_participants[subset]
            data[f'{subset}_chunk_info'] = split_chunk_info[subset]
        else:
            data[f'{subset}_features'] = np.array([])
            data[f'{subset}_labels'] = np.array([])
            data[f'{subset}_participants'] = []
            data[f'{subset}_chunk_info'] = []

    # Get feature names from the first file
    if common_participants:
        first_participant = list(common_participants)[0]
        first_file = glob.glob(os.path.join(feature_dir, first_participant, "*_umap_features.csv"))[0]
        features_df = pd.read_csv(first_file)
        feature_names = [col for col in features_df.columns if col.startswith('umap_')]
        data['feature_names'] = feature_names
    else:
        data['feature_names'] = []

    return data

def train_umap_egemaps_rf_model(train_features, train_labels, dev_features, dev_labels, random_state=42):
    """
    Train a Random Forest model using the predefined train/dev splits on UMAP-transformed eGeMAPS features.
    Handles class imbalance using class weights with optimized RF parameters for speed.

    Args:
        train_features (np.array): Training feature matrix
        train_labels (np.array): Training labels
        dev_features (np.array): Development feature matrix
        dev_labels (np.array): Development labels
        random_state (int): Random seed for reproducibility

    Returns:
        tuple: (trained_model, scaler, dev_predictions)
    """
    # Scale the features
    scaler = StandardScaler()
    train_features_scaled = scaler.fit_transform(train_features)
    dev_features_scaled = scaler.transform(dev_features)

    # Compute class weights to handle imbalance
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(train_labels),
        y=train_labels
    )
    class_weight_dict = dict(zip(np.unique(train_labels), class_weights))

    print(f"Class weights: {class_weight_dict}")
    print(f"Training set class distribution: {np.bincount(train_labels)}")

    # Train Random Forest with optimized parameters for speed and performance
    print("Training Random Forest with optimized parameters...")

    rf_model = RandomForestClassifier(
        n_estimators=100,          # Good balance of performance vs speed
        max_depth=20,              # Limit depth to prevent overfitting
        min_samples_split=10,      # Require more samples to split
        min_samples_leaf=5,        # Require more samples per leaf
        max_features='sqrt',       # Use sqrt(n_features) for speed
        class_weight=class_weight_dict,
        random_state=random_state,
        n_jobs=-1                  # Use all available cores
    )

    # Fit the model with progress tracking
    print(f"Training on {len(train_features_scaled)} samples with {train_features_scaled.shape[1]} features...")
    start_time = time.time()

    with tqdm(total=1, desc="RF Training", unit="model") as pbar:
        pbar.set_postfix({
            'samples': len(train_features_scaled),
            'features': train_features_scaled.shape[1],
            'trees': 100
        })

        # Fit the model
        rf_model.fit(train_features_scaled, train_labels)

        training_time = time.time() - start_time
        pbar.set_postfix({
            'training_time': f"{training_time:.2f}s",
            'samples': len(train_features_scaled),
            'features': train_features_scaled.shape[1]
        })
        pbar.update(1)

    print(f"Random Forest training completed in {training_time:.2f} seconds")

    # Make predictions on dev set
    dev_predictions = rf_model.predict(dev_features_scaled)

    return rf_model, scaler, dev_predictions

def predict_patient_level_from_chunks(model, chunk_features, chunk_participants, chunk_info):
    """
    Predict at chunk level and aggregate to patient level using majority vote.

    Args:
        model: Trained RF model
        chunk_features (np.array): Chunk-level features
        chunk_participants (list): List of participant IDs for each chunk
        chunk_info (list): List of chunk metadata dictionaries

    Returns:
        tuple: (patient_predictions, chunk_predictions_df)
    """
    # Get chunk-level predictions
    chunk_predictions = model.predict(chunk_features)
    chunk_probabilities = model.predict_proba(chunk_features)[:, 1]  # Probability of positive class

    # Create DataFrame with chunk predictions
    chunk_preds_df = pd.DataFrame({
        'participant': chunk_participants,
        'chunk_pred': chunk_predictions,
        'chunk_prob': chunk_probabilities,
        'filename': [info['filename'] for info in chunk_info],
        'true_label': [info['target'] for info in chunk_info]
    })

    # Aggregate to patient level using majority vote
    patient_preds_df = chunk_preds_df.groupby('participant').agg({
        'chunk_pred': lambda x: (x.sum() > len(x)/2).astype(int),  # Majority vote
        'chunk_prob': 'mean',  # Average probability
        'true_label': 'first'  # True label (should be same for all chunks of a patient)
    }).reset_index()

    patient_preds_df.columns = ['participant', 'patient_pred', 'patient_prob', 'true_label']

    return patient_preds_df, chunk_preds_df

def evaluate_chunk_level_model(y_true, y_pred, chunk_info, split_name="dev"):
    """
    Evaluate the model performance at chunk level and patient level.

    Args:
        y_true (np.array): True labels (chunk level)
        y_pred (np.array): Predicted labels (chunk level)
        chunk_info (list): List of chunk metadata dictionaries
        split_name (str): Name of the split being evaluated

    Returns:
        dict: Dictionary containing chunk-level and patient-level evaluation metrics
    """
    # Chunk-level metrics
    chunk_accuracy = accuracy_score(y_true, y_pred)
    chunk_f1_weighted = f1_score(y_true, y_pred, average='weighted')
    chunk_f1_macro = f1_score(y_true, y_pred, average='macro')

    # Get chunk-level classification report
    chunk_report = classification_report(y_true, y_pred, output_dict=True)

    # Patient-level evaluation using majority voting
    unique_participants = set([info['participant'] for info in chunk_info])
    patient_true_labels = []
    patient_pred_labels = []

    print(f"Aggregating predictions for {len(unique_participants)} patients...")
    with tqdm(total=len(unique_participants), desc="Patient Aggregation", unit="patient") as pbar:
        for participant in unique_participants:
            # Get all chunks for this participant
            participant_chunks = [(i, info) for i, info in enumerate(chunk_info) if info['participant'] == participant]

            # True label (same for all chunks)
            true_label = participant_chunks[0][1]['target']

            # Predicted labels for all chunks
            pred_labels = [y_pred[i] for i, _ in participant_chunks]

            # Majority vote
            majority_pred = 1 if sum(pred_labels) > len(pred_labels) / 2 else 0

            patient_true_labels.append(true_label)
            patient_pred_labels.append(majority_pred)

            pbar.set_postfix({
                'participant': participant,
                'chunks': len(participant_chunks),
                'majority_vote': majority_pred
            })
            pbar.update(1)

    # Patient-level metrics
    patient_accuracy = accuracy_score(patient_true_labels, patient_pred_labels)
    patient_f1_weighted = f1_score(patient_true_labels, patient_pred_labels, average='weighted')
    patient_f1_macro = f1_score(patient_true_labels, patient_pred_labels, average='macro')

    # Get patient-level classification report
    patient_report = classification_report(patient_true_labels, patient_pred_labels, output_dict=True)

    return {
        'chunk_accuracy': chunk_accuracy,
        'chunk_f1_weighted': chunk_f1_weighted,
        'chunk_f1_macro': chunk_f1_macro,
        'chunk_classification_report': chunk_report,
        'patient_accuracy': patient_accuracy,
        'patient_f1_weighted': patient_f1_weighted,
        'patient_f1_macro': patient_f1_macro,
        'patient_classification_report': patient_report,
        'split': split_name,
        'n_chunks': len(y_true),
        'n_patients': len(unique_participants),
        'y_true': y_true,
        'y_pred': y_pred,
        'chunk_info': chunk_info,
        'patient_true_labels': patient_true_labels,
        'patient_pred_labels': patient_pred_labels
    }

def save_umap_egemaps_results(model, scaler, results, chunk_length, overlap, results_dir="data/results"):
    """
    Save the trained model, scaler, and results in a timestamped directory.

    Args:
        model: Trained RF model
        scaler: Fitted StandardScaler
        results (dict): Evaluation results
        chunk_length (str): Chunk length used
        overlap (str): Overlap used
        results_dir (str): Directory to save results
    """
    # Create feature-specific subdirectory
    feature_dir = os.path.join(results_dir, "umap_egemaps")
    os.makedirs(feature_dir, exist_ok=True)

    # Create timestamped directory for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_id = f"umap_egemaps_chunk_rf_{chunk_length}_{overlap}_overlap_{timestamp}"
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

    # Save results
    results_path = os.path.join(run_dir, "results.txt")
    with open(results_path, 'w') as f:
        f.write(f"UMAP-eGeMAPS Chunk-Level Random Forest Training Results for {chunk_length}_{overlap}_overlap\n")
        f.write(f"Run timestamp: {timestamp}\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Split: {results['split']}\n")
        f.write(f"Number of chunks: {results['n_chunks']}\n")
        f.write(f"Number of patients: {results['n_patients']}\n\n")

        f.write("CHUNK-LEVEL RESULTS:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Chunk-level Accuracy: {results['chunk_accuracy']:.4f}\n")
        f.write(f"Chunk-level F1 Score (weighted): {results['chunk_f1_weighted']:.4f}\n")
        f.write(f"Chunk-level F1 Score (macro): {results['chunk_f1_macro']:.4f}\n\n")

        f.write("PATIENT-LEVEL RESULTS (Majority Vote):\n")
        f.write("-" * 40 + "\n")
        f.write(f"Patient-level Accuracy: {results['patient_accuracy']:.4f}\n")
        f.write(f"Patient-level F1 Score (weighted): {results['patient_f1_weighted']:.4f}\n")
        f.write(f"Patient-level F1 Score (macro): {results['patient_f1_macro']:.4f}\n\n")

        f.write("Chunk-Level Classification Report:\n")
        f.write("-" * 40 + "\n")
        f.write(classification_report(
            results['y_true'],
            results['y_pred'],
            target_names=['Non-Depressed', 'Depressed']
        ))

        f.write("\nPatient-Level Classification Report:\n")
        f.write("-" * 40 + "\n")
        f.write(classification_report(
            results['patient_true_labels'],
            results['patient_pred_labels'],
            target_names=['Non-Depressed', 'Depressed']
        ))

    # Save chunk-level predictions for analysis
    chunk_preds_df = pd.DataFrame({
        'participant': [info['participant'] for info in results['chunk_info']],
        'filename': [info['filename'] for info in results['chunk_info']],
        'true_label': results['y_true'],
        'predicted_label': results['y_pred'],
        'subset': [info['subset'] for info in results['chunk_info']]
    })

    predictions_path = os.path.join(run_dir, "chunk_predictions.csv")
    chunk_preds_df.to_csv(predictions_path, index=False)

    print(f"Results saved to: {run_dir}")
    print(f"Model saved as: {model_path}")
    print(f"Scaler saved as: {scaler_path}")
    print(f"Results saved as: {results_path}")
    print(f"Chunk predictions saved as: {predictions_path}")

    return run_dir

def get_overlap_for_chunk_length(chunk_length):
    """
    Get the corresponding overlap for a given chunk length.
    This is based on the standard overlap ratios used in the project.

    Args:
        chunk_length (str): Chunk length (e.g., "5.0s")

    Returns:
        str: Corresponding overlap length (e.g., "2.5s")
    """
    chunk_seconds = float(chunk_length.replace("s", ""))

    # Standard overlap ratios: overlap = chunk_length / 2
    overlap_seconds = chunk_seconds / 2
    overlap_str = f"{overlap_seconds:.1f}s"

    return overlap_str

def main():
    """
    Main function to run UMAP-eGeMAPS Random Forest training for all available chunk lengths.
    Uses predefined train/dev/test splits from meta_info.csv.
    Handles class imbalance and excludes specified participants.
    """
    print("UMAP-eGeMAPS Random Forest Trainer for DAIC-WOZ Dataset")
    print("=" * 50)

    try:
        # Get all available chunk lengths
        available_lengths = get_available_chunk_lengths()
        print(f"Found {len(available_lengths)} available chunk lengths: {available_lengths}")

        # Configuration - these can be changed as needed
        # If you want to train on specific lengths only, modify this list
        chunk_lengths_to_train = available_lengths  # Train on all available lengths

        # Process each chunk length
        print(f"\nStarting training across {len(chunk_lengths_to_train)} chunk lengths...")
        chunk_start_time = time.time()

        for i, chunk_length in enumerate(chunk_lengths_to_train):
            chunk_iteration_start = time.time()

            print(f"\n{'='*80}")
            print(f"Training on chunk length {i+1}/{len(chunk_lengths_to_train)}: {chunk_length}")
            print(f"{'='*80}")

            # Get corresponding overlap
            overlap = get_overlap_for_chunk_length(chunk_length)
            print(f"Using overlap: {overlap}")

            try:
                # Load chunk-level features and labels with proper splits
                print("Loading UMAP-eGeMAPS chunk features and labels...")
                data = load_umap_egemaps_chunk_features_and_labels(chunk_length, overlap)

                # Extract data for each split
                train_features = data['train_features']
                train_labels = data['train_labels']
                train_participants = data['train_participants']
                train_chunk_info = data['train_chunk_info']

                dev_features = data['dev_features']
                dev_labels = data['dev_labels']
                dev_participants = data['dev_participants']
                dev_chunk_info = data['dev_chunk_info']

                test_features = data['test_features']
                test_labels = data['test_labels']
                test_participants = data['test_participants']
                test_chunk_info = data['test_chunk_info']

                feature_names = data['feature_names']

                print(f"\nLoaded UMAP-eGeMAPS features with {len(feature_names)} dimensions")
                print(f"Train set: {len(train_features)} chunks from {len(set(train_participants))} patients")
                print(f"Dev set: {len(dev_features)} chunks from {len(set(dev_participants))} patients")
                print(f"Test set: {len(test_features)} chunks from {len(set(test_participants))} patients")

                if len(train_features) > 0:
                    print(f"Train chunk label distribution: {np.bincount(train_labels)}")
                if len(dev_features) > 0:
                    print(f"Dev chunk label distribution: {np.bincount(dev_labels)}")
                if len(test_features) > 0:
                    print(f"Test chunk label distribution: {np.bincount(test_labels)}")

                # Check if we have enough data
                if len(train_features) == 0:
                    print(f"Warning: No training data available for {chunk_length}, skipping...")
                    continue
                if len(dev_features) == 0:
                    print(f"Warning: No development data available for {chunk_length}, skipping...")
                    continue

                # Train Random Forest model on train set
                print("\nTraining UMAP-eGeMAPS Random Forest model on train set...")
                model, scaler, dev_predictions = train_umap_egemaps_rf_model(
                    train_features, train_labels, dev_features, dev_labels
                )

                # Evaluate on dev set (chunk-level and patient-level)
                print("Evaluating model on dev set...")
                dev_results = evaluate_chunk_level_model(dev_labels, dev_predictions, dev_chunk_info, "dev")

                # Print dev results
                print(f"\nDev Set Results (Chunk Level):")
                print(f"Chunk-level Accuracy: {dev_results['chunk_accuracy']:.4f}")
                print(f"Chunk-level F1 Score (weighted): {dev_results['chunk_f1_weighted']:.4f}")
                print(f"Chunk-level F1 Score (macro): {dev_results['chunk_f1_macro']:.4f}")

                print(f"\nDev Set Results (Patient Level - Majority Vote):")
                print(f"Patient-level Accuracy: {dev_results['patient_accuracy']:.4f}")
                print(f"Patient-level F1 Score (weighted): {dev_results['patient_f1_weighted']:.4f}")
                print(f"Patient-level F1 Score (macro): {dev_results['patient_f1_macro']:.4f}")
                print(f"Number of patients: {dev_results['n_patients']}")

                # Evaluate on test set (if available)
                if len(test_features) > 0:
                    print(f"\nEvaluating model on test set ({len(test_features)} chunks)...")
                    with tqdm(total=2, desc="Test Evaluation", unit="step") as pbar:
                        pbar.set_description("Scaling test features")
                        test_features_scaled = scaler.transform(test_features)
                        pbar.update(1)

                        pbar.set_description("Making predictions")
                        test_predictions = model.predict(test_features_scaled)
                        pbar.update(1)

                    test_results = evaluate_chunk_level_model(test_labels, test_predictions, test_chunk_info, "test")

                    print(f"\nTest Set Results (Chunk Level):")
                    print(f"Chunk-level Accuracy: {test_results['chunk_accuracy']:.4f}")
                    print(f"Chunk-level F1 Score (weighted): {test_results['chunk_f1_weighted']:.4f}")
                    print(f"Chunk-level F1 Score (macro): {test_results['chunk_f1_macro']:.4f}")

                    print(f"\nTest Set Results (Patient Level - Majority Vote):")
                    print(f"Patient-level Accuracy: {test_results['patient_accuracy']:.4f}")
                    print(f"Patient-level F1 Score (weighted): {test_results['patient_f1_weighted']:.4f}")
                    print(f"Patient-level F1 Score (macro): {test_results['patient_f1_macro']:.4f}")
                    print(f"Number of patients: {test_results['n_patients']}")
                else:
                    test_results = None

                # Save results (using dev results as primary)
                print("\nSaving results...")
                run_dir = save_umap_egemaps_results(model, scaler, dev_results, chunk_length, overlap)

                # Save test results separately if available
                if test_results:
                    test_results_path = os.path.join(run_dir, "test_results.txt")
                    with open(test_results_path, 'w') as f:
                        f.write(f"UMAP-eGeMAPS Chunk-Level Random Forest Test Results for {chunk_length}_{overlap}_overlap\n")
                        f.write(f"Run timestamp: {datetime.now().strftime('%Y%m%d_%H%M%S')}\n")
                        f.write("=" * 70 + "\n\n")
                        f.write(f"Split: {test_results['split']}\n")
                        f.write(f"Number of chunks: {test_results['n_chunks']}\n")
                        f.write(f"Number of patients: {test_results['n_patients']}\n\n")

                        f.write("CHUNK-LEVEL RESULTS:\n")
                        f.write("-" * 30 + "\n")
                        f.write(f"Chunk-level Accuracy: {test_results['chunk_accuracy']:.4f}\n")
                        f.write(f"Chunk-level F1 Score (weighted): {test_results['chunk_f1_weighted']:.4f}\n")
                        f.write(f"Chunk-level F1 Score (macro): {test_results['chunk_f1_macro']:.4f}\n\n")

                        f.write("PATIENT-LEVEL RESULTS (Majority Vote):\n")
                        f.write("-" * 40 + "\n")
                        f.write(f"Patient-level Accuracy: {test_results['patient_accuracy']:.4f}\n")
                        f.write(f"Patient-level F1 Score (weighted): {test_results['patient_f1_weighted']:.4f}\n")
                        f.write(f"Patient-level F1 Score (macro): {test_results['patient_f1_macro']:.4f}\n\n")

                        f.write("Chunk-Level Classification Report:\n")
                        f.write("-" * 40 + "\n")
                        f.write(classification_report(
                            test_results['y_true'],
                            test_results['y_pred'],
                            target_names=['Non-Depressed', 'Depressed']
                        ))

                        f.write("\nPatient-Level Classification Report:\n")
                        f.write("-" * 40 + "\n")
                        f.write(classification_report(
                            test_results['patient_true_labels'],
                            test_results['patient_pred_labels'],
                            target_names=['Non-Depressed', 'Depressed']
                        ))
                    print(f"Test results saved as: {test_results_path}")

                chunk_iteration_time = time.time() - chunk_iteration_start
                print(f"\nCompleted training for {chunk_length}_{overlap}_overlap in {chunk_iteration_time:.2f} seconds")
                print(f"All results saved in: {run_dir}")

            except Exception as e:
                chunk_iteration_time = time.time() - chunk_iteration_start
                print(f"Error processing {chunk_length} after {chunk_iteration_time:.2f} seconds: {str(e)}")
                print("Continuing with next chunk length...")
                continue

        total_training_time = time.time() - chunk_start_time
        print(f"\n{'='*80}")
        print("UMAP-eGeMAPS Random Forest training completed for all chunk lengths!")
        print(f"Total training time: {total_training_time:.2f} seconds")
        print(f"Average time per chunk length: {total_training_time/len(chunk_lengths_to_train):.2f} seconds")
        print("Check the data/results/umap_egemaps/ directory for all saved models and results.")

    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise

if __name__ == "__main__":
    main()
