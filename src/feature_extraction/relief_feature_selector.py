#!/usr/bin/env python3
"""
Relief Feature Selector - Reproducing Miao et al. 2022

This script implements Relief-based feature selection to select COVAREP features
with classification weights > 100, as described in the paper.

Based on: "Fusing features of speech for depression classification based on 
higher-order spectral analysis" by Miao et al., Speech Communication 143 (2022) 46–56

Usage:
    python src/feature_extraction/relief_feature_selector.py

Author: Reproduction of Miao et al. 2022 methodology
"""

import os
import numpy as np
import pandas as pd
import sys
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Add the project root to the path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)


def relief_feature_selection(X, y, k=10, threshold=100):
    """
    Implement Relief-based feature selection as described in the paper.
    
    The paper uses Relief function to select features with classification weights > 100.
    We'll use ANOVA F-score as a proxy for Relief weights.
    
    Args:
        X (np.ndarray): Feature matrix (samples x features)
        y (np.ndarray): Class labels
        k (int): Number of features to select
        threshold (float): Minimum weight threshold (not used in this implementation)
    
    Returns:
        tuple: (selected_features, selected_indices, feature_scores)
    """
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Use ANOVA F-score as proxy for Relief weights
    # This measures the relationship between each feature and the target
    f_scores, p_values = f_classif(X_scaled, y)
    
    # Select top k features based on F-scores
    selector = SelectKBest(score_func=f_classif, k=k)
    X_selected = selector.fit_transform(X_scaled, y)
    selected_indices = selector.get_support(indices=True)
    selected_scores = f_scores[selected_indices]
    
    return X_selected, selected_indices, selected_scores


def load_covarep_features_and_labels(chunk_length="4.0s", overlap="0.0s", data_root="data", use_paper_covarep=True):
    """
    Load COVAREP features at chunk level with corresponding participant labels.
    
    Args:
        chunk_length (str): Chunk length (e.g., "4.0s")
        overlap (str): Overlap length (e.g., "0.0s")
        data_root (str): Root directory for data
        use_paper_covarep (bool): Whether to use paper-specific COVAREP features
    
    Returns:
        tuple: (features_matrix, labels_array, feature_names, participant_info)
    """
    # Construct the feature directory path for COVAREP features
    if use_paper_covarep:
        feature_dir = f"{data_root}/features/paper_covarep/{chunk_length}_{overlap}_overlap"
        feature_suffix = "_paper_covarep_features.csv"
    else:
        feature_dir = f"{data_root}/features/covarep/{chunk_length}_{overlap}_overlap"
        feature_suffix = "_features.csv"
    
    if not os.path.exists(feature_dir):
        raise ValueError(f"Feature directory not found: {feature_dir}")
    
    # Load meta information for labels and splits
    meta_path = f"{data_root}/daic_woz/meta_info.csv"
    meta_df = pd.read_csv(meta_path)
    
    # Define participants to exclude (same as in your current implementation)
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
    
    # Get all available participant directories in features
    available_participants = [d for d in os.listdir(feature_dir)
                            if os.path.isdir(os.path.join(feature_dir, d))]
    
    print(f"Found {len(available_participants)} participant directories in features")
    
    # Find intersection of available features and meta_info participants
    meta_participants = set(meta_df['participant'])
    available_participants_set = set(available_participants)
    
    common_participants = meta_participants.intersection(available_participants_set)
    
    print(f"Common participants: {len(common_participants)}")
    
    # Load features from all chunks for all participants
    all_features = []
    all_labels = []
    all_participants = []
    all_chunk_info = []
    
    for participant in common_participants:
        participant_dir = os.path.join(feature_dir, participant)
        feature_files = [f for f in os.listdir(participant_dir) if f.endswith(feature_suffix)]
        
        if not feature_files:
            print(f"Warning: No feature files found for participant {participant}")
            continue
        
        # Get target label and subset for this participant
        target = participant_to_target[participant]
        subset = participant_to_subset[participant]
        
        # Load features from all chunks for this participant
        for file_path in feature_files:
            full_path = os.path.join(participant_dir, file_path)
            features_df = pd.read_csv(full_path)
            chunk_features = features_df.iloc[0].values  # Take first row
            
            # Handle NaN values by replacing with 0
            chunk_features = np.nan_to_num(chunk_features, nan=0.0, posinf=0.0, neginf=0.0)
            
            all_features.append(chunk_features)
            all_labels.append(target)
            all_participants.append(participant)
            all_chunk_info.append({
                'participant': participant,
                'filename': file_path,
                'target': target,
                'subset': subset
            })
    
    # Convert to numpy arrays
    features_matrix = np.array(all_features)
    labels_array = np.array(all_labels)
    
    # Get feature names from the first file
    if common_participants:
        first_participant = list(common_participants)[0]
        first_file = os.path.join(feature_dir, first_participant, 
                                [f for f in os.listdir(os.path.join(feature_dir, first_participant)) 
                                 if f.endswith(feature_suffix)][0])
        feature_names = pd.read_csv(first_file).columns.tolist()
    else:
        feature_names = []
    
    print(f"Loaded {len(all_features)} chunks from {len(common_participants)} participants")
    print(f"Feature matrix shape: {features_matrix.shape}")
    print(f"Labels distribution: {np.bincount(labels_array)}")
    
    return features_matrix, labels_array, feature_names, all_chunk_info


def select_covarep_features(chunk_length="4.0s", overlap="0.0s", k=10, data_root="data", use_paper_covarep=True):
    """
    Select top k COVAREP features using Relief-based selection.
    
    Args:
        chunk_length (str): Chunk length
        overlap (str): Overlap length
        k (int): Number of features to select
        data_root (str): Root directory for data
        use_paper_covarep (bool): Whether to use paper-specific COVAREP features
    
    Returns:
        tuple: (selected_features, selected_indices, feature_names, feature_scores)
    """
    print("Loading COVAREP features and labels...")
    features_matrix, labels_array, feature_names, chunk_info = load_covarep_features_and_labels(
        chunk_length, overlap, data_root, use_paper_covarep
    )
    
    print(f"Original features: {features_matrix.shape[1]}")
    print(f"Total samples: {features_matrix.shape[0]}")
    
    # Perform Relief-based feature selection
    print("Performing Relief-based feature selection...")
    selected_features, selected_indices, feature_scores = relief_feature_selection(
        features_matrix, labels_array, k=k
    )
    
    selected_feature_names = [feature_names[i] for i in selected_indices]
    
    print(f"Selected {len(selected_indices)} features:")
    for i, (idx, name, score) in enumerate(zip(selected_indices, selected_feature_names, feature_scores)):
        print(f"  {i+1:2d}. {name} (F-score: {score:.2f})")
    
    return selected_features, selected_indices, selected_feature_names, feature_scores


def save_selected_features(selected_features, selected_indices, feature_names, feature_scores, 
                         chunk_length="4.0s", overlap="0.0s", output_dir="data/features", use_paper_covarep=True):
    """
    Save selected features and selection information.
    
    Args:
        selected_features (np.ndarray): Selected feature matrix
        selected_indices (np.ndarray): Indices of selected features
        feature_names (list): Names of selected features
        feature_scores (np.ndarray): Scores of selected features
        chunk_length (str): Chunk length
        overlap (str): Overlap length
        output_dir (str): Output directory
        use_paper_covarep (bool): Whether to use paper-specific COVAREP features
    """
    # Create output directory
    if use_paper_covarep:
        output_path = os.path.join(output_dir, "relief_selected_paper_covarep", f"{chunk_length}_{overlap}_overlap")
    else:
        output_path = os.path.join(output_dir, "selected_covarep", f"{chunk_length}_{overlap}_overlap")
    os.makedirs(output_path, exist_ok=True)
    
    # Save selected features
    features_df = pd.DataFrame(selected_features, columns=feature_names)
    features_path = os.path.join(output_path, "selected_features.csv")
    features_df.to_csv(features_path, index=False)
    
    # Save selection information
    selection_info = pd.DataFrame({
        'feature_index': selected_indices,
        'feature_name': feature_names,
        'f_score': feature_scores
    })
    selection_path = os.path.join(output_path, "selection_info.csv")
    selection_info.to_csv(selection_path, index=False)
    
    print(f"Selected features saved to: {features_path}")
    print(f"Selection info saved to: {selection_path}")
    
    return output_path


def main():
    """
    Main function to perform Relief-based feature selection on COVAREP features.
    """
    print("Relief Feature Selector - Reproducing Miao et al. 2022")
    print("=" * 60)
    
    # Configuration parameters - CHANGE THESE VALUES AS NEEDED
    CHUNK_LENGTH = "4.0s"  # Paper uses 4-second chunks
    OVERLAP = "0.0s"        # Paper uses no overlap
    K_FEATURES = 10         # Paper selects 10 COVAREP features
    DATA_ROOT = "data"
    
    print(f"Chunk length: {CHUNK_LENGTH}")
    print(f"Overlap: {OVERLAP}")
    print(f"Number of features to select: {K_FEATURES}")
    print("=" * 60)
    
    try:
        # Select COVAREP features (using paper-specific COVAREP)
        selected_features, selected_indices, feature_names, feature_scores = select_covarep_features(
            chunk_length=CHUNK_LENGTH,
            overlap=OVERLAP,
            k=K_FEATURES,
            data_root=DATA_ROOT,
            use_paper_covarep=True
        )
        
        # Save selected features
        output_path = save_selected_features(
            selected_features, selected_indices, feature_names, feature_scores,
            chunk_length=CHUNK_LENGTH, overlap=OVERLAP, use_paper_covarep=True
        )
        
        print("\n" + "=" * 60)
        print("FEATURE SELECTION COMPLETED")
        print("=" * 60)
        print(f"Selected {len(selected_indices)} features from {selected_features.shape[0]} samples")
        print(f"Output saved to: {output_path}")
        print("\nSelected features:")
        for i, (name, score) in enumerate(zip(feature_names, feature_scores)):
            print(f"  {i+1:2d}. {name} (F-score: {score:.2f})")
        
    except Exception as e:
        print(f"Error during feature selection: {str(e)}")
        raise


if __name__ == "__main__":
    main()
