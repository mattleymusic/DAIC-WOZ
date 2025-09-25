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
from sklearn.neighbors import NearestNeighbors
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
    This implements the actual Relief algorithm.
    
    Args:
        X (np.ndarray): Feature matrix (samples x features)
        y (np.ndarray): Class labels
        k (int): Number of features to select
        threshold (float): Minimum weight threshold for Relief weights
    
    Returns:
        tuple: (selected_features, selected_indices, feature_scores)
    """
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    n_samples, n_features = X_scaled.shape
    n_classes = len(np.unique(y))
    
    # Initialize Relief weights
    relief_weights = np.zeros(n_features)
    
    # For each sample, find nearest neighbors
    # Use k=5 nearest neighbors (common choice)
    nn = NearestNeighbors(n_neighbors=6, metric='euclidean')  # 6 because sample itself is included
    nn.fit(X_scaled)
    
    for i in range(n_samples):
        # Find nearest neighbors (including the sample itself)
        distances, indices = nn.kneighbors([X_scaled[i]])
        
        # Remove the sample itself (first neighbor)
        neighbor_indices = indices[0][1:]  # Skip the first (self) neighbor
        
        # Get class of current sample
        current_class = y[i]
        
        # Calculate Relief weights for each feature
        for feature_idx in range(n_features):
            feature_value = X_scaled[i, feature_idx]
            
            # Find nearest hit (same class) and nearest miss (different class)
            nearest_hit_distance = float('inf')
            nearest_miss_distance = float('inf')
            
            for neighbor_idx in neighbor_indices:
                neighbor_class = y[neighbor_idx]
                neighbor_feature_value = X_scaled[neighbor_idx, feature_idx]
                
                # Calculate feature difference (normalized by feature range)
                feature_diff = abs(feature_value - neighbor_feature_value)
                
                if neighbor_class == current_class:
                    # Same class - this is a hit
                    if feature_diff < nearest_hit_distance:
                        nearest_hit_distance = feature_diff
                else:
                    # Different class - this is a miss
                    if feature_diff < nearest_miss_distance:
                        nearest_miss_distance = feature_diff
            
            # Update Relief weight: miss - hit
            # Higher weight means better feature for classification
            if nearest_miss_distance != float('inf') and nearest_hit_distance != float('inf'):
                relief_weights[feature_idx] += nearest_miss_distance - nearest_hit_distance
    
    # Normalize weights by number of samples
    relief_weights = relief_weights / n_samples
    
    # Select top k features based on Relief weights
    # Also apply threshold if specified
    if threshold is not None:
        # Filter features above threshold
        above_threshold = relief_weights >= threshold
        if np.sum(above_threshold) > 0:
            # Select top k from those above threshold
            above_threshold_indices = np.where(above_threshold)[0]
            above_threshold_weights = relief_weights[above_threshold_indices]
            top_k_from_threshold = min(k, len(above_threshold_indices))
            top_indices = np.argsort(above_threshold_weights)[-top_k_from_threshold:][::-1]
            selected_indices = above_threshold_indices[top_indices]
        else:
            # If no features above threshold, select top k overall
            print(f"Warning: No features above threshold {threshold}, selecting top {k} features")
            selected_indices = np.argsort(relief_weights)[-k:][::-1]
    else:
        # Select top k features
        selected_indices = np.argsort(relief_weights)[-k:][::-1]
    
    # Get selected features and their weights
    X_selected = X_scaled[:, selected_indices]
    selected_scores = relief_weights[selected_indices]
    
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
        print(f"  {i+1:2d}. {name} (Relief weight: {score:.4f})")
    
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
        'relief_weight': feature_scores
    })
    selection_path = os.path.join(output_path, "selection_info.csv")
    selection_info.to_csv(selection_path, index=False)
    
    print(f"Selected features saved to: {features_path}")
    print(f"Selection info saved to: {selection_path}")
    
    return output_path


def main():
    """
    Main function to perform Relief-based feature selection on COVAREP features for all chunk configurations.
    """
    print("Relief Feature Selector - Reproducing Miao et al. 2022")
    print("=" * 60)
    
    # Configuration parameters - CHANGE THESE VALUES AS NEEDED
    CHUNK_CONFIGS = [
        ("3.0s", "1.5s"),   # 3-second chunks with 1.5s overlap
        ("4.0s", "0.0s"),   # 4-second chunks with no overlap (paper's approach)
        ("5.0s", "2.5s"),   # 5-second chunks with 2.5s overlap
        ("10.0s", "5.0s"),  # 10-second chunks with 5s overlap
        ("20.0s", "10.0s"), # 20-second chunks with 10s overlap
        ("30.0s", "15.0s")  # 30-second chunks with 15s overlap
    ]
    K_FEATURES = 10         # Paper selects 10 COVAREP features
    DATA_ROOT = "data"
    
    print(f"Chunk configurations: {CHUNK_CONFIGS}")
    print(f"Number of features to select: {K_FEATURES}")
    print("=" * 60)
    
    total_configurations_processed = 0
    total_configurations_failed = 0
    
    # Process each chunk configuration
    for chunk_length, overlap in CHUNK_CONFIGS:
        print(f"\n{'='*20} Processing {chunk_length}_{overlap}_overlap {'='*20}")
        
        try:
            # Select COVAREP features (using paper-specific COVAREP)
            selected_features, selected_indices, feature_names, feature_scores = select_covarep_features(
                chunk_length=chunk_length,
                overlap=overlap,
                k=K_FEATURES,
                data_root=DATA_ROOT,
                use_paper_covarep=True
            )
            
            # Save selected features
            output_path = save_selected_features(
                selected_features, selected_indices, feature_names, feature_scores,
                chunk_length=chunk_length, overlap=overlap, use_paper_covarep=True
            )
            
            print(f"Successfully processed {chunk_length}_{overlap}_overlap")
            print(f"  Selected {len(selected_indices)} features from {selected_features.shape[0]} samples")
            print(f"  Output saved to: {output_path}")
            
            total_configurations_processed += 1
            
        except Exception as e:
            print(f"Error processing {chunk_length}_{overlap}_overlap: {str(e)}")
            total_configurations_failed += 1
            continue
    
    print("\n" + "=" * 60)
    print("FEATURE SELECTION COMPLETED")
    print("=" * 60)
    print(f"Configurations processed successfully: {total_configurations_processed}")
    print(f"Configurations failed: {total_configurations_failed}")
    print(f"\nRelief selection results saved to:")
    print(f"  data/features/relief_selected_paper_covarep/[CONFIG]/")
    print(f"\nEach configuration contains:")
    print(f"  - selected_features.csv: Feature matrix with selected features")
    print(f"  - selection_info.csv: Indices, names, and scores of selected features")
    print(f"\nThis enables the create_paper_fused_features.py script to work!")


if __name__ == "__main__":
    main()
