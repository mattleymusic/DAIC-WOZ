#!/usr/bin/env python3
"""
UMAP Feature Transformer for HOSF Fusion Features in DAIC-WOZ Dataset

This script applies UMAP dimensionality reduction to HOSF fusion features.
It handles NaN values by imputing them with median values and creates new directories 
with UMAP-transformed features for all available chunk lengths.

Features:
- Applies UMAP to HOSF fusion features (COVAREP + HOSF)
- Handles NaN values with median imputation
- Creates new directory: data/features/umap_hosf_fusion
- Processes all available chunk lengths automatically
- Saves UMAP-transformed features as CSV files
- Maintains original data structure and labels

Usage:
    python src/feature_extraction/umap_hosf_fusion_transformer.py

Author: DAIC-WOZ Feature Extraction Pipeline
"""

import os
import pandas as pd
import numpy as np
import glob
from pathlib import Path
from datetime import datetime
import warnings
from tqdm import tqdm
import time
from sklearn.impute import SimpleImputer

# Import UMAP
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    print("UMAP not available. Please install with: pip install umap-learn")
    UMAP_AVAILABLE = False

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

def get_overlap_for_chunk_length(chunk_length):
    """
    Get the corresponding overlap for a given chunk length.

    Args:
        chunk_length (str): Chunk length (e.g., "5.0s")

    Returns:
        str: Corresponding overlap length (e.g., "2.5s")
    """
    chunk_seconds = float(chunk_length.replace("s", ""))
    overlap_seconds = chunk_seconds / 2
    overlap_str = f"{overlap_seconds:.1f}s"
    return overlap_str

def load_hosf_fusion_features_for_umap(chunk_length, overlap, data_root="data"):
    """
    Load HOSF fusion features for UMAP transformation.

    Args:
        chunk_length (str): Chunk length (e.g., "5.0s")
        overlap (str): Overlap length (e.g., "2.5s")
        data_root (str): Root directory for data

    Returns:
        tuple: (features_array, file_paths, participant_ids, labels)
    """
    feature_dir = f"{data_root}/features/hosf_fusion/{chunk_length}_{overlap}_overlap"
    
    if not os.path.exists(feature_dir):
        raise ValueError(f"HOSF fusion feature directory not found: {feature_dir}")

    # Load meta information for labels
    meta_path = f"{data_root}/daic_woz/meta_info.csv"
    meta_df = pd.read_csv(meta_path)

    # Define participants to exclude
    excluded_participants = [
        '324_P', '323_P', '322_P', '321_P',
        '300_P', '305_P', '318_P', '341_P', '362_P', '451_P', '458_P', '480_P'
    ]

    # Filter out excluded participants
    meta_df = meta_df[~meta_df['participant'].isin(excluded_participants)]
    participant_to_target = dict(zip(meta_df['participant'], meta_df['target']))

    # Get all available participant directories
    available_participants = [d for d in os.listdir(feature_dir)
                            if os.path.isdir(os.path.join(feature_dir, d))]

    print(f"Processing {len(available_participants)} participants for HOSF fusion {chunk_length}")

    features_list = []
    file_paths = []
    participant_ids = []
    labels = []

    # Process each participant
    with tqdm(total=len(available_participants), desc="Loading HOSF fusion features", unit="participant") as pbar:
        for participant in available_participants:
            participant_dir = os.path.join(feature_dir, participant)
            feature_files = glob.glob(os.path.join(participant_dir, "*_fused_features.csv"))

            if not feature_files:
                pbar.update(1)
                continue

            # Get label for this participant
            target = participant_to_target.get(participant, 0)  # Default to 0 if not found

            # Load features from all chunks for this participant
            for file_path in feature_files:
                try:
                    features_df = pd.read_csv(file_path)
                    if features_df.empty:
                        continue
                    
                    # Take first row and convert to numpy array
                    chunk_features = features_df.iloc[0].values
                    
                    # Check for NaN values
                    if np.isnan(chunk_features).any():
                        print(f"Warning: Found NaN values in {file_path}")
                    
                    features_list.append(chunk_features)
                    file_paths.append(file_path)
                    participant_ids.append(participant)
                    labels.append(target)
                    
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
                    continue

            pbar.update(1)

    if not features_list:
        print(f"No valid features found for HOSF fusion {chunk_length}")
        return np.array([]), [], [], []

    # Convert to numpy array
    features_array = np.array(features_list)
    
    print(f"Loaded {len(features_array)} feature vectors with {features_array.shape[1]} dimensions")
    
    # Check for NaN values in the entire array
    nan_count = np.isnan(features_array).sum()
    if nan_count > 0:
        print(f"Warning: Found {nan_count} NaN values in the feature array")
        print("These will be imputed with median values during UMAP transformation")

    return features_array, file_paths, participant_ids, labels

def handle_nan_values(features_array):
    """
    Handle NaN values in the feature array using robust imputation and outlier handling.

    Args:
        features_array (np.array): Input features that may contain NaN values

    Returns:
        np.array: Features with NaN values imputed and outliers handled
    """
    if not np.isnan(features_array).any():
        print("No NaN values found, returning original features")
        return features_array
    
    print("Handling NaN values with robust imputation...")
    
    # First, handle infinite values by replacing with NaN
    features_array = np.where(np.isinf(features_array), np.nan, features_array)
    
    # Count NaN values before imputation
    nan_count_before = np.isnan(features_array).sum()
    print(f"Found {nan_count_before} NaN/inf values to impute")
    
    # Use SimpleImputer with median strategy
    imputer = SimpleImputer(strategy='median')
    features_imputed = imputer.fit_transform(features_array)
    
    # Check for any remaining NaN values
    nan_count_after = np.isnan(features_imputed).sum()
    
    if nan_count_after > 0:
        print(f"Warning: {nan_count_after} NaN values remain after median imputation")
        # Fill remaining NaN values with 0
        features_imputed = np.nan_to_num(features_imputed, nan=0.0)
        print("Filled remaining NaN values with 0")
    
    # Handle extreme outliers by clipping to reasonable range
    # Calculate robust statistics (using percentiles instead of mean/std)
    q25 = np.percentile(features_imputed, 25, axis=0)
    q75 = np.percentile(features_imputed, 75, axis=0)
    iqr = q75 - q25
    
    # Define outlier bounds (more conservative than typical 1.5*IQR)
    lower_bound = q25 - 3 * iqr
    upper_bound = q75 + 3 * iqr
    
    # Clip outliers
    features_clipped = np.clip(features_imputed, lower_bound, upper_bound)
    
    # Check if clipping made any changes
    clipping_changes = np.sum(features_imputed != features_clipped)
    if clipping_changes > 0:
        print(f"Clipped {clipping_changes} extreme outlier values")
    
    print(f"Successfully imputed {nan_count_before} NaN/inf values")
    
    return features_clipped

def apply_umap_transformation(features, n_components=50, n_neighbors=15, metric='cosine', random_state=42):
    """
    Apply UMAP transformation to features with robust parameter selection.

    Args:
        features (np.array): Input features
        n_components (int): Number of output dimensions
        n_neighbors (int): Number of neighbors for UMAP
        metric (str): Distance metric
        random_state (int): Random seed

    Returns:
        np.array: UMAP-transformed features
    """
    if not UMAP_AVAILABLE:
        raise ImportError("UMAP is not available. Please install with: pip install umap-learn")

    print(f"Applying UMAP transformation: {features.shape[1]}D → {n_components}D")

    # Ensure we have enough samples for UMAP
    n_samples = features.shape[0]
    if n_samples < 10:
        print(f"Warning: Only {n_samples} samples available, UMAP may not work well")
        # Return zero-filled features if too few samples
        return np.zeros((n_samples, n_components))

    # Adjust n_neighbors based on sample size
    effective_n_neighbors = min(n_neighbors, n_samples - 1)
    if effective_n_neighbors < n_neighbors:
        print(f"Adjusted n_neighbors from {n_neighbors} to {effective_n_neighbors} due to sample size")

    # Initialize UMAP with robust parameters
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=effective_n_neighbors,
        metric=metric,
        min_dist=0.1,
        random_state=random_state,
        verbose=False,
        # Add robust parameters
        spread=1.0,
        local_connectivity=1.0,
        repulsion_strength=1.0,
        negative_sample_rate=5,
        transform_queue_size=4.0,
        angular_rp_forest=False,
        target_n_neighbors=-1,
        target_metric='categorical',
        target_weight=0.5,
        transform_seed=42
    )

    # Fit and transform
    start_time = time.time()
    try:
        umap_features = reducer.fit_transform(features)
        transform_time = time.time() - start_time
        print(f"UMAP transformation completed in {transform_time:.2f} seconds")
        
        # Check for any NaN values in the output
        if np.isnan(umap_features).any():
            print("Warning: UMAP output contains NaN values, filling with zeros")
            umap_features = np.nan_to_num(umap_features, nan=0.0)
        
        return umap_features
        
    except Exception as e:
        print(f"UMAP transformation failed: {str(e)}")
        print("Returning zero-filled features as fallback")
        return np.zeros((features.shape[0], n_components))

def save_umap_features(umap_features, file_paths, participant_ids, labels,
                      chunk_length, overlap, output_dir="data/features"):
    """
    Save UMAP-transformed features to new directory structure.

    Args:
        umap_features (np.array): UMAP-transformed features
        file_paths (list): Original file paths
        participant_ids (list): Participant IDs
        labels (list): Labels
        chunk_length (str): Chunk length
        overlap (str): Overlap length
        output_dir (str): Output directory
    """
    # Create output directory
    output_base = f"{output_dir}/umap_hosf_fusion"
    output_subdir = f"{output_base}/{chunk_length}_{overlap}_overlap"
    os.makedirs(output_subdir, exist_ok=True)

    print(f"Saving UMAP features to: {output_subdir}")

    # Group by participant for directory structure
    participant_groups = {}
    for i, participant in enumerate(participant_ids):
        if participant not in participant_groups:
            participant_groups[participant] = []
        participant_groups[participant].append(i)

    # Save features for each participant
    with tqdm(total=len(participant_groups), desc="Saving UMAP features", unit="participant") as pbar:
        for participant, indices in participant_groups.items():
            # Create participant directory
            participant_dir = os.path.join(output_subdir, participant)
            os.makedirs(participant_dir, exist_ok=True)

            # Save each chunk's UMAP features
            for idx in indices:
                # Get original filename
                original_path = file_paths[idx]
                filename = os.path.basename(original_path)
                base_name = filename.replace('_fused_features.csv', '')

                # Create UMAP feature dataframe
                feature_names = [f'umap_hosf_fusion_{i}' for i in range(umap_features.shape[1])]
                feature_df = pd.DataFrame([umap_features[idx]], columns=feature_names)

                # Add metadata
                feature_df['original_participant'] = participant_ids[idx]
                feature_df['original_label'] = labels[idx]
                feature_df['original_file'] = filename

                # Save as CSV
                output_file = os.path.join(participant_dir, f"{base_name}_umap_features.csv")
                feature_df.to_csv(output_file, index=False)

            pbar.set_postfix({'participant': participant, 'chunks': len(indices)})
            pbar.update(1)

    print(f"Saved {len(umap_features)} UMAP feature files for {len(participant_groups)} participants")

def process_chunk_length(chunk_length, data_root="data"):
    """
    Process a single chunk length for HOSF fusion UMAP transformation.

    Args:
        chunk_length (str): Chunk length to process
        data_root (str): Root directory for data
    """
    print(f"\n--- Processing HOSF fusion {chunk_length} chunks ---")

    try:
        # Get overlap
        overlap = get_overlap_for_chunk_length(chunk_length)

        # Load HOSF fusion features
        print(f"Loading HOSF fusion features for {chunk_length}...")
        features, file_paths, participant_ids, labels = load_hosf_fusion_features_for_umap(
            chunk_length, overlap, data_root
        )

        if len(features) == 0:
            print(f"No features found for HOSF fusion {chunk_length}, skipping...")
            return

        # Handle NaN values
        print("Handling NaN values...")
        features_clean = handle_nan_values(features)

        # Check if we still have valid features after cleaning
        if len(features_clean) == 0 or features_clean.shape[1] == 0:
            print(f"No valid features remaining after cleaning for {chunk_length}, skipping...")
            return

        # Apply UMAP transformation
        print("Applying UMAP transformation...")
        n_components = min(50, features_clean.shape[1] // 2)  # Adaptive component count
        n_neighbors = min(15, max(5, len(features_clean) // 10))  # More conservative neighbor selection
        
        print(f"UMAP parameters: n_components={n_components}, n_neighbors={n_neighbors}")
        
        umap_features = apply_umap_transformation(
            features_clean,
            n_components=n_components,
            n_neighbors=n_neighbors,
            metric='cosine',
            random_state=42
        )

        # Verify UMAP output
        if umap_features is None or len(umap_features) == 0:
            print(f"UMAP transformation failed for {chunk_length}, skipping...")
            return

        # Save UMAP features
        print("Saving UMAP features...")
        save_umap_features(
            umap_features, file_paths, participant_ids, labels,
            chunk_length, overlap, data_root + "/features"
        )

        print(f"✅ Completed HOSF fusion {chunk_length}")

    except Exception as e:
        print(f"❌ Error processing HOSF fusion {chunk_length}: {str(e)}")
        print("Continuing with next chunk length...")
        # Don't raise the exception, just continue with the next chunk length

def main():
    """
    Main function to generate UMAP features for HOSF fusion features.
    """
    if not UMAP_AVAILABLE:
        print("❌ UMAP is not available. Please install with: pip install umap-learn")
        return

    print("🎯 UMAP Feature Transformer for HOSF Fusion Features")
    print("=" * 60)

    # Get available chunk lengths
    try:
        chunk_lengths = get_available_chunk_lengths()
        print(f"Found {len(chunk_lengths)} available chunk lengths: {chunk_lengths}")
    except Exception as e:
        print(f"❌ Error getting chunk lengths: {str(e)}")
        return

    # UMAP configuration
    print("\n🔧 UMAP Configuration:")
    print("  - Metric: cosine (optimal for embeddings)")
    print("  - n_components: adaptive (50 or half of input dimensions)")
    print("  - n_neighbors: adaptive (15 or half of sample count)")
    print("  - min_dist: 0.1")
    print("  - random_state: 42")
    print("  - NaN handling: median imputation")

    start_time = time.time()

    # Process each chunk length
    for chunk_length in chunk_lengths:
        try:
            process_chunk_length(chunk_length)
        except Exception as e:
            print(f"❌ Error processing {chunk_length}: {str(e)}")
            continue

    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print("🎉 UMAP Feature Transformation Completed!")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"UMAP-transformed HOSF fusion features saved to:")
    print(f"   - data/features/umap_hosf_fusion/")
    print(f"\nEach CSV contains 50 UMAP-transformed features for one audio chunk.")

if __name__ == "__main__":
    main()
