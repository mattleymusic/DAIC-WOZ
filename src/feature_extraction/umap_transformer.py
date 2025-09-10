#!/usr/bin/env python3
"""
UMAP Feature Transformer for DAIC-WOZ Dataset

This script applies UMAP dimensionality reduction to already extracted features.
It creates new directories with UMAP-transformed features for all available chunk lengths.

Features:
- Applies UMAP to eGeMAPS, EmotionMSP, and HuBERT features
- Creates new directories: data/features/umap_egemaps, data/features/umap_emotionmsp, data/features/umap_hubert
- Processes all available chunk lengths automatically
- Saves UMAP-transformed features as CSV files
- Maintains original data structure and labels

Usage:
    python src/feature_extraction/umap_transformer.py

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

def load_features_for_umap(feature_type, chunk_length, overlap, data_root="data"):
    """
    Load features for UMAP transformation.

    Args:
        feature_type (str): Type of features ('egemaps', 'emotionmsp', 'hubert')
        chunk_length (str): Chunk length (e.g., "5.0s")
        overlap (str): Overlap length (e.g., "2.5s")
        data_root (str): Root directory for data

    Returns:
        tuple: (features_array, file_paths, participant_ids, labels)
    """
    # Set feature directory based on type
    if feature_type == 'egemaps':
        feature_dir = f"{data_root}/features/egemap/{chunk_length}_{overlap}_overlap"
    elif feature_type == 'emotionmsp':
        feature_dir = f"{data_root}/features/emotionmsp/{chunk_length}_{overlap}_overlap"
    elif feature_type == 'hubert':
        feature_dir = f"{data_root}/features/hubert/{chunk_length}_{overlap}_overlap"
    else:
        raise ValueError(f"Unknown feature type: {feature_type}")

    if not os.path.exists(feature_dir):
        raise ValueError(f"Feature directory not found: {feature_dir}")

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

    print(f"Processing {len(available_participants)} participants for {feature_type} {chunk_length}")

    features_list = []
    file_paths = []
    participant_ids = []
    labels = []

    # Process each participant
    with tqdm(total=len(available_participants), desc="Loading features", unit="participant") as pbar:
        for participant in available_participants:
            participant_dir = os.path.join(feature_dir, participant)
            feature_files = glob.glob(os.path.join(participant_dir, "*_features.csv"))

            if not feature_files:
                pbar.update(1)
                continue

            # Get label for this participant
            target = participant_to_target.get(participant, 0)  # Default to 0 if not found

            # Load features from all chunks for this participant
            for file_path in feature_files:
                features_df = pd.read_csv(file_path)
                chunk_features = features_df.iloc[0].values  # Take first row

                features_list.append(chunk_features)
                file_paths.append(file_path)
                participant_ids.append(participant)
                labels.append(target)

            pbar.update(1)

    # Convert to numpy array
    features_array = np.array(features_list)

    print(f"Loaded {len(features_array)} feature vectors with {features_array.shape[1]} dimensions")

    return features_array, file_paths, participant_ids, labels

def apply_umap_transformation(features, n_components=50, n_neighbors=15, metric='cosine', random_state=42):
    """
    Apply UMAP transformation to features.

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

    # Initialize UMAP
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        metric=metric,
        min_dist=0.1,
        random_state=random_state,
        verbose=False
    )

    # Fit and transform
    start_time = time.time()
    umap_features = reducer.fit_transform(features)
    transform_time = time.time() - start_time

    return umap_features

def save_umap_features(umap_features, file_paths, participant_ids, labels,
                      feature_type, chunk_length, overlap, output_dir="data/features"):
    """
    Save UMAP-transformed features to new directory structure.

    Args:
        umap_features (np.array): UMAP-transformed features
        file_paths (list): Original file paths
        participant_ids (list): Participant IDs
        labels (list): Labels
        feature_type (str): Type of features ('egemaps', 'emotionmsp', 'hubert')
        chunk_length (str): Chunk length
        overlap (str): Overlap length
        output_dir (str): Output directory
    """
    # Create output directory
    output_base = f"{output_dir}/umap_{feature_type}"
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
    with tqdm(total=len(participant_groups), desc="Saving features", unit="participant") as pbar:
        for participant, indices in participant_groups.items():
            # Create participant directory
            participant_dir = os.path.join(output_subdir, participant)
            os.makedirs(participant_dir, exist_ok=True)

            # Save each chunk's UMAP features
            for idx in indices:
                # Get original filename
                original_path = file_paths[idx]
                filename = os.path.basename(original_path)
                base_name = filename.replace('_features.csv', '')

                # Create UMAP feature dataframe
                feature_names = [f'umap_{i}' for i in range(umap_features.shape[1])]
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

def process_feature_type(feature_type, chunk_lengths, data_root="data"):
    """
    Process all chunk lengths for a given feature type.

    Args:
        feature_type (str): Type of features ('egemaps', 'emotionmsp', 'hubert')
        chunk_lengths (list): List of chunk lengths to process
        data_root (str): Root directory for data
    """
    print(f"\n{'='*60}")
    print(f"Processing {feature_type.upper()} features")
    print(f"{'='*60}")

    for chunk_length in chunk_lengths:
        print(f"\n--- Processing {chunk_length} chunks ---")

        try:
            # Get overlap
            overlap = get_overlap_for_chunk_length(chunk_length)

            # Load features
            print(f"Loading {feature_type} features for {chunk_length}...")
            features, file_paths, participant_ids, labels = load_features_for_umap(
                feature_type, chunk_length, overlap, data_root
            )

            if len(features) == 0:
                print(f"No features found for {feature_type} {chunk_length}, skipping...")
                continue

            # Apply UMAP transformation
            print("Applying UMAP transformation...")
            n_components = min(50, features.shape[1] // 2)  # Adaptive component count
            umap_features = apply_umap_transformation(
                features,
                n_components=n_components,
                n_neighbors=min(15, len(features) // 2),  # Adaptive neighbors
                metric='cosine',
                random_state=42
            )

            # Save UMAP features
            print("Saving UMAP features...")
            save_umap_features(
                umap_features, file_paths, participant_ids, labels,
                feature_type, chunk_length, overlap, data_root + "/features"
            )

            print(f"✅ Completed {feature_type} {chunk_length}")

        except Exception as e:
            print(f"❌ Error processing {feature_type} {chunk_length}: {str(e)}")
            continue

def main():
    """
    Main function to generate UMAP features for all feature types and chunk lengths.
    """
    if not UMAP_AVAILABLE:
        print("❌ UMAP is not available. Please install with: pip install umap-learn")
        return

    print("🎯 UMAP Feature Transformer for DAIC-WOZ Dataset")
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

    # Process each feature type
    feature_types = ['egemaps', 'emotionmsp', 'hubert']

    start_time = time.time()

    for feature_type in feature_types:
        try:
            process_feature_type(feature_type, chunk_lengths)
        except Exception as e:
            print(f"❌ Error processing {feature_type}: {str(e)}")
            continue

    total_time = time.time() - start_time
    print(f"Total time: {total_time:.2f} seconds")
    print("🎉 UMAP Feature Transformation Completed!")
    for feature_type in feature_types:
        print(f"   - data/features/umap_{feature_type}/")

if __name__ == "__main__":
    
    main()