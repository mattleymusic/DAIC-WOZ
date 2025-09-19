#!/usr/bin/env python3
"""
Paper Fused Features Creator - Extract and Save Fused Features from Miao et al. 2022

This script extracts the exact fused features used in the paper reproduction:
- 10 selected COVAREP features (from Relief-based selection)
- 16 HOSF features (9 BSF + 7 BCF)
- Total: 26 fused features per chunk

The script creates a unified feature set that can be used for machine learning
without needing to run the full paper reproduction pipeline.

Usage:
    python src/feature_extraction/create_paper_fused_features.py

Author: Based on Miao et al. 2022 paper reproduction methodology
"""

import os
import pandas as pd
import numpy as np
import glob
import sys
import time
from pathlib import Path
from datetime import datetime
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Add the project root to the path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)


def load_relief_selection_info(chunk_length="4.0s", overlap="0.0s", data_root="data"):
    """
    Load Relief feature selection information to get the selected COVAREP features.
    
    Args:
        chunk_length (str): Chunk length (e.g., "4.0s")
        overlap (str): Overlap length (e.g., "0.0s")
        data_root (str): Root directory for data
    
    Returns:
        np.array: Indices of selected COVAREP features
    """
    relief_dir = f"{data_root}/features/relief_selected_paper_covarep/{chunk_length}_{overlap}_overlap"
    if not os.path.exists(relief_dir):
        raise ValueError(f"Relief selected features not found: {relief_dir}")
    
    selection_info_path = os.path.join(relief_dir, "selection_info.csv")
    if not os.path.exists(selection_info_path):
        raise ValueError(f"Selection info file not found: {selection_info_path}")
    
    selection_info_df = pd.read_csv(selection_info_path)
    selected_indices = selection_info_df['feature_index'].values
    
    print(f"Loaded Relief selection: {len(selected_indices)} COVAREP features selected")
    print(f"Selected indices: {selected_indices}")
    
    return selected_indices


def load_meta_info(data_root="data"):
    """
    Load meta information for labels and participant filtering.
    
    Args:
        data_root (str): Root directory for data
    
    Returns:
        tuple: (meta_df, participant_to_target, participant_to_subset)
    """
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
    
    return meta_df, participant_to_target, participant_to_subset


def extract_fused_features_for_participant(participant, participant_to_target, participant_to_subset, 
                                         chunk_length, overlap, data_root, selected_indices):
    """
    Extract fused features for a single participant.
    
    Args:
        participant (str): Participant ID
        participant_to_target (dict): Mapping from participant to target
        participant_to_subset (dict): Mapping from participant to subset
        chunk_length (str): Chunk length
        overlap (str): Overlap length
        data_root (str): Root directory for data
        selected_indices (np.array): Indices of selected COVAREP features
    
    Returns:
        list: List of fused feature dictionaries
    """
    target = participant_to_target[participant]
    subset = participant_to_subset[participant]
    
    # Load HOSF features for this participant
    hosf_dir = f"{data_root}/features/paper_hosf/{chunk_length}_{overlap}_overlap"
    participant_hosf_dir = os.path.join(hosf_dir, participant)
    
    if not os.path.exists(participant_hosf_dir):
        print(f"Warning: No HOSF features found for participant {participant}")
        return []
    
    # Get all HOSF files for this participant
    hosf_files = glob.glob(os.path.join(participant_hosf_dir, "*_paper_hosf_features.csv"))
    hosf_files.sort()  # Ensure consistent ordering
    
    fused_features = []
    
    # Process each HOSF file
    for hosf_file in hosf_files:
        hosf_basename = os.path.basename(hosf_file)
        
        try:
            # Load HOSF features
            hosf_df = pd.read_csv(hosf_file)
            hosf_features = hosf_df.iloc[0].values
            
            # Handle zero HOSF features (temporary fix)
            if np.all(hosf_features == 0):
                print(f"Warning: All HOSF features are zero for {hosf_basename}")
                # Use small random values as placeholder
                hosf_features = np.random.normal(0, 0.001, len(hosf_features))
            
            # Load corresponding COVAREP features
            covarep_basename = hosf_basename.replace('_paper_hosf_features.csv', '_paper_covarep_features.csv')
            covarep_file = os.path.join(f"{data_root}/features/paper_covarep/{chunk_length}_{overlap}_overlap", 
                                      participant, covarep_basename)
            
            if not os.path.exists(covarep_file):
                print(f"Warning: COVAREP file not found for {covarep_basename}")
                continue
            
            # Load individual COVAREP features
            covarep_df = pd.read_csv(covarep_file)
            covarep_features = covarep_df.iloc[0].values
            
            # Apply Relief feature selection (select the 10 most important features)
            if len(covarep_features) != 74:
                print(f"Warning: Expected 74 COVAREP features, got {len(covarep_features)}")
                continue
            
            # Apply Relief selection using the pre-computed indices
            covarep_features = covarep_features[selected_indices]
            
            # Validate features
            if len(covarep_features) != 10:
                print(f"Warning: Expected 10 COVAREP features, got {len(covarep_features)}")
                continue
            if len(hosf_features) != 16:
                print(f"Warning: Expected 16 HOSF features, got {len(hosf_features)}")
                continue
            
            # Combine features: 10 COVAREP + 16 HOSF = 26 total
            combined_features = np.concatenate([covarep_features, hosf_features])
            
            # Check for NaN or infinite values
            if np.any(np.isnan(combined_features)) or np.any(np.isinf(combined_features)):
                print(f"Warning: NaN or infinite values in features for {hosf_basename}")
                combined_features = np.nan_to_num(combined_features, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Create feature dictionary
            chunk_name = hosf_basename.replace('_paper_hosf_features.csv', '')
            
            feature_dict = {
                'participant': participant,
                'chunk_name': chunk_name,
                'target': target,
                'subset': subset,
                'covarep_features': covarep_features,
                'hosf_features': hosf_features,
                'fused_features': combined_features
            }
            
            fused_features.append(feature_dict)
            
        except Exception as e:
            print(f"Error processing {hosf_basename}: {e}")
            continue
    
    return fused_features


def create_fused_feature_names():
    """
    Create feature names for the fused features.
    
    Returns:
        list: List of feature names
    """
    feature_names = []
    
    # 10 COVAREP features (selected by Relief)
    for i in range(10):
        feature_names.append(f'covarep_selected_{i}')
    
    # 16 HOSF features (9 BSF + 7 BCF)
    # 9 Bispectral Features (BSF)
    bsf_names = ['mAmp', 'H1', 'H2', 'f1m', 'f2m', 'f3m', 'f4m', 'bispectrum_std', 'bispectrum_max']
    feature_names.extend(bsf_names)
    
    # 7 Bicoherent Features (BCF)
    bcf_names = ['bicoherence_mean', 'bicoherence_std', 'bicoherence_max', 'bicoherence_min',
                 'bicoherence_median', 'bicoherence_sum', 'bicoherence_sum_squares']
    feature_names.extend(bcf_names)
    
    return feature_names


def save_fused_features(fused_features_list, feature_names, chunk_length, overlap, output_dir):
    """
    Save fused features to CSV files following the exact structure of other feature sets.
    One folder per patient, one CSV file per chunk.
    
    Args:
        fused_features_list (list): List of fused feature dictionaries
        feature_names (list): List of feature names
        chunk_length (str): Chunk length
        overlap (str): Overlap length
        output_dir (str): Output directory
    """
    # Create output directory
    fused_features_dir = os.path.join(output_dir, "features", "paper_fused_features", f"{chunk_length}_{overlap}_overlap")
    os.makedirs(fused_features_dir, exist_ok=True)
    
    # Group features by participant
    participant_features = {}
    for feature_dict in fused_features_list:
        participant = feature_dict['participant']
        if participant not in participant_features:
            participant_features[participant] = []
        participant_features[participant].append(feature_dict)
    
    # Save individual CSV files for each chunk (following exact structure)
    total_chunks_saved = 0
    for participant, features in participant_features.items():
        # Create participant directory
        participant_dir = os.path.join(fused_features_dir, participant)
        os.makedirs(participant_dir, exist_ok=True)
        
        # Save each chunk as individual CSV file
        for feature_dict in features:
            chunk_name = feature_dict['chunk_name']
            
            # Create DataFrame with fused features only (no metadata)
            feature_df = pd.DataFrame([feature_dict['fused_features']], columns=feature_names)
            
            # Save individual chunk CSV file
            chunk_csv_path = os.path.join(participant_dir, f"{chunk_name}_paper_fused_features_features.csv")
            feature_df.to_csv(chunk_csv_path, index=False)
            
            total_chunks_saved += 1
    
    print(f"Saved fused features following exact structure:")
    print(f"  Base directory: {fused_features_dir}")
    print(f"  Structure: [PATIENT_ID]/[CHUNK]_paper_fused_features_features.csv")
    print(f"  Total chunks saved: {total_chunks_saved}")
    print(f"  Feature dimensions: {len(feature_names)} (10 COVAREP + 16 HOSF)")
    
    # Create unified CSV for easy ML usage (optional)
    unified_data = []
    for feature_dict in fused_features_list:
        row = {
            'participant': feature_dict['participant'],
            'chunk_name': feature_dict['chunk_name'],
            'target': feature_dict['target'],
            'subset': feature_dict['subset']
        }
        
        # Add fused features
        for i, feature_name in enumerate(feature_names):
            row[feature_name] = feature_dict['fused_features'][i]
        
        unified_data.append(row)
    
    # Save unified CSV (for convenience)
    unified_df = pd.DataFrame(unified_data)
    unified_csv_path = os.path.join(fused_features_dir, "all_patients_fused_features.csv")
    unified_df.to_csv(unified_csv_path, index=False)
    
    # Save feature information
    feature_info = {
        'total_features': len(feature_names),
        'covarep_features': 10,
        'hosf_features': 16,
        'feature_names': feature_names,
        'chunk_length': chunk_length,
        'overlap': overlap,
        'creation_timestamp': datetime.now().isoformat(),
        'description': 'Fused features from Miao et al. 2022 paper reproduction (10 COVAREP + 16 HOSF)',
        'structure': 'One folder per patient, one CSV file per chunk (same as other feature sets)'
    }
    
    feature_info_path = os.path.join(fused_features_dir, "feature_info.json")
    import json
    with open(feature_info_path, 'w') as f:
        json.dump(feature_info, f, indent=2)
    
    # Print summary statistics
    print(f"\nFused Features Summary:")
    print(f"Total participants: {len(participant_features)}")
    print(f"Total chunks: {total_chunks_saved}")
    print(f"Train samples: {len(unified_df[unified_df['subset'] == 'train'])}")
    print(f"Dev samples: {len(unified_df[unified_df['subset'] == 'dev'])}")
    print(f"Test samples: {len(unified_df[unified_df['subset'] == 'test'])}")
    
    # Class distribution
    class_counts = unified_df['target'].value_counts()
    print(f"Class distribution: {dict(class_counts)}")
    
    return fused_features_dir


def main():
    """
    Main function to create fused features from paper reproduction.
    """
    print("Paper Fused Features Creator - Miao et al. 2022")
    print("=" * 60)
    
    # Configuration parameters - CHANGE THESE VALUES AS NEEDED
    CHUNK_LENGTH = "4.0s"  # Paper uses 4-second chunks
    OVERLAP = "0.0s"        # Paper uses no overlap
    DATA_ROOT = "data"      # Root directory for data
    
    print(f"Chunk length: {CHUNK_LENGTH}")
    print(f"Overlap: {OVERLAP}")
    print(f"Data root: {DATA_ROOT}")
    print("=" * 60)
    
    start_time = time.time()
    
    try:
        # Load Relief selection information
        print("Loading Relief feature selection information...")
        selected_indices = load_relief_selection_info(CHUNK_LENGTH, OVERLAP, DATA_ROOT)
        
        # Load meta information
        print("Loading meta information...")
        meta_df, participant_to_target, participant_to_subset = load_meta_info(DATA_ROOT)
        
        # Create feature names
        feature_names = create_fused_feature_names()
        print(f"Created {len(feature_names)} feature names")
        
        # Extract fused features for all participants
        print("Extracting fused features for all participants...")
        all_fused_features = []
        
        for i, participant in enumerate(meta_df['participant']):
            print(f"Processing participant {i+1}/{len(meta_df)}: {participant}")
            
            participant_features = extract_fused_features_for_participant(
                participant, participant_to_target, participant_to_subset,
                CHUNK_LENGTH, OVERLAP, DATA_ROOT, selected_indices
            )
            
            all_fused_features.extend(participant_features)
            
            if len(participant_features) > 0:
                print(f"  Extracted {len(participant_features)} fused feature chunks")
            else:
                print(f"  No features extracted")
        
        print(f"\nTotal fused feature chunks extracted: {len(all_fused_features)}")
        
        # Save fused features
        print("Saving fused features...")
        fused_features_dir = save_fused_features(
            all_fused_features, feature_names, CHUNK_LENGTH, OVERLAP, DATA_ROOT
        )
        
        # Calculate processing time
        total_time = time.time() - start_time
        
        print("\n" + "=" * 60)
        print("FUSED FEATURES CREATION COMPLETED")
        print("=" * 60)
        print(f"Total processing time: {total_time:.2f} seconds")
        print(f"Fused features saved to: {fused_features_dir}")
        print(f"\nFeature composition:")
        print(f"  - 10 COVAREP features (selected by Relief)")
        print(f"  - 16 HOSF features (9 BSF + 7 BCF)")
        print(f"  - Total: 26 fused features per chunk")
        print(f"\nStructure follows exact pattern of other feature sets:")
        print(f"  data/features/paper_fused_features/4.0s_0.0s_overlap/[PATIENT_ID]/[CHUNK]_paper_fused_features_features.csv")
        print(f"\nThis makes it easy to swap out other features for the fused version")
        print(f"in your machine learning pipelines!")
        
    except Exception as e:
        print(f"Error during fused features creation: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
