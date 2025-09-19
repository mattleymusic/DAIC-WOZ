#!/usr/bin/env python3
"""
Create Unified Features - General Feature Unification Script

This script creates unified feature files for any feature type by combining
individual feature files into a single CSV per chunk. This eliminates the need
to repeatedly load and combine features during machine learning training.

Features:
- Works with any feature type (egemaps, hubert, hosf_fusion, etc.)
- Creates unified CSV files for each chunk
- Maintains the same directory structure as individual feature files
- Handles missing files gracefully
- Configurable parameters in __main__ section

Usage:
    python src/feature_extraction/create_unified_features.py

Author: General feature unification for efficient ML pipeline
"""

import os
import pandas as pd
import numpy as np
import glob
from pathlib import Path
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def create_unified_features(feature_type="egemaps", chunk_length="5.0s", overlap="2.5s", data_root="data"):
    """
    Create unified feature files for any feature type.
    
    Args:
        feature_type (str): Type of features to unify (egemaps, hubert, hosf_fusion, etc.)
        chunk_length (str): Chunk length (e.g., "5.0s")
        overlap (str): Overlap length (e.g., "2.5s")
        data_root (str): Root directory for data
    """
    print("Creating Unified Features")
    print("=" * 50)
    print(f"Feature type: {feature_type}")
    print(f"Chunk length: {chunk_length}")
    print(f"Overlap: {overlap}")
    print(f"Data root: {data_root}")
    print("=" * 50)
    
    # Load meta information for participants
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
    print(f"Processing {len(meta_df)} participants")
    
    # Define directories
    input_dir = f"{data_root}/features/{feature_type}/{chunk_length}_{overlap}_overlap"
    output_dir = f"{data_root}/features/unified_{feature_type}/{chunk_length}_{overlap}_overlap"
    
    if not os.path.exists(input_dir):
        raise ValueError(f"Input feature directory not found: {input_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Statistics
    total_chunks = 0
    successful_chunks = 0
    failed_chunks = 0
    participants_processed = 0
    
    # Process each participant
    for idx, row in meta_df.iterrows():
        participant = row['participant']
        print(f"\nProcessing participant {participant} ({idx+1}/{len(meta_df)})")
        
        # Create participant output directory
        participant_output_dir = os.path.join(output_dir, participant)
        os.makedirs(participant_output_dir, exist_ok=True)
        
        # Get feature files for this participant
        participant_input_dir = os.path.join(input_dir, participant)
        if not os.path.exists(participant_input_dir):
            print(f"  Warning: No features found for participant {participant}")
            continue
        
        # Get all feature files for this participant
        # Try different naming patterns
        feature_files = glob.glob(os.path.join(participant_input_dir, f"*_{feature_type}_features.csv"))
        if not feature_files:
            # Try alternative pattern: participant_chunkXXX_features.csv
            feature_files = glob.glob(os.path.join(participant_input_dir, f"{participant}_chunk*_features.csv"))
        if not feature_files:
            # Try another pattern: chunk_XXX_features.csv
            feature_files = glob.glob(os.path.join(participant_input_dir, "chunk_*_features.csv"))
        feature_files.sort()  # Ensure consistent ordering
        
        participant_chunks = 0
        participant_successful = 0
        
        # Process each feature file
        for feature_file in feature_files:
            feature_basename = os.path.basename(feature_file)
            total_chunks += 1
            participant_chunks += 1
            
            try:
                # Load features
                feature_df = pd.read_csv(feature_file)
                features = feature_df.iloc[0].values
                
                # Check for NaN or infinite values
                if np.any(np.isnan(features)) or np.any(np.isinf(features)):
                    print(f"    Warning: NaN or infinite values in features for {feature_basename}")
                    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
                
                # Create feature names
                feature_names = [f"{feature_type}_{i}" for i in range(len(features))]
                
                # Create unified feature DataFrame
                unified_df = pd.DataFrame([features], columns=feature_names)
                
                # Save unified features
                output_basename = feature_basename.replace(f'_{feature_type}_features.csv', f'_unified_{feature_type}_features.csv')
                output_file = os.path.join(participant_output_dir, output_basename)
                unified_df.to_csv(output_file, index=False)
                
                successful_chunks += 1
                participant_successful += 1
                
            except Exception as e:
                print(f"    Error processing {feature_basename}: {str(e)}")
                failed_chunks += 1
        
        print(f"  Participant {participant}: {participant_successful}/{participant_chunks} chunks processed successfully")
        if participant_successful > 0:
            participants_processed += 1
    
    # Print final statistics
    print("\n" + "=" * 50)
    print("UNIFIED FEATURE CREATION COMPLETED")
    print("=" * 50)
    print(f"Total participants processed: {participants_processed}/{len(meta_df)}")
    print(f"Total chunks processed: {total_chunks}")
    print(f"Successful chunks: {successful_chunks}")
    print(f"Failed chunks: {failed_chunks}")
    if total_chunks > 0:
        print(f"Success rate: {successful_chunks/total_chunks*100:.1f}%")
    else:
        print("Success rate: N/A (no chunks processed)")
    print(f"\nUnified features saved to: {output_dir}")
    
    # Create a summary file with feature information
    summary_file = os.path.join(output_dir, "feature_summary.txt")
    with open(summary_file, 'w') as f:
        f.write(f"Unified {feature_type.upper()} Features Summary\n")
        f.write("=" * 40 + "\n")
        f.write(f"Feature type: {feature_type}\n")
        f.write(f"Chunk length: {chunk_length}\n")
        f.write(f"Overlap: {overlap}\n")
        f.write(f"Total features: {len(feature_names) if 'feature_names' in locals() else 'Unknown'}\n")
        f.write(f"\nStatistics:\n")
        f.write(f"Participants processed: {participants_processed}/{len(meta_df)}\n")
        f.write(f"Total chunks: {total_chunks}\n")
        f.write(f"Successful: {successful_chunks}\n")
        f.write(f"Failed: {failed_chunks}\n")
        if total_chunks > 0:
            f.write(f"Success rate: {successful_chunks/total_chunks*100:.1f}%\n")
        else:
            f.write("Success rate: N/A (no chunks processed)\n")
    
    print(f"Feature summary saved to: {summary_file}")


def main():
    """
    Main function to create unified features.
    """
    # Configuration parameters - CHANGE THESE VALUES AS NEEDED
    FEATURE_TYPE = "egemap"  # Options: egemap, hubert, hosf_fusion, emotionmsp, etc.
    CHUNK_LENGTH = "5.0s"     # Chunk length
    OVERLAP = "2.5s"          # Overlap length
    DATA_ROOT = "data"        # Root directory for data
    
    try:
        create_unified_features(FEATURE_TYPE, CHUNK_LENGTH, OVERLAP, DATA_ROOT)
        print("\n✅ Unified feature creation completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during unified feature creation: {str(e)}")
        raise


if __name__ == "__main__":
    main()
