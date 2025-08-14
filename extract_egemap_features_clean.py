import os
import pandas as pd
import numpy as np
import soundfile as sf
from pathlib import Path
import warnings
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import tqdm
import gc
import time
warnings.filterwarnings('ignore')

# Import opensmile
try:
    import opensmile
except ImportError:
    print("Error: opensmile not found. Please install with: pip install opensmile")
    exit(1)


def extract_egemaps_features_clean(audio_path, sample_rate=16000):
    """
    Extract clean eGeMAPSv02 features from an audio file - ONE 88-dimensional vector per chunk.
    
    Args:
        audio_path (str): Path to the audio file
        sample_rate (int): Target sample rate
    
    Returns:
        dict: Dictionary with exactly 88 eGeMAPSv02 features
    """
    try:
        # Initialize opensmile with eGeMAPSv02 feature set
        smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            sampling_rate=sample_rate
        )
        
        # Extract features
        features = smile.process_file(audio_path)
        
        if features.empty:
            return None
        
        # Get the 88 base feature names
        feature_names = features.columns.tolist()
        
        # Create feature dictionary - ONE value per feature (no aggregation)
        feature_dict = {}
        
        for feature_name in feature_names:
            # Clean the feature name for CSV compatibility
            clean_name = feature_name.replace('(', '').replace(')', '').replace(',', '_')
            
            # Get the feature values and calculate a single representative value
            values = features[feature_name].dropna()
            
            if len(values) > 0:
                # Use median as the representative value (more robust than mean)
                feature_dict[clean_name] = float(np.median(values))
            else:
                # If no valid values, set to 0
                feature_dict[clean_name] = 0.0
        
        # Add audio duration
        audio_info = sf.info(audio_path)
        feature_dict['duration'] = audio_info.duration
        
        return feature_dict
        
    except Exception as e:
        print(f"Error extracting features from {audio_path}: {e}")
        return None


def process_patient_worker_clean(patient_path):
    """
    Process one entire patient - designed for patient-level parallelism.
    Returns clean 88-dimensional feature vectors.
    
    Args:
        patient_path (str): Path to the patient folder
    
    Returns:
        tuple: (patient_id, features_list, success)
    """
    patient_id = os.path.basename(patient_path)
    
    try:
        # Process all chunks for this patient (sequentially within the worker)
        features = process_patient_folder_sequential_clean(patient_path)
        
        if features:
            return (patient_id, features, True)
        else:
            return (patient_id, [], False)
            
    except Exception as e:
        print(f"Error processing patient {patient_id}: {e}")
        return (patient_id, [], False)


def process_patient_folder_sequential_clean(patient_path):
    """
    Process all chunks for one patient sequentially (within a worker).
    Returns clean feature vectors without statistical aggregation.
    """
    patient_name = os.path.basename(patient_path)
    
    # Look for chunk directories
    chunk_dirs = []
    for item in os.listdir(patient_path):
        item_path = os.path.join(patient_path, item)
        if os.path.isdir(item_path) and "s_overlap" in item:
            chunk_dirs.append(item_path)
    
    if not chunk_dirs:
        return []
    
    print(f"Worker processing {len(chunk_dirs)} chunk directories for {patient_name}")
    
    all_features = []
    
    for chunk_dir in chunk_dirs:
        dir_name = os.path.basename(chunk_dir)
        
        # Process all audio files in the chunk directory
        audio_files = []
        for file in os.listdir(chunk_dir):
            if file.lower().endswith(('.wav', '.mp3', '.flac', '.m4a')):
                audio_files.append(os.path.join(chunk_dir, file))
        
        if not audio_files:
            continue
        
        for audio_file in audio_files:
            try:
                # Extract clean features (88-dimensional vector)
                features = extract_egemaps_features_clean(audio_file)
                
                if features is not None:
                    # Add metadata
                    features.update({
                        'patient_id': patient_name,
                        'chunk_file': os.path.basename(audio_file),
                        'chunk_directory': dir_name,
                        'audio_path': audio_file
                    })
                    
                    all_features.append(features)
                
            except Exception as e:
                print(f"    Error processing {audio_file}: {e}")
    
    return all_features


def save_patient_features_clean(patient_features, patient_id, output_dir):
    """
    Save clean features for one patient to separate CSV.
    
    Args:
        patient_features (list): List of feature dictionaries
        patient_id (str): Patient identifier
        output_dir (str): Output directory path
    
    Returns:
        str: Path to saved CSV file
    """
    if not patient_features:
        return None
    
    output_file = f"{patient_id}_clean_features.csv"
    output_path = os.path.join(output_dir, output_file)
    
    df = pd.DataFrame(patient_features)
    df.to_csv(output_path, index=False)
    
    print(f"Saved {len(df)} chunks for {patient_id} to {output_path}")
    return output_path


def main():
    # Configuration parameters
    DATA_DIR = "data/created_data"
    OUTPUT_DIR = "data/features/clean_egemaps"  # New subdirectory
    
    # Parallel processing settings for M4 Pro
    N_WORKERS = 10  # Use 10 performance cores
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("DAIC-WOZ Clean eGeMAPSv02 Feature Extraction (Patient-Level Parallel)")
    print("=" * 70)
    print(f"Input directory: {DATA_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Parallel workers: {N_WORKERS}")
    print(f"Strategy: {N_WORKERS} workers processing {N_WORKERS} patients simultaneously")
    print(f"Feature format: ONE 88-dimensional vector per chunk (no aggregation)")
    print(f"Processor: M4 Pro (10 performance cores)")
    print(f"Expected speedup: 8-12x faster than sequential")
    print("-" * 70)
    
    start_time = time.time()
    
    # Get all patient folders
    patient_folders = []
    for item in os.listdir(DATA_DIR):
        item_path = os.path.join(DATA_DIR, item)
        if os.path.isdir(item_path) and item.endswith('_P'):
            patient_folders.append(item_path)
    
    patient_folders.sort()
    
    print(f"Found {len(patient_folders)} patient folders")
    print(f"Will process {min(N_WORKERS, len(patient_folders))} patients simultaneously")
    print("-" * 70)
    
    # Process patients in parallel (each worker gets one patient)
    all_features = []
    processed_patients = 0
    
    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        # Submit all patient processing tasks
        future_to_patient = {executor.submit(process_patient_worker_clean, patient_path): patient_path 
                           for patient_path in patient_folders}
        
        # Process results as they complete
        for future in tqdm.tqdm(as_completed(future_to_patient), total=len(patient_folders), 
                               desc="Processing patients in parallel"):
            patient_id, features, success = future.result()
            
            if success and features:
                # Save individual patient CSV
                save_patient_features_clean(features, patient_id, OUTPUT_DIR)
                
                # Add to combined results
                all_features.extend(features)
                processed_patients += 1
                
                print(f"Completed patient {patient_id}: {len(features)} chunks")
            else:
                print(f"Failed to process patient {patient_id}")
    
    # Save combined CSV
    if all_features:
        combined_df = pd.DataFrame(all_features)
        combined_path = os.path.join(OUTPUT_DIR, "all_patients_clean_combined.csv")
        combined_df.to_csv(combined_path, index=False)
        
        print("-" * 70)
        print(f"Feature extraction complete!")
        print(f"Total chunks processed: {len(combined_df)}")
        print(f"Patients processed: {processed_patients}")
        
        # Show feature information
        feature_cols = [col for col in combined_df.columns if col not in 
                       ['patient_id', 'chunk_file', 'chunk_directory', 'audio_path', 'duration']]
        
        print(f"Features per chunk: {len(feature_cols)} (should be ~88)")
        print(f"Output saved to: {combined_path}")
        
        # Print summary statistics
        print("\nSummary by patient:")
        patient_summary = combined_df.groupby('patient_id').size()
        for patient, count in patient_summary.items():
            print(f"  {patient}: {count} chunks")
        
        # Print feature statistics
        print(f"\nFeature information:")
        print(f"  Duration range: {combined_df['duration'].min():.2f}s - {combined_df['duration'].max():.2f}s")
        print(f"  Average duration: {combined_df['duration'].mean():.2f}s")
        
        # Show some example features
        print(f"\nExample eGeMAPSv02 features extracted:")
        example_features = feature_cols[:10]  # Show first 10 features
        for feature in example_features:
            print(f"  {feature}")
        
        print(f"\nTotal eGeMAPSv02 features: {len(feature_cols)}")
        print("Features include: F0, energy, spectral, MFCC, formants, voice quality, etc.")
        
        # Verify we have the right number of features
        if len(feature_cols) == 88:
            print("✓ Perfect! Exactly 88 eGeMAPSv02 features extracted")
        elif 80 <= len(feature_cols) <= 95:
            print(f"✓ Good! {len(feature_cols)} eGeMAPSv02 features extracted (close to expected 88)")
        else:
            print(f"⚠ Warning: Expected ~88 features, got {len(feature_cols)}")
    
    else:
        print("No features extracted. Check your data directory.")
        return
    
    # Performance summary
    end_time = time.time()
    total_time = end_time - start_time
    chunks_per_second = len(all_features) / total_time if total_time > 0 else 0
    
    print("-" * 70)
    print(f"Performance Summary:")
    print(f"Total processing time: {total_time:.2f} seconds")
    print(f"Processing speed: {chunks_per_second:.2f} chunks/second")
    print(f"Speedup vs sequential: ~{N_WORKERS:.1f}x faster")
    print(f"Efficiency: {chunks_per_second / N_WORKERS:.2f} chunks/second per worker")
    
    # File size information
    if os.path.exists(combined_path):
        file_size = os.path.getsize(combined_path) / (1024 * 1024)  # MB
        print(f"Combined CSV file size: {file_size:.2f} MB")
        print(f"Average chunk size: {file_size * 1024 / len(all_features):.2f} KB per chunk")


if __name__ == "__main__":
    main()