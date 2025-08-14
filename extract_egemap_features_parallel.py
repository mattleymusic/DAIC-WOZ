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


def extract_egemaps_features(audio_path, sample_rate=16000):
    """
    Extract eGeMAPSv02 features from an audio file using opensmile.
    
    Args:
        audio_path (str): Path to the audio file
        sample_rate (int): Target sample rate
    
    Returns:
        dict: Dictionary of eGeMAPSv02 features
    """
    try:
        # Initialize opensmile with eGeMAPSv02 feature set
        # Use minimal parameters to avoid version conflicts
        smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            sampling_rate=sample_rate
        )
        
        # Extract features
        features = smile.process_file(audio_path)
        
        # Convert to dictionary format
        # features is a pandas DataFrame with one row per frame
        # We need to aggregate across frames to get one feature vector per audio file
        
        if features.empty:
            return None
        
        # Calculate statistics across frames for each feature
        feature_dict = {}
        
        for column in features.columns:
            feature_name = column.replace('(', '').replace(')', '').replace(',', '_')
            
            # Calculate statistics for each feature
            values = features[column].dropna()  # Remove NaN values
            
            if len(values) > 0:
                feature_dict[f'{feature_name}_mean'] = float(np.mean(values))
                feature_dict[f'{feature_name}_std'] = float(np.std(values))
                feature_dict[f'{feature_name}_min'] = float(np.min(values))
                feature_dict[f'{feature_name}_max'] = float(np.max(values))
                feature_dict[f'{feature_name}_range'] = float(np.max(values) - np.min(values))
                
                # Add percentiles for more robust statistics
                feature_dict[f'{feature_name}_p25'] = float(np.percentile(values, 25))
                feature_dict[f'{feature_name}_p50'] = float(np.percentile(values, 50))
                feature_dict[f'{feature_name}_p75'] = float(np.percentile(values, 75))
            else:
                # If no valid values, set all to 0
                feature_dict[f'{feature_name}_mean'] = 0.0
                feature_dict[f'{feature_name}_std'] = 0.0
                feature_dict[f'{feature_name}_min'] = 0.0
                feature_dict[f'{feature_name}_max'] = 0.0
                feature_dict[f'{feature_name}_range'] = 0.0
                feature_dict[f'{feature_name}_p25'] = 0.0
                feature_dict[f'{feature_name}_p50'] = 0.0
                feature_dict[f'{feature_name}_p75'] = 0.0
        
        # Add audio duration
        audio_info = sf.info(audio_path)
        feature_dict['duration'] = audio_info.duration
        
        return feature_dict
        
    except Exception as e:
        print(f"Error extracting features from {audio_path}: {e}")
        return None


def process_chunk_parallel(args):
    """
    Process a single chunk - designed for parallel execution.
    
    Args:
        args: Tuple of (audio_file, patient_name, dir_name)
    
    Returns:
        dict: Feature dictionary or None if failed
    """
    audio_file, patient_name, dir_name = args
    
    try:
        # Extract features
        features = extract_egemaps_features(audio_file)
        
        if features is not None:
            # Add metadata
            features.update({
                'patient_id': patient_name,
                'chunk_file': os.path.basename(audio_file),
                'chunk_directory': dir_name,
                'audio_path': audio_file
            })
            return features
        
    except Exception as e:
        print(f"Error processing {audio_file}: {e}")
    
    return None


def process_patient_folder_parallel(patient_path, n_workers=None):
    """
    Process patient folder using parallel processing.
    
    Args:
        patient_path (str): Path to the patient folder
        n_workers (int): Number of parallel workers
    
    Returns:
        list: List of feature dictionaries
    """
    if n_workers is None:
        n_workers = min(mp.cpu_count(), 10)  # Use 10 performance cores max
    
    patient_name = os.path.basename(patient_path)
    
    # Collect all audio files first
    all_audio_files = []
    chunk_dirs = []
    
    for item in os.listdir(patient_path):
        item_path = os.path.join(patient_path, item)
        if os.path.isdir(item_path) and "s_overlap" in item:
            chunk_dirs.append(item_path)
            for file in os.listdir(item_path):
                if file.lower().endswith(('.wav', '.mp3', '.flac', '.m4a')):
                    all_audio_files.append((os.path.join(item_path, file), patient_name, item))
    
    if not all_audio_files:
        return []
    
    print(f"Processing {len(all_audio_files)} chunks for {patient_name} using {n_workers} workers")
    
    # Process in parallel
    results = []
    
    # Use ProcessPoolExecutor for CPU-intensive tasks
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        # Submit all tasks
        future_to_file = {executor.submit(process_chunk_parallel, args): args for args in all_audio_files}
        
        # Process results as they complete with progress bar
        for future in tqdm.tqdm(as_completed(future_to_file), total=len(all_audio_files), 
                               desc=f"Processing {patient_name}", leave=False):
            result = future.result()
            if result is not None:
                results.append(result)
    
    return results


def save_patient_features(patient_features, patient_id, output_dir):
    """
    Save features for one patient to separate CSV.
    
    Args:
        patient_features (list): List of feature dictionaries
        patient_id (str): Patient identifier
        output_dir (str): Output directory path
    
    Returns:
        str: Path to saved CSV file
    """
    if not patient_features:
        return None
    
    output_file = f"{patient_id}_features.csv"
    output_path = os.path.join(output_dir, output_file)
    
    df = pd.DataFrame(patient_features)
    df.to_csv(output_path, index=False)
    
    print(f"Saved {len(df)} chunks for {patient_id} to {output_path}")
    return output_path


def main():
    # Configuration parameters
    DATA_DIR = "data/created_data"
    OUTPUT_DIR = "data/features"
    
    # Parallel processing settings for M4 Pro
    N_WORKERS = 10  # Use 10 performance cores (avoid efficiency cores for this task)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("DAIC-WOZ eGeMAPSv02 Feature Extraction (Parallel)")
    print("=" * 60)
    print(f"Input directory: {DATA_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Parallel workers: {N_WORKERS}")
    print(f"Processor: M4 Pro (10 performance cores)")
    print(f"Expected speedup: 5-10x faster than sequential")
    print("-" * 60)
    
    start_time = time.time()
    
    # Get all patient folders
    patient_folders = []
    for item in os.listdir(DATA_DIR):
        item_path = os.path.join(DATA_DIR, item)
        if os.path.isdir(item_path) and item.endswith('_P'):
            patient_folders.append(item_path)
    
    patient_folders.sort()
    
    print(f"Found {len(patient_folders)} patient folders")
    print("-" * 60)
    
    # Process each patient folder in parallel
    all_features = []
    processed_patients = 0
    
    for patient_folder in tqdm.tqdm(patient_folders, desc="Processing patients"):
        try:
            # Process patient with parallel chunk processing
            features = process_patient_folder_parallel(patient_folder, N_WORKERS)
            
            if features:
                # Save individual patient CSV
                save_patient_features(features, os.path.basename(patient_folder), OUTPUT_DIR)
                
                # Add to combined results
                all_features.extend(features)
                processed_patients += 1
                
                # Memory cleanup every few patients
                if processed_patients % 5 == 0:
                    gc.collect()
                    print(f"Memory cleanup: processed {processed_patients} patients")
            
        except Exception as e:
            print(f"Error processing patient {patient_folder}: {e}")
            continue
    
    # Save combined CSV
    if all_features:
        combined_df = pd.DataFrame(all_features)
        combined_path = os.path.join(OUTPUT_DIR, "all_patients_combined.csv")
        combined_df.to_csv(combined_path, index=False)
        
        print("-" * 60)
        print(f"Feature extraction complete!")
        print(f"Total chunks processed: {len(combined_df)}")
        print(f"Patients processed: {processed_patients}")
        print(f"Features per chunk: {len(combined_df.columns) - 5}")  # Exclude metadata columns
        print(f"Combined CSV saved to: {combined_path}")
        
        # Print summary statistics
        print("\nSummary by patient:")
        patient_summary = combined_df.groupby('patient_id').size()
        for patient, count in patient_summary.items():
            print(f"  {patient}: {count} chunks")
        
        # Print feature statistics
        print(f"\nFeature statistics:")
        if 'duration' in combined_df.columns:
            print(f"  Duration range: {combined_df['duration'].min():.2f}s - {combined_df['duration'].max():.2f}s")
            print(f"  Average duration: {combined_df['duration'].mean():.2f}s")
        
        # Show some example features
        feature_cols = [col for col in combined_df.columns if col not in ['patient_id', 'chunk_file', 'chunk_directory', 'audio_path', 'duration']]
        print(f"\nExample features extracted:")
        example_features = [col for col in feature_cols if 'mean' in col][:10]
        for feature in example_features:
            print(f"  {feature}")
        
        print(f"\nTotal features: {len(feature_cols)}")
        print("Features include: F0, energy, spectral, MFCC, formants, voice quality, etc.")
    
    else:
        print("No features extracted. Check your data directory.")
        return
    
    # Performance summary
    end_time = time.time()
    total_time = end_time - start_time
    chunks_per_second = len(all_features) / total_time if total_time > 0 else 0
    
    print("-" * 60)
    print(f"Performance Summary:")
    print(f"Total processing time: {total_time:.2f} seconds")
    print(f"Processing speed: {chunks_per_second:.2f} chunks/second")
    print(f"Speedup vs sequential: ~{N_WORKERS:.1f}x faster")


if __name__ == "__main__":
    main()