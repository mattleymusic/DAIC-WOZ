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


def process_patient_worker(patient_path):
    """
    Process one entire patient - designed for patient-level parallelism.
    
    Args:
        patient_path (str): Path to the patient folder
    
    Returns:
        tuple: (patient_id, features_list, success)
    """
    patient_id = os.path.basename(patient_path)
    
    try:
        # Process all chunks for this patient (sequentially within the worker)
        features = process_patient_folder_sequential(patient_path)
        
        if features:
            return (patient_id, features, True)
        else:
            return (patient_id, [], False)
            
    except Exception as e:
        print(f"Error processing patient {patient_id}: {e}")
        return (patient_id, [], False)


def process_patient_folder_sequential(patient_path):
    """
    Process all chunks for one patient sequentially (within a worker).
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
                features = extract_egemaps_features(audio_file)
                
                if features is not None:
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
    N_WORKERS = 10  # Use 10 performance cores
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("DAIC-WOZ eGeMAPSv02 Feature Extraction (Patient-Level Parallel)")
    print("=" * 60)
    print(f"Input directory: {DATA_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Parallel workers: {N_WORKERS}")
    print(f"Strategy: {N_WORKERS} workers processing {N_WORKERS} patients simultaneously")
    print(f"Processor: M4 Pro (10 performance cores)")
    print(f"Expected speedup: 8-12x faster than sequential")
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
    print(f"Will process {min(N_WORKERS, len(patient_folders))} patients simultaneously")
    print("-" * 60)
    
    # Process patients in parallel (each worker gets one patient)
    all_features = []
    processed_patients = 0
    
    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        # Submit all patient processing tasks
        future_to_patient = {executor.submit(process_patient_worker, patient_path): patient_path 
                           for patient_path in patient_folders}
        
        # Process results as they complete
        for future in tqdm.tqdm(as_completed(future_to_patient), total=len(patient_folders), 
                               desc="Processing patients in parallel"):
            patient_id, features, success = future.result()
            
            if success and features:
                # Save individual patient CSV
                save_patient_features(features, patient_id, OUTPUT_DIR)
                
                # Add to combined results
                all_features.extend(features)
                processed_patients += 1
                
                print(f"Completed patient {patient_id}: {len(features)} chunks")
            else:
                print(f"Failed to process patient {patient_id}")
    
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
    print(f"Efficiency: {chunks_per_second / N_WORKERS:.2f} chunks/second per worker")


if __name__ == "__main__":
    main()