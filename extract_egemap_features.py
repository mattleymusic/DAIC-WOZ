import os
import pandas as pd
import numpy as np
import soundfile as sf
from pathlib import Path
import warnings
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


def process_patient_folder(patient_path, output_base_path):
    """
    Process all audio chunks in a patient's folder.
    
    Args:
        patient_path (str): Path to the patient folder (e.g., 300_P)
        output_base_path (str): Base path for output directory
    
    Returns:
        list: List of feature dictionaries
    """
    patient_name = os.path.basename(patient_path)
    
    # Look for chunk directories
    chunk_dirs = []
    for item in os.listdir(patient_path):
        item_path = os.path.join(patient_path, item)
        if os.path.isdir(item_path) and "s_overlap" in item:
            chunk_dirs.append(item_path)
    
    if not chunk_dirs:
        print(f"No chunk directories found in {patient_path}")
        return []
    
    print(f"Processing {len(chunk_dirs)} chunk directories for {patient_name}")
    
    all_features = []
    
    for chunk_dir in chunk_dirs:
        # Get chunk length and overlap from directory name
        dir_name = os.path.basename(chunk_dir)
        
        # Process all audio files in the chunk directory
        audio_files = []
        for file in os.listdir(chunk_dir):
            if file.lower().endswith(('.wav', '.mp3', '.flac', '.m4a')):
                audio_files.append(os.path.join(chunk_dir, file))
        
        if not audio_files:
            continue
        
        print(f"  Processing {len(audio_files)} chunks in {dir_name}")
        
        for audio_file in audio_files:
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
                    
                    all_features.append(features)
                
            except Exception as e:
                print(f"    Error processing {audio_file}: {e}")
    
    return all_features


def main():
    # Configuration parameters
    DATA_DIR = "data/created_data"
    OUTPUT_DIR = "data/features"
    OUTPUT_FILE = "chunk_features_egemaps.csv"
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("DAIC-WOZ eGeMAPSv02 Feature Extraction")
    print("=" * 50)
    print(f"Input directory: {DATA_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Output file: {OUTPUT_FILE}")
    print(f"Feature set: eGeMAPSv02 (opensmile)")
    print("-" * 50)
    
    # Get all patient folders
    patient_folders = []
    for item in os.listdir(DATA_DIR):
        item_path = os.path.join(DATA_DIR, item)
        if os.path.isdir(item_path) and item.endswith('_P'):
            patient_folders.append(item_path)
    
    patient_folders.sort()
    
    print(f"Found {len(patient_folders)} patient folders")
    print("-" * 50)
    
    # Process each patient folder
    all_features = []
    for patient_folder in patient_folders:
        features = process_patient_folder(patient_folder, OUTPUT_DIR)
        all_features.extend(features)
    
    if not all_features:
        print("No features extracted. Check your data directory.")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(all_features)
    
    # Reorder columns for better readability
    metadata_cols = ['patient_id', 'chunk_file', 'chunk_directory', 'audio_path', 'duration']
    feature_cols = [col for col in df.columns if col not in metadata_cols]
    
    df = df[metadata_cols + feature_cols]
    
    # Save to CSV
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
    df.to_csv(output_path, index=False)
    
    print("-" * 50)
    print(f"Feature extraction complete!")
    print(f"Total chunks processed: {len(df)}")
    print(f"Features per chunk: {len(feature_cols)}")
    print(f"Output saved to: {output_path}")
    
    # Print summary statistics
    print("\nSummary by patient:")
    patient_summary = df.groupby('patient_id').size()
    for patient, count in patient_summary.items():
        print(f"  {patient}: {count} chunks")
    
    # Print feature statistics
    print(f"\nFeature statistics:")
    print(f"  Duration range: {df['duration'].min():.2f}s - {df['duration'].max():.2f}s")
    print(f"  Average duration: {df['duration'].mean():.2f}s")
    
    # Show some example features
    print(f"\nExample features extracted:")
    example_features = [col for col in feature_cols if 'mean' in col][:10]
    for feature in example_features:
        print(f"  {feature}")
    
    print(f"\nTotal features: {len(feature_cols)}")
    print("Features include: F0, energy, spectral, MFCC, formants, voice quality, etc.")


if __name__ == "__main__":
    main()