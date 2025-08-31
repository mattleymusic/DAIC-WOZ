import os
import librosa
import numpy as np
import pandas as pd
import sys
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

# Add the project root to the path to import opensmile
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

try:
    import opensmile
except ImportError:
    print("Error: opensmile package not found.")
    print("Please install it with: pip install opensmile")
    sys.exit(1)


def extract_egemaps_features(audio_path, sample_rate=16000):
    """
    Extract eGeMAPS features from a single audio chunk.
    
    Args:
        audio_path (str): Path to the audio chunk file
        sample_rate (int): Sample rate for the audio (default: 16000 Hz)
    
    Returns:
        numpy.ndarray: eGeMAPS feature vector (88 features)
    """
    # Initialize eGeMAPS feature extractor
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        sampling_rate=sample_rate
    )
    
    # Extract features
    features = smile.process_file(audio_path)
    
    # Convert DataFrame to numpy array and handle multiple frames
    if features.shape[0] > 1:
        # If we have multiple time frames, take the mean
        feature_vector = np.mean(features.values, axis=0)
    else:
        # If we have only one frame, use it directly
        feature_vector = features.values.flatten()
    
    return feature_vector


def process_single_patient(patient_folder, output_dir, sample_rate=16000):
    """
    Process all audio chunks for a single patient.
    This function is designed to be called in parallel.
    
    Args:
        patient_folder (str): Path to patient folder containing audio chunks
        output_dir (str): Output directory for this patient's features
        sample_rate (int): Sample rate for audio processing
    
    Returns:
        dict: Processing results for this patient
    """
    patient_name = os.path.basename(patient_folder)
    
    # Create patient output directory
    patient_output_dir = os.path.join(output_dir, patient_name)
    os.makedirs(patient_output_dir, exist_ok=True)
    
    # Get all audio chunks for this patient
    audio_chunks = []
    for file in os.listdir(patient_folder):
        if file.lower().endswith('.wav'):
            audio_chunks.append(os.path.join(patient_folder, file))
    
    audio_chunks.sort()  # Sort to process in order
    
    if not audio_chunks:
        return {
            'patient_name': patient_name,
            'successful_chunks': 0,
            'failed_chunks': 0,
            'total_chunks': 0,
            'status': 'no_chunks_found'
        }
    
    # Create feature names (eGeMAPS feature names)
    feature_names = [
        'F0semitoneFrom27.5Hz_sma3nz_amean', 'F0semitoneFrom27.5Hz_sma3nz_stddevNorm',
        'F0semitoneFrom27.5Hz_sma3nz_percentile20', 'F0semitoneFrom27.5Hz_sma3nz_percentile50',
        'F0semitoneFrom27.5Hz_sma3nz_percentile80', 'F0semitoneFrom27.5Hz_sma3nz_pctlrange0-2',
        'F0semitoneFrom27.5Hz_sma3nz_stddevFallingSlope', 'F0semitoneFrom27.5Hz_sma3nz_stddevRisingSlope',
        'F0semitoneFrom27.5Hz_sma3nz_skewness', 'F0semitoneFrom27.5Hz_sma3nz_kurtosis',
        'F0semitoneFrom27.5Hz_sma3nz_ameanFallingSlope', 'F0semitoneFrom27.5Hz_sma3nz_ameanRisingSlope',
        'F1frequency_sma3nz_amean', 'F1bandwidth_sma3nz_amean', 'F1amplitude_sma3nz_amean',
        'F2frequency_sma3nz_amean', 'F2bandwidth_sma3nz_amean', 'F2amplitude_sma3nz_amean',
        'F3frequency_sma3nz_amean', 'F3bandwidth_sma3nz_amean', 'F3amplitude_sma3nz_amean',
        'FormantDispersion_sma3nz_amean', 'FormantDispersion_sma3nz_stddevNorm',
        'FormantDispersion_sma3nz_percentile50', 'jitterLocal_sma3nz_amean', 'jitterRAP_sma3nz_amean',
        'jitterPPQ5_sma3nz_amean', 'jitterDDP_sma3nz_amean', 'shimmerLocal_sma3nz_amean',
        'shimmerAPQ3_sma3nz_amean', 'shimmerAPQ5_sma3nz_amean', 'shimmerAPQ11_sma3nz_amean',
        'shimmerDDA_sma3nz_amean', 'HNRdBACF_sma3nz_amean', 'HNRdBACF_sma3nz_stddevNorm',
        'HNRdBACF_sma3nz_percentile50', 'spectralCentroid_sma3nz_amean', 'spectralCentroid_sma3nz_stddevNorm',
        'spectralCentroid_sma3nz_percentile50', 'spectralCentroid_sma3nz_percentile80',
        'spectralCentroid_sma3nz_percentile20', 'spectralCentroid_sma3nz_percentile90',
        'spectralCentroid_sma3nz_percentile10', 'spectralCentroid_sma3nz_percentile95',
        'spectralCentroid_sma3nz_percentile05', 'spectralCentroid_sma3nz_percentile98',
        'spectralCentroid_sma3nz_percentile02', 'spectralCentroid_sma3nz_percentile99',
        'spectralSlope_sma3nz_amean', 'spectralSlope_sma3nz_stddevNorm', 'spectralSlope_sma3nz_percentile50',
        'spectralSlope_sma3nz_percentile80', 'spectralSlope_sma3nz_percentile20', 'spectralSlope_sma3nz_percentile90',
        'spectralSlope_sma3nz_percentile10', 'spectralSlope_sma3nz_percentile95', 'spectralSlope_sma3nz_percentile05',
        'spectralSlope_sma3nz_percentile98', 'spectralSlope_sma3nz_percentile02', 'spectralSlope_sma3nz_percentile99',
        'spectralRollOff25_sma3nz_amean', 'spectralRollOff25_sma3nz_stddevNorm', 'spectralRollOff25_sma3nz_percentile50',
        'spectralRollOff25_sma3nz_percentile80', 'spectralRollOff25_sma3nz_percentile20', 'spectralRollOff25_sma3nz_percentile90',
        'spectralRollOff25_sma3nz_percentile10', 'spectralRollOff25_sma3nz_percentile95', 'spectralRollOff25_sma3nz_percentile05',
        'spectralRollOff25_sma3nz_percentile98', 'spectralRollOff25_sma3nz_percentile02', 'spectralRollOff25_sma3nz_percentile99',
        'spectralFlux_sma3nz_amean', 'spectralFlux_sma3nz_stddevNorm', 'spectralFlux_sma3nz_percentile50',
        'spectralFlux_sma3nz_percentile80', 'spectralFlatness_sma3nz_amean', 'spectralFlatness_sma3nz_stddevNorm',
        'spectralFlatness_sma3nz_percentile50', 'spectralFlatness_sma3nz_percentile80', 'spectralVariance_sma3nz_amean',
        'spectralVariance_sma3nz_stddevNorm', 'spectralVariance_sma3nz_percentile50', 'spectralVariance_sma3nz_percentile80',
        'spectralSpread_sma3nz_amean', 'spectralSpread_sma3nz_stddevNorm', 'spectralSpread_sma3nz_percentile50',
        'spectralSpread_sma3nz_percentile80'
    ]
    
    successful_chunks = 0
    failed_chunks = 0
    
    # Process each audio chunk
    for chunk_path in audio_chunks:
        chunk_filename = os.path.basename(chunk_path)
        chunk_name = os.path.splitext(chunk_filename)[0]  # Remove .wav extension
        
        try:
            # Extract eGeMAPS features
            features = extract_egemaps_features(chunk_path, sample_rate)
            
            # Create DataFrame with features
            feature_df = pd.DataFrame([features], columns=feature_names)
            
            # Save to CSV
            output_filename = f"{chunk_name}_features.csv"
            output_path = os.path.join(patient_output_dir, output_filename)
            feature_df.to_csv(output_path, index=False)
            
            successful_chunks += 1
            
        except Exception as e:
            failed_chunks += 1
    
    return {
        'patient_name': patient_name,
        'successful_chunks': successful_chunks,
        'failed_chunks': failed_chunks,
        'total_chunks': len(audio_chunks),
        'status': 'completed'
    }


def process_chunk_configuration(chunk_config_dir, output_base_dir, sample_rate=16000):
    """
    Process all audio chunks in a specific chunk configuration directory using parallel processing.
    
    Args:
        chunk_config_dir (str): Path to chunk configuration directory (e.g., 30.0s_15.0s_overlap)
        output_base_dir (str): Base directory for output features
        sample_rate (int): Sample rate for audio processing
    """
    config_name = os.path.basename(chunk_config_dir)
    
    # Create output directory structure
    output_dir = os.path.join(output_base_dir, "features", "egemap", config_name)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Processing configuration: {config_name}")
    print(f"Output directory: {output_dir}")
    print("-" * 60)
    
    # Get all patient folders in the chunk configuration
    patient_folders = []
    for item in os.listdir(chunk_config_dir):
        item_path = os.path.join(chunk_config_dir, item)
        if os.path.isdir(item_path):
            patient_folders.append(item_path)
    
    patient_folders.sort()  # Sort to process in order
    
    if not patient_folders:
        print("No patient folders found in this configuration.")
        return 0, 0, 0
    
    print(f"Found {len(patient_folders)} patient folders")
    
    # Determine number of workers (use all available cores, but leave some for system)
    num_workers = min(mp.cpu_count(), len(patient_folders), 12)  # Cap at 12 to avoid overwhelming system
    print(f"Using {num_workers} parallel workers")
    
    # Prepare arguments for parallel processing
    args_list = [(patient_folder, output_dir, sample_rate) for patient_folder in patient_folders]
    
    successful_patients = 0
    failed_patients = 0
    total_chunks = 0
    
    # Process patients in parallel
    start_time = time.time()
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        future_to_patient = {executor.submit(process_single_patient, *args): args[0] for args in args_list}
        
        # Process completed tasks
        for future in as_completed(future_to_patient):
            patient_folder = future_to_patient[future]
            patient_name = os.path.basename(patient_folder)
            
            try:
                result = future.result()
                
                if result['status'] == 'completed':
                    if result['failed_chunks'] == 0:
                        print(f"✓ {patient_name}: {result['successful_chunks']}/{result['total_chunks']} chunks processed successfully")
                        successful_patients += 1
                    else:
                        print(f"⚠ {patient_name}: {result['successful_chunks']}/{result['total_chunks']} chunks processed ({result['failed_chunks']} failed)")
                        successful_patients += 1  # Still count as successful if some chunks worked
                    
                    total_chunks += result['total_chunks']
                elif result['status'] == 'no_chunks_found':
                    print(f"⚠ {patient_name}: No audio chunks found")
                    failed_patients += 1
                else:
                    print(f"✗ {patient_name}: Processing failed")
                    failed_patients += 1
                    
            except Exception as e:
                print(f"✗ {patient_name}: Error in parallel processing: {e}")
                failed_patients += 1
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    print(f"\nConfiguration {config_name} completed in {processing_time:.2f} seconds")
    print(f"Average time per patient: {processing_time/len(patient_folders):.2f} seconds")
    
    return successful_patients, failed_patients, total_chunks


def main():
    """
    !!! INPUT STRUCTURE !!!
    
    data/created_data/
    ├── concatenated_diarisation/          # Source files
    ├── 30.0s_15.0s_overlap/             # Audio chunks
    │   ├── 300_P/
    │   │   ├── 300_P_chunk001.wav
    │   │   └── ...
    │   └── 301_P/
    ├── 10.0s_5.0s_overlap/              # Audio chunks
    └── 5.0s_2.5s_overlap/               # Audio chunks
    
    !!! OUTPUT STRUCTURE !!!
    
    data/features/egemap/
    ├── 30.0s_15.0s_overlap/
    │   ├── 300_P/
    │   │   ├── 300_P_chunk001_features.csv
    │   │   ├── 300_P_chunk002_features.csv
    │   │   └── ...
    │   └── 301_P/
    ├── 10.0s_5.0s_overlap/
    └── 5.0s_2.5s_overlap/
    
    !!! PROCESSING DETAILS !!!
    
    - Processes all chunk configurations found in data/created_data/
    - Extracts eGeMAPS features from each audio chunk
    - Saves individual CSV files per chunk with 88 features
    - Maintains the same directory structure as input chunks
    - Each CSV contains one row with 88 eGeMAPS feature values
    """
    # Configuration parameters - CHANGE THESE VALUES AS NEEDED
    CHUNK_CONFIGS = [
        "3.0s_1.5s_overlap",
        "5.0s_2.5s_overlap",
        "10.0s_5.0s_overlap",
        "20.0s_10.0s_overlap",
        "30.0s_15.0s_overlap",

    ]
    SAMPLE_RATE = 16000  # Hz - maintains original DAIC-WOZ format
    
    # Paths - adjusted for src/feature_extraction directory
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    INPUT_BASE_DIR = os.path.join(PROJECT_ROOT, "data", "created_data")
    OUTPUT_BASE_DIR = os.path.join(PROJECT_ROOT, "data")
    
    print(f"Script location: {os.path.abspath(__file__)}")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Input directory: {INPUT_BASE_DIR}")
    print(f"Output directory: {OUTPUT_BASE_DIR}")
    print(f"Sample rate: {SAMPLE_RATE} Hz")
    print(f"Chunk configurations to process: {CHUNK_CONFIGS}")
    print("=" * 80)
    
    total_successful_patients = 0
    total_failed_patients = 0
    total_chunks_processed = 0
    
    # Process each chunk configuration
    for config_name in CHUNK_CONFIGS:
        config_input_dir = os.path.join(INPUT_BASE_DIR, config_name)
        
        # Check if configuration directory exists
        if not os.path.exists(config_input_dir):
            print(f"\n⚠ Configuration directory not found: {config_input_dir}")
            print("Skipping this configuration...")
            continue
        
        print(f"\n{'='*20} Processing {config_name} {'='*20}")
        
        # Process this configuration
        successful, failed, chunks = process_chunk_configuration(
            config_input_dir, 
            OUTPUT_BASE_DIR, 
            SAMPLE_RATE
        )
        
        total_successful_patients += successful
        total_failed_patients += failed
        total_chunks_processed += chunks
    
    print("\n" + "=" * 80)
    print("FINAL PROCESSING SUMMARY")
    print("=" * 80)
    print(f"Total patients processed: {total_successful_patients + total_failed_patients}")
    print(f"Successful patients: {total_successful_patients}")
    print(f"Failed patients: {total_failed_patients}")
    print(f"Total chunks processed: {total_chunks_processed}")
    print(f"Success rate: {(total_successful_patients/(total_successful_patients+total_failed_patients)*100):.1f}%")
    print(f"\neGeMAPS features saved to:")
    print(f"  {OUTPUT_BASE_DIR}/features/egemap/[CONFIG]/[PATIENT_ID]/[CHUNK]_features.csv")
    print(f"\nEach CSV contains 88 eGeMAPS features for one audio chunk.")


if __name__ == "__main__":
    # Set multiprocessing start method for compatibility across platforms
    mp.set_start_method('spawn', force=True)
    main()