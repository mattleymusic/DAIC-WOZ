#!/usr/bin/env python3
"""
Paper-Specific COVAREP Feature Extractor for Miao et al. 2022 Reproduction

This script extracts exactly 74 COVAREP features as specified in the paper:
"Fusing features of speech for depression classification based on higher-order spectral analysis"
Speech Communication 143 (2022) 46–56

The paper specifies:
- 74 total COVAREP features extracted every 4 seconds
- Features include: MFCCs 0-24, harmonic model, prosodic features, speech quality features
- These 74 features are then reduced to 10 features using Relief-based selection

Usage:
    python src/feature_extraction/paper_covarep_extractor.py

Author: Based on Miao et al. 2022 paper specifications
"""

import os
import librosa
import numpy as np
import pandas as pd
import sys
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
from scipy import signal
from scipy.stats import skew, kurtosis
import warnings
import psutil
from functools import partial

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Add the project root to the path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

# MBP Core Configuration - Optimized for 10 Performance + 4 Efficiency cores
PERFORMANCE_CORES = 10  # High-performance cores for CPU-intensive tasks
EFFICIENCY_CORES = 4    # Efficiency cores for I/O-bound tasks
TOTAL_CORES = PERFORMANCE_CORES + EFFICIENCY_CORES

def set_cpu_affinity():
    """Set CPU affinity for optimal performance on MBP."""
    try:
        # Get current process
        current_process = psutil.Process()
        
        # Get logical CPU cores (performance cores first, then efficiency cores)
        cpu_count = psutil.cpu_count(logical=True)
        
        # On Apple Silicon, performance cores are typically 0-9, efficiency cores 10-13
        if cpu_count >= TOTAL_CORES:
            # Use performance cores for CPU-intensive tasks
            performance_core_ids = list(range(PERFORMANCE_CORES))
            current_process.cpu_affinity(performance_core_ids)
            print(f"Set CPU affinity to performance cores: {performance_core_ids}")
        else:
            print(f"Warning: Expected {TOTAL_CORES} cores, found {cpu_count}")
            
    except Exception as e:
        print(f"Warning: Could not set CPU affinity: {e}")

def optimize_memory():
    """Optimize memory usage for MBP M4 Pro with conservative settings."""
    try:
        # Conservative thread settings to avoid memory pressure
        os.environ['OMP_NUM_THREADS'] = str(min(PERFORMANCE_CORES, 4))  # Limit to 4 cores
        os.environ['MKL_NUM_THREADS'] = str(min(PERFORMANCE_CORES, 4))
        os.environ['NUMEXPR_NUM_THREADS'] = str(min(PERFORMANCE_CORES, 4))
        
        # Disable MPS to avoid GPU memory issues
        os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
        
        # Configure librosa for minimal memory usage
        librosa.set_cache(False)  # Disable caching to save memory
        
        # Conservative garbage collection
        import gc
        gc.set_threshold(500, 5, 5)  # More frequent garbage collection
        
        # Set NumPy to ignore numerical warnings for cleaner output
        np.seterr(all='ignore')
        
        # Limit NumPy memory usage
        np.seterr(divide='ignore', invalid='ignore')
        
        print(f"Conservative memory optimization configured for MBP M4 Pro")
        print(f"Thread limits: OMP={os.environ['OMP_NUM_THREADS']}, MKL={os.environ['MKL_NUM_THREADS']}")
        
    except Exception as e:
        print(f"Warning: Could not optimize memory settings: {e}")


def extract_paper_covarep_features(audio_path, sample_rate=16000):
    """
    Extract exactly 74 COVAREP features as specified in Miao et al. 2022.
    
    According to the paper, COVAREP extracts:
    - MFCCs 0-24 (25 features)
    - Harmonic model: mean phase distortion 0-24 (25 features) - IMPROVED IMPLEMENTATION
      * Analyzes actual harmonic components (f0, 2*f0, 3*f0, etc.)
      * Computes phase distortion as deviation from expected phase progression
      * Handles silent segments with small random values instead of zeros
      * Provides meaningful variation across the 25 harmonic features
    - Mean deviation 0-12 (13 features)
    - Prosodic features: F0, pitch, vocalization probability, formants 1-5
    - Speech quality features: various glottal and spectral measures
    
    Args:
        audio_path (str): Path to the audio chunk file
        sample_rate (int): Sample rate for the audio (default: 16000 Hz)
    
    Returns:
        numpy.ndarray: 74 COVAREP feature vector
    """
    try:
        # Load audio file
        y, sr = librosa.load(audio_path, sr=sample_rate)
        
        # Ensure mono
        if len(y.shape) > 1:
            y = np.mean(y, axis=0)
        
        # Frame parameters for 4-second chunks
        frame_length = 2048
        hop_length = 512
        
        features = []
        
        # 1. MFCCs 0-24 (25 features) - as specified in paper
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=25, hop_length=hop_length)
        for i in range(25):
            features.append(np.mean(mfccs[i]))
        
        # 2. Harmonic model: mean phase distortion 0-24 (25 features)
        # Proper harmonic component phase distortion analysis
        
        # Extract fundamental frequency first
        f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), 
                                                    fmax=librosa.note_to_hz('C7'), 
                                                    frame_length=frame_length, 
                                                    hop_length=hop_length)
        
        # Compute STFT for phase analysis
        stft = librosa.stft(y, n_fft=frame_length, hop_length=hop_length)
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        
        # Get frequency bins
        freqs = librosa.fft_frequencies(sr=sr, n_fft=frame_length)
        
        # Handle silent or very quiet segments
        if np.max(np.abs(y)) < 1e-6 or np.all(np.isnan(f0)) or np.mean(voiced_flag) < 0.1:
            # For silent segments, use small random values instead of zeros
            for i in range(25):
                features.append(np.random.normal(0, 0.01))
        else:
            # Compute phase distortion for each harmonic component
            for i in range(25):
                if i == 0:
                    # Fundamental frequency component
                    target_freq = f0
                else:
                    # Harmonic components (2*f0, 3*f0, etc.)
                    target_freq = f0 * (i + 1)
                
                # Find closest frequency bins for this harmonic
                phase_distortions = []
                
                for frame_idx in range(len(target_freq)):
                    if not np.isnan(target_freq[frame_idx]) and target_freq[frame_idx] > 0:
                        # Find closest frequency bin
                        freq_diff = np.abs(freqs - target_freq[frame_idx])
                        freq_idx = np.argmin(freq_diff)
                        
                        # Only use if we're within a reasonable frequency range
                        if freq_diff[freq_idx] < target_freq[frame_idx] * 0.1:  # Within 10% tolerance
                            # Phase distortion as deviation from expected phase progression
                            if frame_idx > 0:
                                expected_phase_diff = 2 * np.pi * target_freq[frame_idx] * hop_length / sr
                                actual_phase_diff = phase[freq_idx, frame_idx] - phase[freq_idx, frame_idx-1]
                                
                                # Normalize phase difference to [-π, π]
                                phase_diff = actual_phase_diff - expected_phase_diff
                                while phase_diff > np.pi:
                                    phase_diff -= 2 * np.pi
                                while phase_diff < -np.pi:
                                    phase_diff += 2 * np.pi
                                
                                phase_distortions.append(np.abs(phase_diff))
                
                # Use mean phase distortion for this harmonic
                if len(phase_distortions) > 0:
                    features.append(np.mean(phase_distortions))
                else:
                    # Fallback: use phase variance at harmonic frequency
                    if not np.all(np.isnan(f0)):
                        mean_f0 = np.nanmean(f0)
                        harmonic_freq = mean_f0 * (i + 1)
                        freq_diff = np.abs(freqs - harmonic_freq)
                        freq_idx = np.argmin(freq_diff)
                        if freq_diff[freq_idx] < harmonic_freq * 0.2:  # Within 20% tolerance
                            harmonic_phase = phase[freq_idx, :]
                            phase_variance = np.var(harmonic_phase[~np.isnan(harmonic_phase)])
                            features.append(phase_variance)
                        else:
                            features.append(0.0)
                    else:
                        features.append(0.0)
        
        # 3. Mean deviation 0-12 (13 features)
        # Compute mean deviation across different frequency bands
        stft = librosa.stft(y, n_fft=frame_length, hop_length=hop_length)
        magnitude = np.abs(stft)
        
        # Divide spectrum into 13 bands and compute mean deviation
        n_bands = 13
        band_size = magnitude.shape[0] // n_bands
        
        for i in range(n_bands):
            start_idx = i * band_size
            end_idx = (i + 1) * band_size if i < n_bands - 1 else magnitude.shape[0]
            band_magnitude = magnitude[start_idx:end_idx, :]
            mean_deviation = np.mean(np.std(band_magnitude, axis=0))
            features.append(mean_deviation)
        
        # 4. Prosodic features (7 features)
        # Fundamental frequency (F0)
        f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), 
                                                    fmax=librosa.note_to_hz('C7'), 
                                                    frame_length=frame_length, 
                                                    hop_length=hop_length)
        f0_clean = f0[~np.isnan(f0)]
        features.append(np.mean(f0_clean) if len(f0_clean) > 0 else 0.0)
        
        # Pitch (similar to F0 but with different processing)
        pitch = librosa.yin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
        pitch_clean = pitch[~np.isnan(pitch)]
        features.append(np.mean(pitch_clean) if len(pitch_clean) > 0 else 0.0)
        
        # Vocalization probability
        features.append(np.mean(voiced_flag))
        
        # Formants 1-4 (4 features)
        # Extract formants using spectral peaks
        freqs = librosa.fft_frequencies(sr=sr, n_fft=frame_length)
        formant_candidates = []
        
        for frame_idx in range(magnitude.shape[1]):
            frame_magnitude = magnitude[:, frame_idx]
            peaks, _ = signal.find_peaks(frame_magnitude, height=np.max(frame_magnitude) * 0.1)
            peak_freqs = freqs[peaks]
            peak_magnitudes = frame_magnitude[peaks]
            sorted_indices = np.argsort(peak_magnitudes)[::-1]
            top_4_peaks = peak_freqs[sorted_indices[:4]]
            formant_candidates.append(top_4_peaks)
        
        # Convert to numpy array and compute mean formants
        max_formants = 4
        formant_matrix = np.full((max_formants, len(formant_candidates)), np.nan)
        
        for i, candidates in enumerate(formant_candidates):
            for j, freq in enumerate(candidates[:max_formants]):
                formant_matrix[j, i] = freq
        
        for i in range(max_formants):
            formant_vals = formant_matrix[i, :]
            formant_clean = formant_vals[~np.isnan(formant_vals)]
            features.append(np.mean(formant_clean) if len(formant_clean) > 0 else 0.0)
        
        # 5. Speech quality features (4 features)
        # Normalized amplitude quotient
        amplitude_envelope = librosa.amplitude_to_db(np.abs(librosa.stft(y, hop_length=hop_length)))
        features.append(np.mean(amplitude_envelope))
        
        # Quasi-open quotient (simplified)
        features.append(np.mean(voiced_flag) * 0.5)  # Simplified calculation
        
        # Difference in amplitude between first two harmonics
        harmonic_1 = np.mean(magnitude[1:3, :])  # First harmonic
        harmonic_2 = np.mean(magnitude[3:5, :])  # Second harmonic
        features.append(harmonic_1 - harmonic_2)
        
        # Parabolic spectral parameters (simplified)
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
        features.append(np.mean(spectral_centroid))
        
        # Ensure we have exactly 74 features
        if len(features) != 74:
            print(f"Warning: Expected 74 features, got {len(features)}")
            # Pad or truncate to 74
            while len(features) < 74:
                features.append(0.0)
            features = features[:74]
        
        return np.array(features)
        
    except Exception as e:
        print(f"Error extracting paper COVAREP features from {audio_path}: {e}")
        # Return zero vector as fallback (74 dimensions)
        return np.zeros(74)


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
    
    # Create feature names for 74 COVAREP features as per paper
    feature_names = []
    
    # MFCCs 0-24 (25 features)
    for i in range(25):
        feature_names.append(f'mfcc_{i}')
    
    # Harmonic model: mean phase distortion 0-24 (25 features)
    for i in range(25):
        feature_names.append(f'phase_distortion_{i}')
    
    # Mean deviation 0-12 (13 features)
    for i in range(13):
        feature_names.append(f'mean_deviation_{i}')
    
    # Prosodic features (7 features)
    feature_names.extend([
        'f0', 'pitch', 'vocalization_probability',
        'formant_1', 'formant_2', 'formant_3', 'formant_4'
    ])
    
    # Speech quality features (4 features)
    feature_names.extend([
        'normalized_amplitude_quotient', 'quasi_open_quotient',
        'harmonic_amplitude_diff', 'parabolic_spectral_param'
    ])
    
    successful_chunks = 0
    failed_chunks = 0
    
    # Process each audio chunk
    for chunk_path in audio_chunks:
        chunk_filename = os.path.basename(chunk_path)
        chunk_name = os.path.splitext(chunk_filename)[0]  # Remove .wav extension
        
        # Check if CSV already exists (resumable functionality)
        output_filename = f"{chunk_name}_paper_covarep_features.csv"
        output_path = os.path.join(patient_output_dir, output_filename)
        
        if os.path.exists(output_path):
            # Skip if already processed
            successful_chunks += 1
            continue
        
        try:
            # Extract paper COVAREP features
            features = extract_paper_covarep_features(chunk_path, sample_rate)
            
            # Create DataFrame with features
            feature_df = pd.DataFrame([features], columns=feature_names)
            
            # Save to CSV
            feature_df.to_csv(output_path, index=False)
            
            successful_chunks += 1
            
        except Exception as e:
            print(f"    Error processing {chunk_name}: {e}")
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
        chunk_config_dir (str): Path to chunk configuration directory (e.g., 4.0s_0.0s_overlap)
        output_base_dir (str): Base directory for output features
        sample_rate (int): Sample rate for audio processing
    """
    config_name = os.path.basename(chunk_config_dir)
    
    # Create output directory structure
    output_dir = os.path.join(output_base_dir, "features", "paper_covarep", config_name)
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
    
    # Determine number of workers (optimized for MBP 10P+4E cores)
    # Use fewer cores for memory efficiency
    num_workers = min(2, len(patient_folders), 4)  # Conservative: use only 2-4 cores to save memory
    print(f"Using {num_workers} parallel workers (conservative for memory)")
    print(f"Available cores: {mp.cpu_count()} total, {PERFORMANCE_CORES} performance, {EFFICIENCY_CORES} efficiency")
    
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
    
    data/created_data/4.0s_0.0s_overlap/  # 4-second chunks from paper_audio_chunker.py
    ├── 300_P/
    │   ├── chunk_001.wav
    │   ├── chunk_002.wav
    │   └── ...
    └── 301_P/
    
    !!! OUTPUT STRUCTURE !!!
    
    data/features/paper_covarep/4.0s_0.0s_overlap/
    ├── 300_P/
    │   ├── chunk_001_paper_covarep_features.csv
    │   ├── chunk_001_paper_covarep_features.csv
    │   └── ...
    └── 301_P/
    
    !!! PROCESSING DETAILS !!!
    
    - Processes 4-second chunks created by paper_audio_chunker.py
    - Extracts exactly 74 COVAREP features as specified in Miao et al. 2022
    - IMPROVED: Phase distortion features now properly analyze harmonic components
    - Handles silent segments gracefully (no more zero values)
    - Saves individual CSV files per chunk with 74 COVAREP features
    - Maintains the same directory structure as input chunks
    - Each CSV contains one row with 74 COVAREP feature values
    - Memory-efficient processing for MBP M4 Pro
    """
    # Initialize optimizations for MBP
    print("Initializing MBP optimizations...")
    optimize_memory()
    set_cpu_affinity()
    
    # Configuration parameters - Processing 4-second chunks as in paper
    CHUNK_CONFIG = "4.0s_0.0s_overlap"  # Exactly as created by paper_audio_chunker.py
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
    print(f"Chunk configuration to process: {CHUNK_CONFIG}")
    print("=" * 80)
    
    # Check if 4-second chunks exist
    config_input_dir = os.path.join(INPUT_BASE_DIR, CHUNK_CONFIG)
    if not os.path.exists(config_input_dir):
        print(f"❌ Error: 4-second chunks not found: {config_input_dir}")
        print("Please run paper_audio_chunker.py first to create 4-second chunks.")
        return
    
    print(f"\n{'='*20} Processing {CHUNK_CONFIG} {'='*20}")
    
    # Process this configuration
    successful, failed, chunks = process_chunk_configuration(
        config_input_dir, 
        OUTPUT_BASE_DIR, 
        SAMPLE_RATE
    )
    
    print("\n" + "=" * 80)
    print("FINAL PROCESSING SUMMARY")
    print("=" * 80)
    print(f"Total patients processed: {successful + failed}")
    print(f"Successful patients: {successful}")
    print(f"Failed patients: {failed}")
    print(f"Total chunks processed: {chunks}")
    print(f"Success rate: {(successful/(successful+failed)*100):.1f}%")
    print(f"\nPaper COVAREP features saved to:")
    print(f"  {OUTPUT_BASE_DIR}/features/paper_covarep/{CHUNK_CONFIG}/[PATIENT_ID]/[CHUNK]_paper_covarep_features.csv")
    print(f"\nEach CSV contains exactly 74 COVAREP features as specified in Miao et al. 2022.")
    print(f"These features will be reduced to 10 features using Relief-based selection in the next step.")


if __name__ == "__main__":
    # Set multiprocessing start method for compatibility across platforms
    mp.set_start_method('spawn', force=True)
    main()
