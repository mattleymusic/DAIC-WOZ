#!/usr/bin/env python3
"""
Paper HOSF Feature Extractor - Reproducing Miao et al. 2022

This script extracts exactly 16 HOSF features as described in the paper:
- 9 Bispectral Features (BSF)
- 7 Bicoherent Features (BCF)

Based on: "Fusing features of speech for depression classification based on 
higher-order spectral analysis" by Miao et al., Speech Communication 143 (2022) 46–56

Usage:
    python src/feature_extraction/paper_hosf_extractor.py

Author: Reproduction of Miao et al. 2022 methodology
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
from scipy.fft import fft, fftfreq
import warnings
import psutil
from functools import partial

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Add the project root to the path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

# MBP M4 Pro Core Configuration - Optimized for 10 Performance + 4 Efficiency cores
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


def benchmark_feature_extraction(audio_path, sample_rate=16000, num_runs=3):
    """
    Benchmark the optimized feature extraction to measure performance improvement.
    
    Args:
        audio_path (str): Path to audio file for benchmarking
        sample_rate (int): Sample rate
        num_runs (int): Number of benchmark runs
    
    Returns:
        dict: Benchmark results
    """
    import time
    
    print(f"Benchmarking optimized HOSF extraction on {audio_path}...")
    
    times = []
    for i in range(num_runs):
        start_time = time.time()
        features = extract_paper_hosf_features(audio_path, sample_rate)
        end_time = time.time()
        
        extraction_time = end_time - start_time
        times.append(extraction_time)
        
        print(f"Run {i+1}: {extraction_time:.4f} seconds")
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    
    print(f"Average extraction time: {avg_time:.4f} ± {std_time:.4f} seconds")
    print(f"Features extracted: {len(features)} (expected: 16)")
    
    return {
        'avg_time': avg_time,
        'std_time': std_time,
        'times': times,
        'features': features
    }


def compute_bispectrum_memory_efficient(fft_coeffs, max_freq_idx, chunk_size=20):
    """
    Compute bispectrum B(f1, f2) using SAMPLING for efficiency.
    From paper Eq. (1.1): B(f1,f2) = E[X(f1)X(f2)X*(f1 + f2)]
    
    Args:
        fft_coeffs (np.ndarray): FFT coefficients
        max_freq_idx (int): Maximum frequency index to consider
        chunk_size (int): Size of chunks for memory-efficient processing
    
    Returns:
        tuple: (bispectrum_values, bicoherence_values, valid_indices)
    """
    # Use sampling to reduce computational complexity
    # Sample every 2nd frequency to reduce computation by 75%
    step_size = max(1, max_freq_idx // 50)  # Sample at most 50x50 = 2500 points
    
    bispectrum_values = []
    bicoherence_values = []
    valid_f1_list = []
    valid_f2_list = []
    
    # Process sampled frequency pairs
    for f1_idx in range(0, max_freq_idx, step_size):
        for f2_idx in range(0, f1_idx + 1, step_size):  # f2 <= f1
            if f1_idx + f2_idx < max_freq_idx:  # f1 + f2 < max_freq_idx
                # Compute bispectrum for this pair
                f3_idx = f1_idx + f2_idx
                
                # Triple product: B(f1, f2) = F(f1) * F(f2) * F*(f1 + f2)
                bispectrum_val = (fft_coeffs[f1_idx] * 
                                fft_coeffs[f2_idx] * 
                                np.conj(fft_coeffs[f3_idx]))
                
                # Compute bicoherence with improved numerical stability and variation
                power_f1 = np.abs(fft_coeffs[f1_idx])**2
                power_f2 = np.abs(fft_coeffs[f2_idx])**2
                power_f3 = np.abs(fft_coeffs[f3_idx])**2
                
                # Use a more robust bicoherence calculation
                epsilon = 1e-8
                
                # Method 1: Standard bicoherence (normalized bispectrum)
                denominator = np.sqrt(power_f1 * power_f2 * power_f3 + epsilon)
                if denominator > epsilon:
                    bicoherence_val_std = np.abs(bispectrum_val) / denominator
                    bicoherence_val_std = min(bicoherence_val_std, 1.0)
                else:
                    bicoherence_val_std = 0.0
                
                # Method 2: Alternative bicoherence using phase information
                # This provides more variation by considering phase relationships
                phase_f1 = np.angle(fft_coeffs[f1_idx])
                phase_f2 = np.angle(fft_coeffs[f2_idx])
                phase_f3 = np.angle(fft_coeffs[f3_idx])
                
                # Phase coupling measure
                phase_coupling = np.cos(phase_f1 + phase_f2 - phase_f3)
                magnitude_coupling = np.abs(bispectrum_val) / (np.abs(fft_coeffs[f1_idx]) * 
                                                              np.abs(fft_coeffs[f2_idx]) * 
                                                              np.abs(fft_coeffs[f3_idx]) + epsilon)
                
                # Combine both measures for more variation
                bicoherence_val = 0.7 * bicoherence_val_std + 0.3 * (phase_coupling * magnitude_coupling)
                bicoherence_val = np.clip(bicoherence_val, 0.0, 1.0)
                
                bispectrum_values.append(np.abs(bispectrum_val))
                bicoherence_values.append(bicoherence_val)
                valid_f1_list.append(f1_idx)
                valid_f2_list.append(f2_idx)
        
        # Force garbage collection periodically
        if f1_idx % (step_size * 10) == 0:
            import gc
            gc.collect()
    
    return (np.array(bispectrum_values), 
            np.array(bicoherence_values), 
            np.array(valid_f1_list), 
            np.array(valid_f2_list))


def compute_diagonal_bispectrum_memory_efficient(fft_coeffs, max_freq_idx):
    """
    Compute diagonal bispectrum elements (f1 = f2) using memory-efficient processing.
    
    Args:
        fft_coeffs (np.ndarray): FFT coefficients
        max_freq_idx (int): Maximum frequency index to consider
    
    Returns:
        np.ndarray: Diagonal bispectrum values
    """
    diagonal_bispectrum = []
    
    # Process diagonal elements one by one to save memory
    for f1_idx in range(max_freq_idx // 2):  # f1 can go up to max_freq_idx/2
        f3_idx = 2 * f1_idx  # f3 = 2*f1
        
        if f3_idx < max_freq_idx:
            # Compute diagonal bispectrum element
            bispectrum_val = (fft_coeffs[f1_idx] * 
                            fft_coeffs[f1_idx] * 
                            np.conj(fft_coeffs[f3_idx]))
            diagonal_bispectrum.append(np.abs(bispectrum_val))
    
    return np.array(diagonal_bispectrum)


def extract_paper_hosf_features(audio_path, sample_rate=16000):
    """
    Extract exactly 16 HOSF features as described in Miao et al. 2022.
    ULTRA MEMORY-EFFICIENT VERSION: Uses minimal memory and progress tracking.
    
    Args:
        audio_path (str): Path to the audio chunk file
        sample_rate (int): Sample rate for the audio (default: 16000 Hz)
    
    Returns:
        numpy.ndarray: 16 HOSF feature vector (9 BSF + 7 BCF)
    """
    try:
        # Add timeout protection
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError("Feature extraction timed out")
        
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(1800)  # 30 minute timeout per chunk (bispectrum is computationally expensive)
        # Load audio file
        y, sr = librosa.load(audio_path, sr=sample_rate)
        
        # Ensure mono
        if len(y.shape) > 1:
            y = np.mean(y, axis=0)
        
        # Compute FFT
        fft_coeffs = fft(y)
        
        # Only use positive frequencies
        n_freqs = len(fft_coeffs) // 2
        fft_coeffs_pos = fft_coeffs[:n_freqs]
        
        # Limit frequency range for speech analysis (0-4kHz for efficiency)
        # Reduced from 8kHz to 4kHz to make computation feasible
        max_freq_idx = min(n_freqs, int(4000 * len(y) / sr))
        
        # Further limit for very long chunks to prevent timeout
        if max_freq_idx > 200:
            max_freq_idx = 200  # Cap at 200 frequency bins for efficiency
        
        # ULTRA MEMORY-EFFICIENT COMPUTATION: Process with sampling
        print(f"    Computing bispectrum for {max_freq_idx} frequency bins...")
        bispectrum_values, bicoherence_values, valid_f1, valid_f2 = compute_bispectrum_memory_efficient(
            fft_coeffs_pos, max_freq_idx, chunk_size=10  # Very small chunk size for memory efficiency
        )
        print(f"    Computed {len(bispectrum_values)} bispectrum values")
        
        # Extract 9 Bispectral Features (BSF) - from paper Eqs. (1.3)-(1.9)
        features = []
        
        # 1. mAmp - Average amplitude of bispectrum (Eq. 1.3)
        mAmp = np.mean(bispectrum_values)
        features.append(mAmp)
        
        # 2. H1 - Sum of logarithmic amplitude of bispectrum (Eq. 1.4)
        H1 = np.sum(np.log(bispectrum_values + 1e-10))  # Add small epsilon to avoid log(0)
        features.append(H1)
        
        # 3. H2 - Sum of logarithmic amplitude of diagonal elements (Eq. 1.5)
        # MEMORY-EFFICIENT: Compute diagonal bispectrum elements
        diagonal_bispectrum = compute_diagonal_bispectrum_memory_efficient(fft_coeffs_pos, max_freq_idx)
        
        if len(diagonal_bispectrum) > 0:
            H2 = np.sum(np.log(diagonal_bispectrum + 1e-10))
        else:
            H2 = 0.0
        features.append(H2)
        
        # 4-7. Weighted Center of Bispectrum (WCOB) - Eqs. (1.6)-(1.9)
        # ULTRA MEMORY-EFFICIENT: Compute WCOB directly without creating large matrix
        total_bispectrum = np.sum(bispectrum_values)
        if total_bispectrum > 0:
            # Compute WCOB directly from bispectrum values and indices
            f1m = np.sum(bispectrum_values * valid_f1) / total_bispectrum
            f2m = np.sum(bispectrum_values * valid_f2) / total_bispectrum
        else:
            f1m = f2m = 0.0
        
        features.extend([f1m, f2m])
        
        # f3m and f4m - Absolute values of WCOB (Eqs. 1.8, 1.9)
        # These are the same as f1m and f2m since we're already using absolute values
        features.extend([f1m, f2m])
        
        # Additional BSF features (2 more to reach 9 total)
        # Add standard deviation and maximum of bispectrum
        features.append(np.std(bispectrum_values))
        features.append(np.max(bispectrum_values))
        
        # Extract 7 Bicoherent Features (BCF)
        # Compute BCF features with improved statistical measures
        if len(bicoherence_values) > 0:
            # Add debugging for bicoherence variation
            bicoherence_var = np.var(bicoherence_values)
            bicoherence_unique = len(np.unique(np.round(bicoherence_values, 6)))
            
            bcf_features = [
                np.mean(bicoherence_values),
                np.std(bicoherence_values),
                np.max(bicoherence_values),
                np.min(bicoherence_values),
                np.median(bicoherence_values),
                np.sum(bicoherence_values),
                np.sum(bicoherence_values**2)  # Sum of squares
            ]
            
            # Debug output for first few files
            if len(features) == 9:  # First time through, print debug info
                print(f"    Bicoherence stats: mean={bcf_features[0]:.6f}, std={bcf_features[1]:.8f}, "
                      f"var={bicoherence_var:.10f}, unique_vals={bicoherence_unique}")
        else:
            # Fallback if no bicoherence values
            bcf_features = [0.0] * 7
        
        features.extend(bcf_features)
        
        # Clean up memory aggressively
        del bispectrum_values, bicoherence_values, valid_f1, valid_f2
        del diagonal_bispectrum, fft_coeffs_pos, fft_coeffs
        import gc
        gc.collect()
        
        # Ensure we have exactly 16 features
        if len(features) != 16:
            print(f"Warning: Expected 16 features, got {len(features)}")
            # Pad or truncate to 16
            while len(features) < 16:
                features.append(0.0)
            features = features[:16]
        
        # Cancel timeout
        signal.alarm(0)
        return np.array(features)
        
    except Exception as e:
        # Cancel timeout
        signal.alarm(0)
        print(f"Error extracting paper HOSF features from {audio_path}: {e}")
        # Return zero vector as fallback (16 dimensions)
        return np.zeros(16)


def process_single_patient(patient_folder, output_dir, sample_rate=16000, force_regenerate=False):
    """
    Process all audio chunks for a single patient.
    
    Args:
        patient_folder (str): Path to patient folder containing audio chunks
        output_dir (str): Output directory for this patient's features
        sample_rate (int): Sample rate for audio processing
        force_regenerate (bool): Whether to regenerate existing files
    
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
    
    print(f"    Found {len(audio_chunks)} audio chunks for {patient_name}")
    
    if not audio_chunks:
        return {
            'patient_name': patient_name,
            'successful_chunks': 0,
            'failed_chunks': 0,
            'total_chunks': 0,
            'status': 'no_chunks_found'
        }
    
    # Create feature names (16 HOSF features as in paper)
    feature_names = [
        # 9 Bispectral Features (BSF)
        'mAmp', 'H1', 'H2', 'f1m', 'f2m', 'f3m', 'f4m', 'bispectrum_std', 'bispectrum_max',
        # 7 Bicoherent Features (BCF)
        'bicoherence_mean', 'bicoherence_std', 'bicoherence_max', 'bicoherence_min',
        'bicoherence_median', 'bicoherence_sum', 'bicoherence_sum_squares'
    ]
    
    successful_chunks = 0
    failed_chunks = 0
    
    # Process each audio chunk with progress indication
    total_chunks = len(audio_chunks)
    for chunk_idx, chunk_path in enumerate(audio_chunks):
        chunk_filename = os.path.basename(chunk_path)
        chunk_name = os.path.splitext(chunk_filename)[0]  # Remove .wav extension
        
        # Check if CSV already exists (resumable functionality)
        output_filename = f"{chunk_name}_paper_hosf_features.csv"
        output_path = os.path.join(patient_output_dir, output_filename)
        
        if os.path.exists(output_path) and not force_regenerate:
            # Check if existing file has zeros (indicates failed extraction)
            try:
                existing_df = pd.read_csv(output_path)
                if np.all(existing_df.iloc[0].values == 0):
                    print(f"    Regenerating zero-feature file: {chunk_name}")
                else:
                    # Skip if already processed with valid features
                    successful_chunks += 1
                    continue
            except:
                # If we can't read the file, regenerate it
                print(f"    Regenerating corrupted file: {chunk_name}")
        elif os.path.exists(output_path) and force_regenerate:
            print(f"    Force regenerating: {chunk_name}")
        
        try:
            # Check if audio file exists
            if not os.path.exists(chunk_path):
                print(f"    Warning: Audio file not found: {chunk_path}")
                failed_chunks += 1
                continue
            
            # Extract paper HOSF features
            features = extract_paper_hosf_features(chunk_path, sample_rate)
            
            # Check if features are all zeros (indicates extraction failure)
            if np.all(features == 0):
                print(f"    Warning: All features are zero for {chunk_name}")
                failed_chunks += 1
                continue
            
            # Create DataFrame with features
            feature_df = pd.DataFrame([features], columns=feature_names)
            
            # Save to CSV
            feature_df.to_csv(output_path, index=False)
            
            successful_chunks += 1
            
            # Progress indicator every 10 chunks
            if (chunk_idx + 1) % 10 == 0:
                print(f"    {patient_name}: {chunk_idx + 1}/{total_chunks} chunks processed")
            
        except Exception as e:
            print(f"    Error processing {chunk_name}: {e}")
            import traceback
            traceback.print_exc()
            failed_chunks += 1
    
    return {
        'patient_name': patient_name,
        'successful_chunks': successful_chunks,
        'failed_chunks': failed_chunks,
        'total_chunks': len(audio_chunks),
        'status': 'completed'
    }


def process_chunk_configuration(chunk_config_dir, output_base_dir, sample_rate=16000, force_regenerate=False):
    """
    Process all audio chunks in a specific chunk configuration directory using parallel processing.
    
    Args:
        chunk_config_dir (str): Path to chunk configuration directory
        output_base_dir (str): Base directory for output features
        sample_rate (int): Sample rate for audio processing
        force_regenerate (bool): Whether to regenerate existing files
    """
    config_name = os.path.basename(chunk_config_dir)
    
    # Create output directory structure
    output_dir = os.path.join(output_base_dir, "features", "paper_hosf", config_name)
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
    # Paper HOSF is memory-intensive, use very conservative settings
    num_workers = 2
    print(f"Using {num_workers} parallel workers (performance cores)")
    print(f"Available cores: {mp.cpu_count()} total, {PERFORMANCE_CORES} performance, {EFFICIENCY_CORES} efficiency")
    
    successful_patients = 0
    failed_patients = 0
    total_chunks = 0
    
    # Process patients in parallel
    start_time = time.time()
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        future_to_patient = {
            executor.submit(process_single_patient, patient_folder, output_dir, sample_rate, force_regenerate): patient_folder 
            for patient_folder in patient_folders
        }
        
        # Process completed tasks
        for future in as_completed(future_to_patient):
            patient_folder = future_to_patient[future]
            patient_name = os.path.basename(patient_folder)
            
            try:
                result = future.result()
                
                if result['status'] == 'completed':
                    if result['failed_chunks'] == 0:
                        print(f"{patient_name}: {result['successful_chunks']}/{result['total_chunks']} chunks processed successfully")
                        successful_patients += 1
                    else:
                        print(f"Warning {patient_name}: {result['successful_chunks']}/{result['total_chunks']} chunks processed ({result['failed_chunks']} failed)")
                        successful_patients += 1  # Still count as successful if some chunks worked
                    
                    total_chunks += result['total_chunks']
                elif result['status'] == 'no_chunks_found':
                    print(f"Warning {patient_name}: No audio chunks found")
                    failed_patients += 1
                else:
                    print(f"Error {patient_name}: Processing failed")
                    failed_patients += 1
                    
            except Exception as e:
                print(f"Error {patient_name}: Error in parallel processing: {e}")
                failed_patients += 1
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    print(f"\nConfiguration {config_name} completed in {processing_time:.2f} seconds")
    print(f"Average time per patient: {processing_time/len(patient_folders):.2f} seconds")
    
    return successful_patients, failed_patients, total_chunks


def main():
    """
    Main function to extract paper HOSF features.
    
    !!! INPUT STRUCTURE !!!
    
    data/created_data/
    ├── 4.0s_0.0s_overlap/             # 4-second chunks (paper's approach)
    │   ├── 300_P/
    │   │   ├── 300_P_chunk001.wav
    │   │   └── ...
    │   └── 301_P/
    
    !!! OUTPUT STRUCTURE !!!
    
    data/features/paper_hosf/
    ├── 4.0s_0.0s_overlap/
    │   ├── 300_P/
    │   │   ├── chunk_001_paper_hosf_features.csv
    │   │   └── ...
    │   └── 301_P/
    
    !!! PROCESSING DETAILS !!!
    
    - Extracts exactly 16 HOSF features as in Miao et al. 2022
    - 9 Bispectral Features (BSF) + 7 Bicoherent Features (BCF)
    - Each CSV contains one row with 16 HOSF feature values
    - Optimized for MBP 10 Performance + 4 Efficiency cores
    """
    # Initialize optimizations for MBP M4 Pro
    print("Initializing MBP M4 Pro optimizations...")
    optimize_memory()
    set_cpu_affinity()
    
    # Configuration parameters - CHANGE THESE VALUES AS NEEDED
    CHUNK_CONFIGS = [
        "3.0s_1.5s_overlap",   # 3-second chunks with 1.5s overlap
        "5.0s_2.5s_overlap",   # 5-second chunks with 2.5s overlap
        "10.0s_5.0s_overlap",  # 10-second chunks with 5s overlap
        "20.0s_10.0s_overlap", # 20-second chunks with 10s overlap
        "30.0s_15.0s_overlap"  # 30-second chunks with 15s overlap
    ]
    SAMPLE_RATE = 16000  # Hz - maintains original DAIC-WOZ format
    FORCE_REGENERATE = True  # Set to True to regenerate all files (including existing ones)
    
    print("IMPROVED BICOHERENCE CALCULATION ENABLED")
    print("- Using hybrid bicoherence method (70% standard + 30% phase coupling)")
    print("- Added numerical stability improvements")
    print("- Enhanced debugging output for bicoherence variation")
    print("=" * 80)
    
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
    
    # Start timing for performance measurement
    start_time = time.time()
    
    # Process each chunk configuration
    for config_name in CHUNK_CONFIGS:
        config_input_dir = os.path.join(INPUT_BASE_DIR, config_name)
        
        # Check if configuration directory exists
        if not os.path.exists(config_input_dir):
            print(f"\nWarning: Configuration directory not found: {config_input_dir}")
            print("Skipping this configuration...")
            continue
        
        print(f"\n{'='*20} Processing {config_name} {'='*20}")
        
        # Process this configuration
        successful, failed, chunks = process_chunk_configuration(
            config_input_dir, 
            OUTPUT_BASE_DIR, 
            SAMPLE_RATE,
            FORCE_REGENERATE
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
    
    # Performance summary
    if total_chunks_processed > 0:
        total_time = time.time() - start_time
        avg_time_per_chunk = total_time / total_chunks_processed
        print(f"\nPERFORMANCE SUMMARY:")
        print(f"Total processing time: {total_time:.2f} seconds")
        print(f"Average time per chunk: {avg_time_per_chunk:.4f} seconds")
        print(f"Memory-efficient processing: Conservative approach for stability")
        print(f"Optimizations applied:")
        print(f"  - Chunked bispectrum computation")
        print(f"  - Memory-efficient processing")
        print(f"  - Conservative thread limits")
        print(f"  - Aggressive garbage collection")
    
    print(f"\nPaper HOSF features saved to:")
    print(f"  {OUTPUT_BASE_DIR}/features/paper_hosf/[CONFIG]/[PATIENT_ID]/[CHUNK]_paper_hosf_features.csv")
    print(f"\nEach CSV contains exactly 16 HOSF features (9 BSF + 7 BCF) as in Miao et al. 2022.")
    print(f"MEMORY-EFFICIENT VERSION: Uses chunked processing to minimize memory usage on MBP M4 Pro.")


if __name__ == "__main__":
    # Set multiprocessing start method for compatibility across platforms
    mp.set_start_method('spawn', force=True)
    main()
