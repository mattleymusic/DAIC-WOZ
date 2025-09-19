#!/usr/bin/env python3
"""
COVAREP-style Feature Extractor for DAIC-WOZ Dataset

This script extracts baseline acoustic features similar to COVAREP (Covariance Matrix REpresentation)
from audio chunks. It processes all chunk configurations in data/created_data/ and saves 
acoustic features to data/features/covarep/ following the same directory structure.

COVAREP features include:
- Fundamental frequency (F0) and related measures
- Formant frequencies and bandwidths
- Voice quality measures (jitter, shimmer, HNR)
- Spectral features (centroid, rolloff, flux, etc.)
- MFCC coefficients
- Energy and intensity measures

Usage:
    python src/feature_extraction/covarep_extractor.py
    or
    ./src/feature_extraction/covarep_extractor.py (if made executable)

Author: Based on COVAREP specifications and DAIC-WOZ requirements
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
    """Optimize memory usage for large-scale processing."""
    try:
        # Set numpy to use fewer threads to avoid oversubscription
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['MKL_NUM_THREADS'] = '1'
        os.environ['NUMEXPR_NUM_THREADS'] = '1'
        
        # Configure librosa for better memory usage
        librosa.set_cache(False)  # Disable caching to save memory
        
    except Exception as e:
        print(f"Warning: Could not optimize memory settings: {e}")


def extract_f0_features(y, sr, frame_length=2048, hop_length=512):
    """
    Extract fundamental frequency (F0) related features.
    
    Args:
        y (np.ndarray): Audio time series
        sr (int): Sample rate
        frame_length (int): Frame length for analysis
        hop_length (int): Hop length for analysis
    
    Returns:
        dict: F0-related features
    """
    features = {}
    
    # Extract F0 using librosa
    f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), 
                                                fmax=librosa.note_to_hz('C7'), 
                                                frame_length=frame_length, 
                                                hop_length=hop_length)
    
    # Remove NaN values
    f0_clean = f0[~np.isnan(f0)]
    
    if len(f0_clean) > 0:
        features['f0_mean'] = np.mean(f0_clean)
        features['f0_std'] = np.std(f0_clean)
        features['f0_min'] = np.min(f0_clean)
        features['f0_max'] = np.max(f0_clean)
        features['f0_range'] = features['f0_max'] - features['f0_min']
        features['f0_median'] = np.median(f0_clean)
        features['f0_skewness'] = skew(f0_clean)
        features['f0_kurtosis'] = kurtosis(f0_clean)
        
        # F0 slope features
        f0_diff = np.diff(f0_clean)
        features['f0_slope_mean'] = np.mean(f0_diff)
        features['f0_slope_std'] = np.std(f0_diff)
        
        # Voicing features
        features['voicing_rate'] = np.mean(voiced_flag)
        features['voicing_std'] = np.std(voiced_flag)
    else:
        # Default values if no F0 detected
        features.update({
            'f0_mean': 0.0, 'f0_std': 0.0, 'f0_min': 0.0, 'f0_max': 0.0,
            'f0_range': 0.0, 'f0_median': 0.0, 'f0_skewness': 0.0, 'f0_kurtosis': 0.0,
            'f0_slope_mean': 0.0, 'f0_slope_std': 0.0, 'voicing_rate': 0.0, 'voicing_std': 0.0
        })
    
    return features


def extract_formant_features(y, sr, frame_length=2048, hop_length=512):
    """
    Extract formant frequency and bandwidth features using spectral peaks.
    
    Args:
        y (np.ndarray): Audio time series
        sr (int): Sample rate
        frame_length (int): Frame length for analysis
        hop_length (int): Hop length for analysis
    
    Returns:
        dict: Formant-related features
    """
    features = {}
    
    try:
        # Extract spectral features and find peaks as formant approximations
        stft = librosa.stft(y, n_fft=frame_length, hop_length=hop_length)
        magnitude = np.abs(stft)
        
        # Get frequency bins
        freqs = librosa.fft_frequencies(sr=sr, n_fft=frame_length)
        
        # Find spectral peaks for each frame (approximate formants)
        formant_candidates = []
        for frame_idx in range(magnitude.shape[1]):
            frame_magnitude = magnitude[:, frame_idx]
            
            # Find peaks in the spectrum
            peaks, _ = signal.find_peaks(frame_magnitude, height=np.max(frame_magnitude) * 0.1)
            
            # Convert peak indices to frequencies
            peak_freqs = freqs[peaks]
            
            # Sort by magnitude and take top 4
            peak_magnitudes = frame_magnitude[peaks]
            sorted_indices = np.argsort(peak_magnitudes)[::-1]
            top_4_peaks = peak_freqs[sorted_indices[:4]]
            
            formant_candidates.append(top_4_peaks)
        
        # Convert to numpy array and pad if necessary
        max_formants = 4
        formant_matrix = np.full((max_formants, len(formant_candidates)), np.nan)
        
        for i, candidates in enumerate(formant_candidates):
            for j, freq in enumerate(candidates[:max_formants]):
                formant_matrix[j, i] = freq
        
        # Extract first 4 formants
        for i in range(max_formants):
            formant_vals = formant_matrix[i, :]
            formant_clean = formant_vals[~np.isnan(formant_vals)]
            
            if len(formant_clean) > 0:
                features[f'f{i+1}_mean'] = np.mean(formant_clean)
                features[f'f{i+1}_std'] = np.std(formant_clean)
                features[f'f{i+1}_min'] = np.min(formant_clean)
                features[f'f{i+1}_max'] = np.max(formant_clean)
                features[f'f{i+1}_range'] = features[f'f{i+1}_max'] - features[f'f{i+1}_min']
            else:
                features.update({
                    f'f{i+1}_mean': 0.0, f'f{i+1}_std': 0.0, f'f{i+1}_min': 0.0,
                    f'f{i+1}_max': 0.0, f'f{i+1}_range': 0.0
                })
        
        # Formant dispersion (F2-F1)
        if 'f2_mean' in features and 'f1_mean' in features:
            features['formant_dispersion'] = features['f2_mean'] - features['f1_mean']
        else:
            features['formant_dispersion'] = 0.0
            
    except Exception as e:
        print(f"Warning: Formant extraction failed: {e}")
        # Set default values
        for i in range(4):
            features.update({
                f'f{i+1}_mean': 0.0, f'f{i+1}_std': 0.0, f'f{i+1}_min': 0.0,
                f'f{i+1}_max': 0.0, f'f{i+1}_range': 0.0
            })
        features['formant_dispersion'] = 0.0
    
    return features


def extract_voice_quality_features(y, sr, frame_length=2048, hop_length=512):
    """
    Extract voice quality features (jitter, shimmer, HNR).
    
    Args:
        y (np.ndarray): Audio time series
        sr (int): Sample rate
        frame_length (int): Frame length for analysis
        hop_length (int): Hop length for analysis
    
    Returns:
        dict: Voice quality features
    """
    features = {}
    
    # Extract F0 for jitter calculation
    f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), 
                                                fmax=librosa.note_to_hz('C7'), 
                                                frame_length=frame_length, 
                                                hop_length=hop_length)
    
    f0_clean = f0[~np.isnan(f0)]
    
    if len(f0_clean) > 1:
        # Jitter (period-to-period variation in F0)
        jitter_values = np.abs(np.diff(f0_clean))
        features['jitter_mean'] = np.mean(jitter_values)
        features['jitter_std'] = np.std(jitter_values)
        features['jitter_rel'] = features['jitter_mean'] / features['f0_mean'] if features.get('f0_mean', 0) > 0 else 0.0
    else:
        features.update({'jitter_mean': 0.0, 'jitter_std': 0.0, 'jitter_rel': 0.0})
    
    # Extract amplitude envelope for shimmer
    amplitude_envelope = librosa.amplitude_to_db(np.abs(librosa.stft(y, hop_length=hop_length)))
    amplitude_mean = np.mean(amplitude_envelope, axis=0)
    
    if len(amplitude_mean) > 1:
        # Shimmer (amplitude variation)
        shimmer_values = np.abs(np.diff(amplitude_mean))
        features['shimmer_mean'] = np.mean(shimmer_values)
        features['shimmer_std'] = np.std(shimmer_values)
        features['shimmer_rel'] = features['shimmer_mean'] / np.mean(amplitude_mean) if np.mean(amplitude_mean) > 0 else 0.0
    else:
        features.update({'shimmer_mean': 0.0, 'shimmer_std': 0.0, 'shimmer_rel': 0.0})
    
    # Harmonic-to-Noise Ratio (HNR)
    try:
        hnr = librosa.effects.harmonic(y)
        noise = librosa.effects.percussive(y)
        if np.mean(noise**2) > 0:
            features['hnr_db'] = 10 * np.log10(np.mean(hnr**2) / np.mean(noise**2))
        else:
            features['hnr_db'] = 0.0
    except:
        features['hnr_db'] = 0.0
    
    return features


def extract_spectral_features(y, sr, frame_length=2048, hop_length=512):
    """
    Extract spectral features (centroid, rolloff, flux, etc.).
    
    Args:
        y (np.ndarray): Audio time series
        sr (int): Sample rate
        frame_length (int): Frame length for analysis
        hop_length (int): Hop length for analysis
    
    Returns:
        dict: Spectral features
    """
    features = {}
    
    # Spectral centroid
    spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
    features['spectral_centroid_mean'] = np.mean(spectral_centroids)
    features['spectral_centroid_std'] = np.std(spectral_centroids)
    
    # Spectral rolloff
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=hop_length)[0]
    features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
    features['spectral_rolloff_std'] = np.std(spectral_rolloff)
    
    # Spectral flux
    spectral_flux = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    features['spectral_flux_mean'] = np.mean(spectral_flux)
    features['spectral_flux_std'] = np.std(spectral_flux)
    
    # Spectral flatness
    spectral_flatness = librosa.feature.spectral_flatness(y=y, hop_length=hop_length)[0]
    features['spectral_flatness_mean'] = np.mean(spectral_flatness)
    features['spectral_flatness_std'] = np.std(spectral_flatness)
    
    # Spectral bandwidth
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=hop_length)[0]
    features['spectral_bandwidth_mean'] = np.mean(spectral_bandwidth)
    features['spectral_bandwidth_std'] = np.std(spectral_bandwidth)
    
    # Zero crossing rate
    zcr = librosa.feature.zero_crossing_rate(y, hop_length=hop_length)[0]
    features['zcr_mean'] = np.mean(zcr)
    features['zcr_std'] = np.std(zcr)
    
    return features


def extract_mfcc_features(y, sr, frame_length=2048, hop_length=512, n_mfcc=13):
    """
    Extract MFCC features.
    
    Args:
        y (np.ndarray): Audio time series
        sr (int): Sample rate
        frame_length (int): Frame length for analysis
        hop_length (int): Hop length for analysis
        n_mfcc (int): Number of MFCC coefficients
    
    Returns:
        dict: MFCC features
    """
    features = {}
    
    # Extract MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
    
    for i in range(n_mfcc):
        features[f'mfcc_{i+1}_mean'] = np.mean(mfccs[i])
        features[f'mfcc_{i+1}_std'] = np.std(mfccs[i])
        features[f'mfcc_{i+1}_min'] = np.min(mfccs[i])
        features[f'mfcc_{i+1}_max'] = np.max(mfccs[i])
        features[f'mfcc_{i+1}_range'] = features[f'mfcc_{i+1}_max'] - features[f'mfcc_{i+1}_min']
    
    return features


def extract_energy_features(y, sr, frame_length=2048, hop_length=512):
    """
    Extract energy and intensity features.
    
    Args:
        y (np.ndarray): Audio time series
        sr (int): Sample rate
        frame_length (int): Frame length for analysis
        hop_length (int): Hop length for analysis
    
    Returns:
        dict: Energy features
    """
    features = {}
    
    # RMS energy
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    features['rms_mean'] = np.mean(rms)
    features['rms_std'] = np.std(rms)
    features['rms_min'] = np.min(rms)
    features['rms_max'] = np.max(rms)
    features['rms_range'] = features['rms_max'] - features['rms_min']
    
    # Energy
    energy = np.sum(y**2)
    features['total_energy'] = energy
    features['energy_per_sample'] = energy / len(y)
    
    # Intensity (log energy)
    features['log_energy'] = np.log(energy + 1e-10)  # Add small value to avoid log(0)
    
    return features


def extract_covarep_features(audio_path, sample_rate=16000):
    """
    Extract COVAREP-style features from a single audio chunk.
    
    Args:
        audio_path (str): Path to the audio chunk file
        sample_rate (int): Sample rate for the audio (default: 16000 Hz)
    
    Returns:
        numpy.ndarray: COVAREP feature vector
    """
    try:
        # Load audio file
        y, sr = librosa.load(audio_path, sr=sample_rate)
        
        # Ensure mono
        if len(y.shape) > 1:
            y = np.mean(y, axis=0)
        
        # Frame parameters
        frame_length = 2048
        hop_length = 512
        
        # Extract all feature groups
        features = {}
        
        # F0 features
        f0_features = extract_f0_features(y, sr, frame_length, hop_length)
        features.update(f0_features)
        
        # Formant features
        formant_features = extract_formant_features(y, sr, frame_length, hop_length)
        features.update(formant_features)
        
        # Voice quality features
        voice_quality_features = extract_voice_quality_features(y, sr, frame_length, hop_length)
        features.update(voice_quality_features)
        
        # Spectral features
        spectral_features = extract_spectral_features(y, sr, frame_length, hop_length)
        features.update(spectral_features)
        
        # MFCC features
        mfcc_features = extract_mfcc_features(y, sr, frame_length, hop_length)
        features.update(mfcc_features)
        
        # Energy features
        energy_features = extract_energy_features(y, sr, frame_length, hop_length)
        features.update(energy_features)
        
        # Convert to numpy array
        feature_vector = np.array(list(features.values()))
        
        return feature_vector
        
    except Exception as e:
        print(f"Error extracting COVAREP features from {audio_path}: {e}")
        # Return zero vector as fallback (exact number of features: 12+21+7+12+65+8=125)
        return np.zeros(125)  # Exact number of features


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
    
    # Create feature names dynamically based on actual feature extraction
    # This ensures the number of features matches the feature names
    feature_names = []
    
    # F0 features (12)
    feature_names.extend([
        'f0_mean', 'f0_std', 'f0_min', 'f0_max', 'f0_range', 'f0_median',
        'f0_skewness', 'f0_kurtosis', 'f0_slope_mean', 'f0_slope_std', 'voicing_rate', 'voicing_std'
    ])
    
    # Formant features (21)
    feature_names.extend([
        'f1_mean', 'f1_std', 'f1_min', 'f1_max', 'f1_range',
        'f2_mean', 'f2_std', 'f2_min', 'f2_max', 'f2_range',
        'f3_mean', 'f3_std', 'f3_min', 'f3_max', 'f3_range',
        'f4_mean', 'f4_std', 'f4_min', 'f4_max', 'f4_range',
        'formant_dispersion'
    ])
    
    # Voice quality features (7)
    feature_names.extend([
        'jitter_mean', 'jitter_std', 'jitter_rel', 'shimmer_mean', 'shimmer_std', 'shimmer_rel', 'hnr_db'
    ])
    
    # Spectral features (12)
    feature_names.extend([
        'spectral_centroid_mean', 'spectral_centroid_std', 'spectral_rolloff_mean', 'spectral_rolloff_std',
        'spectral_flux_mean', 'spectral_flux_std', 'spectral_flatness_mean', 'spectral_flatness_std',
        'spectral_bandwidth_mean', 'spectral_bandwidth_std', 'zcr_mean', 'zcr_std'
    ])
    
    # MFCC features (65)
    for i in range(1, 14):  # 13 MFCC coefficients
        feature_names.extend([
            f'mfcc_{i}_mean', f'mfcc_{i}_std', f'mfcc_{i}_min', f'mfcc_{i}_max', f'mfcc_{i}_range'
        ])
    
    # Energy features (8)
    feature_names.extend([
        'rms_mean', 'rms_std', 'rms_min', 'rms_max', 'rms_range', 'total_energy', 'energy_per_sample', 'log_energy'
    ])
    
    successful_chunks = 0
    failed_chunks = 0
    
    # Process each audio chunk
    for chunk_path in audio_chunks:
        chunk_filename = os.path.basename(chunk_path)
        chunk_name = os.path.splitext(chunk_filename)[0]  # Remove .wav extension
        
        # Check if CSV already exists (resumable functionality)
        output_filename = f"{chunk_name}_covarep_features.csv"
        output_path = os.path.join(patient_output_dir, output_filename)
        
        if os.path.exists(output_path):
            # Skip if already processed
            successful_chunks += 1
            continue
        
        try:
            # Extract COVAREP features
            features = extract_covarep_features(chunk_path, sample_rate)
            
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
        chunk_config_dir (str): Path to chunk configuration directory (e.g., 30.0s_15.0s_overlap)
        output_base_dir (str): Base directory for output features
        sample_rate (int): Sample rate for audio processing
    """
    config_name = os.path.basename(chunk_config_dir)
    
    # Create output directory structure
    output_dir = os.path.join(output_base_dir, "features", "covarep", config_name)
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
    # Use performance cores for CPU-intensive COVAREP feature extraction
    num_workers = min(PERFORMANCE_CORES, len(patient_folders), 10)  # Use up to 10 performance cores
    print(f"Using {num_workers} parallel workers (performance cores)")
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
    
    data/features/covarep/4.0s_0.0s_overlap/
    ├── 300_P/
    │   ├── chunk_001_covarep_features.csv
    │   ├── chunk_001_covarep_features.csv
    │   └── ...
    └── 301_P/
    
    !!! PROCESSING DETAILS !!!
    
    - Processes 4-second chunks created by paper_audio_chunker.py
    - Extracts COVAREP-style features from each audio chunk
    - Saves individual CSV files per chunk with 125 acoustic features
    - Maintains the same directory structure as input chunks
    - Each CSV contains one row with COVAREP feature values
    - Optimized for MBP 10 Performance + 4 Efficiency cores
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
    print(f"\nCOVAREP features saved to:")
    print(f"  {OUTPUT_BASE_DIR}/features/covarep/{CHUNK_CONFIG}/[PATIENT_ID]/[CHUNK]_features.csv")
    print(f"\nEach CSV contains 125 COVAREP-style acoustic features for one 4-second audio chunk.")


if __name__ == "__main__":
    # Set multiprocessing start method for compatibility across platforms
    mp.set_start_method('spawn', force=True)
    main()
