#!/usr/bin/env python3
"""
Higher-Order Spectral Features (HOSF) Extractor for DAIC-WOZ Dataset

This script extracts Higher-Order Spectral Features from audio chunks, including:
- Bispectrum computation using triple product of FFT coefficients
- Bicoherence normalization
- Scalar descriptors (mean, entropy, skewness, flatness, kurtosis, etc.)
- Frame-level analysis with configurable window size and overlap

The HOSF features capture non-linear interactions and phase coupling between 
different frequency components in speech signals.

Usage:
    python src/feature_extraction/hosf_extractor.py
    or
    ./src/feature_extraction/hosf_extractor.py (if made executable)

Author: Based on HOSF specifications and DAIC-WOZ requirements
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
from scipy.stats import skew, kurtosis, entropy
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
SHARED_MEMORY_GB = 48   # M4 Pro shared memory

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
    """Optimize memory usage for MBP M4 Pro large-scale processing."""
    try:
        # Optimize for M4 Pro shared memory architecture
        os.environ['OMP_NUM_THREADS'] = str(PERFORMANCE_CORES)
        os.environ['MKL_NUM_THREADS'] = str(PERFORMANCE_CORES)
        os.environ['NUMEXPR_NUM_THREADS'] = str(PERFORMANCE_CORES)
        
        # Optimize for M4 Pro shared memory architecture
        os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'  # Use MPS efficiently
        
        # Configure librosa for better memory usage
        librosa.set_cache(False)  # Disable caching to save memory
        
        # Configure for large shared memory
        import gc
        gc.set_threshold(700, 10, 10)  # More aggressive garbage collection for large datasets
        
        # Set NumPy to ignore numerical warnings for cleaner output
        np.seterr(all='ignore')
        
        print(f"Memory optimization configured for MBP M4 Pro ({PERFORMANCE_CORES}P+{EFFICIENCY_CORES}E cores, {SHARED_MEMORY_GB}GB shared memory)")
        
    except Exception as e:
        print(f"Warning: Could not optimize memory settings: {e}")


def compute_bispectrum(fft_coeffs, f1_idx, f2_idx):
    """
    Compute bispectrum B(f1, f2) using triple product of FFT coefficients.
    
    Args:
        fft_coeffs (np.ndarray): FFT coefficients
        f1_idx (int): Frequency index 1
        f2_idx (int): Frequency index 2
    
    Returns:
        complex: Bispectrum value
    """
    f3_idx = f1_idx + f2_idx
    
    # Check bounds
    if f3_idx >= len(fft_coeffs):
        return 0.0
    
    # Triple product: B(f1, f2) = F(f1) * F(f2) * F*(f1 + f2)
    bispectrum = fft_coeffs[f1_idx] * fft_coeffs[f2_idx] * np.conj(fft_coeffs[f3_idx])
    
    return bispectrum


def compute_bicoherence(fft_coeffs, f1_idx, f2_idx):
    """
    Compute normalized bicoherence (magnitude of bispectrum normalized by power).
    
    Args:
        fft_coeffs (np.ndarray): FFT coefficients
        f1_idx (int): Frequency index 1
        f2_idx (int): Frequency index 2
    
    Returns:
        float: Bicoherence value
    """
    f3_idx = f1_idx + f2_idx
    
    # Check bounds
    if f3_idx >= len(fft_coeffs):
        return 0.0
    
    # Compute bispectrum
    bispectrum = compute_bispectrum(fft_coeffs, f1_idx, f2_idx)
    
    # Compute normalization factor
    power_f1 = np.abs(fft_coeffs[f1_idx])**2
    power_f2 = np.abs(fft_coeffs[f2_idx])**2
    power_f3 = np.abs(fft_coeffs[f3_idx])**2
    
    # Avoid division by zero
    if power_f1 * power_f2 * power_f3 == 0:
        return 0.0
    
    # Bicoherence = |B(f1, f2)| / sqrt(P(f1) * P(f2) * P(f1 + f2))
    bicoherence = np.abs(bispectrum) / np.sqrt(power_f1 * power_f2 * power_f3)
    
    return bicoherence


def extract_hosf_features_frame(frame, sr, frame_length_ms=20, overlap_ratio=0.5):
    """
    Extract HOSF features from a single frame using optimized vectorized operations.
    
    Args:
        frame (np.ndarray): Audio frame
        sr (int): Sample rate
        frame_length_ms (int): Frame length in milliseconds
        overlap_ratio (float): Overlap ratio (0.5 = 50% overlap)
    
    Returns:
        dict: HOSF features for this frame
    """
    features = {}
    
    # Compute FFT
    fft_coeffs = fft(frame)
    fft_magnitude = np.abs(fft_coeffs)
    fft_phase = np.angle(fft_coeffs)
    
    # Frequency resolution
    freqs = fftfreq(len(frame), 1/sr)
    n_freqs = len(fft_coeffs) // 2  # Only use positive frequencies
    
    # Limit frequency range for speech analysis (0-8kHz)
    max_freq_idx = min(n_freqs, int(8000 * len(frame) / sr))
    
    # VECTORIZED COMPUTATION: Compute bispectrum matrix efficiently
    # Create frequency index grids
    f1_indices = np.arange(max_freq_idx).reshape(-1, 1)
    f2_indices = np.arange(max_freq_idx).reshape(1, -1)
    f3_indices = f1_indices + f2_indices
    
    # Create mask for valid indices (f3 < len(fft_coeffs))
    valid_mask = f3_indices < len(fft_coeffs)
    
    # Vectorized bispectrum computation: B(f1, f2) = F(f1) * F(f2) * F*(f1 + f2)
    bispectrum_matrix = np.zeros((max_freq_idx, max_freq_idx), dtype=complex)
    
    # Use broadcasting for efficient computation
    f1_coeffs = fft_coeffs[f1_indices]  # Shape: (max_freq_idx, max_freq_idx)
    f2_coeffs = fft_coeffs[f2_indices]  # Shape: (max_freq_idx, max_freq_idx)
    f3_coeffs = fft_coeffs[f3_indices]  # Shape: (max_freq_idx, max_freq_idx)
    
    # Compute bispectrum for all valid positions
    bispectrum_matrix = f1_coeffs * f2_coeffs * np.conj(f3_coeffs)
    
    # Set invalid entries to zero
    bispectrum_matrix = np.where(valid_mask, bispectrum_matrix, 0.0)
    
    # Vectorized bicoherence computation (normalized bispectrum)
    # Compute power spectrum for normalization
    power_spectrum = np.abs(fft_coeffs) ** 2
    bicoherence_matrix = np.zeros((max_freq_idx, max_freq_idx))
    
    # Use broadcasting for power spectrum computation
    power_f1 = power_spectrum[f1_indices]  # Shape: (max_freq_idx, max_freq_idx)
    power_f2 = power_spectrum[f2_indices]  # Shape: (max_freq_idx, max_freq_idx)
    power_f3 = power_spectrum[f3_indices]  # Shape: (max_freq_idx, max_freq_idx)
    
    # Compute denominator: sqrt(P(f1) * P(f2) * P(f1+f2))
    denominator = np.sqrt(power_f1 * power_f2 * power_f3)
    
    # Avoid division by zero
    valid_denom = denominator > 1e-10
    
    # Compute bicoherence where valid
    valid_bicoherence = valid_mask & valid_denom
    bicoherence_matrix[valid_bicoherence] = (
        np.abs(bispectrum_matrix[valid_bicoherence]) / 
        denominator[valid_bicoherence]
    )
    
    # Extract scalar descriptors from bispectrum
    bispectrum_magnitude = np.abs(bispectrum_matrix)
    
    # Bispectrum features
    features['bispectrum_mean'] = np.mean(bispectrum_magnitude)
    features['bispectrum_std'] = np.std(bispectrum_magnitude)
    features['bispectrum_max'] = np.max(bispectrum_magnitude)
    features['bispectrum_min'] = np.min(bispectrum_magnitude)
    features['bispectrum_range'] = features['bispectrum_max'] - features['bispectrum_min']
    features['bispectrum_skewness'] = skew(bispectrum_magnitude.flatten())
    features['bispectrum_kurtosis'] = kurtosis(bispectrum_magnitude.flatten())
    
    # Bispectrum entropy
    bispectrum_flat = bispectrum_magnitude.flatten()
    bispectrum_flat = bispectrum_flat[bispectrum_flat > 0]  # Remove zeros for entropy calculation
    if len(bispectrum_flat) > 0:
        bispectrum_prob = bispectrum_flat / np.sum(bispectrum_flat)
        features['bispectrum_entropy'] = entropy(bispectrum_prob)
    else:
        features['bispectrum_entropy'] = 0.0
    
    # Bicoherence features
    bicoherence_flat = bicoherence_matrix.flatten()
    bicoherence_flat = bicoherence_flat[bicoherence_flat > 0]  # Remove zeros
    
    features['bicoherence_mean'] = np.mean(bicoherence_flat) if len(bicoherence_flat) > 0 else 0.0
    features['bicoherence_std'] = np.std(bicoherence_flat) if len(bicoherence_flat) > 0 else 0.0
    features['bicoherence_max'] = np.max(bicoherence_flat) if len(bicoherence_flat) > 0 else 0.0
    features['bicoherence_min'] = np.min(bicoherence_flat) if len(bicoherence_flat) > 0 else 0.0
    features['bicoherence_range'] = features['bicoherence_max'] - features['bicoherence_min']
    features['bicoherence_skewness'] = skew(bicoherence_flat) if len(bicoherence_flat) > 0 else 0.0
    features['bicoherence_kurtosis'] = kurtosis(bicoherence_flat) if len(bicoherence_flat) > 0 else 0.0
    
    # Bicoherence entropy
    if len(bicoherence_flat) > 0:
        bicoherence_prob = bicoherence_flat / np.sum(bicoherence_flat)
        features['bicoherence_entropy'] = entropy(bicoherence_prob)
    else:
        features['bicoherence_entropy'] = 0.0
    
    # Spectral flatness of bispectrum
    if np.mean(bispectrum_magnitude) > 0:
        geometric_mean = np.exp(np.mean(np.log(bispectrum_magnitude[bispectrum_magnitude > 0])))
        arithmetic_mean = np.mean(bispectrum_magnitude)
        features['bispectrum_flatness'] = geometric_mean / arithmetic_mean
    else:
        features['bispectrum_flatness'] = 0.0
    
    # Spectral flatness of bicoherence
    if len(bicoherence_flat) > 0 and np.mean(bicoherence_flat) > 0:
        geometric_mean = np.exp(np.mean(np.log(bicoherence_flat)))
        arithmetic_mean = np.mean(bicoherence_flat)
        features['bicoherence_flatness'] = geometric_mean / arithmetic_mean
    else:
        features['bicoherence_flatness'] = 0.0
    
    # Phase coupling strength
    features['phase_coupling_strength'] = np.mean(bicoherence_matrix)
    
    # Diagonal features (f1 = f2)
    diagonal_bicoherence = np.diag(bicoherence_matrix)
    features['diagonal_bicoherence_mean'] = np.mean(diagonal_bicoherence)
    features['diagonal_bicoherence_std'] = np.std(diagonal_bicoherence)
    
    # Off-diagonal features (f1 != f2)
    mask = ~np.eye(bicoherence_matrix.shape[0], dtype=bool)
    off_diagonal_bicoherence = bicoherence_matrix[mask]
    features['off_diagonal_bicoherence_mean'] = np.mean(off_diagonal_bicoherence)
    features['off_diagonal_bicoherence_std'] = np.std(off_diagonal_bicoherence)
    
    return features


def extract_hosf_features_utterance(y, sr, frame_length_ms=20, overlap_ratio=0.5):
    """
    Extract HOSF features from an entire utterance by analyzing frames.
    
    Args:
        y (np.ndarray): Audio time series
        sr (int): Sample rate
        frame_length_ms (int): Frame length in milliseconds
        overlap_ratio (float): Overlap ratio (0.5 = 50% overlap)
    
    Returns:
        dict: Aggregated HOSF features for the utterance
    """
    # Convert frame length to samples
    frame_length_samples = int(frame_length_ms * sr / 1000)
    hop_length_samples = int(frame_length_samples * (1 - overlap_ratio))
    
    # Ensure minimum frame length
    if frame_length_samples < 64:
        frame_length_samples = 64
        hop_length_samples = int(frame_length_samples * (1 - overlap_ratio))
    
    # Extract frames
    frames = []
    for i in range(0, len(y) - frame_length_samples + 1, hop_length_samples):
        frame = y[i:i + frame_length_samples]
        frames.append(frame)
    
    if not frames:
        # If utterance is too short, pad it
        if len(y) < frame_length_samples:
            padded_y = np.pad(y, (0, frame_length_samples - len(y)), mode='constant')
            frames = [padded_y]
        else:
            frames = [y[:frame_length_samples]]
    
    # Extract features from each frame
    frame_features = []
    for frame in frames:
        frame_feat = extract_hosf_features_frame(frame, sr, frame_length_ms, overlap_ratio)
        frame_features.append(frame_feat)
    
    # Aggregate features across frames
    aggregated_features = {}
    
    for key in frame_features[0].keys():
        values = [feat[key] for feat in frame_features]
        
        # Statistical aggregations
        aggregated_features[f'{key}_mean'] = np.mean(values)
        aggregated_features[f'{key}_std'] = np.std(values)
        aggregated_features[f'{key}_min'] = np.min(values)
        aggregated_features[f'{key}_max'] = np.max(values)
        aggregated_features[f'{key}_range'] = np.max(values) - np.min(values)
        aggregated_features[f'{key}_median'] = np.median(values)
        # Handle single-value case for skewness and kurtosis
        if len(values) > 1:
            aggregated_features[f'{key}_skewness'] = skew(values)
            aggregated_features[f'{key}_kurtosis'] = kurtosis(values)
        else:
            # For single value, skewness and kurtosis are undefined
            aggregated_features[f'{key}_skewness'] = 0.0
            aggregated_features[f'{key}_kurtosis'] = 0.0
    
    # Add frame count (aggregated like other features)
    frame_counts = [len(frames)]  # Single value, but we'll aggregate it for consistency
    aggregated_features['num_frames_mean'] = np.mean(frame_counts)
    aggregated_features['num_frames_std'] = np.std(frame_counts)
    aggregated_features['num_frames_min'] = np.min(frame_counts)
    aggregated_features['num_frames_max'] = np.max(frame_counts)
    aggregated_features['num_frames_range'] = np.max(frame_counts) - np.min(frame_counts)
    aggregated_features['num_frames_median'] = np.median(frame_counts)
    # Handle single-value case for frame count skewness and kurtosis
    if len(frame_counts) > 1:
        aggregated_features['num_frames_skewness'] = skew(frame_counts)
        aggregated_features['num_frames_kurtosis'] = kurtosis(frame_counts)
    else:
        # For single value, skewness and kurtosis are undefined
        aggregated_features['num_frames_skewness'] = 0.0
        aggregated_features['num_frames_kurtosis'] = 0.0
    
    return aggregated_features


def extract_hosf_features(audio_path, sample_rate=16000, frame_length_ms=20, overlap_ratio=0.5):
    """
    Extract HOSF features from a single audio chunk.
    
    Args:
        audio_path (str): Path to the audio chunk file
        sample_rate (int): Sample rate for the audio (default: 16000 Hz)
        frame_length_ms (int): Frame length in milliseconds (default: 20ms)
        overlap_ratio (float): Overlap ratio (default: 0.5 = 50% overlap)
    
    Returns:
        numpy.ndarray: HOSF feature vector
    """
    try:
        # Load audio file
        y, sr = librosa.load(audio_path, sr=sample_rate)
        
        # Ensure mono
        if len(y.shape) > 1:
            y = np.mean(y, axis=0)
        
        # Extract HOSF features
        features = extract_hosf_features_utterance(y, sr, frame_length_ms, overlap_ratio)
        
        # Convert to numpy array
        feature_vector = np.array(list(features.values()))
        
        return feature_vector
        
    except Exception as e:
        print(f"Error extracting HOSF features from {audio_path}: {e}")
        # Return zero vector as fallback (estimated number of features)
        return np.zeros(200)  # Approximate number of features


def process_single_patient(patient_folder, output_dir, sample_rate=16000, frame_length_ms=20, overlap_ratio=0.5):
    """
    Process all audio chunks for a single patient.
    This function is designed to be called in parallel.
    
    Args:
        patient_folder (str): Path to patient folder containing audio chunks
        output_dir (str): Output directory for this patient's features
        sample_rate (int): Sample rate for audio processing
        frame_length_ms (int): Frame length in milliseconds
        overlap_ratio (float): Overlap ratio
    
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
    
    # Create feature names (HOSF feature names)
    feature_names = [
        # Bispectrum features (8 features * 7 aggregations = 56)
        'bispectrum_mean_mean', 'bispectrum_mean_std', 'bispectrum_mean_min', 'bispectrum_mean_max', 
        'bispectrum_mean_range', 'bispectrum_mean_median', 'bispectrum_mean_skewness', 'bispectrum_mean_kurtosis',
        'bispectrum_std_mean', 'bispectrum_std_std', 'bispectrum_std_min', 'bispectrum_std_max',
        'bispectrum_std_range', 'bispectrum_std_median', 'bispectrum_std_skewness', 'bispectrum_std_kurtosis',
        'bispectrum_max_mean', 'bispectrum_max_std', 'bispectrum_max_min', 'bispectrum_max_max',
        'bispectrum_max_range', 'bispectrum_max_median', 'bispectrum_max_skewness', 'bispectrum_max_kurtosis',
        'bispectrum_min_mean', 'bispectrum_min_std', 'bispectrum_min_min', 'bispectrum_min_max',
        'bispectrum_min_range', 'bispectrum_min_median', 'bispectrum_min_skewness', 'bispectrum_min_kurtosis',
        'bispectrum_range_mean', 'bispectrum_range_std', 'bispectrum_range_min', 'bispectrum_range_max',
        'bispectrum_range_range', 'bispectrum_range_median', 'bispectrum_range_skewness', 'bispectrum_range_kurtosis',
        'bispectrum_skewness_mean', 'bispectrum_skewness_std', 'bispectrum_skewness_min', 'bispectrum_skewness_max',
        'bispectrum_skewness_range', 'bispectrum_skewness_median', 'bispectrum_skewness_skewness', 'bispectrum_skewness_kurtosis',
        'bispectrum_kurtosis_mean', 'bispectrum_kurtosis_std', 'bispectrum_kurtosis_min', 'bispectrum_kurtosis_max',
        'bispectrum_kurtosis_range', 'bispectrum_kurtosis_median', 'bispectrum_kurtosis_skewness', 'bispectrum_kurtosis_kurtosis',
        'bispectrum_entropy_mean', 'bispectrum_entropy_std', 'bispectrum_entropy_min', 'bispectrum_entropy_max',
        'bispectrum_entropy_range', 'bispectrum_entropy_median', 'bispectrum_entropy_skewness', 'bispectrum_entropy_kurtosis',
        
        # Bicoherence features (8 features * 7 aggregations = 56)
        'bicoherence_mean_mean', 'bicoherence_mean_std', 'bicoherence_mean_min', 'bicoherence_mean_max',
        'bicoherence_mean_range', 'bicoherence_mean_median', 'bicoherence_mean_skewness', 'bicoherence_mean_kurtosis',
        'bicoherence_std_mean', 'bicoherence_std_std', 'bicoherence_std_min', 'bicoherence_std_max',
        'bicoherence_std_range', 'bicoherence_std_median', 'bicoherence_std_skewness', 'bicoherence_std_kurtosis',
        'bicoherence_max_mean', 'bicoherence_max_std', 'bicoherence_max_min', 'bicoherence_max_max',
        'bicoherence_max_range', 'bicoherence_max_median', 'bicoherence_max_skewness', 'bicoherence_max_kurtosis',
        'bicoherence_min_mean', 'bicoherence_min_std', 'bicoherence_min_min', 'bicoherence_min_max',
        'bicoherence_min_range', 'bicoherence_min_median', 'bicoherence_min_skewness', 'bicoherence_min_kurtosis',
        'bicoherence_range_mean', 'bicoherence_range_std', 'bicoherence_range_min', 'bicoherence_range_max',
        'bicoherence_range_range', 'bicoherence_range_median', 'bicoherence_range_skewness', 'bicoherence_range_kurtosis',
        'bicoherence_skewness_mean', 'bicoherence_skewness_std', 'bicoherence_skewness_min', 'bicoherence_skewness_max',
        'bicoherence_skewness_range', 'bicoherence_skewness_median', 'bicoherence_skewness_skewness', 'bicoherence_skewness_kurtosis',
        'bicoherence_kurtosis_mean', 'bicoherence_kurtosis_std', 'bicoherence_kurtosis_min', 'bicoherence_kurtosis_max',
        'bicoherence_kurtosis_range', 'bicoherence_kurtosis_median', 'bicoherence_kurtosis_skewness', 'bicoherence_kurtosis_kurtosis',
        'bicoherence_entropy_mean', 'bicoherence_entropy_std', 'bicoherence_entropy_min', 'bicoherence_entropy_max',
        'bicoherence_entropy_range', 'bicoherence_entropy_median', 'bicoherence_entropy_skewness', 'bicoherence_entropy_kurtosis',
        
        # Additional HOSF features (8 features * 7 aggregations = 56)
        'bispectrum_flatness_mean', 'bispectrum_flatness_std', 'bispectrum_flatness_min', 'bispectrum_flatness_max',
        'bispectrum_flatness_range', 'bispectrum_flatness_median', 'bispectrum_flatness_skewness', 'bispectrum_flatness_kurtosis',
        'bicoherence_flatness_mean', 'bicoherence_flatness_std', 'bicoherence_flatness_min', 'bicoherence_flatness_max',
        'bicoherence_flatness_range', 'bicoherence_flatness_median', 'bicoherence_flatness_skewness', 'bicoherence_flatness_kurtosis',
        'phase_coupling_strength_mean', 'phase_coupling_strength_std', 'phase_coupling_strength_min', 'phase_coupling_strength_max',
        'phase_coupling_strength_range', 'phase_coupling_strength_median', 'phase_coupling_strength_skewness', 'phase_coupling_strength_kurtosis',
        'diagonal_bicoherence_mean_mean', 'diagonal_bicoherence_mean_std', 'diagonal_bicoherence_mean_min', 'diagonal_bicoherence_mean_max',
        'diagonal_bicoherence_mean_range', 'diagonal_bicoherence_mean_median', 'diagonal_bicoherence_mean_skewness', 'diagonal_bicoherence_mean_kurtosis',
        'diagonal_bicoherence_std_mean', 'diagonal_bicoherence_std_std', 'diagonal_bicoherence_std_min', 'diagonal_bicoherence_std_max',
        'diagonal_bicoherence_std_range', 'diagonal_bicoherence_std_median', 'diagonal_bicoherence_std_skewness', 'diagonal_bicoherence_std_kurtosis',
        'off_diagonal_bicoherence_mean_mean', 'off_diagonal_bicoherence_mean_std', 'off_diagonal_bicoherence_mean_min', 'off_diagonal_bicoherence_mean_max',
        'off_diagonal_bicoherence_mean_range', 'off_diagonal_bicoherence_mean_median', 'off_diagonal_bicoherence_mean_skewness', 'off_diagonal_bicoherence_mean_kurtosis',
        'off_diagonal_bicoherence_std_mean', 'off_diagonal_bicoherence_std_std', 'off_diagonal_bicoherence_std_min', 'off_diagonal_bicoherence_std_max',
        'off_diagonal_bicoherence_std_range', 'off_diagonal_bicoherence_std_median', 'off_diagonal_bicoherence_std_skewness', 'off_diagonal_bicoherence_std_kurtosis',
        
        # Frame count
        'num_frames_mean', 'num_frames_std', 'num_frames_min', 'num_frames_max', 'num_frames_range', 'num_frames_median', 'num_frames_skewness', 'num_frames_kurtosis'
    ]
    
    successful_chunks = 0
    failed_chunks = 0
    
    # Process each audio chunk
    for chunk_path in audio_chunks:
        chunk_filename = os.path.basename(chunk_path)
        chunk_name = os.path.splitext(chunk_filename)[0]  # Remove .wav extension
        
        # Check if CSV already exists (resumable functionality)
        output_filename = f"{chunk_name}_hosf_features.csv"
        output_path = os.path.join(patient_output_dir, output_filename)
        
        if os.path.exists(output_path):
            # Skip if already processed
            successful_chunks += 1
            continue
        
        try:
            # Extract HOSF features
            features = extract_hosf_features(chunk_path, sample_rate, frame_length_ms, overlap_ratio)
            
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


def process_chunk_configuration(chunk_config_dir, output_base_dir, sample_rate=16000, frame_length_ms=20, overlap_ratio=0.5):
    """
    Process all audio chunks in a specific chunk configuration directory using parallel processing.
    
    Args:
        chunk_config_dir (str): Path to chunk configuration directory (e.g., 30.0s_15.0s_overlap)
        output_base_dir (str): Base directory for output features
        sample_rate (int): Sample rate for audio processing
        frame_length_ms (int): Frame length in milliseconds
        overlap_ratio (float): Overlap ratio
    """
    config_name = os.path.basename(chunk_config_dir)
    
    # Create output directory structure
    output_dir = os.path.join(output_base_dir, "features", "hosf", config_name)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Processing configuration: {config_name}")
    print(f"Output directory: {output_dir}")
    print(f"Frame length: {frame_length_ms}ms, Overlap: {overlap_ratio*100:.1f}%")
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
    # HOSF is more computationally intensive, use fewer cores to avoid memory issues
    # Optimized for M4 Pro: Use most performance cores but leave 1-2 for system
    num_workers = min(PERFORMANCE_CORES - 2, len(patient_folders), 8)  # Use up to 8 performance cores for HOSF
    print(f"Using {num_workers} parallel workers (performance cores)")
    print(f"Available cores: {mp.cpu_count()} total, {PERFORMANCE_CORES} performance, {EFFICIENCY_CORES} efficiency")
    
    # Prepare arguments for parallel processing
    args_list = [(patient_folder, output_dir, sample_rate, frame_length_ms, overlap_ratio) for patient_folder in patient_folders]
    
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
    
    data/features/hosf/
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
    - Extracts HOSF features from each audio chunk
    - Saves individual CSV files per chunk with ~200 HOSF features
    - Maintains the same directory structure as input chunks
    - Each CSV contains one row with HOSF feature values
    - Optimized for MBP 10 Performance + 4 Efficiency cores
    """
    # Initialize optimizations for MBP M4 Pro
    print("Initializing MBP M4 Pro optimizations...")
    optimize_memory()
    set_cpu_affinity()
    
    # Configuration parameters - CHANGE THESE VALUES AS NEEDED
    CHUNK_CONFIGS = [
        "3.0s_1.5s_overlap",
        "5.0s_2.5s_overlap",
        "10.0s_5.0s_overlap",
        "20.0s_10.0s_overlap",
        "30.0s_15.0s_overlap",
    ]
    SAMPLE_RATE = 16000  # Hz - maintains original DAIC-WOZ format
    FRAME_LENGTH_MS = 20  # ms - frame length for HOSF analysis
    OVERLAP_RATIO = 0.5  # 50% overlap between frames
    
    # Paths - adjusted for src/feature_extraction directory
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    INPUT_BASE_DIR = os.path.join(PROJECT_ROOT, "data", "created_data")
    OUTPUT_BASE_DIR = os.path.join(PROJECT_ROOT, "data")
    
    print(f"Script location: {os.path.abspath(__file__)}")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Input directory: {INPUT_BASE_DIR}")
    print(f"Output directory: {OUTPUT_BASE_DIR}")
    print(f"Sample rate: {SAMPLE_RATE} Hz")
    print(f"Frame length: {FRAME_LENGTH_MS} ms")
    print(f"Overlap ratio: {OVERLAP_RATIO*100:.1f}%")
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
            SAMPLE_RATE,
            FRAME_LENGTH_MS,
            OVERLAP_RATIO
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
    print(f"\nHOSF features saved to:")
    print(f"  {OUTPUT_BASE_DIR}/features/hosf/[CONFIG]/[PATIENT_ID]/[CHUNK]_features.csv")
    print(f"\nEach CSV contains ~200 HOSF features for one audio chunk.")


if __name__ == "__main__":
    # Set multiprocessing start method for compatibility across platforms
    mp.set_start_method('spawn', force=True)
    main()
