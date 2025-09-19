#!/usr/bin/env python3
"""
HOSF Fusion Feature Extractor for DAIC-WOZ Dataset

This script combines COVAREP baseline acoustic features with Higher-Order Spectral Features (HOSF)
and optionally performs feature selection using ANOVA F-score ranking. It processes all chunk 
configurations and creates fused feature sets for machine learning.

Features:
- Loads COVAREP features from data/features/covarep/
- Loads HOSF features from data/features/hosf/
- Concatenates features (COVAREP + HOSF)
- Optional feature selection using ANOVA F-score ranking
- Saves fused features to data/features/hosf_fusion/

Usage:
    python src/feature_extraction/hosf_fusion_extractor.py
    or
    ./src/feature_extraction/hosf_fusion_extractor.py (if made executable)

Author: Based on HOSF fusion specifications and DAIC-WOZ requirements
"""

import os
import numpy as np
import pandas as pd
import sys
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
from sklearn.feature_selection import f_classif
from sklearn.preprocessing import StandardScaler
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
        
    except Exception as e:
        print(f"Warning: Could not optimize memory settings: {e}")


def load_features_from_csv(csv_path):
    """
    Load features from a CSV file.
    
    Args:
        csv_path (str): Path to CSV file
    
    Returns:
        numpy.ndarray: Feature vector
    """
    try:
        df = pd.read_csv(csv_path)
        if df.empty:
            return None
        return df.values.flatten()
    except Exception as e:
        print(f"Error loading features from {csv_path}: {e}")
        return None


def load_covarep_features(patient_dir, chunk_name):
    """
    Load COVAREP features for a specific chunk.
    
    Args:
        patient_dir (str): Patient directory path
        chunk_name (str): Chunk name (without extension)
    
    Returns:
        numpy.ndarray: COVAREP feature vector
    """
    csv_path = os.path.join(patient_dir, f"{chunk_name}_covarep_features.csv")
    return load_features_from_csv(csv_path)


def load_hosf_features(patient_dir, chunk_name):
    """
    Load HOSF features for a specific chunk.
    
    Args:
        patient_dir (str): Patient directory path
        chunk_name (str): Chunk name (without extension)
    
    Returns:
        numpy.ndarray: HOSF feature vector
    """
    csv_path = os.path.join(patient_dir, f"{chunk_name}_hosf_features.csv")
    return load_features_from_csv(csv_path)


def fuse_features(covarep_features, hosf_features):
    """
    Fuse COVAREP and HOSF features by concatenation.
    
    Args:
        covarep_features (np.ndarray): COVAREP feature vector
        hosf_features (np.ndarray): HOSF feature vector
    
    Returns:
        numpy.ndarray: Fused feature vector
    """
    if covarep_features is None or hosf_features is None:
        return None
    
    # Concatenate features
    fused_features = np.concatenate([covarep_features, hosf_features])
    
    return fused_features


def perform_feature_selection(features_matrix, labels, top_n=None, f_score_threshold=1.0):
    """
    Perform feature selection using ANOVA F-score ranking.
    
    Args:
        features_matrix (np.ndarray): Feature matrix (samples x features)
        labels (np.ndarray): Class labels
        top_n (int): Number of top features to select (None for threshold-based)
        f_score_threshold (float): F-score threshold for feature selection
    
    Returns:
        tuple: (selected_features_matrix, selected_feature_indices, f_scores)
    """
    # Remove features with zero variance
    feature_vars = np.var(features_matrix, axis=0)
    non_zero_var_mask = feature_vars > 1e-10
    
    if np.sum(non_zero_var_mask) == 0:
        print("Warning: All features have zero variance")
        return features_matrix, np.arange(features_matrix.shape[1]), np.zeros(features_matrix.shape[1])
    
    # Filter features
    filtered_features = features_matrix[:, non_zero_var_mask]
    filtered_indices = np.where(non_zero_var_mask)[0]
    
    # Compute F-scores
    f_scores, p_values = f_classif(filtered_features, labels)
    
    if top_n is not None:
        # Select top N features
        top_indices = np.argsort(f_scores)[-top_n:]
        selected_features = filtered_features[:, top_indices]
        selected_feature_indices = filtered_indices[top_indices]
        selected_f_scores = f_scores[top_indices]
    else:
        # Select features above threshold
        threshold_mask = f_scores >= f_score_threshold
        selected_features = filtered_features[:, threshold_mask]
        selected_feature_indices = filtered_indices[threshold_mask]
        selected_f_scores = f_scores[threshold_mask]
    
    return selected_features, selected_feature_indices, selected_f_scores


def process_single_patient(patient_folder, covarep_dir, hosf_dir, output_dir, 
                          perform_selection=False, top_n=None, f_score_threshold=1.0,
                          load_labels=False, labels_df=None):
    """
    Process all audio chunks for a single patient.
    This function is designed to be called in parallel.
    
    Args:
        patient_folder (str): Path to patient folder containing audio chunks
        covarep_dir (str): COVAREP features directory for this patient
        hosf_dir (str): HOSF features directory for this patient
        output_dir (str): Output directory for this patient's fused features
        perform_selection (bool): Whether to perform feature selection
        top_n (int): Number of top features to select
        f_score_threshold (float): F-score threshold for feature selection
        load_labels (bool): Whether to load labels for feature selection
        labels_df (pd.DataFrame): Labels dataframe
    
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
            chunk_name = os.path.splitext(file)[0]  # Remove .wav extension
            audio_chunks.append(chunk_name)
    
    audio_chunks.sort()  # Sort to process in order
    
    if not audio_chunks:
        return {
            'patient_name': patient_name,
            'successful_chunks': 0,
            'failed_chunks': 0,
            'total_chunks': 0,
            'status': 'no_chunks_found'
        }
    
    # Load all features for this patient
    all_covarep_features = []
    all_hosf_features = []
    all_labels = []
    valid_chunks = []
    
    for chunk_name in audio_chunks:
        # Load COVAREP features
        covarep_features = load_covarep_features(covarep_dir, chunk_name)
        
        # Load HOSF features
        hosf_features = load_hosf_features(hosf_dir, chunk_name)
        
        if covarep_features is not None and hosf_features is not None:
            all_covarep_features.append(covarep_features)
            all_hosf_features.append(hosf_features)
            valid_chunks.append(chunk_name)
            
            # Load labels if needed
            if load_labels and labels_df is not None:
                # Extract patient ID from patient name (e.g., "300_P" -> "300")
                patient_id = patient_name.split('_')[0]
                patient_labels = labels_df[labels_df['Participant_ID'] == int(patient_id)]
                if not patient_labels.empty:
                    all_labels.append(patient_labels.iloc[0]['PHQ8_Binary'])
                else:
                    all_labels.append(0)  # Default label
            else:
                all_labels.append(0)  # Default label
    
    if not valid_chunks:
        return {
            'patient_name': patient_name,
            'successful_chunks': 0,
            'failed_chunks': len(audio_chunks),
            'total_chunks': len(audio_chunks),
            'status': 'no_valid_features'
        }
    
    # Convert to numpy arrays
    covarep_matrix = np.array(all_covarep_features)
    hosf_matrix = np.array(all_hosf_features)
    labels_array = np.array(all_labels)
    
    # Fuse features
    fused_features_matrix = np.concatenate([covarep_matrix, hosf_matrix], axis=1)
    
    # Perform feature selection if requested
    if perform_selection and len(valid_chunks) > 1:
        try:
            selected_features, selected_indices, f_scores = perform_feature_selection(
                fused_features_matrix, labels_array, top_n, f_score_threshold
            )
            
            # Save feature selection info
            selection_info = {
                'original_features': fused_features_matrix.shape[1],
                'selected_features': selected_features.shape[1],
                'selected_indices': selected_indices,
                'f_scores': f_scores
            }
            
            # Save selection info to file
            selection_info_path = os.path.join(patient_output_dir, f"{patient_name}_feature_selection_info.npz")
            np.savez(selection_info_path, **selection_info)
            
            final_features_matrix = selected_features
        except Exception as e:
            print(f"    Warning: Feature selection failed for {patient_name}: {e}")
            final_features_matrix = fused_features_matrix
    else:
        final_features_matrix = fused_features_matrix
    
    # Create feature names
    covarep_feature_names = [f'covarep_{i:03d}' for i in range(covarep_matrix.shape[1])]
    hosf_feature_names = [f'hosf_{i:03d}' for i in range(hosf_matrix.shape[1])]
    all_feature_names = covarep_feature_names + hosf_feature_names
    
    # If feature selection was performed, use only selected feature names
    if perform_selection and len(valid_chunks) > 1:
        try:
            selected_feature_names = [all_feature_names[i] for i in selected_indices]
        except:
            selected_feature_names = all_feature_names
    else:
        selected_feature_names = all_feature_names
    
    successful_chunks = 0
    failed_chunks = 0
    
    # Save fused features for each chunk
    for i, chunk_name in enumerate(valid_chunks):
        # Check if CSV already exists (resumable functionality)
        output_filename = f"{chunk_name}_hosf_fusion_features.csv"
        output_path = os.path.join(patient_output_dir, output_filename)
        
        if os.path.exists(output_path):
            # Skip if already processed
            successful_chunks += 1
            continue
        
        try:
            # Get features for this chunk
            chunk_features = final_features_matrix[i]
            
            # Create DataFrame with features
            feature_df = pd.DataFrame([chunk_features], columns=selected_feature_names)
            
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


def load_labels_data(project_root):
    """
    Load labels data for feature selection.
    
    Args:
        project_root (str): Project root directory
    
    Returns:
        pd.DataFrame: Labels dataframe
    """
    try:
        labels_path = os.path.join(project_root, "data", "daic_woz", "meta_info.csv")
        labels_df = pd.read_csv(labels_path)
        return labels_df
    except Exception as e:
        print(f"Warning: Could not load labels data: {e}")
        return None


def process_chunk_configuration(chunk_config_dir, covarep_base_dir, hosf_base_dir, output_base_dir,
                              config_name, perform_selection=False, top_n=None, f_score_threshold=1.0,
                              load_labels=False, labels_df=None):
    """
    Process all audio chunks in a specific chunk configuration directory using parallel processing.
    
    Args:
        chunk_config_dir (str): Path to chunk configuration directory (e.g., 30.0s_15.0s_overlap)
        covarep_base_dir (str): Base directory for COVAREP features
        hosf_base_dir (str): Base directory for HOSF features
        output_base_dir (str): Base directory for output features
        config_name (str): Configuration name
        perform_selection (bool): Whether to perform feature selection
        top_n (int): Number of top features to select
        f_score_threshold (float): F-score threshold for feature selection
        load_labels (bool): Whether to load labels for feature selection
        labels_df (pd.DataFrame): Labels dataframe
    """
    # Create output directory structure
    output_dir = os.path.join(output_base_dir, "features", "hosf_fusion", config_name)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Processing configuration: {config_name}")
    print(f"Output directory: {output_dir}")
    print(f"Feature selection: {'Yes' if perform_selection else 'No'}")
    if perform_selection:
        if top_n is not None:
            print(f"Top N features: {top_n}")
        else:
            print(f"F-score threshold: {f_score_threshold}")
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
    # Fusion is I/O intensive, use efficiency cores for file operations
    num_workers = min(EFFICIENCY_CORES + 2, len(patient_folders), 6)  # Use up to 6 cores for fusion (mix of P+E)
    print(f"Using {num_workers} parallel workers (mixed cores for I/O)")
    print(f"Available cores: {mp.cpu_count()} total, {PERFORMANCE_CORES} performance, {EFFICIENCY_CORES} efficiency")
    
    successful_patients = 0
    failed_patients = 0
    total_chunks = 0
    
    # Process patients in parallel
    start_time = time.time()
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Prepare arguments for parallel processing
        futures = []
        for patient_folder in patient_folders:
            patient_name = os.path.basename(patient_folder)
            covarep_dir = os.path.join(covarep_base_dir, "features", "covarep", config_name, patient_name)
            hosf_dir = os.path.join(hosf_base_dir, "features", "hosf", config_name, patient_name)
            
            future = executor.submit(
                process_single_patient,
                patient_folder, covarep_dir, hosf_dir, output_dir,
                perform_selection, top_n, f_score_threshold,
                load_labels, labels_df
            )
            futures.append((future, patient_name))
        
        # Process completed tasks
        for future, patient_name in futures:
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
                elif result['status'] in ['no_chunks_found', 'no_valid_features']:
                    print(f"⚠ {patient_name}: {result['status']}")
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
    
    data/features/covarep/[CONFIG]/[PATIENT_ID]/[CHUNK]_features.csv
    data/features/hosf/[CONFIG]/[PATIENT_ID]/[CHUNK]_features.csv
    
    !!! OUTPUT STRUCTURE !!!
    
    data/features/hosf_fusion/[CONFIG]/[PATIENT_ID]/[CHUNK]_fused_features.csv
    
    !!! PROCESSING DETAILS !!!
    
    - Loads COVAREP and HOSF features from their respective directories
    - Concatenates features (COVAREP + HOSF)
    - Optionally performs feature selection using ANOVA F-score ranking
    - Saves fused features maintaining the same directory structure
    - Each CSV contains one row with fused feature values
    - Optimized for MBP 10 Performance + 4 Efficiency cores
    """
    # Initialize optimizations for MBP
    print("Initializing MBP optimizations...")
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
    
    # Feature selection parameters
    PERFORM_FEATURE_SELECTION = False  # Set to True to enable feature selection
    TOP_N_FEATURES = 100  # Number of top features to select (None for threshold-based)
    F_SCORE_THRESHOLD = 1.0  # F-score threshold for feature selection
    LOAD_LABELS = False  # Set to True to load labels for feature selection
    
    # Paths - adjusted for src/feature_extraction directory
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    INPUT_BASE_DIR = os.path.join(PROJECT_ROOT, "data", "created_data")
    OUTPUT_BASE_DIR = os.path.join(PROJECT_ROOT, "data")
    
    print(f"Script location: {os.path.abspath(__file__)}")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Input directory: {INPUT_BASE_DIR}")
    print(f"Output directory: {OUTPUT_BASE_DIR}")
    print(f"Feature selection: {'Enabled' if PERFORM_FEATURE_SELECTION else 'Disabled'}")
    if PERFORM_FEATURE_SELECTION:
        if TOP_N_FEATURES is not None:
            print(f"Top N features: {TOP_N_FEATURES}")
        else:
            print(f"F-score threshold: {F_SCORE_THRESHOLD}")
    print(f"Chunk configurations to process: {CHUNK_CONFIGS}")
    print("=" * 80)
    
    # Load labels data if needed
    labels_df = None
    if LOAD_LABELS:
        labels_df = load_labels_data(PROJECT_ROOT)
        if labels_df is not None:
            print(f"Loaded labels data: {len(labels_df)} participants")
        else:
            print("Warning: Could not load labels data, feature selection will use default labels")
    
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
            OUTPUT_BASE_DIR,
            OUTPUT_BASE_DIR,
            config_name,
            PERFORM_FEATURE_SELECTION,
            TOP_N_FEATURES,
            F_SCORE_THRESHOLD,
            LOAD_LABELS,
            labels_df
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
    print(f"\nFused features saved to:")
    print(f"  {OUTPUT_BASE_DIR}/features/hosf_fusion/[CONFIG]/[PATIENT_ID]/[CHUNK]_fused_features.csv")
    print(f"\nEach CSV contains fused COVAREP + HOSF features for one audio chunk.")


if __name__ == "__main__":
    # Set multiprocessing start method for compatibility across platforms
    mp.set_start_method('spawn', force=True)
    main()
