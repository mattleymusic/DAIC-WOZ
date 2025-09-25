#!/usr/bin/env python3
"""
Paper Audio Chunker - Reproducing Miao et al. 2022

This script creates 4-second audio chunks with no overlap, exactly as described
in the paper: "Fusing features of speech for depression classification based on 
higher-order spectral analysis" by Miao et al., Speech Communication 143 (2022) 46–56

The paper uses:
- 4-second chunks (S = 4 s)
- No overlap
- Balanced sampling (equal number of samples per speaker)

Usage:
    python src/preprocessing/paper_audio_chunker.py

Author: Reproduction of Miao et al. 2022 methodology
"""

import os
import librosa
import soundfile as sf
import numpy as np
import pandas as pd
import sys
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import time
import warnings
import psutil

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
    """Optimize memory usage for MBP M4 Pro large-scale processing."""
    try:
        # Optimize for M4 Pro shared memory architecture
        os.environ['OMP_NUM_THREADS'] = str(PERFORMANCE_CORES)
        os.environ['MKL_NUM_THREADS'] = str(PERFORMANCE_CORES)
        os.environ['NUMEXPR_NUM_THREADS'] = str(PERFORMANCE_CORES)
        
        # Configure librosa for better memory usage
        librosa.set_cache(False)  # Disable caching to save memory
        
        # Configure for large shared memory
        import gc
        gc.set_threshold(700, 10, 10)  # More aggressive garbage collection for large datasets
        
        # Set NumPy to ignore numerical warnings for cleaner output
        np.seterr(all='ignore')
        
        print(f"Memory optimization configured for MBP M4 Pro ({PERFORMANCE_CORES}P+{EFFICIENCY_CORES}E cores)")
        
    except Exception as e:
        print(f"Warning: Could not optimize memory settings: {e}")


def create_4s_chunks(audio_path, output_dir, chunk_length=4.0, sample_rate=16000):
    """
    Create 4-second chunks from audio file as described in the paper.
    
    Args:
        audio_path (str): Path to the input audio file
        output_dir (str): Output directory for chunks
        chunk_length (float): Chunk length in seconds (default: 4.0s)
        sample_rate (int): Sample rate for audio processing
    
    Returns:
        list: List of created chunk file paths
    """
    try:
        # Load audio file
        y, sr = librosa.load(audio_path, sr=sample_rate)
        
        # Ensure mono
        if len(y.shape) > 1:
            y = np.mean(y, axis=0)
        
        # Calculate chunk parameters
        chunk_samples = int(chunk_length * sr)
        total_samples = len(y)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate chunks
        chunk_paths = []
        chunk_count = 0
        
        for start_sample in range(0, total_samples, chunk_samples):
            end_sample = min(start_sample + chunk_samples, total_samples)
            chunk_audio = y[start_sample:end_sample]
            
            # Pad if necessary (for last chunk if it's shorter than 4s)
            if len(chunk_audio) < chunk_samples:
                chunk_audio = np.pad(chunk_audio, (0, chunk_samples - len(chunk_audio)), mode='constant')
            
            # Create chunk filename
            chunk_filename = f"chunk_{chunk_count+1:03d}.wav"
            chunk_path = os.path.join(output_dir, chunk_filename)
            
            # Save chunk
            sf.write(chunk_path, chunk_audio, sr)
            chunk_paths.append(chunk_path)
            chunk_count += 1
        
        return chunk_paths
        
    except Exception as e:
        print(f"Error creating chunks from {audio_path}: {e}")
        return []


def process_single_patient(patient_folder, output_base_dir, chunk_length=4.0, sample_rate=16000):
    """
    Process all audio files for a single patient.
    
    Args:
        patient_folder (str): Path to patient folder containing audio files
        output_base_dir (str): Base directory for output chunks
        chunk_length (float): Chunk length in seconds
        sample_rate (int): Sample rate for audio processing
    
    Returns:
        dict: Processing results for this patient
    """
    patient_name = os.path.basename(patient_folder)
    
    # Create patient output directory
    patient_output_dir = os.path.join(output_base_dir, patient_name)
    
    # Get all audio files for this patient
    audio_files = []
    for file in os.listdir(patient_folder):
        if file.lower().endswith('.wav'):
            audio_files.append(os.path.join(patient_folder, file))
    
    audio_files.sort()  # Sort to process in order
    
    if not audio_files:
        return {
            'patient_name': patient_name,
            'successful_files': 0,
            'failed_files': 0,
            'total_files': 0,
            'total_chunks': 0,
            'status': 'no_files_found'
        }
    
    successful_files = 0
    failed_files = 0
    total_chunks = 0
    
    # Process each audio file
    for audio_file in audio_files:
        filename = os.path.basename(audio_file)
        file_name = os.path.splitext(filename)[0]  # Remove .wav extension
        
        try:
            # Create chunks for this file
            chunk_paths = create_4s_chunks(audio_file, patient_output_dir, chunk_length, sample_rate)
            
            if chunk_paths:
                successful_files += 1
                total_chunks += len(chunk_paths)
            else:
                failed_files += 1
                
        except Exception as e:
            print(f"    Error processing {filename}: {e}")
            failed_files += 1
    
    return {
        'patient_name': patient_name,
        'successful_files': successful_files,
        'failed_files': failed_files,
        'total_files': len(audio_files),
        'total_chunks': total_chunks,
        'status': 'completed'
    }


def process_chunk_configuration(input_dir, output_base_dir, chunk_length=4.0, sample_rate=16000):
    """
    Process all patients in the input directory using parallel processing.
    
    Args:
        input_dir (str): Path to input directory containing patient folders
        output_base_dir (str): Base directory for output chunks
        chunk_length (float): Chunk length in seconds
        sample_rate (int): Sample rate for audio processing
    """
    config_name = f"{chunk_length:.1f}s_0.0s_overlap"
    
    # Create output directory structure
    output_dir = os.path.join(output_base_dir, config_name)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Processing configuration: {config_name}")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Chunk length: {chunk_length} seconds")
    print(f"Overlap: 0.0 seconds (no overlap)")
    print("-" * 60)
    
    # Get all patient folders
    patient_folders = []
    for item in os.listdir(input_dir):
        item_path = os.path.join(input_dir, item)
        if os.path.isdir(item_path):
            patient_folders.append(item_path)
    
    patient_folders.sort()  # Sort to process in order
    
    if not patient_folders:
        print("No patient folders found.")
        return 0, 0, 0, 0
    
    print(f"Found {len(patient_folders)} patient folders")
    
    # Determine number of workers (optimized for MBP 10P+4E cores)
    # Audio chunking is I/O intensive, use efficiency cores
    num_workers = min(EFFICIENCY_CORES + 2, len(patient_folders), 6)  # Use up to 6 cores for chunking
    print(f"Using {num_workers} parallel workers (mixed cores for I/O)")
    print(f"Available cores: {mp.cpu_count()} total, {PERFORMANCE_CORES} performance, {EFFICIENCY_CORES} efficiency")
    
    successful_patients = 0
    failed_patients = 0
    total_files = 0
    total_chunks = 0
    
    # Process patients in parallel
    start_time = time.time()
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        future_to_patient = {
            executor.submit(process_single_patient, patient_folder, output_dir, chunk_length, sample_rate): patient_folder 
            for patient_folder in patient_folders
        }
        
        # Process completed tasks
        for future in as_completed(future_to_patient):
            patient_folder = future_to_patient[future]
            patient_name = os.path.basename(patient_folder)
            
            try:
                result = future.result()
                
                if result['status'] == 'completed':
                    if result['failed_files'] == 0:
                        print(f"{patient_name}: {result['successful_files']}/{result['total_files']} files processed, {result['total_chunks']} chunks created")
                        successful_patients += 1
                    else:
                        print(f"Warning {patient_name}: {result['successful_files']}/{result['total_files']} files processed ({result['failed_files']} failed), {result['total_chunks']} chunks created")
                        successful_patients += 1  # Still count as successful if some files worked
                    
                    total_files += result['total_files']
                    total_chunks += result['total_chunks']
                elif result['status'] == 'no_files_found':
                    print(f"Warning {patient_name}: No audio files found")
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
    
    return successful_patients, failed_patients, total_files, total_chunks


def main():
    """
    Main function to create 4-second chunks as described in the paper.
    
    !!! INPUT STRUCTURE !!!
    
    data/created_data/concatenated_diarisation/
    ├── 300_P/
    │   └── 300_P_concatenated.wav
    ├── 301_P/
    │   └── 301_P_concatenated.wav
    └── ...
    
    !!! OUTPUT STRUCTURE !!!
    
    data/created_data/4.0s_0.0s_overlap/
    ├── 300_P/
    │   ├── chunk_001.wav
    │   ├── chunk_002.wav
    │   └── ...
    ├── 301_P/
    │   ├── chunk_001.wav
    │   ├── chunk_002.wav
    │   └── ...
    └── ...
    
    !!! PROCESSING DETAILS !!!
    
    - Creates 4-second chunks with no overlap (as in paper)
    - Pads shorter chunks to 4 seconds
    - Maintains 16kHz sample rate
    - Each chunk is exactly 4 seconds long
    - Optimized for MBP 10 Performance + 4 Efficiency cores
    """
    # Initialize optimizations for MBP M4 Pro
    print("Initializing MBP M4 Pro optimizations...")
    optimize_memory()
    set_cpu_affinity()
    
    # Configuration parameters - CHANGE THESE VALUES AS NEEDED
    CHUNK_LENGTH = 4.0  # seconds - exactly as in paper
    OVERLAP_LENGTH = 0.0  # seconds - no overlap as in paper
    SAMPLE_RATE = 16000  # Hz - maintains original DAIC-WOZ format
    
    # Paths - adjusted for src/preprocessing directory
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    INPUT_BASE_DIR = os.path.join(PROJECT_ROOT, "data", "created_data", "concatenated_diarisation")
    OUTPUT_BASE_DIR = os.path.join(PROJECT_ROOT, "data", "created_data")
    
    print(f"Script location: {os.path.abspath(__file__)}")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Input directory: {INPUT_BASE_DIR}")
    print(f"Output directory: {OUTPUT_BASE_DIR}")
    print(f"Chunk length: {CHUNK_LENGTH} seconds")
    print(f"Overlap length: {OVERLAP_LENGTH} seconds")
    print(f"Sample rate: {SAMPLE_RATE} Hz")
    print("=" * 80)
    
    # Check if input directory exists
    if not os.path.exists(INPUT_BASE_DIR):
        print(f"Error: Input directory not found: {INPUT_BASE_DIR}")
        print("Please run the audio concatenator first to create concatenated audio files.")
        return
    
    total_successful_patients = 0
    total_failed_patients = 0
    total_files_processed = 0
    total_chunks_created = 0
    
    # Process the concatenated audio files
    print(f"\n{'='*20} Processing Concatenated Audio Files {'='*20}")
    
    successful, failed, files, chunks = process_chunk_configuration(
        INPUT_BASE_DIR,
        OUTPUT_BASE_DIR,
        CHUNK_LENGTH,
        SAMPLE_RATE
    )
    
    total_successful_patients += successful
    total_failed_patients += failed
    total_files_processed += files
    total_chunks_created += chunks
    
    print("\n" + "=" * 80)
    print("FINAL PROCESSING SUMMARY")
    print("=" * 80)
    print(f"Total patients processed: {total_successful_patients + total_failed_patients}")
    print(f"Successful patients: {total_successful_patients}")
    print(f"Failed patients: {total_failed_patients}")
    print(f"Total files processed: {total_files_processed}")
    print(f"Total chunks created: {total_chunks_created}")
    print(f"Success rate: {(total_successful_patients/(total_successful_patients+total_failed_patients)*100):.1f}%")
    print(f"\n4-second chunks saved to:")
    print(f"  {OUTPUT_BASE_DIR}/4.0s_0.0s_overlap/[PATIENT_ID]/chunk_XXX.wav")
    print(f"\nEach chunk is exactly 4 seconds long with no overlap, as described in Miao et al. 2022.")


if __name__ == "__main__":
    # Set multiprocessing start method for compatibility across platforms
    mp.set_start_method('spawn', force=True)
    main()
