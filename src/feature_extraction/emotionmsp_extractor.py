#!/usr/bin/env python3
"""
EmotionMSP Feature Extractor for DAIC-WOZ Dataset

This script extracts EmotionMSP features from audio chunks using the audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim model.
It processes all chunk configurations in data/created_data/ and saves 1024-dimensional emotion features
to data/features/emotionmsp/ following the same directory structure.

Usage:
    python src/feature_extraction/emotionmsp_extractor.py
    or
    ./src/feature_extraction/emotionmsp_extractor.py (if made executable)

Author: Based on Danila-OG.pdf specifications
"""

import os
import torch
import torchaudio
import numpy as np
import pandas as pd
import sys
import time
from transformers import Wav2Vec2Model, Wav2Vec2Config
import warnings
import argparse

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Add the project root to the path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)


def load_emotionmsp_model(force_cpu=False):
    """
    Load the EmotionMSP model from Hugging Face.
    
    Args:
        force_cpu (bool): Force CPU usage even if GPU is available
    
    Returns:
        tuple: (Wav2Vec2Model, torch.device) - Loaded model and device
    """
    try:
        # Load the EmotionMSP model
        model_name = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
        model = Wav2Vec2Model.from_pretrained(model_name)
        model.eval()  # Set to evaluation mode
        
        # Choose device
        if force_cpu:
            device = torch.device("cpu")
            print("Forcing CPU usage as requested")
        else:
            # Check for CUDA first, then MPS (Apple Silicon), then CPU
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
        
        model = model.to(device)
        
        print(f"EmotionMSP model loaded successfully on {device}")
        return model, device
        
    except Exception as e:
        print(f"Error loading EmotionMSP model: {e}")
        print("Please ensure you have the transformers library installed:")
        print("pip install transformers")
        sys.exit(1)


def extract_emotionmsp_features(audio_path, model, device, sample_rate=16000):
    """
    Extract EmotionMSP features from a single audio chunk.
    
    Args:
        audio_path (str): Path to the audio chunk file
        model (Wav2Vec2Model): Loaded EmotionMSP model
        device (torch.device): Device to run the model on
        sample_rate (int): Sample rate for the audio (default: 16000 Hz)
    
    Returns:
        numpy.ndarray: EmotionMSP feature vector (1024 features)
    """
    try:
        # Load audio file
        waveform, sr = torchaudio.load(audio_path)
        
        # Resample if necessary (EmotionMSP expects 16kHz)
        if sr != sample_rate:
            resampler = torchaudio.transforms.Resample(sr, sample_rate)
            waveform = resampler(waveform)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Move to device
        waveform = waveform.to(device)
        
        # Extract features using EmotionMSP
        with torch.no_grad():
            # Get the hidden states from the last layer
            outputs = model(waveform, output_hidden_states=True)
            # Extract the final hidden layer (last layer)
            hidden_states = outputs.hidden_states[-1]
            # Extract hidden state of the final token for classification
            feature_vector = hidden_states[:, -1, :]
            
            # Convert to numpy and flatten
            feature_vector = feature_vector.cpu().numpy().flatten()
        
        return feature_vector
        
    except Exception as e:
        print(f"Error extracting features from {audio_path}: {e}")
        # Return zero vector as fallback (1024 dimensions for EmotionMSP)
        return np.zeros(1024)


def process_single_patient(patient_folder, output_dir, model, device, sample_rate=16000):
    """
    Process all audio chunks for a single patient.
    This function is designed to be called in parallel.
    
    Args:
        patient_folder (str): Path to patient folder containing audio chunks
        output_dir (str): Output directory for this patient's features
        model (Wav2Vec2Model): Loaded EmotionMSP model
        device (torch.device): Device to run the model on
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
    
    # Create feature names (EmotionMSP feature names - 1024 dimensions)
    feature_names = [f'emotionmsp_feature_{i:03d}' for i in range(1024)]
    
    successful_chunks = 0
    failed_chunks = 0
    skipped_chunks = 0

    # Process each audio chunk
    for i, chunk_path in enumerate(audio_chunks, 1):
        chunk_filename = os.path.basename(chunk_path)
        chunk_name = os.path.splitext(chunk_filename)[0]  # Remove .wav extension

        # Check if output CSV already exists
        output_filename = f"{chunk_name}_emotionmsp_features.csv"
        output_path = os.path.join(patient_output_dir, output_filename)

        if os.path.exists(output_path):
            skipped_chunks += 1
            continue

        try:
            # Extract EmotionMSP features
            features = extract_emotionmsp_features(chunk_path, model, device, sample_rate)

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
        'skipped_chunks': skipped_chunks,
        'total_chunks': len(audio_chunks),
        'status': 'completed'
    }


def process_chunk_configuration(chunk_config_dir, output_base_dir, model, device, sample_rate=16000):
    """
    Process all audio chunks in a specific chunk configuration directory using sequential processing.
    
    Args:
        chunk_config_dir (str): Path to chunk configuration directory (e.g., 30.0s_15.0s_overlap)
        output_base_dir (str): Base directory for output features
        model (Wav2Vec2Model): Loaded EmotionMSP model
        device (torch.device): Device to run the model on
        sample_rate (int): Sample rate for audio processing
    """
    config_name = os.path.basename(chunk_config_dir)
    
    # Create output directory structure
    output_dir = os.path.join(output_base_dir, "features", "emotionmsp", config_name)
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
    print("Processing patients sequentially...")
    
    successful_patients = 0
    failed_patients = 0
    total_chunks = 0
    
    # Process patients sequentially (more reliable with PyTorch models)
    start_time = time.time()
    
    for i, patient_folder in enumerate(patient_folders, 1):
        patient_name = os.path.basename(patient_folder)
        print(f"[{i}/{len(patient_folders)}] Processing {patient_name}...")
        
        try:
            result = process_single_patient(patient_folder, output_dir, model, device, sample_rate)
            
            if result['status'] == 'completed':
                if result['failed_chunks'] == 0 and result['skipped_chunks'] == 0:
                    print(f"  {patient_name}: {result['successful_chunks']}/{result['total_chunks']} chunks processed successfully")
                    successful_patients += 1
                elif result['failed_chunks'] == 0:
                    print(f"  {patient_name}: {result['successful_chunks']}/{result['total_chunks']} chunks processed ({result['skipped_chunks']} skipped)")
                    successful_patients += 1
                else:
                    print(f"  Warning {patient_name}: {result['successful_chunks']}/{result['total_chunks']} chunks processed ({result['failed_chunks']} failed, {result['skipped_chunks']} skipped)")
                    successful_patients += 1  # Still count as successful if some chunks worked
                
                total_chunks += result['total_chunks']
            elif result['status'] == 'no_chunks_found':
                print(f"  Warning {patient_name}: No audio chunks found")
                failed_patients += 1
            else:
                print(f"  Error {patient_name}: Processing failed")
                failed_patients += 1
                
        except Exception as e:
            print(f"  Error {patient_name}: Error in processing: {e}")
            failed_patients += 1
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    print(f"\nConfiguration {config_name} completed in {processing_time:.2f} seconds")
    print(f"Average time per patient: {processing_time/len(patient_folders):.2f} seconds")
    
    return successful_patients, failed_patients, total_chunks


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract EmotionMSP features from DAIC-WOZ audio chunks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all configurations
  python emotionmsp_extractor.py
  
  # Process only specific configurations
  python emotionmsp_extractor.py --configs 3.0s_1.5s_overlap 5.0s_2.5s_overlap
  
  # Use CPU only (if GPU memory is limited)
  python emotionmsp_extractor.py --cpu-only
  
  # Set custom sample rate
  python emotionmsp_extractor.py --sample-rate 22050
        """
    )
    
    parser.add_argument(
        '--configs', 
        nargs='+',
        default=[
            "3.0s_1.5s_overlap",
            "5.0s_2.5s_overlap", 
            "10.0s_5.0s_overlap",
            "20.0s_10.0s_overlap",
            "30.0s_15.0s_overlap",
        ],
        help='Chunk configurations to process (default: all configurations)'
    )
    
    parser.add_argument(
        '--sample-rate',
        type=int,
        default=16000,
        help='Sample rate for audio processing (default: 16000 Hz)'
    )
    
    parser.add_argument(
        '--cpu-only',
        action='store_true',
        help='Force CPU usage even if GPU is available'
    )
    
    parser.add_argument(
        '--input-dir',
        type=str,
        default=None,
        help='Custom input directory (default: data/created_data)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Custom output directory (default: data/features/emotionmsp)'
    )
    
    return parser.parse_args()


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
    
    data/features/emotionmsp/
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
    - Extracts EmotionMSP features from each audio chunk using audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim
    - Saves individual CSV files per chunk with 1024 features
    - Maintains the same directory structure as input chunks
    - Each CSV contains one row with 1024 EmotionMSP feature values
    """
    # Parse command line arguments
    args = parse_arguments()
    
    # Configuration parameters
    CHUNK_CONFIGS = args.configs
    SAMPLE_RATE = args.sample_rate
    
    # Paths - adjusted for src/feature_extraction directory
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    if args.input_dir:
        INPUT_BASE_DIR = args.input_dir
    else:
        INPUT_BASE_DIR = os.path.join(PROJECT_ROOT, "data", "created_data")
    
    if args.output_dir:
        OUTPUT_BASE_DIR = args.output_dir
    else:
        OUTPUT_BASE_DIR = os.path.join(PROJECT_ROOT, "data")
    
    print(f"Script location: {os.path.abspath(__file__)}")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Input directory: {INPUT_BASE_DIR}")
    print(f"Output directory: {OUTPUT_BASE_DIR}")
    print(f"Sample rate: {SAMPLE_RATE} Hz")
    print(f"Chunk configurations to process: {CHUNK_CONFIGS}")
    print("=" * 80)
    
    # Load EmotionMSP model
    print("Loading EmotionMSP model...")
    model, device = load_emotionmsp_model(force_cpu=args.cpu_only)
    print(f"Model loaded on device: {device}")
    print("=" * 80)
    
    total_successful_patients = 0
    total_failed_patients = 0
    total_chunks_processed = 0
    total_chunks_skipped = 0
    
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
            model,
            device,
            SAMPLE_RATE
        )
        
        total_successful_patients += successful
        total_failed_patients += failed
        total_chunks_processed += chunks
        # Note: skipped chunks are tracked per patient, not returned by process_chunk_configuration
        # They will be shown in individual patient messages
    
    print("\n" + "=" * 80)
    print("FINAL PROCESSING SUMMARY")
    print("=" * 80)
    print(f"Total patients processed: {total_successful_patients + total_failed_patients}")
    print(f"Successful patients: {total_successful_patients}")
    print(f"Failed patients: {total_failed_patients}")
    print(f"Total chunks processed: {total_chunks_processed}")
    print(f"Success rate: {(total_successful_patients/(total_successful_patients+total_failed_patients)*100):.1f}%")
    print(f"\nEmotionMSP features saved to:")
    print(f"  {OUTPUT_BASE_DIR}/features/emotionmsp/[CONFIG]/[PATIENT_ID]/[CHUNK]_features.csv")
    print(f"\nEach CSV contains 1024 EmotionMSP features for one audio chunk.")
    print(f"\nResumable: The script will skip chunks for which feature CSVs already exist.")


if __name__ == "__main__":
    main()
