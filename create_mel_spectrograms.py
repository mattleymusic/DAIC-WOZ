#!/usr/bin/env python3
"""
Create mel-spectrograms from diarized audio chunks for DAIC-WOZ dataset.
Saves mel-spectrograms in data/features/mel_spectrograms with organized structure.
"""

import os
import json
import time
from pathlib import Path
import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm
import argparse

# -----------------------------
# Configuration
# -----------------------------
# Mel-spectrogram parameters (hard-coded as per user preference)
MEL_PARAMS = {
    'n_mels': 128,           # Number of mel frequency bins
    'n_fft': 2048,           # FFT window size
    'hop_length': 512,       # Hop length between frames
    'win_length': 2048,      # Window length
    'fmin': 0,               # Minimum frequency
    'fmax': 8000,            # Maximum frequency
    'power': 2.0,            # Power for spectrogram
    'norm': 'slaney',        # Mel filterbank normalization
    'htk': False             # HTK-style mel scaling
}

# Audio processing parameters
AUDIO_PARAMS = {
    'sample_rate': 16000,    # Target sample rate
}

# -----------------------------
# Helpers
# -----------------------------
def ensure_dir(path: Path):
    """Create directory if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)
    return path

def load_audio_safe(audio_path, target_sr=16000):
    """Load audio file safely with error handling."""
    try:
        # Load audio with librosa (handles various formats)
        audio, sr = librosa.load(audio_path, sr=target_sr, mono=True)
        return audio, sr
    except Exception as e:
        print(f"Error loading {audio_path}: {e}")
        return None, None

def create_mel_spectrogram(audio, sr, **params):
    """Create mel-spectrogram from audio."""
    try:
        # Create mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio, 
            sr=sr,
            **params
        )
        
        # Convert to log scale (dB)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        return mel_spec_db
    except Exception as e:
        print(f"Error creating mel-spectrogram: {e}")
        return None

def get_audio_files(patient_dir):
    """Get all audio files from patient diarization directory."""
    diarization_dir = patient_dir / 'diarisation_participant'
    if not diarization_dir.exists():
        return []
    
    # Get all WAV files
    audio_files = sorted(diarization_dir.glob('*.wav'))
    return audio_files

def process_patient_audio(patient_dir, output_dir, mel_params, audio_params):
    """Process all audio files for a single patient."""
    patient_id = patient_dir.name
    print(f"Processing patient: {patient_id}")
    
    # Get audio files
    audio_files = get_audio_files(patient_dir)
    if not audio_files:
        print(f"No audio files found for {patient_id}")
        return 0
    
    # Create output directory for this patient
    patient_output_dir = output_dir / patient_id
    ensure_dir(patient_output_dir)
    
    # Process each audio file
    processed_files = []
    for audio_file in tqdm(audio_files, desc=f"Processing {patient_id}"):
        try:
            # Load audio
            audio, sr = load_audio_safe(audio_file, audio_params['sample_rate'])
            if audio is None:
                continue
            
            # Create mel-spectrogram
            mel_spec = create_mel_spectrogram(audio, sr, **mel_params)
            if mel_spec is None:
                continue
            
            # Generate output filename
            audio_stem = audio_file.stem
            output_filename = f"{audio_stem}_mel.npy"
            output_path = patient_output_dir / output_filename
            
            # Save mel-spectrogram
            np.save(output_path, mel_spec)
            processed_files.append(output_filename)
            
        except Exception as e:
            print(f"Error processing {audio_file}: {e}")
            continue
    
    print(f"Completed {patient_id}: {len(processed_files)} files processed")
    return len(processed_files)

def save_specifications_metadata(output_dir, mel_params, audio_params):
    """Save a single metadata file with all specifications."""
    metadata = {
        'mel_spectrogram_specifications': {
            'description': 'Mel-spectrogram parameters for DAIC-WOZ audio processing',
            'parameters': mel_params,
            'audio_processing': audio_params,
            'notes': {
                'n_mels': 'Number of mel frequency bins (128 provides good balance)',
                'n_fft': 'FFT window size (2048 for good frequency resolution)',
                'hop_length': 'Hop length between frames (512 = ~32ms frames)',
                'win_length': 'Window length for STFT (2048 for good temporal resolution)',
                'fmin': 'Minimum frequency in Hz (0 = include all low frequencies)',
                'fmax': 'Maximum frequency in Hz (8000 covers most speech frequencies)',
                'power': 'Power for spectrogram calculation (2.0 = power spectrum)',
                'norm': 'Mel filterbank normalization (slaney = standard normalization)',
                'htk': 'HTK-style mel scaling (False = use standard mel scale)',
                'sample_rate': 'Target audio sample rate (16000 Hz = standard for speech)'
            }
        },
        'created_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'dataset': 'DAIC-WOZ',
        'purpose': 'Audio feature extraction for depression classification'
    }
    
    metadata_path = output_dir / 'mel_spectrogram_specifications.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Saved specifications metadata to: {metadata_path}")

def main():
    parser = argparse.ArgumentParser(description='Create mel-spectrograms from DAIC-WOZ audio')
    parser.add_argument('--input_dir', type=str, default='data/daic_woz',
                       help='Input directory containing patient data')
    parser.add_argument('--output_dir', type=str, default='data/features/mel_spectrograms',
                       help='Output directory for mel-spectrograms')
    parser.add_argument('--patients', type=str, nargs='+',
                       help='Specific patient IDs to process (e.g., 300_P 301_P)')
    parser.add_argument('--max_patients', type=int, default=None,
                       help='Maximum number of patients to process')
    
    args = parser.parse_args()
    
    # Setup paths
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    if not input_dir.exists():
        print(f"Input directory does not exist: {input_dir}")
        return
    
    # Create output directory
    ensure_dir(output_dir)
    
    # Save specifications metadata first
    save_specifications_metadata(output_dir, MEL_PARAMS, AUDIO_PARAMS)
    
    # Get patient directories
    if args.patients:
        patient_dirs = [input_dir / patient_id for patient_id in args.patients]
        patient_dirs = [d for d in patient_dirs if d.exists()]
    else:
        patient_dirs = [d for d in input_dir.iterdir() 
                       if d.is_dir() and d.name.endswith('_P')]
    
    # Limit number of patients if specified
    if args.max_patients:
        patient_dirs = patient_dirs[:args.max_patients]
    
    print(f"Found {len(patient_dirs)} patient directories")
    print(f"Mel-spectrogram parameters: {MEL_PARAMS}")
    print(f"Audio parameters: {AUDIO_PARAMS}")
    print(f"Output directory: {output_dir}")
    
    # Process each patient
    total_processed = 0
    start_time = time.time()
    
    for patient_dir in patient_dirs:
        try:
            files_processed = process_patient_audio(patient_dir, output_dir, MEL_PARAMS, AUDIO_PARAMS)
            if files_processed:
                total_processed += files_processed
        except Exception as e:
            print(f"Error processing {patient_dir}: {e}")
            continue
    
    # Save processing summary
    processing_summary = {
        'processing_summary': {
            'total_patients': len(patient_dirs),
            'total_files_processed': total_processed,
            'processing_time_seconds': time.time() - start_time,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
    }
    
    summary_path = output_dir / 'processing_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(processing_summary, f, indent=2)
    
    print(f"\nProcessing completed!")
    print(f"Total patients: {len(patient_dirs)}")
    print(f"Total files processed: {total_processed}")
    print(f"Processing time: {time.time() - start_time:.2f} seconds")
    print(f"Output directory: {output_dir}")
    print(f"Specifications saved to: {output_dir / 'mel_spectrogram_specifications.json'}")
    print(f"Processing summary saved to: {output_dir / 'processing_summary.json'}")

if __name__ == "__main__":
    main()