import os
import librosa
import numpy as np
import soundfile as sf
from pathlib import Path
import argparse


def create_audio_chunks(audio_path, chunk_length, overlap, sample_rate=16000):
    """
    Create overlapping audio chunks from an audio file.
    
    Args:
        audio_path (str): Path to the input audio file
        chunk_length (float): Length of each chunk in seconds
        overlap (float): Overlap between chunks in seconds
        sample_rate (int): Sample rate for the audio
    
    Returns:
        list: List of audio chunks as numpy arrays
    """
    # Load audio file
    audio, sr = librosa.load(audio_path, sr=sample_rate)
    
    # Calculate chunk parameters
    chunk_samples = int(chunk_length * sr)
    overlap_samples = int(overlap * sr)
    step_samples = chunk_samples - overlap_samples
    
    chunks = []
    start_sample = 0
    
    while start_sample < len(audio):
        end_sample = start_sample + chunk_samples
        
        if end_sample > len(audio):
            # Pad with silence if the last chunk is too short
            chunk = audio[start_sample:]
            padding_length = chunk_samples - len(chunk)
            chunk = np.pad(chunk, (0, padding_length), mode='constant')
        else:
            chunk = audio[start_sample:end_sample]
        
        chunks.append(chunk)
        start_sample += step_samples
    
    return chunks


def process_patient_folder(patient_path, output_base_path, chunk_length, overlap, sample_rate=16000):
    """
    Process all audio files in a patient's diarisation_participant folder.
    
    Args:
        patient_path (str): Path to the patient folder (e.g., 300_P)
        output_base_path (str): Base path for output directory
        chunk_length (float): Length of each chunk in seconds
        overlap (float): Overlap between chunks in seconds
        sample_rate (int): Sample rate for the audio
    """
    patient_name = os.path.basename(patient_path)
    
    # Look for diarisation_participant folder
    diarisation_path = os.path.join(patient_path, "diarisation_participant")
    if not os.path.exists(diarisation_path):
        print(f"Warning: diarisation_participant folder not found in {patient_path}")
        return
    
    # Create output directory
    output_dir_name = f"{chunk_length:.1f}s_{overlap:.1f}s_overlap"
    output_path = os.path.join(output_base_path, patient_name, output_dir_name)
    os.makedirs(output_path, exist_ok=True)
    
    # Process all audio files in the diarisation_participant folder
    audio_files = []
    for file in os.listdir(diarisation_path):
        if file.lower().endswith(('.wav', '.mp3', '.flac', '.m4a')):
            audio_files.append(os.path.join(diarisation_path, file))
    
    if not audio_files:
        print(f"No audio files found in {diarisation_path}")
        return
    
    print(f"Processing {len(audio_files)} audio files for {patient_name}")
    
    for audio_file in audio_files:
        try:
            # Create chunks
            chunks = create_audio_chunks(audio_file, chunk_length, overlap, sample_rate)
            
            # Save chunks
            base_filename = os.path.splitext(os.path.basename(audio_file))[0]
            
            for i, chunk in enumerate(chunks):
                chunk_filename = f"{base_filename}_chunk{i+1:02d}.wav"
                chunk_path = os.path.join(output_path, chunk_filename)
                
                sf.write(chunk_path, chunk, sample_rate)
            
            print(f"  Created {len(chunks)} chunks from {os.path.basename(audio_file)}")
            
        except Exception as e:
            print(f"  Error processing {audio_file}: {e}")


def main():
    # Configuration parameters
    CHUNK_LENGTH = 3.0  # seconds
    OVERLAP = 1.5       # seconds
    SAMPLE_RATE = 16000 # Hz
    
    # Paths
    DATA_DIR = "data/daic_woz"
    OUTPUT_DIR = "data/created_data"
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Get all patient folders
    patient_folders = []
    for item in os.listdir(DATA_DIR):
        item_path = os.path.join(DATA_DIR, item)
        if os.path.isdir(item_path) and item.endswith('_P'):
            patient_folders.append(item_path)
    
    patient_folders.sort()  # Sort to process in order
    
    print(f"Found {len(patient_folders)} patient folders")
    print(f"Chunk length: {CHUNK_LENGTH}s")
    print(f"Overlap: {OVERLAP}s")
    print(f"Sample rate: {SAMPLE_RATE} Hz")
    print("-" * 50)
    
    # Process each patient folder
    for patient_folder in patient_folders:
        process_patient_folder(
            patient_folder, 
            OUTPUT_DIR, 
            CHUNK_LENGTH, 
            OVERLAP, 
            SAMPLE_RATE
        )
    
    print("-" * 50)
    print("Processing complete!")


if __name__ == "__main__":
    main()