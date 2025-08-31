import os
import librosa
import soundfile as sf
import numpy as np
from pathlib import Path

"""
    !!! INPUT STRUCTURE !!!
    
    data/created_data/concatenated_diarisation/
    ├── 300_P/
    │   └── 300_P_concatenated.wav
    ├── 301_P/
    │   └── 301_P_concatenated.wav
    └── ...
    
    !!! OUTPUT STRUCTURE !!!
    
    data/created_data/
    ├── concatenated_diarisation/
    │   └── [patient_folders]/
    │       └── [concatenated_files]
    ├── 10.0s_5.0s_overlap/
    │   ├── 300_P/
    │   │   ├── 300_P_chunk001.wav
    │   │   ├── 300_P_chunk002.wav
    │   │   └── ...
    │   ├── 301_P/
    │   │   └── ...
    │   └── ...
    └── [other_chunk_sizes]/
        └── [patient_folders]/
            └── [chunk_files]
    
    !!! PROCESSING DETAILS !!!
    
    - Loads concatenated audio files from concatenated_diarisation folders
    - Creates overlapping chunks of specified length and overlap
    - Last chunk may be silence-padded if it's shorter than chunk_length
    - Maintains original 16kHz sample rate and audio quality
    - Output organized by chunk configuration, then by patient
    - Easy to delete all chunks of a specific size by removing the config folder
    """

def create_audio_chunks(audio_path, chunk_length, overlap, sample_rate=16000):
    """
    Create overlapping audio chunks from a concatenated audio file.
    
    Args:
        audio_path (str): Path to the input concatenated audio file
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


def process_concatenated_audio(concatenated_audio_path, output_base_path, chunk_length, overlap, sample_rate=16000):
    """
    Process a concatenated audio file and create chunks.
    
    Args:
        concatenated_audio_path (str): Path to the concatenated audio file
        output_base_path (str): Base path for output directory
        chunk_length (float): Length of each chunk in seconds
        overlap (float): Overlap between chunks in seconds
        sample_rate (int): Sample rate for the audio
    """
    # Extract patient name from path (e.g., "300_P" from "300_P_concatenated.wav")
    filename = os.path.basename(concatenated_audio_path)
    patient_name = filename.replace('_concatenated.wav', '')
    
    # Create output directory with descriptive name
    output_dir_name = f"{chunk_length}s_{overlap}s_overlap"
    output_path = os.path.join(output_base_path, output_dir_name, patient_name)
    os.makedirs(output_path, exist_ok=True)
    
    try:
        # Create chunks
        chunks = create_audio_chunks(concatenated_audio_path, chunk_length, overlap, sample_rate)
        
        # Save chunks
        for i, chunk in enumerate(chunks):
            chunk_filename = f"{patient_name}_chunk{i+1:03d}.wav"
            chunk_path = os.path.join(output_path, chunk_filename)
            
            sf.write(chunk_path, chunk, sample_rate)
        
        print(f"  Created {len(chunks)} chunks from {filename}")
        print(f"  Saved to: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"  Error processing {filename}: {e}")
        return False


def main():
    # Configuration parameters - CHANGE THESE VALUES AS NEEDED
    CHUNK_LENGTH = 30.0  # seconds
    OVERLAP = 15.0        # seconds
    SAMPLE_RATE = 16000  # Hz - maintains original DAIC-WOZ format
    
    # Paths - adjusted for src/preprocessing directory
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    CONCATENATED_DIR = os.path.join(PROJECT_ROOT, "data", "created_data", "concatenated_diarisation")
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "created_data")
    
    # Verify concatenated audio directory exists
    if not os.path.exists(CONCATENATED_DIR):
        print(f"Error: Concatenated audio directory not found: {CONCATENATED_DIR}")
        print("Please run audio_concatenator.py first to create concatenated audio files.")
        return
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Get all patient folders in concatenated_diarisation
    patient_folders = []
    for item in os.listdir(CONCATENATED_DIR):
        item_path = os.path.join(CONCATENATED_DIR, item)
        if os.path.isdir(item_path):
            patient_folders.append(item_path)
    
    patient_folders.sort()  # Sort to process in order
    
    print(f"Script location: {os.path.abspath(__file__)}")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Concatenated audio directory: {CONCATENATED_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Found {len(patient_folders)} patient folders")
    print(f"Chunk length: {CHUNK_LENGTH}s")
    print(f"Overlap: {OVERLAP}s")
    print(f"Sample rate: {SAMPLE_RATE} Hz")
    print("-" * 60)
    
    successful = 0
    failed = 0
    
    # Process each patient folder
    for patient_folder in patient_folders:
        patient_name = os.path.basename(patient_folder)
        print(f"\nProcessing: {patient_name}")
        print("-" * 40)
        
        # Look for concatenated audio file
        concatenated_audio_path = os.path.join(patient_folder, f"{patient_name}_concatenated.wav")
        
        if not os.path.exists(concatenated_audio_path):
            print(f"  Warning: Concatenated audio file not found: {concatenated_audio_path}")
            failed += 1
            continue
        
        if process_concatenated_audio(concatenated_audio_path, OUTPUT_DIR, CHUNK_LENGTH, OVERLAP, SAMPLE_RATE):
            successful += 1
        else:
            failed += 1
    
    print("\n" + "=" * 60)
    print("PROCESSING SUMMARY")
    print("=" * 60)
    print(f"Total patients: {len(patient_folders)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Success rate: {(successful/len(patient_folders)*100):.1f}%")
    print(f"\nAudio chunks saved to:")
    print(f"  {OUTPUT_DIR}/{CHUNK_LENGTH:.0f}s_{OVERLAP:.0f}s_overlap/[PATIENT_ID]/")
    print(f"\nTo change chunk parameters, modify CHUNK_LENGTH and OVERLAP in the script.")
    print(f"To delete all chunks of this size, remove:")
    print(f"  {OUTPUT_DIR}/{CHUNK_LENGTH:.0f}s_{OVERLAP:.0f}s_overlap/")


if __name__ == "__main__":
    main()