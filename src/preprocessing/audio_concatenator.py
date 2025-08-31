import os
import librosa
import soundfile as sf
import numpy as np
from pathlib import Path

"""
    !!! INPUT STRUCTURE !!!
    
    data/daic_woz/
    ├── 300_P/
    │   └── diarisation_participant/
    │       ├── part_0_62.328_63.178.wav
    │       ├── part_1_68.978_70.288.wav
    │       ├── part_2_75.028_78.128.wav
    │       └── ...
    ├── 301_P/
    │   └── diarisation_participant/
    │       └── ...
    
    !!! OUTPUT STRUCTURE !!!
    
    data/created_data/
    ├── 300_P/
    │   └── concatenated_diarisation/
    │       └── 300_P_concatenated_diarisation.wav
    ├── 301_P/
    │   └── concatenated_diarisation/
    │       └── 301_P_concatenated_diarisation.wav
    └── ...
    
    !!! PROCESSING DETAILS !!!
    
    - Loads all diarised audio snippets from diarisation_participant folders
    - Sorts them chronologically by part number (part_0, part_1, part_2, etc.)
    - Concatenates them into one continuous audio file per patient
    - Maintains original 16kHz sample rate and audio quality
    - Output files contain everything the participant said in chronological order
    - No padding or gaps between audio segments
"""

def concatenate_diarised_audio(patient_path, output_base_path, sample_rate=16000):
    """
    Concatenate all diarised audio snippets from a patient's diarisation_participant folder
    into one continuous audio file.
    
    Args:
        patient_path (str): Path to the patient folder (e.g., 300_P)
        output_base_path (str): Base path for output directory
        sample_rate (int): Sample rate for the audio (maintains original)
    
    Returns:
        bool: True if successful, False otherwise
    """
    patient_name = os.path.basename(patient_path)
    
    # Look for diarisation_participant folder
    diarisation_path = os.path.join(patient_path, "diarisation_participant")
    if not os.path.exists(diarisation_path):
        print(f"Warning: diarisation_participant folder not found in {patient_path}")
        return False
    
    # Create output directory
    output_path = os.path.join(output_base_path, patient_name, "concatenated_diarisation")
    os.makedirs(output_path, exist_ok=True)
    
    # Get all audio files and sort them by part number
    audio_files = []
    for file in os.listdir(diarisation_path):
        if file.lower().endswith(('.wav', '.mp3', '.flac', '.m4a')):
            audio_files.append(os.path.join(diarisation_path, file))
    
    if not audio_files:
        print(f"No audio files found in {diarisation_path}")
        return False
    
    # Sort files by part number to maintain chronological order
    audio_files.sort(key=lambda x: extract_part_number(os.path.basename(x)))
    
    print(f"Processing {len(audio_files)} audio files for {patient_name}")
    
    try:
        # Load and concatenate all audio files
        concatenated_audio = []
        total_duration = 0
        
        for i, audio_file in enumerate(audio_files):
            print(f"  Loading {os.path.basename(audio_file)} ({i+1}/{len(audio_files)})")
            
            # Load audio file
            audio, sr = librosa.load(audio_file, sr=sample_rate)
            
            # Verify sample rate consistency
            if sr != sample_rate:
                print(f"    Warning: {audio_file} has sample rate {sr} Hz, expected {sample_rate} Hz")
                # Resample if necessary
                audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)
            
            concatenated_audio.append(audio)
            duration = len(audio) / sample_rate
            total_duration += duration
            
            print(f"    Duration: {duration:.2f}s, Total so far: {total_duration:.2f}s")
        
        # Concatenate all audio segments
        print(f"  Concatenating {len(concatenated_audio)} audio segments...")
        final_audio = np.concatenate(concatenated_audio)
        
        # Save concatenated audio
        output_filename = f"{patient_name}_concatenated_diarisation.wav"
        output_filepath = os.path.join(output_path, output_filename)
        
        sf.write(output_filepath, final_audio, sample_rate)
        
        final_duration = len(final_audio) / sample_rate
        print(f"  Successfully created: {output_filename}")
        print(f"  Final duration: {final_duration:.2f}s")
        print(f"  Saved to: {output_filepath}")
        
        return True
        
    except Exception as e:
        print(f"  Error processing {patient_name}: {e}")
        return False


def extract_part_number(filename):
    """
    Extract the part number from filenames like 'part_0_62.328_63.178.wav'
    
    Args:
        filename (str): The filename to extract part number from
    
    Returns:
        int: The part number for sorting
    """
    try:
        # Extract part number (e.g., "0" from "part_0_62.328_63.178.wav")
        parts = filename.split('_')
        if len(parts) >= 2 and parts[0] == 'part':
            return int(parts[1])
        return 0
    except (ValueError, IndexError):
        return 0


def main():
    # Configuration parameters
    SAMPLE_RATE = 16000  # Hz - maintains original DAIC-WOZ format
    
    # Paths - adjusted for src/preprocessing directory
    # Go up two levels from src/preprocessing to reach project root
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    DATA_DIR = os.path.join(PROJECT_ROOT, "data", "daic_woz")
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "created_data")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Get all patient folders
    patient_folders = []
    for item in os.listdir(DATA_DIR):
        item_path = os.path.join(DATA_DIR, item)
        if os.path.isdir(item_path) and item.endswith('_P'):
            patient_folders.append(item_path)
    
    patient_folders.sort()  # Sort to process in order
    
    print(f"Script location: {os.path.abspath(__file__)}")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Data directory: {DATA_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Found {len(patient_folders)} patient folders")
    print(f"Sample rate: {SAMPLE_RATE} Hz")
    print("-" * 60)
    
    successful = 0
    failed = 0
    
    # Process each patient folder
    for patient_folder in patient_folders:
        print(f"\nProcessing: {os.path.basename(patient_folder)}")
        print("-" * 40)
        
        if concatenate_diarised_audio(patient_folder, OUTPUT_DIR, SAMPLE_RATE):
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
    print("\nConcatenated audio files saved to:")
    print(f"  {OUTPUT_DIR}/[PATIENT_ID]/concatenated_diarisation/")
    print("\nNext step: Use these concatenated files for chunking with different lengths!")


if __name__ == "__main__":
    main()