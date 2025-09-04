#!/usr/bin/env python3
"""
Simple test for HuBERT feature extractor.
Tests with a single audio file to ensure the basic functionality works.
"""

import os
import sys
import tempfile
import shutil
import numpy as np
import pandas as pd

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from feature_extraction.hubert_extractor import load_hubert_model, extract_hubert_features


def create_test_audio():
    """Create a simple test audio file."""
    import torch
    import torchaudio
    
    # Create a simple sine wave
    sample_rate = 16000
    duration = 3.0
    frequency = 440  # A4 note
    
    t = torch.linspace(0, duration, int(sample_rate * duration))
    waveform = torch.sin(2 * np.pi * frequency * t)
    waveform = waveform.unsqueeze(0)  # Add channel dimension
    
    # Create temporary file
    temp_dir = tempfile.mkdtemp()
    temp_file = os.path.join(temp_dir, "test_audio.wav")
    
    # Save audio file
    torchaudio.save(temp_file, waveform, sample_rate)
    
    return temp_file, temp_dir


def test_single_file():
    """Test HuBERT feature extraction on a single file."""
    print("Testing HuBERT feature extraction...")
    
    # Create test audio
    test_audio_path, temp_dir = create_test_audio()
    print(f"Created test audio: {test_audio_path}")
    
    try:
        # Load HuBERT model
        print("Loading HuBERT model...")
        model, device = load_hubert_model(force_cpu=True)  # Force CPU for testing
        print(f"Model loaded on device: {device}")
        
        # Extract features
        print("Extracting features...")
        features = extract_hubert_features(test_audio_path, model, device)
        
        # Check feature dimensions
        print(f"Feature vector shape: {features.shape}")
        print(f"Feature vector length: {len(features)}")
        
        if len(features) == 1024:
            print("✓ Feature extraction successful! Got 1024-dimensional HuBERT features.")
            
            # Print some statistics
            print(f"Feature statistics:")
            print(f"  Mean: {np.mean(features):.6f}")
            print(f"  Std: {np.std(features):.6f}")
            print(f"  Min: {np.min(features):.6f}")
            print(f"  Max: {np.max(features):.6f}")
            
            return True
            
        else:
            print(f"✗ Feature extraction failed! Expected 1024 features, got {len(features)}")
            return False
            
    except Exception as e:
        print(f"✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Clean up
        shutil.rmtree(temp_dir)
        print("Cleaned up temporary files")


if __name__ == "__main__":
    print("=" * 60)
    print("Simple HuBERT Feature Extractor Test")
    print("=" * 60)
    
    success = test_single_file()
    
    print("\n" + "=" * 60)
    if success:
        print("✓ Test passed! HuBERT feature extractor is working correctly.")
        print("\nYou can now run the full feature extraction:")
        print("python src/feature_extraction/hubert_extractor.py")
    else:
        print("✗ Test failed! Please check the error messages above.")
    print("=" * 60)
