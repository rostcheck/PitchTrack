#!/usr/bin/env python3
"""
Test script for Spleeter vocal extraction with timing measurements.

This script uses the Spleeter model to extract vocals from audio files
and measures the execution time.
"""

import os
import sys
import argparse
import numpy as np
import time
from pathlib import Path

def extract_vocals_with_spleeter(input_file, output_dir):
    """Extract vocals using Spleeter and measure execution time."""
    try:
        import librosa
        import soundfile as sf
        from spleeter.separator import Separator
        
        print("Using Spleeter for vocal extraction")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Start timing
        start_time = time.time()
        
        # Load the separator
        print("Loading Spleeter model...")
        separator = Separator('spleeter:2stems')
        
        # Load the audio file
        print("Loading audio file...")
        waveform, sr = librosa.load(input_file, sr=44100, mono=False)
        
        # Print waveform shape for debugging
        print(f"Original waveform shape: {waveform.shape}, sample rate: {sr}")
        
        # Ensure waveform is in the correct format for Spleeter (samples, channels)
        if waveform.ndim == 1:
            # Convert mono to stereo
            waveform = np.stack([waveform, waveform])
        
        # Transpose to (samples, channels) if needed
        if waveform.shape[0] == 2:
            waveform = waveform.T
        
        print(f"Waveform shape after formatting: {waveform.shape}")
        
        # Save original audio
        orig_path = os.path.join(output_dir, "original.wav")
        sf.write(orig_path, waveform, sr)
        
        # Separate the audio
        print("Separating audio with Spleeter...")
        prediction = separator.separate(waveform)
        
        # Print prediction shape for debugging
        for key, value in prediction.items():
            print(f"Prediction shape: {key}={value.shape}")
        
        # Get the vocals and accompaniment
        vocals = prediction['vocals']
        accompaniment = prediction['accompaniment']
        
        # End timing
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Save the vocals
        vocals_path = os.path.join(output_dir, "vocals_spleeter.wav")
        sf.write(vocals_path, vocals, sr)
        print(f"Saved vocals to {vocals_path}")
        
        # Save the instrumental
        instrumental_path = os.path.join(output_dir, "instrumental_spleeter.wav")
        sf.write(instrumental_path, accompaniment, sr)
        print(f"Saved instrumental to {instrumental_path}")
        
        # Save timing information
        timing_path = os.path.join(output_dir, "timing_spleeter.txt")
        with open(timing_path, 'w') as f:
            f.write(f"Execution time: {execution_time:.2f} seconds\n")
        print(f"Execution time: {execution_time:.2f} seconds")
        
        return vocals_path, execution_time
    except Exception as e:
        print(f"Error using Spleeter: {e}")
        import traceback
        traceback.print_exc()
        return None, 0

def main():
    parser = argparse.ArgumentParser(description="Extract vocals using Spleeter with timing")
    parser.add_argument("input_file", help="Path to the input audio file")
    parser.add_argument("--output", help="Output directory (default: test_results in current directory)")
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' does not exist")
        return 1
    
    # Create output directory
    base_name = os.path.splitext(os.path.basename(args.input_file))[0]
    output_dir = args.output or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 
        "test_results", 
        base_name,
        "spleeter"
    )
    
    # Extract vocals
    result, execution_time = extract_vocals_with_spleeter(args.input_file, output_dir)
    
    # Print summary
    print("\n=== Extraction Summary ===")
    status = "SUCCESS" if result else "FAILED"
    print(f"SPLEETER: {status}")
    if result:
        print(f"  - Output: {result}")
        print(f"  - Execution time: {execution_time:.2f} seconds")
    
    print(f"\nResults saved in: {output_dir}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
