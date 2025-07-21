#!/usr/bin/env python3
"""
Test script for Demucs vocal extraction with timing measurements.

This script uses the Demucs model to extract vocals from audio files
and measures the execution time.
"""

import os
import sys
import argparse
import numpy as np
import librosa
import soundfile as sf
import torch
import time
from pathlib import Path

def extract_vocals_with_demucs(input_file, output_dir):
    """Extract vocals using Demucs and measure execution time."""
    try:
        from demucs.pretrained import get_model
        from demucs.apply import apply_model
        
        print("Using Demucs for vocal extraction")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load the audio file for reference
        y, sr = librosa.load(input_file, sr=None)
        
        # Save original audio
        orig_path = os.path.join(output_dir, "original.wav")
        sf.write(orig_path, y, sr)
        
        # Start timing
        start_time = time.time()
        
        # Load the model
        print("Loading Demucs model...")
        model = get_model('htdemucs')
        model.cpu()
        model.eval()
        
        # Load the audio file
        print("Loading audio file...")
        wav, sr = librosa.load(input_file, sr=model.samplerate, mono=False)
        
        # If mono, convert to stereo
        if wav.ndim == 1:
            wav = np.stack([wav, wav])
        
        # Apply the model
        print("Separating audio with Demucs...")
        ref = wav.mean(0)
        wav = torch.tensor(wav)
        with torch.no_grad():
            sources = apply_model(model, wav[None])[0]
        
        # Get the vocals and instrumental
        sources = sources.cpu().numpy()
        vocals = sources[model.sources.index('vocals')]
        
        # Create instrumental by subtracting vocals from original
        instrumental = wav.numpy() - vocals
        
        # End timing
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Save the vocals
        vocals_path = os.path.join(output_dir, "vocals_demucs.wav")
        sf.write(vocals_path, vocals.T, model.samplerate)
        print(f"Saved vocals to {vocals_path}")
        
        # Save the instrumental
        instrumental_path = os.path.join(output_dir, "instrumental_demucs.wav")
        sf.write(instrumental_path, instrumental.T, model.samplerate)
        print(f"Saved instrumental to {instrumental_path}")
        
        # Save timing information
        timing_path = os.path.join(output_dir, "timing_demucs.txt")
        with open(timing_path, 'w') as f:
            f.write(f"Execution time: {execution_time:.2f} seconds\n")
        print(f"Execution time: {execution_time:.2f} seconds")
        
        return vocals_path, execution_time
    except Exception as e:
        print(f"Error using Demucs: {e}")
        import traceback
        traceback.print_exc()
        return None, 0

def main():
    parser = argparse.ArgumentParser(description="Extract vocals using Demucs with timing")
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
        "demucs"
    )
    
    # Extract vocals
    result, execution_time = extract_vocals_with_demucs(args.input_file, output_dir)
    
    # Print summary
    print("\n=== Extraction Summary ===")
    status = "SUCCESS" if result else "FAILED"
    print(f"DEMUCS: {status}")
    if result:
        print(f"  - Output: {result}")
        print(f"  - Execution time: {execution_time:.2f} seconds")
    
    print(f"\nResults saved in: {output_dir}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
