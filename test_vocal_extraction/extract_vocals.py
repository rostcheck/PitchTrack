#!/usr/bin/env python3
"""
Improved Vocal Extraction Test Script

This script extracts vocals from music files using different methods and
saves the results in clearly labeled directories for comparison.

Usage:
  python extract_vocals.py path/to/music.mp3 [--methods method1,method2,...]
"""

import os
import sys
import argparse
import librosa
import numpy as np
import soundfile as sf
from pathlib import Path

# Import the local vocal extractor test module
from vocal_extractor_test import VocalExtractor

def extract_with_method(input_file, method, output_dir):
    """
    Extract vocals using the specified method and save to output directory.
    
    Args:
        input_file: Path to the input audio file
        method: Extraction method to use
        output_dir: Directory to save the extracted audio
        
    Returns:
        Path to the extracted vocals file or None if extraction failed
    """
    print(f"\n=== Extracting vocals using {method.upper()} method ===")
    
    # Create method-specific output directory
    method_dir = os.path.join(output_dir, method)
    os.makedirs(method_dir, exist_ok=True)
    
    try:
        # Create the extractor
        extractor = VocalExtractor(method=method, output_dir=method_dir)
        
        # Load the original audio for comparison
        y_orig, sr = librosa.load(input_file, sr=None)
        
        # Save the original audio for reference
        orig_path = os.path.join(method_dir, "original.wav")
        sf.write(orig_path, y_orig, sr)
        
        # Extract vocals
        temp_vocals_path = extractor.extract_vocals(input_file, return_audio=False)
        
        # Ensure the vocals file is in the expected location with the expected name
        final_vocals_path = os.path.join(method_dir, f"vocals_{method}.wav")
        
        # If the vocals file exists but is not in the expected location, move it
        if temp_vocals_path and os.path.exists(temp_vocals_path):
            if temp_vocals_path != final_vocals_path:
                # Load the vocals audio
                y_vocals, sr_vocals = librosa.load(temp_vocals_path, sr=sr)
                
                # Save to the final location
                sf.write(final_vocals_path, y_vocals, sr_vocals)
                
                # Remove the temporary file
                if os.path.exists(temp_vocals_path):
                    os.remove(temp_vocals_path)
            
            # If we have both the original and vocals, create an instrumental track
            try:
                y_vocals, sr_vocals = librosa.load(final_vocals_path, sr=sr)
                
                # Make sure lengths match
                min_length = min(len(y_orig), len(y_vocals))
                y_orig_trimmed = y_orig[:min_length]
                y_vocals_trimmed = y_vocals[:min_length]
                
                # Create instrumental by subtracting vocals from original
                # Phase-aware subtraction would be better, but this is a simple approach
                y_instrumental = y_orig_trimmed - y_vocals_trimmed
                
                # Save instrumental
                instrumental_path = os.path.join(method_dir, f"instrumental_{method}.wav")
                sf.write(instrumental_path, y_instrumental, sr)
                print(f"Created instrumental track: {os.path.basename(instrumental_path)}")
            except Exception as e:
                print(f"Could not create instrumental track: {e}")
        
        print(f"Extraction complete. Files saved in: {method_dir}")
        return final_vocals_path
        
    except Exception as e:
        print(f"Error extracting vocals with {method}: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    parser = argparse.ArgumentParser(description="Extract vocals using different methods")
    parser.add_argument("input_file", help="Path to the input audio file")
    parser.add_argument("--methods", default="librosa,spleeter,demucs",
                       help="Comma-separated list of methods to try (librosa,spleeter,demucs)")
    parser.add_argument("--output", help="Output directory (default: test_results in current directory)")
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' does not exist")
        return 1
    
    # Create output directory
    output_dir = args.output or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 
        "test_results", 
        os.path.splitext(os.path.basename(args.input_file))[0]
    )
    os.makedirs(output_dir, exist_ok=True)
    
    # Get methods to test
    methods = [m.strip() for m in args.methods.split(",")]
    
    # Extract vocals with each method
    results = {}
    for method in methods:
        if method not in ["librosa", "spleeter", "demucs"]:
            print(f"Warning: Unknown method '{method}', skipping")
            continue
            
        result_path = extract_with_method(args.input_file, method, output_dir)
        results[method] = result_path
    
    # Print summary
    print("\n=== Extraction Summary ===")
    for method, path in results.items():
        status = "SUCCESS" if path else "FAILED"
        print(f"{method.upper()}: {status}")
        if path:
            print(f"  - Output: {path}")
    
    print(f"\nAll results saved in: {output_dir}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
