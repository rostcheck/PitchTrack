#!/usr/bin/env python3
"""
Improved test script for Spleeter vocal extraction with best practices.

This script uses the Spleeter model with improved settings to extract vocals
from audio files and measures the execution time.
"""

import os
import sys
import argparse
import numpy as np
import time
from pathlib import Path
from scipy import signal

def apply_highpass_filter(audio, sr, cutoff=150):
    """Apply a high-pass filter to the audio to improve vocal clarity."""
    # Design a high-pass filter
    nyquist = 0.5 * sr
    normal_cutoff = cutoff / nyquist
    b, a = signal.butter(4, normal_cutoff, btype='high', analog=False)
    
    # Apply the filter
    if audio.ndim == 1:
        return signal.filtfilt(b, a, audio)
    else:
        # For multi-channel audio
        filtered = np.zeros_like(audio)
        for i in range(audio.shape[1]):
            filtered[:, i] = signal.filtfilt(b, a, audio[:, i])
        return filtered

def extract_vocals_with_spleeter(input_file, output_dir, model='4stems', high_pass=True):
    """Extract vocals using Spleeter with improved settings."""
    try:
        import librosa
        import soundfile as sf
        from spleeter.separator import Separator
        
        print(f"Using Spleeter with {model} model for vocal extraction")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Start timing
        start_time = time.time()
        
        # Load the separator with the specified model
        print(f"Loading Spleeter {model} model...")
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pretrained_models')
        os.environ['MODEL_PATH'] = model_path
        separator = Separator(f'spleeter:{model}', multiprocess=False)
        
        # Load the audio file
        print("Loading audio file...")
        waveform, sr = librosa.load(input_file, sr=44100, mono=False)
        
        # Print waveform shape for debugging
        print(f"Original waveform shape: {waveform.shape}, sample rate: {sr}")
        
        # Ensure waveform is in the correct format for Spleeter (samples, channels)
        if waveform.ndim == 1:
            # Convert mono to stereo
            waveform = np.stack([waveform, waveform]).T
        elif waveform.shape[0] == 2:
            waveform = waveform.T
        
        print(f"Waveform shape after formatting: {waveform.shape}")
        
        # Apply high-pass filter if requested
        if high_pass:
            print("Applying high-pass filter (150 Hz)...")
            waveform = apply_highpass_filter(waveform, sr)
        
        # Save original audio
        orig_path = os.path.join(output_dir, "original.wav")
        sf.write(orig_path, waveform, sr)
        
        # Separate the audio
        print("Separating audio with Spleeter...")
        prediction = separator.separate(waveform)
        
        # Print prediction shape for debugging
        for key, value in prediction.items():
            print(f"Prediction shape: {key}={value.shape}")
        
        # Get the vocals
        vocals = prediction['vocals']
        
        # Create accompaniment by combining all other stems
        if model == '2stems':
            accompaniment = prediction['accompaniment']
        elif model == '4stems':
            accompaniment = prediction['drums'] + prediction['bass'] + prediction['other']
        elif model == '5stems':
            accompaniment = prediction['drums'] + prediction['bass'] + prediction['piano'] + prediction['other']
        
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
        
        # Save individual stems if using 4stems or 5stems
        if model != '2stems':
            for stem_name, stem_audio in prediction.items():
                if stem_name != 'vocals':  # Vocals already saved
                    stem_path = os.path.join(output_dir, f"{stem_name}_spleeter.wav")
                    sf.write(stem_path, stem_audio, sr)
                    print(f"Saved {stem_name} to {stem_path}")
        
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
    parser = argparse.ArgumentParser(description="Extract vocals using Spleeter with improved settings")
    parser.add_argument("input_file", help="Path to the input audio file")
    parser.add_argument("--output", help="Output directory (default: test_results in current directory)")
    parser.add_argument("--model", choices=['2stems', '4stems', '5stems'], default='4stems',
                      help="Spleeter model to use (default: 4stems)")
    parser.add_argument("--no-highpass", action="store_true", 
                      help="Disable high-pass filter")
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' does not exist")
        return 1
    
    # Create output directory
    base_name = os.path.splitext(os.path.basename(args.input_file))[0]
    model_name = args.model.replace('stems', '')
    output_dir = args.output or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 
        "test_results", 
        base_name,
        f"spleeter_{model_name}"
    )
    
    # Extract vocals
    result, execution_time = extract_vocals_with_spleeter(
        args.input_file, 
        output_dir, 
        model=args.model, 
        high_pass=not args.no_highpass
    )
    
    # Print summary
    print("\n=== Extraction Summary ===")
    status = "SUCCESS" if result else "FAILED"
    print(f"SPLEETER ({args.model}): {status}")
    if result:
        print(f"  - Output: {result}")
        print(f"  - Execution time: {execution_time:.2f} seconds")
    
    print(f"\nResults saved in: {output_dir}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
