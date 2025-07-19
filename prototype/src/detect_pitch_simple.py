#!/usr/bin/env python3
"""
Simplified pitch detection using librosa instead of aubio.
This script processes audio files and extracts pitch information.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import argparse
import json
import time

def detect_pitch_librosa(file_path, hop_length=512, fmin=50, fmax=2000):
    """
    Detect pitch using librosa library.
    
    Parameters:
    - file_path: Path to the audio file
    - hop_length: Hop size between frames
    - fmin: Minimum frequency to detect
    - fmax: Maximum frequency to detect
    
    Returns:
    - times: Array of time points
    - pitches: Array of detected pitches (in Hz)
    - confidences: Array of confidence values (magnitude of pitch)
    """
    # Load audio file
    y, sr = librosa.load(file_path, sr=None)
    
    # Extract pitch
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr, hop_length=hop_length, 
                                          fmin=fmin, fmax=fmax)
    
    # Convert frame indices to time
    times = librosa.times_like(pitches[0], sr=sr, hop_length=hop_length)
    
    # Extract the most prominent pitch for each frame
    pitch_values = []
    confidence_values = []
    
    for t in range(pitches.shape[1]):
        index = magnitudes[:, t].argmax()
        pitch = pitches[index, t]
        confidence = magnitudes[index, t]
        
        # Only include pitches with sufficient magnitude
        if confidence > 0:
            pitch_values.append(float(pitch))
        else:
            pitch_values.append(0.0)
        
        confidence_values.append(float(confidence))
    
    return list(times), pitch_values, confidence_values

def plot_pitch(times, pitches, confidences, title, output_path=None):
    """Plot pitch over time with confidence visualization."""
    plt.figure(figsize=(12, 6))
    
    # Create a colormap based on confidence
    norm_confidences = np.array(confidences) / max(confidences) if max(confidences) > 0 else np.zeros_like(confidences)
    
    # Plot pitch points with confidence-based coloring
    scatter = plt.scatter([t for t, p in zip(times, pitches) if p > 0],
                         [p for p in pitches if p > 0],
                         c=[c for c, p in zip(norm_confidences, pitches) if p > 0],
                         cmap='viridis', alpha=0.7, s=5)
    
    # Add a colorbar for confidence
    cbar = plt.colorbar(scatter)
    cbar.set_label('Normalized Confidence')
    
    # Add piano key frequencies as horizontal lines
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    A4_freq = 440.0  # A4 = 440Hz
    A4_midi = 69     # MIDI note number for A4
    
    # Add horizontal lines for piano keys in the detected pitch range
    min_pitch = min([p for p in pitches if p > 0]) if any(p > 0 for p in pitches) else 0
    max_pitch = max(pitches) if any(p > 0 for p in pitches) else 1000
    
    # Extend range slightly for better visualization
    min_midi = int(12 * np.log2(min_pitch / A4_freq) + A4_midi - 2) if min_pitch > 0 else 48
    max_midi = int(12 * np.log2(max_pitch / A4_freq) + A4_midi + 2) if max_pitch > 0 else 84
    
    for midi_note in range(min_midi, max_midi + 1):
        freq = A4_freq * 2 ** ((midi_note - A4_midi) / 12)
        note_name = note_names[midi_note % 12]
        octave = midi_note // 12 - 1
        label = f"{note_name}{octave}" if midi_note % 12 == 0 else None  # Only label C notes
        plt.axhline(y=freq, color='gray', linestyle='--', alpha=0.3, label=label)
    
    # Set plot properties
    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.yscale('log')  # Logarithmic scale for better visualization of musical intervals
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # Set y-axis limits to focus on the relevant pitch range
    plt.ylim(min_pitch * 0.8 if min_pitch > 0 else 50, max_pitch * 1.2)
    
    # Add legend for C notes
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='upper right')
    
    # Save or show the plot
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    else:
        plt.show()
    
    plt.close()

def save_results(times, pitches, confidences, output_path):
    """Save pitch detection results to a JSON file."""
    results = {
        "times": times,
        "pitches": pitches,
        "confidences": confidences
    }
    
    with open(output_path, 'w') as f:
        json.dump(results, f)
    
    print(f"Results saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Detect pitch in audio files using librosa')
    parser.add_argument('input_file', help='Path to input audio file')
    parser.add_argument('--hop-length', type=int, default=512, 
                        help='Hop size between frames')
    parser.add_argument('--fmin', type=float, default=50.0, 
                        help='Minimum frequency to detect (Hz)')
    parser.add_argument('--fmax', type=float, default=2000.0, 
                        help='Maximum frequency to detect (Hz)')
    parser.add_argument('--plot', action='store_true', 
                        help='Plot the detected pitch')
    parser.add_argument('--output-dir', default=None, 
                        help='Directory to save results and plots')
    
    args = parser.parse_args()
    
    # Create output directory if specified
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    else:
        # Default to results directory in project
        args.output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Get base filename without extension
    base_filename = os.path.splitext(os.path.basename(args.input_file))[0]
    
    # Start timing
    start_time = time.time()
    
    # Detect pitch
    print(f"Detecting pitch in {args.input_file} using librosa...")
    times, pitches, confidences = detect_pitch_librosa(
        args.input_file, 
        hop_length=args.hop_length,
        fmin=args.fmin,
        fmax=args.fmax
    )
    
    # End timing
    elapsed_time = time.time() - start_time
    print(f"Pitch detection completed in {elapsed_time:.2f} seconds")
    
    # Save results
    output_json = os.path.join(args.output_dir, f"{base_filename}_librosa.json")
    save_results(times, pitches, confidences, output_json)
    
    # Plot results if requested
    if args.plot:
        output_plot = os.path.join(args.output_dir, f"{base_filename}_librosa.png")
        title = f"Pitch Detection: {base_filename} (librosa)"
        plot_pitch(times, pitches, confidences, title, output_plot)

if __name__ == "__main__":
    main()
