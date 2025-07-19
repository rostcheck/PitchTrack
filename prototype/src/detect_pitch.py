#!/usr/bin/env python3
"""
Detect pitch in audio files using various algorithms.
This script processes audio files and extracts pitch information.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import argparse
import json
import time

# Check if aubio is installed, otherwise provide installation instructions
try:
    import aubio
except ImportError:
    print("Error: aubio is not installed. Please install it using:")
    print("pip install aubio")
    sys.exit(1)

def detect_pitch_aubio(file_path, method="yin", buffer_size=2048, hop_size=512, sample_rate=44100):
    """
    Detect pitch using aubio library.
    
    Parameters:
    - file_path: Path to the audio file
    - method: Pitch detection method ('yin', 'yinfft', 'mcomb', 'fcomb', 'schmitt')
    - buffer_size: Buffer size for analysis
    - hop_size: Hop size between frames
    - sample_rate: Sample rate to use (0 for original file's rate)
    
    Returns:
    - times: Array of time points
    - pitches: Array of detected pitches (in Hz)
    - confidences: Array of confidence values
    """
    # Create pitch object
    pitch_o = aubio.pitch(method, buffer_size, hop_size, sample_rate)
    pitch_o.set_unit("Hz")
    pitch_o.set_silence(-40)
    pitch_o.set_tolerance(0.8)
    
    # Load audio file
    source = aubio.source(file_path, sample_rate, hop_size)
    sample_rate = source.samplerate
    
    # Lists to store results
    pitches = []
    confidences = []
    total_frames = 0
    
    # Process audio file
    while True:
        samples, read = source()
        pitch = pitch_o(samples)[0]
        confidence = pitch_o.get_confidence()
        
        pitches.append(float(pitch))
        confidences.append(float(confidence))
        total_frames += read
        
        if read < hop_size:
            break
    
    # Convert frame indices to time
    times = [t * hop_size / float(sample_rate) for t in range(len(pitches))]
    
    return times, pitches, confidences

def plot_pitch(times, pitches, confidences, title, output_path=None):
    """Plot pitch over time with confidence visualization."""
    plt.figure(figsize=(12, 6))
    
    # Create a colormap based on confidence
    colors = plt.cm.viridis(confidences)
    
    # Plot pitch points with confidence-based coloring
    for i in range(len(times)):
        if pitches[i] > 0:  # Only plot detected pitches
            plt.scatter(times[i], pitches[i], color=colors[i], alpha=0.7, s=5)
    
    # Add a colorbar for confidence
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis)
    sm.set_array(confidences)
    cbar = plt.colorbar(sm)
    cbar.set_label('Confidence')
    
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
    parser = argparse.ArgumentParser(description='Detect pitch in audio files')
    parser.add_argument('input_file', help='Path to input audio file')
    parser.add_argument('--method', choices=['yin', 'yinfft', 'mcomb', 'fcomb', 'schmitt'], 
                        default='yin', help='Pitch detection method')
    parser.add_argument('--buffer-size', type=int, default=2048, 
                        help='Buffer size for analysis')
    parser.add_argument('--hop-size', type=int, default=512, 
                        help='Hop size between frames')
    parser.add_argument('--sample-rate', type=int, default=0, 
                        help='Sample rate (0 for original file\'s rate)')
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
    print(f"Detecting pitch in {args.input_file} using {args.method} method...")
    times, pitches, confidences = detect_pitch_aubio(
        args.input_file, 
        method=args.method,
        buffer_size=args.buffer_size,
        hop_size=args.hop_size,
        sample_rate=args.sample_rate
    )
    
    # End timing
    elapsed_time = time.time() - start_time
    print(f"Pitch detection completed in {elapsed_time:.2f} seconds")
    
    # Save results
    output_json = os.path.join(args.output_dir, f"{base_filename}_{args.method}.json")
    save_results(times, pitches, confidences, output_json)
    
    # Plot results if requested
    if args.plot:
        output_plot = os.path.join(args.output_dir, f"{base_filename}_{args.method}.png")
        title = f"Pitch Detection: {base_filename} ({args.method})"
        plot_pitch(times, pitches, confidences, title, output_plot)

if __name__ == "__main__":
    main()
