#!/usr/bin/env python3
"""
Enhanced pitch detection for vocal fundamental tracking.
This script focuses on identifying the fundamental pitch of vocal lines.
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
from scipy.signal import medfilt
from scipy.ndimage import gaussian_filter1d

def detect_vocal_pitch(file_path, hop_length=512, fmin=80.0, fmax=800.0, 
                      energy_threshold=0.05, median_filter_size=11, 
                      continuity_tolerance=0.2, octave_cost=0.9):
    """
    Enhanced pitch detection optimized for vocal fundamental tracking.
    
    Parameters:
    - file_path: Path to the audio file
    - hop_length: Hop size between frames
    - fmin: Minimum frequency to detect
    - fmax: Maximum frequency to detect
    - energy_threshold: Threshold for voice activity detection
    - median_filter_size: Size of the median filter for smoothing
    - continuity_tolerance: Maximum allowed pitch change between consecutive frames (in octaves)
    - octave_cost: Cost factor for octave jumps (higher values discourage octave jumps)
    
    Returns:
    - times: Array of time points
    - pitches: Array of detected pitches (in Hz)
    - confidences: Array of confidence values
    """
    print(f"Loading audio file: {file_path}")
    # Load audio file
    y, sr = librosa.load(file_path, sr=None)
    
    print(f"Calculating energy...")
    # Calculate energy for voice activity detection
    energy = librosa.feature.rms(y=y, frame_length=hop_length*2, hop_length=hop_length)[0]
    energy = energy / np.max(energy) if np.max(energy) > 0 else energy
    
    print(f"Extracting pitch using pYIN algorithm...")
    # Use pYIN algorithm for more accurate fundamental frequency estimation
    # This is better for vocal pitch tracking than the standard piptrack
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y, 
        fmin=fmin,
        fmax=fmax,
        sr=sr,
        hop_length=hop_length,
        fill_na=None  # Don't fill unvoiced sections
    )
    
    # Convert frame indices to time
    times = librosa.times_like(f0, sr=sr, hop_length=hop_length)
    
    print(f"Post-processing pitch data...")
    # Initialize arrays for processed pitch and confidence
    processed_pitch = np.zeros_like(f0)
    confidence = np.zeros_like(f0)
    
    # Fill in confidence values
    for i in range(len(f0)):
        if voiced_flag[i] and f0[i] > 0:
            # Combine voicing probability with energy for confidence
            confidence[i] = voiced_probs[i] * (0.5 + 0.5 * energy[i])
        else:
            confidence[i] = 0
    
    # Apply threshold to remove low-confidence segments
    for i in range(len(f0)):
        if confidence[i] > energy_threshold and f0[i] > 0:
            processed_pitch[i] = f0[i]
        else:
            processed_pitch[i] = 0
    
    # Apply continuity constraints to avoid octave jumps
    for i in range(1, len(processed_pitch)):
        if processed_pitch[i] > 0 and processed_pitch[i-1] > 0:
            # Calculate octave difference
            octave_diff = np.abs(np.log2(processed_pitch[i] / processed_pitch[i-1]))
            
            # If jump is too large, try to correct it
            if octave_diff > continuity_tolerance:
                # Check if it's likely an octave error
                if abs(octave_diff - 1.0) < 0.1:  # Close to an octave jump
                    # Adjust to previous octave if confidence allows
                    if confidence[i] < confidence[i-1] * (1 + octave_cost):
                        if processed_pitch[i] > processed_pitch[i-1]:
                            processed_pitch[i] = processed_pitch[i] / 2.0
                        else:
                            processed_pitch[i] = processed_pitch[i] * 2.0
    
    # Apply median filtering to smooth the pitch contour
    valid_indices = processed_pitch > 0
    if np.any(valid_indices):
        # Create a copy for filtering
        smoothed_pitch = np.copy(processed_pitch)
        
        # Only apply filtering to segments with valid pitch
        segments = []
        segment_start = None
        
        # Find continuous segments
        for i in range(len(valid_indices)):
            if valid_indices[i] and segment_start is None:
                segment_start = i
            elif not valid_indices[i] and segment_start is not None:
                segments.append((segment_start, i))
                segment_start = None
        
        # Add the last segment if it exists
        if segment_start is not None:
            segments.append((segment_start, len(valid_indices)))
        
        # Apply median filtering to each segment
        for start, end in segments:
            if end - start > median_filter_size:
                segment = processed_pitch[start:end]
                smoothed_segment = medfilt(segment, median_filter_size)
                smoothed_pitch[start:end] = smoothed_segment
        
        processed_pitch = smoothed_pitch
    
    # Convert to lists for JSON serialization
    times_list = times.tolist()
    pitch_list = processed_pitch.tolist()
    confidence_list = confidence.tolist()
    
    return times_list, pitch_list, confidence_list

def plot_pitch(times, pitches, confidences, title, output_path=None):
    """Plot pitch over time with confidence visualization."""
    plt.figure(figsize=(12, 6))
    
    # Create a colormap based on confidence
    norm_confidences = np.array(confidences) / max(confidences) if max(confidences) > 0 else np.zeros_like(confidences)
    
    # Plot pitch points with confidence-based coloring
    valid_indices = [i for i, p in enumerate(pitches) if p > 0]
    if valid_indices:
        valid_times = [times[i] for i in valid_indices]
        valid_pitches = [pitches[i] for i in valid_indices]
        valid_confidences = [norm_confidences[i] for i in valid_indices]
        
        scatter = plt.scatter(valid_times, valid_pitches, 
                             c=valid_confidences, cmap='viridis', 
                             alpha=0.7, s=5)
        
        # Add a colorbar for confidence
        cbar = plt.colorbar(scatter)
        cbar.set_label('Normalized Confidence')
    
    # Add piano key frequencies as horizontal lines
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    A4_freq = 440.0  # A4 = 440Hz
    A4_midi = 69     # MIDI note number for A4
    
    # Add horizontal lines for piano keys in the detected pitch range
    valid_pitches = [p for p in pitches if p > 0]
    if valid_pitches:
        min_pitch = min(valid_pitches)
        max_pitch = max(valid_pitches)
        
        # Extend range slightly for better visualization
        min_midi = int(12 * np.log2(min_pitch / A4_freq) + A4_midi - 2)
        max_midi = int(12 * np.log2(max_pitch / A4_freq) + A4_midi + 2)
        
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
    if valid_pitches:
        plt.ylim(min_pitch * 0.8, max_pitch * 1.2)
    
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
    parser = argparse.ArgumentParser(description='Enhanced pitch detection for vocal fundamental tracking')
    parser.add_argument('input_file', help='Path to input audio file')
    parser.add_argument('--hop-length', type=int, default=512, 
                        help='Hop size between frames')
    parser.add_argument('--fmin', type=float, default=80.0, 
                        help='Minimum frequency to detect (Hz)')
    parser.add_argument('--fmax', type=float, default=800.0, 
                        help='Maximum frequency to detect (Hz)')
    parser.add_argument('--energy-threshold', type=float, default=0.05, 
                        help='Threshold for voice activity detection')
    parser.add_argument('--median-filter-size', type=int, default=11, 
                        help='Size of the median filter for smoothing')
    parser.add_argument('--continuity-tolerance', type=float, default=0.2, 
                        help='Maximum allowed pitch change between consecutive frames (in octaves)')
    parser.add_argument('--octave-cost', type=float, default=0.9, 
                        help='Cost factor for octave jumps')
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
    print(f"Detecting pitch in {args.input_file} using enhanced vocal pitch detection...")
    times, pitches, confidences = detect_vocal_pitch(
        args.input_file, 
        hop_length=args.hop_length,
        fmin=args.fmin,
        fmax=args.fmax,
        energy_threshold=args.energy_threshold,
        median_filter_size=args.median_filter_size,
        continuity_tolerance=args.continuity_tolerance,
        octave_cost=args.octave_cost
    )
    
    # End timing
    elapsed_time = time.time() - start_time
    print(f"Pitch detection completed in {elapsed_time:.2f} seconds")
    
    # Save results
    output_json = os.path.join(args.output_dir, f"{base_filename}_vocal.json")
    save_results(times, pitches, confidences, output_json)
    
    # Plot results if requested
    if args.plot:
        output_plot = os.path.join(args.output_dir, f"{base_filename}_vocal.png")
        title = f"Vocal Pitch Detection: {base_filename}"
        plot_pitch(times, pitches, confidences, title, output_plot)

if __name__ == "__main__":
    main()
