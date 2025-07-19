#!/usr/bin/env python3
"""
Utility functions for the PitchTrack prototype.
"""

import os
import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
import subprocess
import platform

def note_to_freq(note_name):
    """
    Convert a note name to its frequency in Hz.
    Example: note_to_freq("A4") returns 440.0
    
    Parameters:
    - note_name: String in format "NoteOctave" (e.g., "A4", "C5", "F#3")
    
    Returns:
    - Frequency in Hz
    """
    notes = {"C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, 
             "E": 4, "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8, 
             "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11}
    
    # Extract note and octave
    if len(note_name) == 2:
        note = note_name[0]
        octave = int(note_name[1])
    else:
        note = note_name[:2]
        octave = int(note_name[2])
    
    # Calculate frequency
    note_number = notes[note]
    frequency = 440.0 * (2.0 ** ((note_number - 9) / 12.0 + (octave - 4)))
    
    return frequency

def freq_to_note(frequency):
    """
    Convert a frequency to the closest note name.
    Example: freq_to_note(440.0) returns "A4"
    
    Parameters:
    - frequency: Frequency in Hz
    
    Returns:
    - Note name in format "NoteOctave" (e.g., "A4", "C5", "F#3")
    """
    if frequency <= 0:
        return "N/A"
    
    notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    
    # Calculate note number
    note_number = 12 * (np.log2(frequency / 440.0)) + 69
    
    # Round to nearest note
    note_number = round(note_number)
    
    # Calculate octave and note
    octave = (note_number // 12) - 1
    note = notes[note_number % 12]
    
    return f"{note}{octave}"

def cents_deviation(detected_freq, reference_freq):
    """
    Calculate the deviation in cents between detected and reference frequencies.
    
    Parameters:
    - detected_freq: Detected frequency in Hz
    - reference_freq: Reference frequency in Hz
    
    Returns:
    - Deviation in cents (100 cents = 1 semitone)
    """
    if detected_freq <= 0 or reference_freq <= 0:
        return float('nan')
    
    return 1200 * np.log2(detected_freq / reference_freq)

def play_audio(file_path):
    """
    Play an audio file using the system's default audio player.
    
    Parameters:
    - file_path: Path to the audio file
    """
    system = platform.system()
    
    try:
        if system == "Darwin":  # macOS
            subprocess.call(["afplay", file_path])
        elif system == "Linux":
            subprocess.call(["aplay", file_path])
        elif system == "Windows":
            os.startfile(file_path)
        else:
            print(f"Unsupported platform: {system}")
    except Exception as e:
        print(f"Error playing audio: {e}")

def visualize_waveform(file_path, output_path=None):
    """
    Visualize the waveform of an audio file.
    
    Parameters:
    - file_path: Path to the audio file
    - output_path: Path to save the visualization (if None, display instead)
    """
    # Read audio file
    sample_rate, data = wavfile.read(file_path)
    
    # Convert to mono if stereo
    if len(data.shape) > 1:
        data = data.mean(axis=1)
    
    # Normalize data
    data = data / np.max(np.abs(data))
    
    # Create time axis
    time = np.arange(0, len(data)) / sample_rate
    
    # Plot waveform
    plt.figure(figsize=(12, 4))
    plt.plot(time, data, color='blue', alpha=0.7)
    plt.title(f"Waveform: {os.path.basename(file_path)}")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    
    # Add file info
    duration = len(data) / sample_rate
    plt.figtext(0.01, 0.01, f"Sample Rate: {sample_rate} Hz, Duration: {duration:.2f} s", 
                fontsize=8, ha='left')
    
    # Save or show the plot
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Waveform visualization saved to {output_path}")
    else:
        plt.show()
    
    plt.close()

def create_spectrogram(file_path, output_path=None):
    """
    Create a spectrogram visualization of an audio file.
    
    Parameters:
    - file_path: Path to the audio file
    - output_path: Path to save the visualization (if None, display instead)
    """
    # Read audio file
    sample_rate, data = wavfile.read(file_path)
    
    # Convert to mono if stereo
    if len(data.shape) > 1:
        data = data.mean(axis=1)
    
    # Normalize data
    data = data / np.max(np.abs(data))
    
    # Create spectrogram
    plt.figure(figsize=(12, 6))
    plt.specgram(data, NFFT=2048, Fs=sample_rate, noverlap=1024, cmap='viridis')
    plt.title(f"Spectrogram: {os.path.basename(file_path)}")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.colorbar(label='Intensity (dB)')
    
    # Add file info
    duration = len(data) / sample_rate
    plt.figtext(0.01, 0.01, f"Sample Rate: {sample_rate} Hz, Duration: {duration:.2f} s", 
                fontsize=8, ha='left')
    
    # Save or show the plot
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Spectrogram saved to {output_path}")
    else:
        plt.show()
    
    plt.close()

def batch_process(input_dir, output_dir, process_func, **kwargs):
    """
    Process all audio files in a directory.
    
    Parameters:
    - input_dir: Directory containing audio files
    - output_dir: Directory to save results
    - process_func: Function to process each file
    - **kwargs: Additional arguments to pass to process_func
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all audio files
    audio_files = [f for f in os.listdir(input_dir) if f.endswith(('.wav', '.mp3', '.aiff'))]
    
    # Process each file
    for audio_file in audio_files:
        input_path = os.path.join(input_dir, audio_file)
        base_name = os.path.splitext(audio_file)[0]
        
        # Call the processing function
        process_func(input_path, output_dir=output_dir, base_name=base_name, **kwargs)
        
        print(f"Processed {audio_file}")

if __name__ == "__main__":
    # Example usage
    print(f"A4 = {note_to_freq('A4')} Hz")
    print(f"440 Hz = {freq_to_note(440.0)}")
    print(f"Deviation between 440 Hz and 445 Hz: {cents_deviation(445.0, 440.0):.2f} cents")
