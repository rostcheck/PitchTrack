#!/usr/bin/env python3
"""
Generate reference tones for pitch detection testing.
This script creates various audio files with known pitch content.
"""

import os
import numpy as np
from scipy.io import wavfile
import argparse

# Create output directory if it doesn't exist
output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reference_tones")
os.makedirs(output_dir, exist_ok=True)

def generate_sine_wave(frequency, duration, sample_rate=44100, amplitude=0.5):
    """Generate a sine wave at the specified frequency."""
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    sine_wave = amplitude * np.sin(2 * np.pi * frequency * t)
    return sine_wave

def generate_pure_tone(frequency, duration=3.0, sample_rate=44100, filename=None):
    """Generate a pure tone at the specified frequency and save to WAV file."""
    if filename is None:
        filename = f"tone_{frequency}Hz.wav"
    
    sine_wave = generate_sine_wave(frequency, duration, sample_rate)
    
    # Apply fade in/out to avoid clicks
    fade_samples = int(0.01 * sample_rate)  # 10ms fade
    fade_in = np.linspace(0, 1, fade_samples)
    fade_out = np.linspace(1, 0, fade_samples)
    
    sine_wave[:fade_samples] *= fade_in
    sine_wave[-fade_samples:] *= fade_out
    
    # Convert to 16-bit PCM
    audio = np.int16(sine_wave * 32767)
    
    # Save to file
    output_path = os.path.join(output_dir, filename)
    wavfile.write(output_path, sample_rate, audio)
    print(f"Generated {output_path}")
    
    return output_path

def generate_chromatic_scale(start_freq=220.0, num_notes=13, duration_per_note=0.5, sample_rate=44100):
    """Generate a chromatic scale starting at the specified frequency."""
    scale_audio = np.array([], dtype=np.float32)
    
    for i in range(num_notes):
        # Calculate frequency using equal temperament formula: f = f0 * 2^(n/12)
        freq = start_freq * (2 ** (i / 12))
        note_audio = generate_sine_wave(freq, duration_per_note, sample_rate)
        
        # Apply fade in/out to avoid clicks between notes
        fade_samples = int(0.01 * sample_rate)  # 10ms fade
        if len(note_audio) > 2 * fade_samples:
            fade_in = np.linspace(0, 1, fade_samples)
            fade_out = np.linspace(1, 0, fade_samples)
            note_audio[:fade_samples] *= fade_in
            note_audio[-fade_samples:] *= fade_out
        
        scale_audio = np.append(scale_audio, note_audio)
    
    # Convert to 16-bit PCM
    audio = np.int16(scale_audio * 32767)
    
    # Save to file
    output_path = os.path.join(output_dir, f"chromatic_scale_{start_freq}Hz.wav")
    wavfile.write(output_path, sample_rate, audio)
    print(f"Generated {output_path}")
    
    return output_path

def generate_vibrato_tone(frequency, vibrato_rate=5.0, vibrato_depth=0.05, duration=3.0, sample_rate=44100):
    """Generate a tone with vibrato at the specified frequency."""
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    
    # Calculate frequency modulation for vibrato
    vibrato = frequency * vibrato_depth * np.sin(2 * np.pi * vibrato_rate * t)
    instantaneous_freq = frequency + vibrato
    
    # Generate phase by integrating frequency
    phase = 2 * np.pi * np.cumsum(instantaneous_freq) / sample_rate
    
    # Generate sine wave with modulated phase
    sine_wave = 0.5 * np.sin(phase)
    
    # Apply fade in/out
    fade_samples = int(0.01 * sample_rate)  # 10ms fade
    fade_in = np.linspace(0, 1, fade_samples)
    fade_out = np.linspace(1, 0, fade_samples)
    
    sine_wave[:fade_samples] *= fade_in
    sine_wave[-fade_samples:] *= fade_out
    
    # Convert to 16-bit PCM
    audio = np.int16(sine_wave * 32767)
    
    # Save to file
    output_path = os.path.join(output_dir, f"vibrato_{frequency}Hz.wav")
    wavfile.write(output_path, sample_rate, audio)
    print(f"Generated {output_path}")
    
    return output_path

def generate_simple_melody():
    """Generate a simple melody with known pitches."""
    # C major scale frequencies
    c_major = [261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88, 523.25]
    
    # Simple melody pattern (indices into c_major)
    melody_pattern = [0, 1, 2, 0, 0, 1, 2, 0, 2, 3, 4, 2, 3, 4, 4, 5, 4, 3, 2, 0, 4, 5, 4, 3, 2, 0, 0, 7, 0]
    
    sample_rate = 44100
    duration_per_note = 0.3
    melody_audio = np.array([], dtype=np.float32)
    
    for note_idx in melody_pattern:
        freq = c_major[note_idx]
        note_audio = generate_sine_wave(freq, duration_per_note, sample_rate)
        
        # Apply fade in/out to avoid clicks between notes
        fade_samples = int(0.01 * sample_rate)  # 10ms fade
        fade_in = np.linspace(0, 1, fade_samples)
        fade_out = np.linspace(1, 0, fade_samples)
        note_audio[:fade_samples] *= fade_in
        note_audio[-fade_samples:] *= fade_out
        
        melody_audio = np.append(melody_audio, note_audio)
    
    # Convert to 16-bit PCM
    audio = np.int16(melody_audio * 32767)
    
    # Save to file
    output_path = os.path.join(output_dir, "simple_melody.wav")
    wavfile.write(output_path, sample_rate, audio)
    print(f"Generated {output_path}")
    
    return output_path

def generate_glissando(start_freq=220.0, end_freq=440.0, duration=3.0, sample_rate=44100):
    """Generate a continuous pitch glide from start_freq to end_freq."""
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    
    # Calculate logarithmic frequency sweep
    exponent = np.log2(end_freq / start_freq)
    instantaneous_freq = start_freq * (2 ** (exponent * t / duration))
    
    # Generate phase by integrating frequency
    phase = 2 * np.pi * np.cumsum(instantaneous_freq) / sample_rate
    
    # Generate sine wave with modulated phase
    sine_wave = 0.5 * np.sin(phase)
    
    # Apply fade in/out
    fade_samples = int(0.01 * sample_rate)  # 10ms fade
    fade_in = np.linspace(0, 1, fade_samples)
    fade_out = np.linspace(1, 0, fade_samples)
    
    sine_wave[:fade_samples] *= fade_in
    sine_wave[-fade_samples:] *= fade_out
    
    # Convert to 16-bit PCM
    audio = np.int16(sine_wave * 32767)
    
    # Save to file
    output_path = os.path.join(output_dir, f"glissando_{start_freq}Hz_to_{end_freq}Hz.wav")
    wavfile.write(output_path, sample_rate, audio)
    print(f"Generated {output_path}")
    
    return output_path

def main():
    parser = argparse.ArgumentParser(description='Generate reference tones for pitch detection testing')
    parser.add_argument('--all', action='store_true', help='Generate all reference tones')
    parser.add_argument('--pure', action='store_true', help='Generate pure tones')
    parser.add_argument('--scale', action='store_true', help='Generate chromatic scale')
    parser.add_argument('--vibrato', action='store_true', help='Generate vibrato tones')
    parser.add_argument('--melody', action='store_true', help='Generate simple melody')
    parser.add_argument('--glissando', action='store_true', help='Generate glissando (pitch slide)')
    
    args = parser.parse_args()
    
    # If no specific option is selected, generate all
    if not (args.pure or args.scale or args.vibrato or args.melody or args.glissando):
        args.all = True
    
    if args.all or args.pure:
        # Generate standard reference tones (A4=440Hz, etc.)
        standard_frequencies = [
            261.63,  # C4
            293.66,  # D4
            329.63,  # E4
            349.23,  # F4
            392.00,  # G4
            440.00,  # A4
            493.88,  # B4
            523.25   # C5
        ]
        
        for freq in standard_frequencies:
            generate_pure_tone(freq)
    
    if args.all or args.scale:
        # Generate chromatic scales
        generate_chromatic_scale(220.0)  # A3 to A4
        generate_chromatic_scale(440.0)  # A4 to A5
    
    if args.all or args.vibrato:
        # Generate vibrato tones
        generate_vibrato_tone(440.0)  # A4 with vibrato
        generate_vibrato_tone(261.63)  # C4 with vibrato
    
    if args.all or args.melody:
        # Generate simple melody
        generate_simple_melody()
    
    if args.all or args.glissando:
        # Generate glissando (continuous pitch slide)
        generate_glissando(220.0, 440.0)  # A3 to A4
        generate_glissando(440.0, 880.0)  # A4 to A5

if __name__ == "__main__":
    main()
