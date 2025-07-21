#!/usr/bin/env python3
"""
Simplified Vocal Extractor Module for Testing

This module provides functions to extract vocal tracks from music files
using various source separation techniques.
"""

import os
import sys
import numpy as np
import librosa
import tempfile
import subprocess
import shutil
import soundfile as sf
from pathlib import Path

class VocalExtractor:
    """Class for extracting vocals from music files using different methods."""
    
    def __init__(self, method="librosa", output_dir=None):
        """
        Initialize the vocal extractor.
        
        Args:
            method (str): The extraction method to use ('spleeter', 'demucs', or 'librosa')
            output_dir (str): Directory to store extracted files (temporary dir if None)
        """
        self.method = method
        self.output_dir = output_dir or tempfile.mkdtemp(prefix="vocal_extract_")
        
        # Check if the selected method is available
        self._check_dependencies()
    
    def _check_dependencies(self):
        """Check if the required dependencies for the selected method are installed."""
        if self.method == "spleeter":
            try:
                import tensorflow as tf
                print(f"TensorFlow version: {tf.__version__}")
                
                try:
                    import spleeter
                    from spleeter.separator import Separator
                    print(f"Spleeter is available")
                except ImportError:
                    print("Spleeter not found. Please install with: pip install spleeter")
                    raise ImportError("Spleeter is required for this extraction method")
            except ImportError as e:
                print("TensorFlow not found. Spleeter requires TensorFlow.")
                raise ImportError("TensorFlow is required for Spleeter extraction method") from e
        
        elif self.method == "demucs":
            try:
                # Try to run demucs --version to check if it's installed
                result = subprocess.run(["demucs", "--version"], 
                                      stdout=subprocess.PIPE, 
                                      stderr=subprocess.PIPE, 
                                      text=True,
                                      check=False)
                if result.returncode == 0:
                    print(f"Demucs version: {result.stdout.strip()}")
                else:
                    raise FileNotFoundError("Demucs command returned non-zero exit code")
            except (subprocess.SubprocessError, FileNotFoundError) as e:
                print("Demucs not found. Please install with: pip install demucs")
                raise ImportError("Demucs is required for this extraction method") from e
    
    def extract_vocals(self, audio_path, return_audio=True):
        """
        Extract vocals from the given audio file.
        
        Args:
            audio_path (str): Path to the audio file
            return_audio (bool): If True, return the extracted audio as numpy array
                                If False, return the path to the extracted file
        
        Returns:
            If return_audio is True: tuple (vocals_audio, sample_rate)
            If return_audio is False: str (path to extracted vocals file)
        """
        if self.method == "spleeter":
            return self._extract_with_spleeter(audio_path, return_audio)
        elif self.method == "demucs":
            return self._extract_with_demucs(audio_path, return_audio)
        else:  # Default to librosa
            return self._extract_with_librosa(audio_path, return_audio)
    
    def _extract_with_spleeter(self, audio_path, return_audio=True):
        """Extract vocals using Spleeter."""
        from spleeter.separator import Separator
        
        # Create a unique output directory for this file
        file_name = os.path.basename(audio_path)
        base_name = os.path.splitext(file_name)[0]
        output_dir = os.path.join(self.output_dir, base_name)
        
        print(f"Using Spleeter to separate vocals")
        print(f"Output directory: {output_dir}")
        
        # Initialize the separator with the 2stems model (vocals + accompaniment)
        separator = Separator('spleeter:2stems')
        
        # Perform the separation
        separator.separate_to_file(audio_path, output_dir)
        
        # Path to the extracted vocals
        vocals_path = os.path.join(output_dir, base_name, 'vocals.wav')
        
        if return_audio:
            # Load the extracted vocals
            vocals, sr = librosa.load(vocals_path, sr=None)
            return vocals, sr
        else:
            return vocals_path
    
    def _extract_with_demucs(self, audio_path, return_audio=True):
        """Extract vocals using Demucs."""
        # Create a unique output directory for this file
        file_name = os.path.basename(audio_path)
        base_name = os.path.splitext(file_name)[0]
        output_dir = os.path.join(self.output_dir, base_name)
        
        print(f"Using Demucs to separate vocals")
        print(f"Output directory: {output_dir}")
        
        # Run demucs command
        cmd = ["demucs", "--out", output_dir, audio_path]
        print(f"Running command: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, 
                              stdout=subprocess.PIPE, 
                              stderr=subprocess.PIPE, 
                              text=True,
                              check=True)  # Will raise CalledProcessError if command fails
        
        # Demucs creates a directory structure like:
        # output_dir/separated/model_name/track_name/vocals.wav
        # Find the vocals file
        separated_dir = os.path.join(output_dir, "separated")
        if not os.path.exists(separated_dir):
            raise FileNotFoundError(f"Expected directory not found: {separated_dir}")
            
        model_dirs = os.listdir(separated_dir)
        if not model_dirs:
            raise FileNotFoundError("Demucs did not produce any output")
        
        vocals_path = os.path.join(separated_dir, model_dirs[0], base_name, "vocals.wav")
        
        if not os.path.exists(vocals_path):
            raise FileNotFoundError(f"Vocals file not found at expected path: {vocals_path}")
        
        if return_audio:
            # Load the extracted vocals
            vocals, sr = librosa.load(vocals_path, sr=None)
            return vocals, sr
        else:
            return vocals_path
    
    def _extract_with_librosa(self, audio_path, return_audio=True):
        """
        Extract vocals using librosa's HPSS (Harmonic-Percussive Source Separation).
        This is a basic approach and won't be as effective as Spleeter or Demucs.
        """
        print("Using librosa HPSS for vocal extraction")
        
        # Load the audio file
        y, sr = librosa.load(audio_path, sr=None)
        
        # Compute the spectrogram
        S_full = librosa.stft(y)
        
        # Compute the magnitude spectrogram
        S_mag = np.abs(S_full)
        
        # Compute the harmonic and percussive components
        print("Separating harmonic and percussive components...")
        H, P = librosa.decompose.hpss(S_mag)
        
        # Reconstruct the harmonic component (which often contains vocals)
        print("Reconstructing audio from harmonic component...")
        y_harm = librosa.istft(H * np.exp(1.j * np.angle(S_full)))
        
        if return_audio:
            return y_harm, sr
        else:
            # Save the harmonic component to a file
            vocals_path = os.path.join(self.output_dir, "vocals.wav")
            print(f"Saving extracted vocals to {vocals_path}")
            sf.write(vocals_path, y_harm, sr)
            return vocals_path
    
    def cleanup(self):
        """Remove temporary files."""
        if os.path.exists(self.output_dir) and self.output_dir.startswith(tempfile.gettempdir()):
            shutil.rmtree(self.output_dir)


def extract_vocals_from_file(file_path, method="librosa", output_dir=None, return_audio=True):
    """
    Convenience function to extract vocals from a file.
    
    Args:
        file_path (str): Path to the audio file
        method (str): Extraction method ('spleeter', 'demucs', or 'librosa')
        output_dir (str): Directory to store extracted files
        return_audio (bool): If True, return the extracted audio as numpy array
    
    Returns:
        If return_audio is True: tuple (vocals_audio, sample_rate)
        If return_audio is False: str (path to extracted vocals file)
    """
    extractor = VocalExtractor(method=method, output_dir=output_dir)
    result = extractor.extract_vocals(file_path, return_audio=return_audio)
    
    # Don't cleanup if we're returning a file path
    if return_audio:
        extractor.cleanup()
    
    return result


if __name__ == "__main__":
    # Example usage
    if len(sys.argv) < 2:
        print("Usage: python vocal_extractor.py <audio_file> [method] [output_dir]")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    method = sys.argv[2] if len(sys.argv) > 2 else "librosa"
    output_dir = sys.argv[3] if len(sys.argv) > 3 else None
    
    print(f"Extracting vocals from {audio_file} using {method}...")
    result = extract_vocals_from_file(audio_file, method=method, 
                                     output_dir=output_dir, return_audio=False)
    print(f"Vocals extracted to: {result}")
