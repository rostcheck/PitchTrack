#!/usr/bin/env python3
"""
Real-time audio processing module for PitchTrack.
This module handles audio input capture and pitch detection.
"""

import numpy as np
import librosa
import sounddevice as sd
import threading
import queue
import time

class AudioProcessor:
    """
    Handles real-time audio input and pitch detection.
    """
    
    def __init__(self, callback=None, buffer_size=2048, hop_length=512, 
                 sample_rate=44100, fmin=50.0, fmax=2000.0):
        """
        Initialize the audio processor.
        
        Parameters:
        - callback: Function to call with pitch detection results (frequency, confidence)
        - buffer_size: Size of audio buffer for processing
        - hop_length: Hop length for pitch detection
        - sample_rate: Audio sample rate
        - fmin: Minimum frequency to detect
        - fmax: Maximum frequency to detect
        """
        self.callback = callback
        self.buffer_size = buffer_size
        self.hop_length = hop_length
        self.sample_rate = sample_rate
        self.fmin = fmin
        self.fmax = fmax
        
        # Audio buffer
        self.audio_buffer = np.zeros(buffer_size, dtype=np.float32)
        
        # Processing queue and thread
        self.queue = queue.Queue()
        self.running = False
        self.thread = None
    
    def start(self):
        """Start audio capture and processing."""
        if self.running:
            return
        
        self.running = True
        
        # Start processing thread
        self.thread = threading.Thread(target=self._processing_thread)
        self.thread.daemon = True
        self.thread.start()
        
        # Start audio capture
        sd.InputStream(
            callback=self._audio_callback,
            channels=1,
            samplerate=self.sample_rate,
            blocksize=self.hop_length
        ).start()
    
    def stop(self):
        """Stop audio capture and processing."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
            self.thread = None
    
    def _audio_callback(self, indata, frames, time_info, status):
        """Callback for audio input."""
        if status:
            print(f"Audio input status: {status}")
        
        # Get audio data as mono and convert to float32
        audio_data = indata[:, 0].copy().astype(np.float32)
        
        # Add to queue for processing
        self.queue.put(audio_data)
    
    def _processing_thread(self):
        """Thread for processing audio data."""
        while self.running:
            try:
                # Get audio data from queue with timeout
                audio_data = self.queue.get(timeout=0.1)
                
                # Update buffer (shift old data and add new data)
                self.audio_buffer = np.roll(self.audio_buffer, -len(audio_data))
                self.audio_buffer[-len(audio_data):] = audio_data
                
                # Detect pitch
                frequency, confidence = self._detect_pitch(self.audio_buffer)
                
                # Call callback with results
                if self.callback:
                    self.callback(frequency, confidence)
                
            except queue.Empty:
                # No data available, continue
                continue
            except Exception as e:
                print(f"Error in audio processing: {e}")
    
    def _detect_pitch(self, audio_data):
        """
        Detect pitch in audio data using librosa.
        
        Returns:
        - frequency: Detected frequency in Hz
        - confidence: Confidence value (0.0 to 1.0)
        """
        # Use librosa's piptrack for pitch detection
        pitches, magnitudes = librosa.piptrack(
            y=audio_data, 
            sr=self.sample_rate,
            fmin=self.fmin,
            fmax=self.fmax
        )
        
        # Find the highest magnitude for each frame
        if magnitudes.size > 0:
            index = magnitudes[:, 0].argmax()
            pitch = pitches[index, 0]
            confidence = magnitudes[index, 0]
            
            # Normalize confidence
            max_possible = np.abs(audio_data).max()
            if max_possible > 0:
                confidence = min(confidence / max_possible, 1.0)
            else:
                confidence = 0.0
            
            return float(pitch), float(confidence)
        else:
            return 0.0, 0.0

# Example usage
if __name__ == "__main__":
    def print_pitch(frequency, confidence):
        print(f"Frequency: {frequency:.2f} Hz, Confidence: {confidence:.2f}")
    
    processor = AudioProcessor(callback=print_pitch)
    processor.start()
    
    try:
        print("Listening for audio... Press Ctrl+C to stop.")
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        processor.stop()
