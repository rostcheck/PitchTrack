#!/usr/bin/env python3
"""
File-based pitch visualizer with enhanced vocal pitch tracking.
Processes audio files and visualizes the fundamental pitch of vocal lines.
"""

import sys
import os
import numpy as np
import librosa
import sounddevice as sd
import threading
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QSlider, QLabel, 
                            QComboBox, QFrame, QFileDialog, QProgressDialog,
                            QMessageBox, QCheckBox)
from PyQt6.QtGui import QPainter, QColor, QPen, QBrush, QFont, QLinearGradient
from PyQt6.QtCore import Qt, QTimer, QRect, QPointF, pyqtSignal, QThread
from scipy.signal import medfilt
from scipy.ndimage import gaussian_filter1d

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our modules
from src.pitch_visualizer import PianoKeyboard, PitchDisplay

# Print available audio devices for debugging
print("Available audio devices:")
print(sd.query_devices())

class VocalProcessingThread(QThread):
    """Thread for processing audio files with enhanced vocal pitch tracking."""
    progress_signal = pyqtSignal(int)
    finished_signal = pyqtSignal(object, object, object, int)
    
    def __init__(self, file_path, hop_length=512, fmin=80.0, fmax=800.0, 
                 energy_threshold=0.05, median_filter_size=11,
                 continuity_tolerance=0.2, octave_cost=0.9):
        super().__init__()
        self.file_path = file_path
        self.hop_length = hop_length
        self.fmin = fmin
        self.fmax = fmax
        self.energy_threshold = energy_threshold
        self.median_filter_size = median_filter_size
        self.continuity_tolerance = continuity_tolerance
        self.octave_cost = octave_cost
    
    def run(self):
        try:
            # Load audio file
            self.progress_signal.emit(10)
            audio_data, sample_rate = librosa.load(self.file_path, sr=None)
            
            # Calculate energy for voice activity detection
            self.progress_signal.emit(20)
            energy = librosa.feature.rms(y=audio_data, frame_length=self.hop_length*2, 
                                        hop_length=self.hop_length)[0]
            energy = energy / np.max(energy) if np.max(energy) > 0 else energy
            
            # Use pYIN algorithm for more accurate fundamental frequency estimation
            self.progress_signal.emit(30)
            f0, voiced_flag, voiced_probs = librosa.pyin(
                audio_data, 
                fmin=self.fmin,
                fmax=self.fmax,
                sr=sample_rate,
                hop_length=self.hop_length,
                fill_na=None  # Don't fill unvoiced sections
            )
            
            # Convert frame indices to time
            times = librosa.times_like(f0, sr=sample_rate, hop_length=self.hop_length)
            
            # Initialize arrays for processed pitch and confidence
            self.progress_signal.emit(60)
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
                if confidence[i] > self.energy_threshold and f0[i] > 0:
                    processed_pitch[i] = f0[i]
                else:
                    processed_pitch[i] = 0
            
            # Apply continuity constraints to avoid octave jumps
            self.progress_signal.emit(70)
            for i in range(1, len(processed_pitch)):
                if processed_pitch[i] > 0 and processed_pitch[i-1] > 0:
                    # Calculate octave difference
                    octave_diff = np.abs(np.log2(processed_pitch[i] / processed_pitch[i-1]))
                    
                    # If jump is too large, try to correct it
                    if octave_diff > self.continuity_tolerance:
                        # Check if it's likely an octave error
                        if abs(octave_diff - 1.0) < 0.1:  # Close to an octave jump
                            # Adjust to previous octave if confidence allows
                            if confidence[i] < confidence[i-1] * (1 + self.octave_cost):
                                if processed_pitch[i] > processed_pitch[i-1]:
                                    processed_pitch[i] = processed_pitch[i] / 2.0
                                else:
                                    processed_pitch[i] = processed_pitch[i] * 2.0
            
            # Apply median filtering to smooth the pitch contour
            self.progress_signal.emit(80)
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
                    if end - start > self.median_filter_size:
                        segment = processed_pitch[start:end]
                        smoothed_segment = medfilt(segment, self.median_filter_size)
                        smoothed_pitch[start:end] = smoothed_segment
                
                processed_pitch = smoothed_pitch
            
            # Convert to lists for JSON serialization
            self.progress_signal.emit(90)
            times_list = times.tolist()
            pitch_list = processed_pitch.tolist()
            confidence_list = confidence.tolist()
            
            # Emit finished signal with results
            self.progress_signal.emit(100)
            self.finished_signal.emit(audio_data, pitch_list, confidence_list, sample_rate)
            
        except Exception as e:
            print(f"Error processing file: {e}")
            import traceback
            traceback.print_exc()
            self.finished_signal.emit(None, None, None, 0)

class VocalVisualizer(QMainWindow):
    """Main window for the vocal pitch visualizer application."""
    
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("PitchTrack Vocal Visualizer")
        self.setMinimumSize(800, 500)
        
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout(central_widget)
        
        # Create piano keyboard widget
        self.piano = PianoKeyboard()
        main_layout.addWidget(self.piano)
        
        # Add separator
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        main_layout.addWidget(separator)
        
        # Create pitch display widget
        self.pitch_display = PitchDisplay()
        main_layout.addWidget(self.pitch_display)
        
        # Add controls at the bottom
        controls_layout = QHBoxLayout()
        
        # Open File button
        self.open_button = QPushButton("Open File")
        self.open_button.clicked.connect(self.open_file)
        controls_layout.addWidget(self.open_button)
        
        # Rewind button
        self.rewind_button = QPushButton("⏮ Rewind")
        self.rewind_button.clicked.connect(self.rewind)
        self.rewind_button.setEnabled(False)
        controls_layout.addWidget(self.rewind_button)
        
        # Play/Pause button
        self.play_button = QPushButton("▶ Play")
        self.play_button.clicked.connect(lambda: self.toggle_playback())
        self.play_button.setEnabled(False)
        controls_layout.addWidget(self.play_button)
        
        # Volume slider
        self.volume_label = QLabel("Volume:")
        controls_layout.addWidget(self.volume_label)
        
        self.volume_slider = QSlider(Qt.Orientation.Horizontal)
        self.volume_slider.setMinimum(0)
        self.volume_slider.setMaximum(100)
        self.volume_slider.setValue(80)
        self.volume_slider.setFixedWidth(100)
        self.volume_slider.valueChanged.connect(self.set_volume)
        controls_layout.addWidget(self.volume_slider)
        
        # Add spacer
        controls_layout.addStretch(1)
        
        # Add file info label
        self.file_label = QLabel("No file loaded")
        controls_layout.addWidget(self.file_label)
        
        # Add controls layout to main layout
        main_layout.addLayout(controls_layout)
        
        # Add advanced settings layout
        advanced_layout = QHBoxLayout()
        
        # Energy threshold slider
        self.energy_label = QLabel("Energy Threshold:")
        advanced_layout.addWidget(self.energy_label)
        
        self.energy_slider = QSlider(Qt.Orientation.Horizontal)
        self.energy_slider.setMinimum(1)
        self.energy_slider.setMaximum(20)
        self.energy_slider.setValue(5)  # Default 0.05
        self.energy_slider.setFixedWidth(100)
        self.energy_slider.valueChanged.connect(self.update_settings)
        advanced_layout.addWidget(self.energy_slider)
        
        # Median filter size slider
        self.filter_label = QLabel("Smoothing:")
        advanced_layout.addWidget(self.filter_label)
        
        self.filter_slider = QSlider(Qt.Orientation.Horizontal)
        self.filter_slider.setMinimum(3)
        self.filter_slider.setMaximum(21)
        self.filter_slider.setValue(11)  # Default 11
        self.filter_slider.setFixedWidth(100)
        self.filter_slider.valueChanged.connect(self.update_settings)
        advanced_layout.addWidget(self.filter_slider)
        
        # Continuity tolerance slider
        self.continuity_label = QLabel("Continuity:")
        advanced_layout.addWidget(self.continuity_label)
        
        self.continuity_slider = QSlider(Qt.Orientation.Horizontal)
        self.continuity_slider.setMinimum(5)
        self.continuity_slider.setMaximum(50)
        self.continuity_slider.setValue(20)  # Default 0.2
        self.continuity_slider.setFixedWidth(100)
        self.continuity_slider.valueChanged.connect(self.update_settings)
        advanced_layout.addWidget(self.continuity_slider)
        
        # Add advanced layout to main layout
        main_layout.addLayout(advanced_layout)
        
        # Set up timer for playback
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_playback)
        
        # Set up timer for device monitoring
        self.device_monitor_timer = QTimer()
        self.device_monitor_timer.timeout.connect(self.check_audio_device)
        self.device_monitor_timer.start(1000)  # Check every second
        
        # Store current default device
        hostapi_info = sd.query_hostapis(sd.default.hostapi)
        self.current_default_device = hostapi_info['default_output_device']
        device_info = sd.query_devices(self.current_default_device)
        print(f"Initial default output device: {device_info['name']} (ID: {self.current_default_device})")
        
        # Audio data
        self.audio_data = None
        self.sample_rate = 44100
        self.pitch_data = None
        self.confidence_data = None
        self.current_frame = 0
        self.is_playing = False
        self.hop_length = 512
        self.volume = 0.8  # Default volume (0.0 to 1.0)
        
        # Pitch detection settings
        self.energy_threshold = 0.05
        self.median_filter_size = 11
        self.continuity_tolerance = 0.2
        self.octave_cost = 0.9
        
        # Audio playback
        self.audio_stream = None
        self.audio_buffer_size = 1024
        self.audio_position = 0
        
        # File processing
        self.processing_thread = None
        self.progress_dialog = None
        
        # Apply macOS style
        self.apply_mac_style()
    
    def apply_mac_style(self):
        """Apply macOS-like styling to the application."""
        # Set the fusion style which looks more modern
        QApplication.setStyle("Fusion")
        
        # Set stylesheet for macOS-like appearance
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f5f7;
            }
            QWidget {
                font-family: 'Helvetica Neue', Arial, sans-serif;
            }
            QPushButton {
                background-color: #0071e3;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: #0077ed;
            }
            QPushButton:pressed {
                background-color: #0068d1;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #999999;
            }
            QSlider::groove:horizontal {
                border: 1px solid #bdbdbd;
                height: 4px;
                background: #e0e0e0;
                margin: 0px;
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                background: #0071e3;
                border: none;
                width: 16px;
                height: 16px;
                margin: -6px 0;
                border-radius: 8px;
            }
            QComboBox {
                border: 1px solid #bdbdbd;
                border-radius: 4px;
                padding: 4px 8px;
                background: white;
            }
            QLabel {
                color: #333333;
            }
            QProgressDialog {
                background-color: #f5f5f7;
                border: 1px solid #cccccc;
                border-radius: 6px;
            }
            QProgressDialog QLabel {
                font-size: 14px;
                color: #333333;
            }
            QProgressDialog QProgressBar {
                border: 1px solid #bdbdbd;
                border-radius: 4px;
                background-color: #e0e0e0;
                text-align: center;
                color: #333333;
            }
            QProgressDialog QProgressBar::chunk {
                background-color: #0071e3;
                border-radius: 3px;
            }
        """)
    
    def update_settings(self):
        """Update pitch detection settings from sliders."""
        self.energy_threshold = self.energy_slider.value() / 100.0
        self.median_filter_size = self.filter_slider.value()
        if self.median_filter_size % 2 == 0:
            self.median_filter_size += 1  # Ensure odd number for median filter
        self.continuity_tolerance = self.continuity_slider.value() / 100.0
        
        # Update labels with current values
        self.energy_label.setText(f"Energy Threshold: {self.energy_threshold:.2f}")
        self.filter_label.setText(f"Smoothing: {self.median_filter_size}")
        self.continuity_label.setText(f"Continuity: {self.continuity_tolerance:.2f}")
        
        print(f"Settings updated: energy={self.energy_threshold}, "
              f"filter={self.median_filter_size}, continuity={self.continuity_tolerance}")
    
    def open_file(self):
        """Open an audio file and process it."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Audio File", "", "Audio Files (*.wav *.mp3 *.aiff *.ogg)"
        )
        
        if not file_path:
            return
        
        # Update settings before processing
        self.update_settings()
        
        # Disable buttons during processing
        self.open_button.setEnabled(False)
        self.play_button.setEnabled(False)
        self.rewind_button.setEnabled(False)
        
        # Create progress dialog
        self.progress_dialog = QProgressDialog("Processing audio file...", "Cancel", 0, 100, self)
        self.progress_dialog.setWindowTitle("Loading File")
        self.progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
        self.progress_dialog.setMinimumDuration(0)  # Show immediately
        self.progress_dialog.setValue(0)
        
        # Create and start processing thread
        self.processing_thread = VocalProcessingThread(
            file_path,
            energy_threshold=self.energy_threshold,
            median_filter_size=self.median_filter_size,
            continuity_tolerance=self.continuity_tolerance,
            octave_cost=self.octave_cost
        )
        self.processing_thread.progress_signal.connect(self.update_progress)
        self.processing_thread.finished_signal.connect(self.processing_finished)
        self.processing_thread.start()
    
    def update_progress(self, value):
        """Update progress dialog."""
        if self.progress_dialog:
            self.progress_dialog.setValue(value)
    
    def processing_finished(self, audio_data, pitch_data, confidence_data, sample_rate):
        """Handle completion of file processing."""
        # Close progress dialog
        if self.progress_dialog:
            self.progress_dialog.close()
            self.progress_dialog = None
        
        # Re-enable buttons
        self.open_button.setEnabled(True)
        
        if audio_data is not None:
            # Store processed data
            self.audio_data = audio_data
            self.pitch_data = pitch_data
            self.confidence_data = confidence_data
            self.sample_rate = sample_rate
            
            # Update UI
            filename = os.path.basename(self.processing_thread.file_path)
            self.file_label.setText(f"File: {filename}")
            self.play_button.setEnabled(True)
            self.rewind_button.setEnabled(True)
            self.current_frame = 0
            
            # Reset display
            self.piano.set_current_note(0)
            for _ in range(self.pitch_display.history_size):
                self.pitch_display.add_pitch(0, 0)
        else:
            # Show error
            self.file_label.setText("Error loading file")
        
        # Clean up thread
        self.processing_thread = None
    
    def rewind(self):
        """Rewind to the beginning of the file."""
        # Stop playback if playing
        if self.is_playing:
            self.stop_audio()
            self.timer.stop()
            self.play_button.setText("▶ Play")
            self.is_playing = False
        
        # Reset position
        self.current_frame = 0
        self.audio_position = 0
        
        # Reset display
        self.piano.set_current_note(0)
        for _ in range(self.pitch_display.history_size):
            self.pitch_display.add_pitch(0, 0)
    
    def toggle_playback(self, force_stop=False):
        """Toggle between playing and paused states."""
        if self.is_playing or force_stop:
            self.stop_audio()
            self.timer.stop()
            self.play_button.setText("▶ Play")
            self.is_playing = False
        else:
            # Start audio playback (this will handle errors internally)
            self.start_audio()
            
            # Only start timer and update UI if audio started successfully
            if self.audio_stream is not None and self.audio_stream.active:
                self.timer.start(30)  # 30ms = ~33fps
                self.play_button.setText("⏸ Pause")
                self.is_playing = True
    
    def check_audio_device(self):
        """Check if the default audio device has changed."""
        try:
            # Get current default output device
            hostapi_info = sd.query_hostapis(sd.default.hostapi)
            default_device = hostapi_info['default_output_device']
            
            # Check if default device has changed
            if default_device != self.current_default_device:
                old_device_info = sd.query_devices(self.current_default_device)
                new_device_info = sd.query_devices(default_device)
                print(f"Default audio device changed from {old_device_info['name']} to {new_device_info['name']}")
                
                # Update stored default device
                self.current_default_device = default_device
                
                # If currently playing, restart with new device
                if self.is_playing:
                    # Remember position
                    current_position = self.audio_position
                    
                    # Stop current playback
                    self.stop_audio()
                    
                    # Restart with new device
                    self.audio_position = current_position
                    self.start_audio()
        except Exception as e:
            print(f"Error checking audio device: {e}")
    
    def start_audio(self):
        """Start audio playback."""
        if self.audio_data is None:
            return
        
        # Calculate starting position in audio data
        self.audio_position = int(self.current_frame * self.hop_length)
        
        try:
            # Get the current default output device
            hostapi_info = sd.query_hostapis(sd.default.hostapi)
            device_id = hostapi_info['default_output_device']
            device_info = sd.query_devices(device_id)
            
            print(f"Using current default output device: {device_info['name']} (ID: {device_id})")
            
            # Start audio stream with the current default output device
            self.audio_stream = sd.OutputStream(
                samplerate=self.sample_rate,
                channels=1,
                callback=self.audio_callback,
                blocksize=self.audio_buffer_size,
                device=device_id  # Explicitly use the current default output device
            )
            self.audio_stream.start()
            print("Audio playback started")
        except Exception as e:
            print(f"Error starting audio playback: {e}")
            # Show error message to user
            QMessageBox.warning(self, "Audio Error", 
                               f"Could not start audio playback: {str(e)}")
            # Reset playback state
            self.audio_stream = None
    
    def stop_audio(self):
        """Stop audio playback."""
        if self.audio_stream is not None:
            try:
                self.audio_stream.stop()
                self.audio_stream.close()
            except Exception as e:
                print(f"Error stopping audio stream: {e}")
            finally:
                self.audio_stream = None
    
    def audio_callback(self, outdata, frames, time, status):
        """Callback for audio output."""
        if status:
            print(f"Audio output status: {status}")
        
        if self.audio_position + frames > len(self.audio_data):
            # End of file reached
            available_frames = len(self.audio_data) - self.audio_position
            if available_frames <= 0:
                # No more data
                outdata.fill(0)
                return
            
            # Fill with remaining data and then zeros
            outdata[:available_frames, 0] = self.audio_data[self.audio_position:] * self.volume
            outdata[available_frames:, 0] = 0
        else:
            # Fill with audio data
            outdata[:, 0] = self.audio_data[self.audio_position:self.audio_position + frames] * self.volume
        
        # Update position
        self.audio_position += frames
    
    def set_volume(self, value):
        """Set the playback volume."""
        self.volume = value / 100.0  # Convert from 0-100 to 0.0-1.0
    
    def update_playback(self):
        """Update playback position and visualization."""
        if self.pitch_data is None:
            return
            
        # Calculate current frame based on audio position
        if self.audio_stream is not None:
            self.current_frame = min(self.audio_position // self.hop_length, len(self.pitch_data) - 1)
        
        # Check if we've reached the end
        if self.current_frame >= len(self.pitch_data):
            self.toggle_playback(force_stop=True)  # Stop at end
            self.current_frame = 0
            return
        
        # Get current pitch and confidence
        pitch = self.pitch_data[self.current_frame]
        confidence = self.confidence_data[self.current_frame]
        
        # Update visualization
        self.piano.set_current_note(pitch)
        self.pitch_display.add_pitch(pitch, confidence)
        
        # If not using audio stream, manually increment frame
        if self.audio_stream is None:
            self.current_frame += 1
            
    def closeEvent(self, event):
        """Handle window close event."""
        if self.is_playing:
            self.stop_audio()
        
        # Stop device monitoring
        self.device_monitor_timer.stop()
        
        # Cancel processing if in progress
        if self.processing_thread and self.processing_thread.isRunning():
            self.processing_thread.terminate()
            self.processing_thread.wait()
        
        event.accept()

def main():
    app = QApplication(sys.argv)
    window = VocalVisualizer()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
