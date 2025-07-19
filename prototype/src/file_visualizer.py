#!/usr/bin/env python3
"""
File-based pitch visualizer for PitchTrack.
Processes audio files and visualizes pitch in real-time.
"""

import sys
import os
import numpy as np
import librosa
import sounddevice as sd
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QSlider, QLabel, 
                            QComboBox, QFrame, QFileDialog)
from PyQt6.QtGui import QPainter, QColor, QPen, QBrush, QFont, QLinearGradient
from PyQt6.QtCore import Qt, QTimer, QRect, QPointF, pyqtSignal

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our modules
from src.pitch_visualizer import PianoKeyboard, PitchDisplay

class FileVisualizer(QMainWindow):
    """Main window for the file-based pitch visualizer application."""
    
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("PitchTrack File Visualizer")
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
        
        # Play/Pause button
        self.play_button = QPushButton("Play")
        self.play_button.clicked.connect(self.toggle_playback)
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
        
        # Set up timer for playback
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_playback)
        
        # Audio data
        self.audio_data = None
        self.sample_rate = 44100
        self.pitch_data = None
        self.confidence_data = None
        self.current_frame = 0
        self.is_playing = False
        self.hop_length = 512
        self.volume = 0.8  # Default volume (0.0 to 1.0)
        
        # Audio playback
        self.audio_stream = None
        self.audio_buffer_size = 1024
        self.audio_position = 0
        
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
        """)
    
    def open_file(self):
        """Open an audio file and process it."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Audio File", "", "Audio Files (*.wav *.mp3 *.aiff *.ogg)"
        )
        
        if not file_path:
            return
        
        try:
            # Load audio file
            self.audio_data, self.sample_rate = librosa.load(file_path, sr=None)
            
            # Process pitch
            self.process_pitch()
            
            # Update UI
            filename = os.path.basename(file_path)
            self.file_label.setText(f"File: {filename}")
            self.play_button.setEnabled(True)
            self.current_frame = 0
            
            # Reset display
            self.piano.set_current_note(0)
            for _ in range(self.pitch_display.history_size):
                self.pitch_display.add_pitch(0, 0)
            
        except Exception as e:
            print(f"Error loading file: {e}")
            self.file_label.setText(f"Error: {str(e)}")
    
    def process_pitch(self):
        """Process pitch data from audio file."""
        if self.audio_data is None:
            return
        
        # Extract pitch using librosa
        pitches, magnitudes = librosa.piptrack(
            y=self.audio_data, 
            sr=self.sample_rate,
            hop_length=self.hop_length,
            fmin=80.0,
            fmax=1000.0
        )
        
        # Extract the most prominent pitch for each frame
        self.pitch_data = []
        self.confidence_data = []
        
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            confidence = magnitudes[index, t]
            
            # Only include pitches with sufficient magnitude
            if confidence > 0:
                self.pitch_data.append(float(pitch))
            else:
                self.pitch_data.append(0.0)
            
            # Normalize confidence
            max_possible = np.abs(self.audio_data).max()
            if max_possible > 0:
                self.confidence_data.append(min(float(confidence / max_possible), 1.0))
            else:
                self.confidence_data.append(0.0)
    
    def toggle_playback(self):
        """Toggle between playing and paused states."""
        if self.is_playing:
            self.stop_audio()
            self.timer.stop()
            self.play_button.setText("Play")
            self.is_playing = False
        else:
            self.start_audio()
            self.timer.start(30)  # 30ms = ~33fps
            self.play_button.setText("Pause")
            self.is_playing = True
    
    def start_audio(self):
        """Start audio playback."""
        if self.audio_data is None:
            return
        
        # Calculate starting position in audio data
        self.audio_position = int(self.current_frame * self.hop_length)
        
        # Start audio stream
        self.audio_stream = sd.OutputStream(
            samplerate=self.sample_rate,
            channels=1,
            callback=self.audio_callback,
            blocksize=self.audio_buffer_size
        )
        self.audio_stream.start()
    
    def stop_audio(self):
        """Stop audio playback."""
        if self.audio_stream is not None:
            self.audio_stream.stop()
            self.audio_stream.close()
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
            self.toggle_playback()  # Stop at end
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
        event.accept()

def main():
    app = QApplication(sys.argv)
    window = FileVisualizer()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
