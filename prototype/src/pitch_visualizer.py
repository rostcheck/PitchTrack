#!/usr/bin/env python3
"""
PitchTrack Visualizer - A real-time pitch visualization tool similar to SingAndSee.
"""

import sys
import os
import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QSlider, QLabel, 
                            QComboBox, QFrame)
from PyQt6.QtGui import QPainter, QColor, QPen, QBrush, QFont, QLinearGradient
from PyQt6.QtCore import Qt, QTimer, QRect, QPointF, pyqtSignal, QSize

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Constants for visualization
MIN_FREQUENCY = 80.0   # Hz (E2)
MAX_FREQUENCY = 1000.0  # Hz (B5)
HISTORY_SECONDS = 5.0   # How many seconds of pitch history to display
UPDATE_INTERVAL = 30    # Update interval in milliseconds (approx 30 fps)
SAMPLE_RATE = 44100     # Audio sample rate
HOP_LENGTH = 512        # Hop length for pitch detection

# Piano key colors
WHITE_KEY_COLOR = QColor(250, 250, 250)
BLACK_KEY_COLOR = QColor(40, 40, 40)
KEY_HIGHLIGHT_COLOR = QColor(102, 204, 255)  # Light blue highlight

# Note frequencies (C4 = middle C = 261.63 Hz)
A4_FREQ = 440.0  # A4 = 440Hz
A4_MIDI = 69     # MIDI note number for A4

def freq_to_midi(frequency):
    """Convert frequency to MIDI note number."""
    if frequency <= 0:
        return 0
    return 12 * np.log2(frequency / A4_FREQ) + A4_MIDI

def midi_to_freq(midi_note):
    """Convert MIDI note number to frequency."""
    return A4_FREQ * 2 ** ((midi_note - A4_MIDI) / 12)

def note_name(midi_note):
    """Get note name from MIDI note number."""
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    octave = midi_note // 12 - 1
    note = notes[midi_note % 12]
    return f"{note}{octave}"

class PianoKeyboard(QWidget):
    """Widget that displays a piano keyboard with highlighted keys based on pitch."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(80)
        self.setMaximumHeight(100)
        
        # Current highlighted key (MIDI note number)
        self.current_note = None
        
        # Define range of keys to display
        self.min_midi = int(freq_to_midi(MIN_FREQUENCY))
        self.max_midi = int(freq_to_midi(MAX_FREQUENCY))
        
        # Set background color
        self.setAutoFillBackground(True)
        palette = self.palette()
        palette.setColor(self.backgroundRole(), Qt.GlobalColor.white)
        self.setPalette(palette)
    
    def set_current_note(self, frequency):
        """Set the currently highlighted key based on frequency."""
        if frequency <= 0:
            self.current_note = None
        else:
            self.current_note = int(round(freq_to_midi(frequency)))
        self.update()
    
    def paintEvent(self, event):
        """Draw the piano keyboard."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Calculate key dimensions
        width = self.width()
        height = self.height()
        
        # Number of keys to display
        num_keys = self.max_midi - self.min_midi + 1
        
        # Calculate key width
        white_key_count = sum(1 for i in range(self.min_midi, self.max_midi + 1) 
                             if i % 12 not in [1, 3, 6, 8, 10])  # Not black keys
        white_key_width = width / white_key_count
        
        # Draw white keys first
        x_pos = 0
        key_positions = {}  # Store key positions for black keys
        
        for midi_note in range(self.min_midi, self.max_midi + 1):
            note_index = midi_note % 12
            
            # Skip black keys for now
            if note_index in [1, 3, 6, 8, 10]:
                continue
            
            # Store position for this white key
            key_positions[midi_note] = x_pos
            
            # Set color based on whether this key is highlighted
            if midi_note == self.current_note:
                painter.setBrush(QBrush(KEY_HIGHLIGHT_COLOR))
            else:
                painter.setBrush(QBrush(WHITE_KEY_COLOR))
            
            # Draw white key
            painter.setPen(QPen(Qt.GlobalColor.black, 1))
            painter.drawRect(int(x_pos), 0, int(white_key_width), height)
            
            # Draw note name at bottom of key
            if note_index == 0:  # Only label C notes
                painter.setPen(QPen(Qt.GlobalColor.black, 1))
                painter.setFont(QFont("Arial", 8))
                note = note_name(midi_note)
                painter.drawText(QRect(int(x_pos), height - 20, int(white_key_width), 20), 
                                Qt.AlignmentFlag.AlignCenter, note)
            
            x_pos += white_key_width
        
        # Now draw black keys on top
        black_key_width = white_key_width * 0.6
        
        for midi_note in range(self.min_midi, self.max_midi + 1):
            note_index = midi_note % 12
            
            # Only draw black keys
            if note_index not in [1, 3, 6, 8, 10]:
                continue
            
            # Find position based on adjacent white keys
            if note_index == 1:  # C#
                x_pos = key_positions.get(midi_note - 1, 0) + white_key_width * 0.7
            elif note_index == 3:  # D#
                x_pos = key_positions.get(midi_note - 1, 0) + white_key_width * 0.7
            elif note_index == 6:  # F#
                x_pos = key_positions.get(midi_note - 1, 0) + white_key_width * 0.7
            elif note_index == 8:  # G#
                x_pos = key_positions.get(midi_note - 1, 0) + white_key_width * 0.7
            elif note_index == 10:  # A#
                x_pos = key_positions.get(midi_note - 1, 0) + white_key_width * 0.7
            
            # Set color based on whether this key is highlighted
            if midi_note == self.current_note:
                painter.setBrush(QBrush(KEY_HIGHLIGHT_COLOR))
            else:
                painter.setBrush(QBrush(BLACK_KEY_COLOR))
            
            # Draw black key
            painter.setPen(QPen(Qt.GlobalColor.black, 1))
            painter.drawRect(int(x_pos), 0, int(black_key_width), int(height * 0.6))

class PitchDisplay(QWidget):
    """Widget that displays the pitch history as a scrolling line graph."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(300)
        
        # Buffer for pitch history
        self.pitch_history = []
        self.confidence_history = []
        
        # Calculate how many data points to store based on history length
        self.history_size = int(HISTORY_SECONDS * SAMPLE_RATE / HOP_LENGTH)
        
        # Initialize with zeros
        self.pitch_history = [0.0] * self.history_size
        self.confidence_history = [0.0] * self.history_size
        
        # Set background color
        self.setAutoFillBackground(True)
        palette = self.palette()
        palette.setColor(self.backgroundRole(), Qt.GlobalColor.white)
        self.setPalette(palette)
    
    def add_pitch(self, frequency, confidence):
        """Add a new pitch data point to the history."""
        self.pitch_history.append(frequency)
        self.confidence_history.append(confidence)
        
        # Remove oldest data point
        if len(self.pitch_history) > self.history_size:
            self.pitch_history.pop(0)
            self.confidence_history.pop(0)
        
        self.update()
    
    def paintEvent(self, event):
        """Draw the pitch history display."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        width = self.width()
        height = self.height()
        
        # Draw grid lines for piano keys
        painter.setPen(QPen(QColor(220, 220, 220), 1))
        
        min_midi = int(freq_to_midi(MIN_FREQUENCY))
        max_midi = int(freq_to_midi(MAX_FREQUENCY))
        
        for midi_note in range(min_midi, max_midi + 1):
            y_pos = self.freq_to_y(midi_to_freq(midi_note))
            painter.drawLine(0, y_pos, width, y_pos)
            
            # Add note labels on the left side
            note_index = midi_note % 12
            if note_index == 0:  # Only label C notes
                painter.setPen(QPen(Qt.GlobalColor.black, 1))
                painter.setFont(QFont("Arial", 8))
                note = note_name(midi_note)
                painter.drawText(QRect(5, int(y_pos - 15), 30, 15), 
                                Qt.AlignmentFlag.AlignLeft, note)
                painter.setPen(QPen(QColor(220, 220, 220), 1))
        
        # Draw pitch history line
        if len(self.pitch_history) > 1:
            # Create gradient for line color based on confidence
            for i in range(len(self.pitch_history) - 1):
                if self.pitch_history[i] <= 0 or self.pitch_history[i+1] <= 0:
                    continue
                
                # Calculate positions
                x1 = int(width * i / self.history_size)
                y1 = self.freq_to_y(self.pitch_history[i])
                x2 = int(width * (i + 1) / self.history_size)
                y2 = self.freq_to_y(self.pitch_history[i+1])
                
                # Set color based on confidence
                confidence = self.confidence_history[i]
                color = QColor(
                    int(255 * (1 - confidence)),  # Red
                    int(255 * confidence),        # Green
                    255,                          # Blue
                    255                           # Alpha
                )
                
                painter.setPen(QPen(color, 2))
                painter.drawLine(x1, y1, x2, y2)
    
    def freq_to_y(self, frequency):
        """Convert frequency to y-coordinate."""
        if frequency <= 0:
            return self.height()
        
        # Use logarithmic scale for frequency
        log_min = np.log2(MIN_FREQUENCY)
        log_max = np.log2(MAX_FREQUENCY)
        log_freq = np.log2(frequency)
        
        # Normalize and invert (0 = bottom, 1 = top)
        normalized = (log_freq - log_min) / (log_max - log_min)
        inverted = 1 - normalized
        
        return int(inverted * self.height())

class PitchVisualizer(QMainWindow):
    """Main window for the pitch visualizer application."""
    
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("PitchTrack Visualizer")
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
        
        # Start/Stop button
        self.start_stop_button = QPushButton("Start")
        self.start_stop_button.clicked.connect(self.toggle_recording)
        controls_layout.addWidget(self.start_stop_button)
        
        # Add spacer
        controls_layout.addStretch(1)
        
        # Add controls layout to main layout
        main_layout.addLayout(controls_layout)
        
        # Set up timer for demo animation
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_demo)
        self.is_recording = False
        
        # Demo data
        self.demo_index = 0
        self.demo_data = self.generate_demo_data()
        
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
        """)
    
    def toggle_recording(self):
        """Toggle between recording and stopped states."""
        if self.is_recording:
            self.timer.stop()
            self.start_stop_button.setText("Start")
            self.is_recording = False
        else:
            self.timer.start(UPDATE_INTERVAL)
            self.start_stop_button.setText("Stop")
            self.is_recording = True
    
    def generate_demo_data(self):
        """Generate demo pitch data for testing."""
        demo_data = []
        
        # Add a chromatic scale
        base_freq = 220.0  # A3
        for i in range(13):
            freq = base_freq * (2 ** (i / 12))
            for _ in range(10):  # Hold each note for 10 frames
                demo_data.append((freq, 1.0))  # (frequency, confidence)
        
        # Add a simple melody (Twinkle Twinkle Little Star)
        notes = [0, 0, 7, 7, 9, 9, 7, 5, 5, 4, 4, 2, 2, 0]  # Scale degrees
        for note in notes:
            freq = base_freq * (2 ** (note / 12))
            for _ in range(15):  # Hold each note for 15 frames
                demo_data.append((freq, 1.0))
        
        # Add a vibrato note
        center_freq = 440.0  # A4
        for i in range(100):
            # Add sine wave modulation
            vibrato_depth = 20.0  # Hz
            vibrato = vibrato_depth * np.sin(2 * np.pi * i / 20)
            freq = center_freq + vibrato
            demo_data.append((freq, 1.0))
        
        # Add a glissando
        start_freq = 220.0
        end_freq = 440.0
        steps = 100
        for i in range(steps):
            # Logarithmic interpolation
            t = i / (steps - 1)
            freq = start_freq * (end_freq / start_freq) ** t
            demo_data.append((freq, 1.0))
        
        return demo_data
    
    def update_demo(self):
        """Update the visualization with the next demo data point."""
        if self.demo_index >= len(self.demo_data):
            self.demo_index = 0
        
        frequency, confidence = self.demo_data[self.demo_index]
        self.piano.set_current_note(frequency)
        self.pitch_display.add_pitch(frequency, confidence)
        
        self.demo_index += 1

def main():
    app = QApplication(sys.argv)
    window = PitchVisualizer()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
