#!/usr/bin/env python3
"""
Real-time pitch visualizer for PitchTrack.
Integrates audio processing with visualization.
"""

import sys
import os
import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QSlider, QLabel, 
                            QComboBox, QFrame)
from PyQt6.QtGui import QPainter, QColor, QPen, QBrush, QFont, QLinearGradient
from PyQt6.QtCore import Qt, QTimer, QRect, QPointF, pyqtSignal

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our modules
from src.pitch_visualizer import PianoKeyboard, PitchDisplay
from src.audio_processor import AudioProcessor

class RealtimeVisualizer(QMainWindow):
    """Main window for the real-time pitch visualizer application."""
    
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("PitchTrack Real-time Visualizer")
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
        
        # Set up audio processor
        self.audio_processor = AudioProcessor(callback=self.update_pitch)
        self.is_recording = False
        
        # Set up timer for UI updates
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_ui)
        self.timer.start(30)  # 30ms = ~33fps
        
        # Store latest pitch data
        self.current_frequency = 0.0
        self.current_confidence = 0.0
        
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
            self.audio_processor.stop()
            self.start_stop_button.setText("Start")
            self.is_recording = False
        else:
            self.audio_processor.start()
            self.start_stop_button.setText("Stop")
            self.is_recording = True
    
    def update_pitch(self, frequency, confidence):
        """Callback for audio processor to update pitch data."""
        self.current_frequency = frequency
        self.current_confidence = confidence
    
    def update_ui(self):
        """Update UI with current pitch data."""
        self.piano.set_current_note(self.current_frequency)
        self.pitch_display.add_pitch(self.current_frequency, self.current_confidence)
    
    def closeEvent(self, event):
        """Handle window close event."""
        if self.is_recording:
            self.audio_processor.stop()
        event.accept()

def main():
    app = QApplication(sys.argv)
    window = RealtimeVisualizer()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
