#!/usr/bin/env python3
"""
Vertical piano keyboard widget for PitchTrack.
"""

import numpy as np
from PyQt6.QtWidgets import QWidget
from PyQt6.QtGui import QPainter, QColor, QPen, QBrush, QFont
from PyQt6.QtCore import Qt, QRect

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

class VerticalPianoKeyboard(QWidget):
    """Widget that displays a vertical piano keyboard with highlighted keys based on pitch."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(80)
        self.setMaximumWidth(100)
        
        # Current highlighted key (MIDI note number)
        self.current_note = None
        
        # Define range of keys to display (C3 to C6)
        self.min_midi = 48  # C3
        self.max_midi = 84  # C6
        
        # Store key positions for external access
        self.key_positions = {}
        
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
    
    def get_y_for_midi(self, midi_note):
        """Get the y-coordinate for a given MIDI note number."""
        if midi_note in self.key_positions:
            return self.key_positions[midi_note]
        return None
    
    def get_y_for_frequency(self, frequency):
        """Get the y-coordinate for a given frequency."""
        if frequency <= 0:
            return None
        midi_note = int(round(freq_to_midi(frequency)))
        return self.get_y_for_midi(midi_note)
    
    def paintEvent(self, event):
        """Draw the piano keyboard."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Calculate key dimensions
        width = self.width()
        height = self.height()
        
        # Count white keys
        white_keys = [i for i in range(self.min_midi, self.max_midi + 1) 
                     if i % 12 not in [1, 3, 6, 8, 10]]  # Not black keys
        white_key_count = len(white_keys)
        
        # Calculate key height
        white_key_height = height / white_key_count
        
        # Clear previous key positions
        self.key_positions = {}
        
        # Draw white keys first (from high to low)
        # Reverse the order so C6 is at the top and C3 is at the bottom
        white_keys.reverse()  # Now C6 is first, C3 is last
        
        y_pos = 0  # Start from the top (high notes)
        
        for midi_note in white_keys:
            # Store position for this white key
            self.key_positions[midi_note] = y_pos + (white_key_height / 2)  # Store middle of key
            
            # Set color based on whether this key is highlighted
            if midi_note == self.current_note:
                painter.setBrush(QBrush(KEY_HIGHLIGHT_COLOR))
            else:
                painter.setBrush(QBrush(WHITE_KEY_COLOR))
            
            # Draw white key
            painter.setPen(QPen(Qt.GlobalColor.black, 1))
            painter.drawRect(0, int(y_pos), width, int(white_key_height))
            
            # Draw note name at right side of key
            painter.setPen(QPen(Qt.GlobalColor.black, 1))
            painter.setFont(QFont("Arial", 8))
            note = note_name(midi_note)
            painter.drawText(QRect(width - 30, int(y_pos), 25, int(white_key_height)), 
                            Qt.AlignmentFlag.AlignCenter, note)
            
            # Move down for the next key
            y_pos += white_key_height
        
        # Now draw black keys on top
        black_key_width = width * 0.6
        black_key_height = white_key_height * 0.6
        
        # Get all MIDI notes in our range
        all_notes = list(range(self.min_midi, self.max_midi + 1))
        all_notes.reverse()  # High to low
        
        for midi_note in all_notes:
            note_index = midi_note % 12
            
            # Only draw black keys
            if note_index not in [1, 3, 6, 8, 10]:
                continue
            
            # Find position based on adjacent white keys
            # For each black key, find the white key above it
            if note_index == 1:  # C#
                base_note = midi_note + 1  # D
            elif note_index == 3:  # D#
                base_note = midi_note + 1  # E
            elif note_index == 6:  # F#
                base_note = midi_note + 1  # G
            elif note_index == 8:  # G#
                base_note = midi_note + 1  # A
            elif note_index == 10:  # A#
                base_note = midi_note + 1  # B
            
            if base_note in self.key_positions:
                # Position black key between the white keys
                if note_index == 1:  # C#
                    y_pos = self.key_positions[base_note] - (white_key_height * 0.7)
                elif note_index == 3:  # D#
                    y_pos = self.key_positions[base_note] - (white_key_height * 0.7)
                elif note_index == 6:  # F#
                    y_pos = self.key_positions[base_note] - (white_key_height * 0.7)
                elif note_index == 8:  # G#
                    y_pos = self.key_positions[base_note] - (white_key_height * 0.7)
                elif note_index == 10:  # A#
                    y_pos = self.key_positions[base_note] - (white_key_height * 0.7)
                
                # Store position for this black key
                self.key_positions[midi_note] = y_pos
                
                # Set color based on whether this key is highlighted
                if midi_note == self.current_note:
                    painter.setBrush(QBrush(KEY_HIGHLIGHT_COLOR))
                else:
                    painter.setBrush(QBrush(BLACK_KEY_COLOR))
                
                # Draw black key
                painter.setPen(QPen(Qt.GlobalColor.black, 1))
                painter.drawRect(0, int(y_pos - black_key_height/2), int(black_key_width), int(black_key_height))
