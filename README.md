# PitchTrack - Karaoke Trainer Application

PitchTrack is an interactive application designed to help users improve their singing by providing real-time visual feedback on vocal pitch.

## Features

- **Piano Roll Visualization**: Clear display of notes with highlighted keys showing the current sung note
- **Connected Pitch Lines**: Smooth visualization of continuous singing with pitch tracking
- **File-Based Analysis**: Practice with existing audio recordings
- **Real-Time Audio Playback**: Listen to the original audio while seeing the pitch visualization
- **Enhanced Pitch Detection**: Accurate pitch tracking with stabilization to reduce jitter

## Usage

1. Run the application: `python pitch_track.py`
2. Click "Open File" to load an audio file (WAV, MP3, AIFF, OGG supported)
3. Use the playback controls to play, pause, and rewind the audio
4. Sing along with the recording and watch the pitch visualization
5. Adjust settings as needed for optimal pitch detection

## Requirements

- Python 3.8+
- PyQt6
- NumPy
- librosa
- sounddevice
- scipy

## Installation

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the application: `python pitch_track.py`

## Project Structure

- **pitch_track.py**: Main application with piano roll visualization and audio playback
- **requirements.txt**: List of required Python packages
- **prototype/**: Directory containing research and development code
  - **src/**: Source code for pitch detection algorithms and experimental features
  - **setup.sh**: Script to set up the prototype environment
  - Various test and reference files

## Development

This project evolved from a research prototype to a functional application. The development process included:

1. Research on pitch detection algorithms (see pitch_research.md)
2. Prototype implementation of various approaches (in prototype/src)
3. Development of the visualization component
4. Integration of audio playback and pitch tracking

## Future Enhancements

- Microphone input for real-time singing practice
- Score tracking and performance metrics
- Multiple visualization modes
- Customizable appearance and settings
- Song library management
