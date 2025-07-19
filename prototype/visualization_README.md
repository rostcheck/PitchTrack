# PitchTrack Visualization Component

This directory contains the visualization component for the PitchTrack application, which provides a real-time display of pitch similar to SingAndSee.

## Components

### 1. Basic Visualization Framework (`pitch_visualizer.py`)
- Piano keyboard display
- Scrolling pitch history display
- Demo mode with synthetic pitch data

### 2. File-Based Visualizer (`file_visualizer.py`)
- Load and analyze audio files
- Visualize pitch in a piano roll format
- Play back audio files with synchronized visualization

### 3. Real-Time Visualizer (`realtime_visualizer.py`)
- Capture audio from microphone
- Process pitch in real-time
- Display pitch on piano keyboard and scrolling display

## Usage

### Demo Mode
```bash
python src/pitch_visualizer.py
```
This runs the basic visualization with synthetic demo data.

### File-Based Visualization
```bash
python src/file_visualizer.py
```
1. Click "Open File" to select an audio file
2. Click "Play" to start visualization
3. The piano keyboard will highlight the current note
4. The pitch display will show the pitch history

### Real-Time Visualization (Experimental)
```bash
python src/realtime_visualizer.py
```
1. Click "Start" to begin capturing audio
2. Sing or play an instrument
3. The visualization will display your pitch in real-time

## Technical Details

### Piano Keyboard Display
- Shows a range of notes from E2 (82.4 Hz) to B5 (987.8 Hz)
- Highlights the current note being played/sung
- Labels C notes for easy reference

### Pitch History Display
- Shows pitch over time (5 seconds of history)
- Logarithmic frequency scale for musical accuracy
- Color-coded by confidence level
- Grid lines corresponding to semitones

### Pitch Detection
- Uses librosa's `piptrack` function for pitch detection
- Configurable parameters for minimum/maximum frequency
- Confidence values based on magnitude of detected pitch

## Next Steps

1. **Improve Real-Time Performance**:
   - Optimize audio processing for lower latency
   - Implement more efficient pitch detection algorithms

2. **Add Karaoke Features**:
   - Load and display reference pitch tracks
   - Compare sung pitch with reference pitch
   - Provide feedback on pitch accuracy

3. **Enhanced Visualization**:
   - Add waveform display
   - Implement spectrogram view
   - Show note accuracy scoring

4. **User Interface Improvements**:
   - Add settings for display parameters
   - Implement recording and playback of user performances
   - Create a more polished, native macOS look and feel

## Dependencies

- PyQt6: UI framework
- librosa: Audio processing and pitch detection
- numpy: Numerical processing
- sounddevice: Audio capture (for real-time mode)
