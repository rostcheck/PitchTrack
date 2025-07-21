# PitchTrack Prototype

This directory contains the research prototype and exploratory code for the PitchTrack application. It serves as a testbed for pitch detection algorithms and visualization approaches.

## Directory Structure

- **src/**: Source code for various components
  - **analyze_results.py**: Tools for analyzing pitch detection results
  - **audio_processor.py**: Audio processing utilities
  - **detect_pitch.py**: Comprehensive pitch detection implementation
  - **detect_pitch_simple.py**: Simplified pitch detection for testing
  - **file_visualizer.py**: Original visualization implementation (now moved to main directory)
  - **generate_tones.py**: Script to generate reference tones for testing
  - **utils.py**: Utility functions
  - **vertical_piano.py**: Piano keyboard visualization component
  - **vocal_pitch_detector.py**: Specialized pitch detection for vocals
- **reference_tones/**: Generated reference audio files
- **test_audio/**: Test audio files for evaluation
- **results/**: Analysis results and visualizations

## Getting Started

1. Run the setup script to prepare the environment:
   ```
   ./setup.sh
   ```

2. Generate reference tones for testing:
   ```
   ./src/generate_tones.py --all
   ```

3. Test pitch detection on a reference tone:
   ```
   ./src/detect_pitch.py reference_tones/tone_440Hz.wav --method yin --plot
   ```

## Relationship to Main Application

The research and code in this prototype directory informed the development of the main PitchTrack application. The main application (pitch_track.py) in the parent directory is the evolved version of the prototype, with refined features and a more polished user interface.

This prototype code is maintained for reference and continued research purposes.
