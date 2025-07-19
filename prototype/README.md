# PitchTrack Prototype

This prototype implements and evaluates pitch detection algorithms for the PitchTrack application. The goal is to identify the most suitable algorithm for real-time vocal pitch tracking.

## Project Structure

```
/prototype/
├── reference_tones/           # Generated reference audio files
├── test_audio/                # Test audio files for evaluation
├── results/                   # Analysis results and visualizations
├── src/                       # Source code
│   ├── generate_tones.py      # Script to generate reference tones
│   ├── detect_pitch.py        # Pitch detection implementation
│   ├── analyze_results.py     # Analysis and visualization tools
│   └── utils.py               # Utility functions
└── README.md                  # Project documentation
```

## Setup

1. Install required Python packages:

```bash
pip install numpy matplotlib aubio librosa scipy pandas
```

2. Make the scripts executable:

```bash
chmod +x src/*.py
```

## Usage

### Step 1: Generate Reference Tones

Generate a set of reference tones with known pitch content:

```bash
./src/generate_tones.py --all
```

Options:
- `--all`: Generate all reference tones
- `--pure`: Generate pure tones
- `--scale`: Generate chromatic scales
- `--vibrato`: Generate vibrato tones
- `--melody`: Generate simple melody
- `--glissando`: Generate glissando (pitch slide)

### Step 2: Detect Pitch

Process audio files with pitch detection algorithms:

```bash
./src/detect_pitch.py reference_tones/tone_440Hz.wav --method yin --plot
```

Options:
- `--method`: Pitch detection method ('yin', 'yinfft', 'mcomb', 'fcomb', 'schmitt')
- `--buffer-size`: Buffer size for analysis (default: 2048)
- `--hop-size`: Hop size between frames (default: 512)
- `--sample-rate`: Sample rate (0 for original file's rate)
- `--plot`: Generate visualization of detected pitch
- `--output-dir`: Directory to save results and plots

### Step 3: Analyze Results

Compare different algorithms or analyze confidence impact:

```bash
# Compare algorithms
./src/analyze_results.py compare results/reference.json results/algorithm1.json results/algorithm2.json

# Analyze confidence impact
./src/analyze_results.py analyze results/algorithm1.json
```

## Example Workflow

1. Generate reference tones:

```bash
./src/generate_tones.py --all
```

2. Detect pitch using different algorithms:

```bash
./src/detect_pitch.py reference_tones/tone_440Hz.wav --method yin --plot
./src/detect_pitch.py reference_tones/tone_440Hz.wav --method yinfft --plot
./src/detect_pitch.py reference_tones/tone_440Hz.wav --method mcomb --plot
```

3. Compare algorithm performance:

```bash
./src/analyze_results.py compare reference_tones/tone_440Hz.json results/tone_440Hz_yin.json results/tone_440Hz_yinfft.json results/tone_440Hz_mcomb.json
```

## Evaluating Algorithms

When evaluating pitch detection algorithms, consider:

1. **Accuracy**: How close are the detected pitches to the reference?
2. **Latency**: How quickly can the algorithm detect pitch changes?
3. **Robustness**: How well does it handle vibrato, transitions, and noise?
4. **Confidence**: How reliable are the confidence values?

## Next Steps

After identifying the best algorithm:

1. Implement real-time processing with microphone input
2. Integrate with visualization similar to SingAndSee
3. Add karaoke track playback and synchronization
4. Implement feedback mechanisms for pitch accuracy
