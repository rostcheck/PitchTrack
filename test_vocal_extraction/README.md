# Vocal Extraction Testing

This directory contains scripts for testing different vocal extraction methods (Spleeter and Demucs) for the PitchTrack application.

## Overview

We evaluated multiple vocal extraction methods to determine the best approach for isolating vocals from mixed audio recordings. Our findings indicate that Demucs provides superior vocal isolation quality, while Spleeter offers faster processing times.

For detailed findings, see:
- [CURRENT_STATE.md](CURRENT_STATE.md): Summary of our conclusions
- [RESEARCH_SUMMARY.md](RESEARCH_SUMMARY.md): Comprehensive research overview
- [ARM64_ENVIRONMENT_SETUP.md](ARM64_ENVIRONMENT_SETUP.md): Details on environment setup

## Environment Setup (Apple Silicon Mac)

### Prerequisites

- macOS on Apple Silicon (M1/M2/M3)
- Homebrew installed in `/opt/homebrew` (arm64 version)
- Python 3.11 installed via arm64 Homebrew

### Setting Up the Environment

1. Install Python 3.11 with arm64 architecture:
   ```bash
   brew install python@3.11
   ```

2. Verify the architecture:
   ```bash
   /opt/homebrew/bin/python3.11 -c "import platform; print(platform.machine())"
   # Should output: arm64
   ```

3. Create a virtual environment:
   ```bash
   /opt/homebrew/bin/python3.11 -m venv venv_py311_arm64
   ```

4. Activate the environment:
   ```bash
   source venv_py311_arm64/bin/activate
   ```

5. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Test Scripts

### Demucs Testing

1. Run the Demucs test:
   ```bash
   source venv_py311_arm64/bin/activate
   python test_demucs.py "/path/to/audio/file.mp3"
   ```

   Note: The first time you run Demucs, it will automatically download the pretrained models to `~/.cache/torch/hub/checkpoints/`.

### Spleeter Testing

1. Download pretrained models (first time only):
   ```bash
   source venv_py311_arm64/bin/activate
   mkdir -p pretrained_models
   
   # For 2-stem model
   curl -L -o pretrained_models/2stems.tar.gz https://github.com/deezer/spleeter/releases/download/v1.4.0/2stems.tar.gz
   mkdir -p pretrained_models/2stems
   tar -xzf pretrained_models/2stems.tar.gz -C pretrained_models/2stems
   
   # For 4-stem model
   curl -L -o pretrained_models/4stems.tar.gz https://github.com/deezer/spleeter/releases/download/v1.4.0/4stems.tar.gz
   mkdir -p pretrained_models/4stems
   tar -xzf pretrained_models/4stems.tar.gz -C pretrained_models/4stems
   
   # For 5-stem model
   curl -L -o pretrained_models/5stems.tar.gz https://github.com/deezer/spleeter/releases/download/v1.4.0/5stems.tar.gz
   mkdir -p pretrained_models/5stems
   tar -xzf pretrained_models/5stems.tar.gz -C pretrained_models/5stems
   ```

2. Basic Spleeter test:
   ```bash
   source venv_py311_arm64/bin/activate
   python test_spleeter.py "/path/to/audio/file.mp3"
   ```

3. Improved Spleeter test with 4-stem or 5-stem model:
   ```bash
   source venv_py311_arm64/bin/activate
   python test_spleeter_improved.py "/path/to/audio/file.mp3" --model 4stems
   ```

## Output Structure

The extracted files are organized in a clear directory structure:

```
test_results/
└── Song Name/
    ├── spleeter/
    │   ├── original.wav
    │   ├── vocals_spleeter.wav
    │   ├── instrumental_spleeter.wav
    │   └── timing_spleeter.txt
    ├── spleeter_4/
    │   ├── original.wav
    │   ├── vocals_spleeter.wav
    │   ├── instrumental_spleeter.wav
    │   ├── drums_spleeter.wav
    │   ├── bass_spleeter.wav
    │   ├── other_spleeter.wav
    │   └── timing_spleeter.txt
    └── demucs/
        ├── original.wav
        ├── vocals_demucs.wav
        ├── instrumental_demucs.wav
        └── timing_demucs.txt
```

## Troubleshooting

### Spleeter Issues

- Spleeter requires specific versions of TensorFlow and NumPy to work on arm64
- If you encounter errors with Spleeter, refer to [ARM64_ENVIRONMENT_SETUP.md](ARM64_ENVIRONMENT_SETUP.md)
- If you get model download errors, manually download the models as described in the "Spleeter Testing" section
- The test scripts expect pretrained models in the `pretrained_models` directory with subdirectories for each model type

### Demucs Issues

- Ensure PyTorch is installed with MPS support for hardware acceleration
- Verify with: `python -c "import torch; print(torch.backends.mps.is_available())"`
- Demucs models are stored in `~/.cache/torch/hub/checkpoints/`
- If model download fails, you can manually download them from the [Demucs GitHub repository](https://github.com/facebookresearch/demucs/releases)

## Conclusion

Based on our testing, Demucs is recommended for the PitchTrack application due to its superior vocal isolation quality, despite being slower than Spleeter.
