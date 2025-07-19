#!/bin/bash

# Make all Python scripts executable
chmod +x src/*.py

# Create necessary directories
mkdir -p reference_tones
mkdir -p test_audio
mkdir -p results

# Check if required Python packages are installed
echo "Checking required Python packages..."

# Function to check if a package is installed
check_package() {
  python3 -c "import $1" 2>/dev/null
  if [ $? -ne 0 ]; then
    echo "Package $1 is not installed. Please install it using: pip install $1"
    MISSING_PACKAGES=1
  else
    echo "✓ $1"
  fi
}

MISSING_PACKAGES=0

# Check required packages
check_package numpy
check_package matplotlib
check_package scipy
check_package pandas

# Check if aubio is installed
python3 -c "import aubio" 2>/dev/null
if [ $? -ne 0 ]; then
  echo "Package aubio is not installed. Please install it using: pip install aubio"
  echo "Note: If installation fails, you may need to install portaudio first:"
  echo "  macOS: brew install portaudio"
  echo "  Linux: sudo apt-get install libportaudio2 portaudio19-dev"
  MISSING_PACKAGES=1
else
  echo "✓ aubio"
fi

# Check if librosa is installed
check_package librosa

if [ $MISSING_PACKAGES -eq 0 ]; then
  echo "All required packages are installed."
  echo "Setup complete! You can now use the prototype."
  echo ""
  echo "Try generating some reference tones:"
  echo "  ./src/generate_tones.py --all"
  echo ""
  echo "Then detect pitch in the generated tones:"
  echo "  ./src/detect_pitch.py reference_tones/tone_440Hz.wav --method yin --plot"
  echo ""
else
  echo "Please install the missing packages and run this script again."
fi
