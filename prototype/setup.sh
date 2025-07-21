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
check_package librosa
check_package sounddevice
check_package PyQt6

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
  echo "Or run the main application:"
  echo "  python ../pitch_track.py"
  echo ""
else
  echo "Please install the missing packages and run this script again."
fi
