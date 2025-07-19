# Pitch Detection Research

## Overview of Pitch Detection Algorithms

This document outlines research on pitch detection algorithms suitable for the PitchTrack application, focusing on their strengths, weaknesses, and applicability to real-time vocal pitch tracking.

## Key Algorithms

### 1. YIN Algorithm
- **Approach**: Time-domain algorithm based on autocorrelation with additional processing
- **Strengths**:
  - Relatively efficient computation
  - Good accuracy for monophonic audio
  - Well-established with many implementations
- **Weaknesses**:
  - Can struggle with vibrato and rapid transitions
  - Less accurate at extreme pitch ranges
- **Implementation**: Available in libraries like Aubio, Essentia, and librosa
- **Suitability**: Good baseline option, especially for resource-constrained environments

### 2. pYIN (Probabilistic YIN)
- **Approach**: Enhanced version of YIN using probabilistic modeling
- **Strengths**:
  - Better handling of note transitions
  - Improved accuracy for vibrato and ornaments
  - Good for vocal applications
- **Weaknesses**:
  - Higher computational cost than YIN
  - More complex implementation
- **Implementation**: Available in Sonic Annotator, pYIN library
- **Suitability**: Excellent choice for vocal pitch tracking with moderate computational resources

### 3. CREPE (Convolutional Representation for Pitch Estimation)
- **Approach**: Deep learning approach using convolutional neural networks
- **Strengths**:
  - State-of-the-art accuracy
  - Robust to noise and varied vocal timbres
  - Handles vibrato and complex vocal techniques well
- **Weaknesses**:
  - Highest computational requirements
  - Requires GPU for optimal performance in real-time applications
- **Implementation**: TensorFlow implementation available on GitHub
- **Suitability**: Best accuracy but may require optimization for real-time use

### 4. McLeod Pitch Method (MPM)
- **Approach**: Based on normalized square difference function (NSDF)
- **Strengths**:
  - Good performance for real-time applications
  - Handles noisy signals well
  - Efficient implementation possible
- **Weaknesses**:
  - Less accurate than pYIN or CREPE for vocals
  - May require parameter tuning
- **Implementation**: Available in TarsosDSP, Aubio
- **Suitability**: Good alternative if YIN performance is insufficient

### 5. Harmonic Product Spectrum (HPS)
- **Approach**: Frequency-domain method analyzing harmonic structure
- **Strengths**:
  - Works well for strongly harmonic sounds
  - Conceptually straightforward
- **Weaknesses**:
  - Less suitable for vocals with weak harmonics
  - Struggles with breathy or noisy vocals
- **Implementation**: Can be implemented using FFT libraries
- **Suitability**: Better for instrumental pitch tracking than vocals

## Evaluation Criteria for PitchTrack

When selecting an algorithm, consider:

1. **Accuracy**: How precisely does it track pitch variations?
2. **Latency**: How quickly can it detect pitch (critical for real-time feedback)?
3. **Robustness**: How well does it handle different voices, microphones, and environments?
4. **Computational Efficiency**: Will it run smoothly on target hardware?
5. **Implementation Complexity**: Development time and maintenance considerations

## Recommendations

### Primary Recommendation: pYIN
- Best balance of accuracy and performance for vocal applications
- Handles the nuances of singing well
- Reasonable computational requirements
- Good existing implementations to build upon

### Alternative: CREPE
- Consider if accuracy is paramount and computational resources are available
- May require optimization for real-time performance
- Could be used as a high-quality offline analysis option

### Implementation Strategy
1. Start with pYIN implementation for core functionality
2. Benchmark performance and accuracy with test recordings
3. Optimize parameters for vocal tracking specifically
4. Consider a hybrid approach: real-time feedback with pYIN, detailed post-performance analysis with CREPE

## Available Libraries and Resources

- **Aubio**: C library with Python bindings implementing YIN and other algorithms
- **librosa**: Python package with pitch detection capabilities
- **TarsosDSP**: Java library implementing MPM and other algorithms
- **CREPE**: TensorFlow implementation available on GitHub
- **Essentia**: C++ library with Python bindings for audio analysis including pitch detection

## Next Steps

1. Prototype implementations of pYIN and CREPE
2. Develop evaluation framework with test vocal recordings
3. Measure accuracy, latency, and resource usage
4. Select final algorithm based on empirical results
5. Optimize implementation for real-time performance
