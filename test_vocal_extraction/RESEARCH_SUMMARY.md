# Vocal Extraction Research Summary

## Project Goal
Evaluate and compare vocal extraction methods (Spleeter and Demucs) to determine the best option for the PitchTrack karaoke trainer application.

## Research Process

### 1. Environment Setup
- Created Python 3.11 virtual environment with arm64 architecture support
- Resolved complex dependency conflicts:
  - TensorFlow 2.12.0 for Spleeter (to maintain tf.estimator compatibility)
  - NumPy 1.23.5 and Numba 0.57.1 for compatibility
  - PyTorch with MPS acceleration for Demucs

### 2. Implementation
- Successfully implemented both methods:
  - Demucs with default settings
  - Spleeter with multiple configurations:
    - 2-stem model (vocals + accompaniment)
    - 4-stem model (vocals, drums, bass, other)
    - 5-stem model (vocals, drums, bass, piano, other)
  - Applied best practices:
    - High-pass filtering (150 Hz)
    - Proper audio format handling

### 3. Performance Metrics
- Processing times (for "Jonathan Coulton - Code Monkey.mp3"):
  - Spleeter (2-stem): ~8.9 seconds
  - Spleeter (4-stem): ~11.3 seconds
  - Spleeter (5-stem): ~13.0 seconds
  - Demucs: ~56.7 seconds

### 4. Quality Assessment
- Conducted listening tests on all outputs
- Quality ranking (best to worst):
  1. Demucs: Superior vocal isolation with minimal bleed-through
  2. Spleeter (5-stem): Better than other Spleeter models but still has noticeable bleed-through
  3. Spleeter (4-stem): Improved over 2-stem but significant instrumental presence
  4. Spleeter (2-stem): Most instrumental bleed-through

## Conclusions

### Key Findings
1. **Quality vs. Speed Trade-off**:
   - Demucs provides significantly better vocal isolation but is ~5x slower
   - Spleeter is much faster but has more instrumental bleed-through

2. **Model Complexity Impact**:
   - More complex Spleeter models (4-stem, 5-stem) improve vocal isolation
   - Even the best Spleeter model doesn't match Demucs quality

3. **Technical Considerations**:
   - Spleeter uses deprecated TensorFlow APIs that may cause future compatibility issues
   - Demucs works well with modern PyTorch and benefits from MPS acceleration on Apple Silicon

### Final Recommendation
**Demucs is the recommended solution for the PitchTrack application** due to its superior vocal isolation quality. While it's slower than Spleeter, the quality of vocal isolation is critical for accurate pitch tracking, making Demucs the better choice.

## Implementation Files
- `test_demucs.py`: Demucs implementation
- `test_spleeter.py`: Basic Spleeter implementation
- `test_spleeter_improved.py`: Enhanced Spleeter with best practices
- `test_results/`: Output files for comparison

## Future Considerations
- Explore optimizations to improve Demucs processing speed
- Consider offering Spleeter as a faster alternative when speed is prioritized over quality
- Monitor updates to both libraries for improvements
- Investigate potential hybrid approaches for different use cases
