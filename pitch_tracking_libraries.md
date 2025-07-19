# MacOS Pitch Tracking Libraries Research

Based on Perplexity search results, here are the best pitch tracking libraries for MacOS with a focus on low latency:

## 1. **YIN Pitch Detection Algorithm**
   - **Description**: The YIN algorithm is a popular choice for real-time pitch detection. It uses autocorrelation in the time domain, which results in latency equal to twice the period of the minimum detectable frequency.
   - **Implementation**: Available as part of various projects, such as Limes, which provides a modified implementation of the YIN algorithm.
   - **Source**: [JUCE Forum discussion on low-latency pitch detection](https://forum.juce.com/t/lowest-latency-real-time-pitch-detection/51741)

## 2. **Bitstream Autocorrelation (BACF)**
   - **Description**: BACF is an efficient and fast algorithm that operates on zero-crossings, making it suitable for low-latency applications. It is reported to be as accurate as standard autocorrelation methods but faster, consuming about 50 nanoseconds per sample.
   - **Implementation**: This algorithm is not directly mentioned as a standalone library but could be implemented in a custom solution for MacOS.
   - **Source**: [Fast and Efficient Pitch Detection: Revisited](https://www.cycfi.com/2020/07/fast-and-efficient-pitch-detection-revisited/)

## 3. **Bungee Audio Library**
   - **Description**: While primarily designed for audio stretching with high controllability, Bungee also manipulates pitch in real-time. It offers reasonable latency, typically around 20 ms for speed and pitch controls, and 40 ms for input to output.
   - **Features**: It uses a frequency-domain phase-vocoder-based algorithm and is available as a modern C++ library with a permissive license.
   - **Source**: [GitHub - Bungee Audio Library](https://github.com/kupix/bungee)

## Additional Libraries Worth Considering

### 4. **Aubio**
   - Open-source library for audio labeling with pitch detection capabilities
   - Cross-platform, works well on macOS
   - Implements YIN and other algorithms
   - C library with Python bindings

### 5. **TarsosDSP**
   - Java library with pitch detection algorithms
   - Implements MPM (McLeod Pitch Method) which offers good performance
   - Cross-platform, works on macOS

### 6. **CREPE**
   - Deep learning approach to pitch detection
   - Very accurate but higher computational requirements
   - TensorFlow implementation available

## Comparison Table

| Library/Algorithm | Latency | Features | Best For |
|-------------------|---------|----------|----------|
| **YIN**           | Variable latency, minimum of 2x the period of the minimum detectable frequency | Time-domain autocorrelation | General purpose pitch detection |
| **BACF**          | Extremely low computation time (about 50 ns/sample) | Works on zero-crossings, fast and efficient | Ultra-low latency applications |
| **Bungee**        | Approx. 20-40 ms | Frequency-domain phase-vocoder, real-time pitch manipulation | Applications that need both pitch and time manipulation |
| **Aubio**         | Moderate | Multiple algorithms, good documentation | Projects needing a well-established library |
| **TarsosDSP**     | Low to moderate | Java-based, good for cross-platform | Java applications |
| **CREPE**         | Higher (depends on hardware) | Most accurate results | Applications where accuracy is paramount |

## Recommendations for PitchTrack

For the PitchTrack application, which requires real-time pitch tracking for vocal training:

1. **Best for Accuracy**: CREPE (if latency requirements allow)
2. **Best for Low Latency**: Custom implementation using BACF algorithm
3. **Best Balance**: pYIN (probabilistic YIN) implementation, which offers improvements over standard YIN

For ultra-low latency on MacOS, implementing a custom solution using BACF or optimizing existing libraries like YIN might be necessary. For applications that allow slightly higher latency (20-40ms), Bungee could be a suitable choice for its robust features in real-time audio manipulation.

## Additional Considerations for MacOS

- Core Audio framework provides low-level audio capabilities on macOS
- Audio Unit plugins can be developed with minimal latency
- Consider buffer size and sample rate settings for optimizing latency
- MacOS has specific audio routing considerations that may affect overall system latency

---
*Research based on Perplexity search results from July 19, 2025*
