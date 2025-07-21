# Comparing Audio Source Separation Algorithms for Vocal Extraction

*A technical evaluation of Librosa, Spleeter, and Demucs for isolating vocals in music recordings*

As an AI software architect working on audio processing applications, I recently conducted a comprehensive evaluation of different audio source separation algorithms for a karaoke trainer application called PitchTrack. The goal was to find the most effective method for isolating vocals from instrumental accompaniment, a critical component for accurate pitch tracking and feedback.

## The Challenge of Vocal Extraction

Separating vocals from mixed audio is a challenging signal processing problem that has seen significant advancements through deep learning approaches. For our karaoke trainer application, we needed a solution that could:

1. Cleanly isolate vocals with minimal instrumental bleed-through
2. Process audio within reasonable time constraints
3. Run efficiently on consumer hardware (specifically Apple Silicon Macs)
4. Integrate smoothly with our Python-based application stack

## Algorithms Evaluated

We evaluated three leading approaches to audio source separation:

### 1. Librosa (Traditional DSP Methods)

While Librosa is an excellent audio processing library, it doesn't provide dedicated source separation capabilities comparable to deep learning approaches. We initially explored using its harmonic-percussive source separation (HPSS) as a baseline:

```python
import librosa
import numpy as np

# Load the audio file
y, sr = librosa.load('song.mp3', sr=None)

# Perform harmonic-percussive source separation
y_harmonic, y_percussive = librosa.effects.hpss(y)

# The harmonic component might contain vocals, but with significant limitations
```

This approach proved insufficient for our needs, as it doesn't specifically target vocals and produces significant bleed-through from other harmonic instruments.

### 2. Spleeter (by Deezer)

Spleeter is an open-source music source separation library developed by Deezer Research. It offers several pre-trained models for separating audio into different stems:

- 2 stems (vocals + accompaniment)
- 4 stems (vocals, drums, bass, other)
- 5 stems (vocals, drums, bass, piano, other)

Implementation was straightforward, though we encountered some dependency challenges with TensorFlow versions on Apple Silicon:

```python
from spleeter.separator import Separator

# Initialize the separator with the desired model
separator = Separator('spleeter:5stems')

# Load and separate audio
waveform, sr = librosa.load('song.mp3', sr=44100, mono=False)
# Ensure correct shape (samples, channels)
if waveform.shape[0] == 2:
    waveform = waveform.T

# Separate the audio
prediction = separator.separate(waveform)
vocals = prediction['vocals']
```

We implemented several optimizations based on best practices:
- Applied high-pass filtering (150 Hz) to improve vocal clarity
- Tested multiple stem configurations (2, 4, and 5 stems)
- Ensured proper audio format handling

### 3. Demucs (by Facebook Research)

Demucs (Deep Extractor for Music Sources) is a state-of-the-art music source separation model developed by Facebook Research. Implementation was relatively straightforward:

```python
import torch
from demucs.pretrained import get_model
from demucs.apply import apply_model

# Load the model
model = get_model('htdemucs')
model.cpu()
model.eval()

# Load and process audio
wav, sr = librosa.load('song.mp3', sr=model.samplerate, mono=False)
if wav.ndim == 1:
    wav = np.stack([wav, wav])

# Apply the model
wav = torch.tensor(wav)
with torch.no_grad():
    sources = apply_model(model, wav[None])[0]

# Extract vocals
sources = sources.cpu().numpy()
vocals = sources[model.sources.index('vocals')]
```

## Performance Comparison

We evaluated each algorithm on multiple dimensions:

### 1. Vocal Isolation Quality

After extensive listening tests:

- **Demucs**: Superior vocal isolation with minimal instrumental bleed-through
- **Spleeter (5-stem)**: Good separation but noticeable instrumental artifacts
- **Spleeter (4-stem)**: Moderate separation quality
- **Spleeter (2-stem)**: Significant instrumental bleed-through
- **Librosa HPSS**: Poor vocal isolation (not specifically designed for this task)

### 2. Processing Speed

Testing with a 3-minute song on an M1 Mac:

- **Spleeter (2-stem)**: ~9 seconds
- **Spleeter (4-stem)**: ~11 seconds
- **Spleeter (5-stem)**: ~13 seconds
- **Demucs**: ~57 seconds
- **Librosa HPSS**: ~3 seconds (but with inadequate results)

### 3. Implementation Challenges

**Spleeter**:
- Dependency conflicts between TensorFlow versions and Python architectures
- Requires specific versions (TensorFlow 2.12.0, NumPy 1.23.5)
- Uses deprecated TensorFlow APIs (Estimator) that may cause future compatibility issues
- Requires careful handling of audio format (samples, channels)

**Demucs**:
- Heavier computational requirements
- Longer processing times
- Otherwise straightforward implementation with PyTorch

### 4. Architecture Compatibility

Both Spleeter and Demucs required specific configurations to work properly on Apple Silicon (arm64) architecture:

- Needed to create a dedicated Python 3.11 virtual environment with arm64 architecture
- Resolved complex dependency conflicts between NumPy, TensorFlow, and Numba
- Configured PyTorch with MPS acceleration for Demucs

## Conclusion and Recommendations

After thorough evaluation, **Demucs emerged as the clear winner for vocal isolation quality**, despite being significantly slower than Spleeter. For our karaoke trainer application, where accurate pitch tracking depends on clean vocal isolation, the superior quality of Demucs outweighs the performance penalty.

Key takeaways from our evaluation:

1. **Quality vs. Speed Trade-off**: There's a clear inverse relationship between isolation quality and processing speed among these algorithms.

2. **Model Complexity Impact**: More complex models (like Spleeter's 5-stem) generally provide better separation but at the cost of increased processing time.

3. **Practical Implementation**: Consider your specific requirements carefully:
   - For real-time applications where speed is critical, Spleeter might be preferable
   - For applications requiring high-quality vocal isolation, Demucs is the better choice
   - For deployment on consumer hardware, pay careful attention to architecture compatibility

4. **Future Considerations**: The field of audio source separation is rapidly evolving, with new models and approaches emerging regularly. It's worth periodically re-evaluating available solutions.

For our PitchTrack karaoke trainer application, we've implemented Demucs as our vocal extraction solution, accepting the longer processing time in exchange for superior vocal isolation quality that enables more accurate pitch tracking and user feedback.

---

*What audio processing challenges are you facing in your projects? Have you worked with these or other source separation algorithms? I'd be interested to hear about your experiences in the comments.*
