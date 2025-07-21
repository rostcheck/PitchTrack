# Building a Karaoke Pitch Tracker: From Concept to Application

## Introduction

Have you ever wanted to improve your singing but found it difficult to know if you're hitting the right notes? That's the problem we set out to solve with PitchTrack, a karaoke trainer application that provides real-time visual feedback on vocal pitch. In this article, I'll share our journey from initial concept to functional application, including the research, prototyping, and development process.

## The Vision

The vision for PitchTrack was simple yet ambitious: create an application that helps users improve their singing by visualizing their vocal pitch in real-time, similar to professional tools like SingAndSee but more accessible and focused on the karaoke experience.

Our goals were to:

1. Accurately detect and visualize vocal pitch in real-time
2. Provide an intuitive interface that singers of all levels could understand
3. Allow users to practice with their favorite songs
4. Give immediate visual feedback on pitch accuracy

## Research Phase

Before writing a single line of code, we needed to understand the technical challenges involved in pitch detection. This led us down a fascinating rabbit hole of signal processing algorithms and audio analysis techniques.

### Exploring Pitch Detection Algorithms

We researched several pitch detection algorithms, each with its own strengths and weaknesses:

- **YIN Algorithm**: A time-domain algorithm based on autocorrelation, offering a good balance of efficiency and accuracy
- **pYIN (Probabilistic YIN)**: An enhanced version with better handling of note transitions and vibrato
- **CREPE**: A deep learning approach using convolutional neural networks, offering state-of-the-art accuracy but higher computational requirements
- **McLeod Pitch Method (MPM)**: Based on normalized square difference function, good for real-time applications

After evaluating these options, we decided to start with pYIN for our prototype due to its good performance with vocal audio and reasonable computational requirements.

### Library Selection

We also needed to choose the right libraries for our implementation:

- **librosa**: A Python package for music and audio analysis
- **PyQt6**: For building the graphical user interface
- **sounddevice**: For audio playback and capture
- **numpy** and **scipy**: For numerical processing and signal analysis

## Prototyping

With our research complete, we moved on to prototyping. We followed a systematic approach:

1. Generate reference tones with known frequencies
2. Implement pitch detection algorithms
3. Analyze and visualize the results
4. Evaluate performance and accuracy

### The First Prototype

Our first prototype focused on processing pre-recorded audio files rather than real-time microphone input to simplify initial development. We created a structured project with scripts for:

- Generating reference tones
- Detecting pitch using different algorithms
- Analyzing and visualizing results

This allowed us to compare different approaches and fine-tune our pitch detection parameters.

### Visualization Experiments

The next challenge was visualizing the pitch data in a way that would be intuitive for singers. We experimented with several approaches:

1. **Simple line graphs**: Showing pitch over time
2. **Piano roll display**: Mapping pitch to piano keys
3. **Combined approach**: Piano keyboard with scrolling pitch history

The piano roll visualization proved most intuitive, as it directly maps to musical notes and provides clear visual feedback.

## From Prototype to Application

As our prototype evolved, we began to refine the user interface and add features that would make the application more useful for singers.

### Key Development Milestones

Looking at our git history, we can see the progression:

1. **Initial commit**: Basic project structure and research documentation
2. **Added noise filtering**: Implemented RMS energy filtering to focus on vocal parts
3. **Tuned for lead vocal line**: Optimized detection parameters for vocal audio
4. **Improved visualization**: Added keyboard highlighting to show the current note being sung

### Design Iterations

One of the most significant design decisions was the visualization approach. We initially created a vertical piano keyboard on the left side of the display, but found several issues:

1. The keyboard orientation was inverted (high notes at bottom)
2. Grid lines didn't align properly with piano keys
3. The visualization wasn't intuitive for users

After several iterations, we arrived at a piano roll style visualization with:

- Evenly spaced pitch lines for better readability
- Colored key indicators on the left showing black and white piano keys
- Note names displayed directly on the keys
- High notes at the top and low notes at the bottom
- Key highlighting when a note is sung, with smoothing to reduce jitter

This design proved much more intuitive and effective for users.

## Technical Challenges and Solutions

Throughout the development process, we encountered several technical challenges:

### Challenge 1: Pitch Detection Accuracy

**Problem**: Initial pitch detection was jumpy and inaccurate, especially with vocal audio that contains vibrato and transitions.

**Solution**: We implemented several improvements:
- Added pitch stabilization by rounding to the nearest semitone
- Applied median filtering to smooth the pitch contour
- Added confidence thresholds to ignore uncertain pitch estimates
- Implemented energy thresholds to filter out non-vocal audio

### Challenge 2: Visualization Clarity

**Problem**: Early visualizations were confusing and didn't clearly show the relationship between the sung pitch and target notes.

**Solution**: We evolved the visualization through several iterations:
- Started with a basic piano keyboard and separate pitch display
- Moved to an integrated piano roll style visualization
- Added connected pitch lines to show continuous singing
- Implemented key highlighting with smoothing to reduce jitter
- Made keys larger and added note names for better readability

### Challenge 3: User Interface Responsiveness

**Problem**: The application needed to be responsive while processing audio and updating visualizations.

**Solution**: We implemented:
- Background processing for audio file analysis
- Progress dialog for file loading
- Efficient drawing routines for the visualization
- Optimized data structures for pitch history

## Lessons Learned

Throughout this project, we learned several valuable lessons:

1. **Start with research**: Understanding the problem domain (pitch detection algorithms) before coding saved us time in the long run.

2. **Prototype iteratively**: Building a simple prototype first allowed us to experiment with different approaches before committing to a design.

3. **Focus on user experience**: The most technically accurate solution isn't always the most useful. We prioritized a clear, intuitive visualization over perfect pitch detection.

4. **Evolve don't multiply**: Instead of creating multiple parallel implementations, we focused on evolving our code, using version control to track changes.

5. **Regular refactoring**: Periodically reviewing and refactoring our code helped maintain quality and prevent technical debt.

## Current State and Future Plans

PitchTrack has evolved from a research prototype to a functional application that allows users to:

- Load audio files in various formats (WAV, MP3, AIFF, OGG)
- Visualize the pitch of the vocal line
- Play back the audio while seeing the pitch visualization
- Adjust settings for optimal pitch detection

But we're not done yet! Future enhancements include:

- Microphone input for real-time singing practice
- Score tracking and performance metrics
- Multiple visualization modes
- Customizable appearance and settings
- Song library management

## Conclusion

Building PitchTrack has been a fascinating journey through signal processing, audio analysis, and user interface design. The project demonstrates how research, prototyping, and iterative development can transform a concept into a useful application.

Whether you're a developer interested in audio processing or a singer looking to improve your pitch, we hope this journey provides insights and inspiration for your own projects.

---

*This blog post documents the development of PitchTrack as of July 2025. The application continues to evolve, and we welcome contributions and feedback from the community.*
