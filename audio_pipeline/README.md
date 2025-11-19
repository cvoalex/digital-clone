# Audio Processing Pipeline

A standalone, testable implementation of the SyncTalk_2D audio processing pipeline. This module is designed to be portable to iOS/CoreML and provides reference outputs for validation.

## Overview

This pipeline converts audio waveforms into feature tensors suitable for the SyncTalk_2D image generation model. It consists of two main stages:

1. **Mel Spectrogram Processing**: Converts audio to mel spectrograms
2. **Audio Encoding**: Extracts deep features using a pre-trained neural network

## Directory Structure

```
audio_pipeline/
├── __init__.py                 # Package initialization
├── mel_processor.py            # Mel spectrogram processing
├── audio_encoder.py            # AudioEncoder neural network
├── pipeline.py                 # Complete pipeline integration
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── tests/                      # Test suite
│   ├── test_mel_processor.py  # Mel processing tests
│   ├── test_audio_encoder.py  # Encoder tests
│   └── test_pipeline.py       # Integration tests
├── test_data/                  # Test audio and reference outputs
│   └── reference_output/       # Generated reference dataset
└── reference_outputs/          # Pre-generated reference outputs
```

## Installation

### Prerequisites

- Python 3.10
- CUDA-capable GPU (optional, but recommended)
- Access to the pretrained AudioEncoder checkpoint

### Setup

```bash
# Navigate to the audio_pipeline directory
cd audio_pipeline

# Install dependencies
pip install -r requirements.txt

# Or install from the parent directory
pip install -e .
```

## Usage

### Basic Usage

```python
from audio_pipeline import AudioPipeline

# Initialize the pipeline
pipeline = AudioPipeline(
    checkpoint_path="model/checkpoints/audio_visual_encoder.pth",
    mode="ave",  # or "hubert", "wenet"
    fps=25
)

# Process an audio file
audio_features, metadata = pipeline.process_audio_file(
    audio_path="demo/talk_hb.wav",
    save_intermediates=True,
    output_dir="output"
)

# Get features for a specific frame
frame_idx = 10
frame_features = pipeline.get_frame_features(
    audio_features, 
    frame_idx, 
    reshape=True  # Shape: (32, 16, 16) for AVE mode
)
```

### Generate Reference Dataset for iOS

```python
from audio_pipeline import AudioPipeline

pipeline = AudioPipeline(
    checkpoint_path="model/checkpoints/audio_visual_encoder.pth",
    mode="ave"
)

# Generate complete reference dataset
pipeline.process_and_save_all_frames(
    audio_path="test_audio.wav",
    output_dir="reference_output"
)
```

This creates:
- `mel_spectrogram.npy` - Raw mel spectrogram
- `audio_features_padded.npy` - Encoded audio features
- `frames/frame_XXXXX_reshaped.npy` - Per-frame features
- `metadata.json` - Processing metadata
- `VALIDATION_INSTRUCTIONS.json` - iOS validation guide

## Running Tests

### Using pytest

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_mel_processor.py -v

# Run with coverage
pytest tests/ --cov=audio_pipeline --cov-report=html
```

### Running test scripts directly

```bash
# Test mel processor
python tests/test_mel_processor.py

# Test audio encoder
python tests/test_audio_encoder.py

# Run integration tests and generate reference data
python tests/test_pipeline.py
```

## Pipeline Details

### Mel Spectrogram Processing

**Parameters:**
- Sample rate: 16,000 Hz
- FFT size: 800
- Hop length: 200 (12.5ms)
- Window length: 800 (50ms)
- Mel bands: 80
- Frequency range: 55-7600 Hz
- Pre-emphasis coefficient: 0.97

**Output:** Shape `(80, n_frames)`, normalized to `[-4, 4]`

### AudioEncoder Network

**Architecture:**
- Input: `[batch, 1, 80, 16]` (mel spectrogram window)
- Conv blocks with progressive downsampling
- Residual connections for stability
- Output: `[batch, 512]` (feature vector)

**Processing:**
1. Extract 16-frame mel windows
2. Process through AudioEncoder
3. Add temporal padding (repeat first/last frame)
4. Extract ±8 frame context window per frame
5. Reshape for model input

### Mode-Specific Output Shapes

| Mode    | Reshape Dimensions | Total Elements |
|---------|-------------------|----------------|
| AVE     | (32, 16, 16)      | 8,192         |
| Hubert  | (32, 32, 32)      | 32,768        |
| WeNet   | (256, 16, 32)     | 131,072       |

## iOS/CoreML Porting Guide

### Step 1: Convert AudioEncoder to CoreML

```python
import torch
import coremltools as ct

# Load the model
model = AudioEncoder()
checkpoint = torch.load('model/checkpoints/audio_visual_encoder.pth')
model.load_state_dict(checkpoint)
model.eval()

# Trace the model
example_input = torch.randn(1, 1, 80, 16)
traced_model = torch.jit.trace(model, example_input)

# Convert to CoreML
mlmodel = ct.convert(
    traced_model,
    inputs=[ct.TensorType(shape=(1, 1, 80, 16), name="mel_input")],
    outputs=[ct.TensorType(name="features")]
)

# Save
mlmodel.save("AudioEncoder.mlmodel")
```

### Step 2: Implement Mel Spectrogram Processing

The mel spectrogram processing can be implemented in Swift using Accelerate framework:

```swift
import Accelerate

class MelSpectrogramProcessor {
    let sampleRate: Int = 16000
    let nFFT: Int = 800
    let hopLength: Int = 200
    let nMels: Int = 80
    
    func processs(audio: [Float]) -> [[Float]] {
        // 1. Apply pre-emphasis
        // 2. Compute STFT
        // 3. Convert to mel scale
        // 4. Convert to dB
        // 5. Normalize to [-4, 4]
    }
}
```

### Step 3: Validate Against Reference Outputs

```swift
// Load reference data
let referenceMel = loadNumpyArray("mel_spectrogram.npy")
let yourMel = processMelSpectrogram(audio)

// Compare
let maxDiff = computeMaxDifference(referenceMel, yourMel)
assert(maxDiff < 1e-3, "Mel spectrogram validation failed")
```

### Step 4: Integration

```swift
class AudioPipeline {
    let melProcessor = MelSpectrogramProcessor()
    let audioEncoder: AudioEncoderModel // CoreML model
    
    func processFrame(audio: [Float], frameIndex: Int) -> MLMultiArray {
        // 1. Convert to mel spectrogram
        let melSpec = melProcessor.process(audio: audio)
        
        // 2. Extract window for frame
        let melWindow = extractWindow(melSpec, frameIndex: frameIndex)
        
        // 3. Run through AudioEncoder
        let features = audioEncoder.prediction(from: melWindow)
        
        // 4. Extract temporal context
        let contextFeatures = extractContext(features, frameIndex: frameIndex)
        
        // 5. Reshape for U-Net
        let reshaped = reshape(contextFeatures, mode: .ave)
        
        return reshaped
    }
}
```

## Reference Outputs

Run the test suite to generate reference outputs:

```bash
python tests/test_pipeline.py
```

This creates a complete reference dataset in `test_data/reference_output/` with:
- All intermediate processing outputs
- Per-frame features for validation
- Metadata and validation instructions

## Validation Criteria

When validating your iOS implementation:

1. **Mel Spectrogram**:
   - Shape matches: `(80, n_frames)`
   - Value range: `[-4, 4]`
   - Max difference from reference: `< 1e-3`

2. **Audio Features**:
   - Shape matches: `(n_frames, 512)`
   - Max difference from reference: `< 1e-3`

3. **Frame Features**:
   - Shape matches mode-specific dimensions
   - Max difference from reference: `< 1e-3`

## Troubleshooting

### Import Errors

```bash
# Make sure you're in the project root
cd /path/to/digital-clone

# Install in development mode
pip install -e .
```

### Missing Checkpoint

The AudioEncoder checkpoint should be at:
```
model/checkpoints/audio_visual_encoder.pth
```

If missing, ensure you've downloaded the pre-trained models.

### GPU Out of Memory

Reduce batch size in `AudioEncoderWrapper.process_mel_windows()`:

```python
audio_features = audio_encoder.process_mel_windows(mel_windows, batch_size=32)
```

## Contributing

When modifying the pipeline:

1. Ensure backward compatibility
2. Update tests
3. Regenerate reference outputs
4. Update this README

## License

This code is part of the SyncTalk_2D project. See the main repository LICENSE file.

## Acknowledgements

Based on the audio processing pipeline from:
- SyncTalk
- Ultralight-Digital-Human
- SyncTalk_2D

## Contact

For questions about iOS porting or validation, please refer to the generated `VALIDATION_INSTRUCTIONS.json` in the reference output directory.

