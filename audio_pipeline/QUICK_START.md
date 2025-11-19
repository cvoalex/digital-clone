# Audio Pipeline - Quick Start Guide

## What This Is

This is a **standalone, testable** implementation of the SyncTalk_2D audio processing pipeline, designed specifically for:

1. **Understanding** the exact audio processing steps
2. **Testing** with known inputs and outputs
3. **Porting** to iOS/CoreML with validation

## What It Does

Converts audio → model-ready tensors in 2 stages:

```
Audio File (.wav)
    ↓
[Mel Spectrogram Processing]
    ↓
Mel Spectrogram (80 bands × time)
    ↓
[AudioEncoder Neural Network]
    ↓
Feature Tensors (32, 16, 16) per frame
```

## Installation

```bash
cd audio_pipeline
pip install -r requirements.txt
```

## Quick Test

```bash
# Test the pipeline (uses demo audio)
python3 tests/test_pipeline.py
```

This generates:
- ✅ `test_data/reference_output/` - Complete reference dataset
- ✅ All intermediate outputs saved
- ✅ Per-frame features for validation

## Generated Reference Dataset

After running tests, you'll have:

```
audio_pipeline/test_data/reference_output/
├── reference_audio.wav              # 2-second test audio
├── mel_spectrogram.npy             # Shape: (80, 161)
├── mel_windows.npy                 # Shape: (47, 16, 80)
├── audio_features_raw.npy          # Shape: (47, 512)
├── audio_features_padded.npy       # Shape: (49, 512)
├── frames/                         # Per-frame features
│   ├── frame_00000_window.npy     # (16, 512)
│   ├── frame_00000_reshaped.npy   # (32, 16, 16)
│   ├── frame_00001_window.npy
│   ├── frame_00001_reshaped.npy
│   └── ... (47 frames total)
├── metadata.json                   # Processing info
├── summary.json                    # Dataset summary
└── VALIDATION_INSTRUCTIONS.json    # iOS validation guide
```

## Using in Python

```python
from audio_pipeline import AudioPipeline

# Initialize
pipeline = AudioPipeline(
    checkpoint_path="model/checkpoints/audio_visual_encoder.pth",
    mode="ave",
    fps=25
)

# Process audio file
audio_features, metadata = pipeline.process_audio_file(
    "your_audio.wav",
    save_intermediates=True,
    output_dir="output"
)

# Get features for a specific video frame
frame_features = pipeline.get_frame_features(
    audio_features, 
    frame_idx=0,
    reshape=True  # Returns (32, 16, 16) for AVE mode
)
```

## For iOS Development

### Step 1: Generate Reference Data

```bash
python3 tests/test_pipeline.py
```

### Step 2: Convert AudioEncoder to CoreML

```python
# See README.md for full CoreML conversion code
import coremltools as ct
# ... convert PyTorch model to CoreML
```

### Step 3: Implement Mel Processing in Swift

```swift
// See audio_pipeline/README.md for Swift implementation guide
class MelSpectrogramProcessor {
    // Implement using Accelerate framework
}
```

### Step 4: Validate Your iOS Implementation

```bash
# After generating iOS outputs
python3 validate_ios_port.py \
    audio_pipeline/test_data/reference_output \
    your_ios_outputs
```

This compares:
- Mel spectrograms
- Audio features
- Per-frame outputs

**Tolerance**: Max difference should be < 1e-3

## File Structure

```
audio_pipeline/
├── __init__.py              # Package init
├── mel_processor.py         # Mel spectrogram conversion
├── audio_encoder.py         # AudioEncoder model
├── pipeline.py              # Complete pipeline
├── requirements.txt         # Dependencies
├── README.md               # Full documentation
├── QUICK_START.md          # This file
├── run_tests.py            # Test runner
├── validate_ios_port.py    # iOS validation
└── tests/                  # Test suite
    ├── test_mel_processor.py
    ├── test_audio_encoder.py
    └── test_pipeline.py
```

## Key Parameters

### Mel Spectrogram
- Sample rate: 16,000 Hz
- FFT size: 800
- Hop: 200 (12.5ms)
- Mel bands: 80
- Range: 55-7600 Hz
- Output range: [-4, 4]

### AudioEncoder
- Input: (batch, 1, 80, 16)
- Output: (batch, 512)
- Parameters: 2.8M

### Modes
- **AVE**: (32, 16, 16) - Default, recommended
- **Hubert**: (32, 32, 32) - Better speech understanding
- **WeNet**: (256, 16, 32) - Detailed phonemes

## Common Issues

**Q: Tests fail with "checkpoint not found"**
A: Make sure `model/checkpoints/audio_visual_encoder.pth` exists

**Q: How do I use my own audio?**
A: 
```python
pipeline.process_audio_file("your_audio.wav")
```

**Q: What's the expected output shape?**
A: For AVE mode (default), each frame gets (32, 16, 16)

**Q: How accurate should iOS port be?**
A: Max difference < 1e-3 is acceptable for mobile

## Next Steps

1. ✅ Run tests: `python3 tests/test_pipeline.py`
2. ✅ Check reference outputs in `test_data/reference_output/`
3. 📱 Convert AudioEncoder to CoreML
4. 📱 Implement mel processing in Swift
5. ✅ Validate with `validate_ios_port.py`

## Support

For detailed information:
- Full docs: See `README.md`
- iOS porting: See `VALIDATION_INSTRUCTIONS.json` in reference output
- Architecture: See parent `docs/` directory

---

**Ready to go!** Run the tests and start porting to iOS. 🚀

