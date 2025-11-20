# Audio Usage Explanation

## What Audio Are We Using?

### The Sanders Package Has:

1. **`aud.wav`** (1.3 MB)
   - Original audio file
   - Used in final video (merged with generated frames)
   - NOT used for frame generation

2. **`aud_ave.npy`** (1.0 MB)  
   - **PRE-COMPUTED audio features**
   - Shape: (522, 512)
   - Already processed through the audio encoder
   - **This is what we use for frame generation!**

## What We're NOT Doing

❌ Running the audio encoder on `aud.wav`  
❌ Processing audio in real-time  
❌ Using the audio encoder ONNX model  

## What We ARE Doing

✅ Using **pre-computed** audio features (`aud_ave.npy`)  
✅ These were created by someone else ahead of time  
✅ We just load them and use them directly  

## The Flow

```
Audio Processing (ALREADY DONE):
  aud.wav → [Audio Encoder] → aud_ave.npy (522 × 512 features)
                                    ↓
                              [STORED IN PACKAGE]
                                    ↓
Frame Generation (WHAT WE'RE DOING):
  aud_ave.npy → [Reshape] → (1, 32, 16, 16) per frame
       +
  Pre-cut frames → [U-Net Model] → Generated frames
                                    ↓
  Generated frames + aud.wav → [ffmpeg] → Final video
```

## Why This Works

The Sanders package is **complete** - it includes:
- ✅ Pre-processed audio features (no need to run encoder!)
- ✅ Pre-cut image frames (no need to crop!)
- ✅ U-Net ONNX model (no need for PyTorch!)
- ✅ Original audio for final video

## What Each File Does

| File | Purpose | Used When |
|------|---------|-----------|
| `aud.wav` | Original audio | Final video assembly (ffmpeg) |
| `aud_ave.npy` | Audio features | Frame generation (ONNX input) |
| `aud_ave.bin` | Audio features (binary) | Go/Swift (same data, different format) |
| `models/audio_encoder.onnx` | Audio encoder | NOT USED (features pre-computed) |
| `models/generator.onnx` | U-Net model | Frame generation (main model) |

## Current Implementation

### Python:
```python
# Load pre-computed features (NOT running encoder)
audio_feats = np.load('aud_ave.npy')  # (522, 512)

# For each frame
audio_frame = audio_feats[i]  # (512,)

# Reshape to match model input
audio_reshaped = tile_and_reshape(audio_frame)  # (1, 32, 16, 16)

# Run U-Net (NOT audio encoder)
output = unet_model.run(image_input, audio_reshaped)
```

### Go:
```go
// Load pre-computed features (NOT running encoder)
audioFeats := loadBinary('aud_ave.bin')  // 522 × 512

// For each frame
audioFrame := audioFeats[i]  // 512 floats

// Reshape to match model input
audioReshaped := reshape(audioFrame)  // (1, 32, 16, 16)

// Run U-Net (NOT audio encoder)
output := unetModel.Run(imageInput, audioReshaped)
```

## Why We Don't Need Audio Encoder

The Sanders package **already ran** the audio encoder:

```
THEY DID (pre-packaged):
  aud.wav → [Audio Encoder ONNX] → aud_ave.npy ✅

WE DO (frame generation):
  aud_ave.npy → [Reshape] → [U-Net] → Frames ✅
```

## Two Separate Models

1. **Audio Encoder** (`audio_encoder.onnx`)
   - Input: Mel spectrograms
   - Output: 512-dim features
   - **Not used** - features pre-computed

2. **U-Net Generator** (`generator.onnx`)  
   - Input: 6-channel image + audio features
   - Output: Generated lip region
   - **This is what we use!**

## Summary

**Audio being used:**
- ✅ Pre-computed features from `aud_ave.npy` (for frame generation)
- ✅ Original `aud.wav` (for final video audio track)

**Models being used:**
- ✅ U-Net generator.onnx (for frame generation)
- ❌ Audio encoder.onnx (NOT used - features pre-computed)

**Audio encoder:**
- ❌ Not running it
- ✅ Using its pre-computed outputs

---

**TL;DR:** We're using **pre-computed audio features** from the Sanders package. No audio encoding needed - just frame generation! 🎬

