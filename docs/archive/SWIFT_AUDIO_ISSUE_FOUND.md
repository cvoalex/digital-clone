# Swift Audio Issue - ROOT CAUSE FOUND

## Problem

Swift generates frames with **closed mouth** (no lip sync) even though:
- ✅ Audio features ARE changing between frames
- ✅ Generator output IS changing
- ✅ Core ML and Metal are working

## Root Cause: Audio Encoder Output Scale

**The Swift audio encoder produces values 200x smaller than Go!**

### Comparison:

**Go audio encoder output (frame 1):**
```
First 10 values: [0.0, 0.347, 0.179, 0.742, 0.0, 0.0, 0.289, 0.0, 0.0, 0.0]
Range: [0, ~10]
```

**Swift audio encoder output (frame 1):**
```
First 10 values: [0.0, 0.044, 0.013, 0.0, 0.0, 0.031, 0.0, 0.011, 0.017, 0.0]
Range: [0, ~0.05]
```

**Difference: Swift values are ~200x smaller!**

### After Reshaping to [1, 32, 16, 16]:

**Go:** Values up to ~9.6  
**Swift:** Values up to ~0.04  

**Mean difference: 0.48 (huge!)**

## Why This Causes Closed Mouth

The U-Net generator expects audio features in a certain range (0-10).

When Swift feeds values 200x smaller (0-0.05), the generator interprets this as:
- Very quiet/no audio
- No speech signal
- **→ Generates closed mouth!**

## Possible Causes

### 1. Core ML Model Weights
The AudioEncoder.mlpackage might have:
- Different weights than the ONNX model
- Been converted from an untrained model
- Lost weight precision during conversion

### 2. Normalization Issue
The Core ML conversion might have:
- Added automatic normalization
- Changed the output scale
- Applied FP16 precision (half precision)

### 3. Model Architecture Mismatch
The PyTorch→Core ML conversion might have:
- Changed layer operations
- Modified activations
- Altered the output range

## Solutions

### Option 1: Scale Swift Audio Features (Quick Fix)
Multiply Swift audio encoder output by 200:
```swift
features = features.map { $0 * 200.0 }
```

### Option 2: Use ONNX Audio Encoder in Swift
Instead of Core ML for audio encoder, use ONNX:
- Load audio_encoder.onnx with ONNX Runtime
- Should match Go/Python exactly

### Option 3: Reconvert Audio Encoder
Check the conversion in `convertmodeltocoreml/`:
- Verify it loaded correct weights
- Check if it applied normalization
- Try reconverting with different settings

### Option 4: Use Pre-computed Features
Load `aud_ave.npy` directly (like optimized Go does):
- Guaranteed to match
- Faster (no encoding needed)
- But only works for one audio

## Recommendation

**Try Option 1 first** (scale by 200) to see if that fixes the lip sync.

If that works, then investigate why the Core ML audio encoder produces different scale.

## Verification Needed

Check `convertmodeltocoreml/convert_pytorch_to_coreml.py`:
- Did it load the audio encoder weights correctly?
- Are the weights from `audio_visual_encoder.pth`?
- Was there any normalization applied?

---

**The Swift generator IS working - it's just getting wrong audio features from the Core ML audio encoder!**

