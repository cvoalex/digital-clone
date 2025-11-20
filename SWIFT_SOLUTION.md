# Swift Audio Issue - Solution

## Root Cause

The Core ML AudioEncoder.mlpackage was converted with **random/incorrect weights**!

From `convertmodeltocoreml/convert_pytorch_to_coreml.py` line 42:
```python
print("🔄 Using randomly initialized model for AudioEncoder")
```

**The Core ML audio encoder has random weights, not trained weights!**

This is why:
- Output is 200x smaller
- Doesn't match Go/Python
- Causes closed mouth

## Why This Happened

The conversion script looked for `audio_visual_encoder.pth` in the `convertmodeltocoreml/` directory, but:
- It wasn't there, OR
- The weights didn't load correctly

So it converted a **randomly initialized model** to Core ML!

## Solutions

### Option 1: Use ONNX for Audio Encoder (Recommended)

Swift should use:
- **ONNX Runtime for audio encoder** (audio_encoder.onnx) ← Has correct weights
- **Core ML for generator** (Generator.mlpackage) ← Works fine

This matches what Go does and guarantees correctness.

### Option 2: Reconvert Audio Encoder Properly

1. Copy `model/checkpoints/audio_visual_encoder.pth` to `convertmodeltocoreml/`
2. Run conversion again
3. Verify it loads weights correctly

### Option 3: Scale Factor (Temporary)

Current quick fix: Multiply by 200
- Works but is a hack
- Doesn't address root cause
- May not be exactly right

## Recommended Implementation

**Use ONNX for audio, Core ML for generator:**

```swift
// Audio encoder: Use ONNX Runtime (same as Go)
let audioEncoder = ONNXRuntimeSession(modelPath: "audio_encoder.onnx")
let audioFeatures = audioEncoder.run(melWindow)  // Correct weights!

// Generator: Use Core ML (faster on Apple Silicon)
let generator = try MLModel(contentsOf: generatorURL)
let output = generator.prediction(...)  // Neural Engine!
```

This gives us:
- ✅ Correct audio features (ONNX has trained weights)
- ✅ Fast generation (Core ML + Neural Engine)
- ✅ Guaranteed to match Go/Python

## Why Not All Core ML?

The generator Core ML works fine (from Sanders conversion).
Only the audio encoder has wrong weights.

**So: Use ONNX audio encoder + Core ML generator = Best of both worlds!**

## Implementation

The Swift code already has ONNXWrapper from the audio pipeline.
Just need to:
1. Keep using ONNX for audio encoder
2. Keep using Core ML for generator
3. This matches the Go approach exactly

---

**Bottom line:** Core ML audio encoder has random weights. Use ONNX audio encoder instead (like Go does).

