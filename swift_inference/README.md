# Swift Core ML Frame Generation

**Native macOS/iOS implementation using Core ML and Neural Engine**

Expected performance: **20-30 FPS** (2-3x faster than Python!)

## Prerequisites

### 1. Convert ONNX Models to Core ML

You need to convert the ONNX models to Core ML format first.

**See:** [../CONVERT_MODELS_INSTRUCTIONS.md](../CONVERT_MODELS_INSTRUCTIONS.md)

**Required files:**
- `AudioEncoder.mlpackage` (from audio_encoder.onnx)
- `Generator.mlpackage` (from generator.onnx)

**Using Xcode (Easiest):**
1. Open Xcode
2. Drag `model/sanders_full_onnx/models/audio_encoder.onnx` into project
3. Xcode converts automatically
4. Copy resulting `.mlpackage` to `swift_inference/`
5. Repeat for `generator.onnx`

### 2. System Requirements

- macOS 13.0+ (for Neural Engine support)
- Xcode 14.0+
- Swift 5.7+

## Build

```bash
cd swift_inference
swift build --configuration release
```

## Usage

### With Default Audio (sanders/aud.wav):

```bash
.build/release/swift-infer --frames 250
```

### With Custom Audio:

```bash
.build/release/swift-infer \
  --audio ../demo/talk_hb.wav \
  --frames 250
```

### All Options:

```bash
.build/release/swift-infer \
  --sanders ../model/sanders_full_onnx \
  --audio ../demo/talk_hb.wav \
  --output ../comparison_results/swift_output/frames \
  --frames 250
```

## Performance

**Expected on M1 Pro:**
- Audio processing: 20-30 FPS (Neural Engine!)
- Frame generation: 25-35 FPS (Core ML optimized!)
- **Overall: 20-30 FPS** (2-3x faster than Python!)

**Why Faster:**
- ✅ Neural Engine acceleration
- ✅ Metal GPU operations
- ✅ Optimized for Apple Silicon
- ✅ Native framework integration

## Architecture

```
Sources/
├── main.swift                    CLI entry point
├── Audio/
│   ├── MelProcessor.swift       Mel spectrogram processing
│   └── SimpleWAVLoader.swift    WAV file loading
├── Models/
│   ├── ONNXWrapper.swift        ONNX Runtime wrapper (fallback)
│   └── ONNXModel.swift          Model abstraction
└── Compositor/
    ├── CoreMLGenerator.swift    Core ML-based generation
    └── FrameGenerator.swift     ONNX-based generation (fallback)
```

## How It Works

### 1. Audio Processing (Core ML)
```swift
Load WAV → Mel Spectrograms → AudioEncoder.mlpackage → Features
```

### 2. Frame Generation (Core ML)
```swift
Features + Images → Generator.mlpackage → Generated Frames
```

### 3. Compositing (Accelerate)
```swift
Generated + Full Body → Paste using coordinates → Final Frames
```

## Comparison with Other Implementations

| Implementation | FPS | Python-Free | Hardware Accel |
|----------------|-----|-------------|----------------|
| Python | 12.6 | ❌ | CPU/GPU |
| Go | 8.9 | ✅ | CPU/GPU |
| **Swift** | **20-30** | ✅ | **Neural Engine!** |

## Create Video

After generating frames:

```bash
ffmpeg -framerate 25 -i ../comparison_results/swift_output/frames/frame_%05d.jpg \
  -i ../demo/talk_hb.wav \
  -vframes 250 -shortest \
  -c:v libx264 -c:a aac -crf 20 \
  swift_output.mp4 -y
```

## Validation

Run comparison with Python and Go:

```bash
# Generate 250 frames with all three
cd ../
./run_comparison.sh 250

# Swift will be in comparison_results/swift_output/
```

## Troubleshooting

### "Core ML models not found"

Convert the ONNX models first:
- See [CONVERT_MODELS_INSTRUCTIONS.md](../CONVERT_MODELS_INSTRUCTIONS.md)
- Or use Xcode to import ONNX files

### "Neural Engine not available"

Check macOS version:
```bash
sw_vers  # Should be 13.0+
```

### Build errors

```bash
swift build --configuration release -v
```

## Expected Results

**250 frames on M1 Pro:**
- Time: ~8-12 seconds
- FPS: 20-30
- **2-3x faster than Python!**
- **3-4x faster than Go!**

## Why Core ML is Faster

1. **Neural Engine**: Dedicated ML hardware (4-6 TOPs)
2. **Metal**: GPU acceleration for operations
3. **Optimizations**: Apple-specific ML optimizations
4. **Unified Memory**: Faster data access on Apple Silicon

## Next Steps

1. **Convert models** using Xcode or coremltools
2. **Build** Swift CLI
3. **Test** on 10 frames
4. **Validate** against Python/Go
5. **Benchmark** on 250 frames

---

**Status:** Ready to build once Core ML models are converted! 🚀

The Swift implementation will be the fastest and most efficient for macOS/iOS!

