# Swift/macOS Implementation Plan

## Current Status

We have working implementations:
- ✅ **Python**: 12.6 FPS (fastest, but needs Python)
- ✅ **Go**: 8.9 FPS (Python-free, standalone binary)
- ⏭️ **Swift**: To be implemented (Native macOS/iOS)

## Swift/macOS Advantages

### Performance Potential

**With ONNX Runtime:**
- Expected: ~8-10 FPS (similar to Go)
- Uses same ONNX Runtime C library
- Standard image processing

**With Core ML + Neural Engine:**
- Expected: **20-30 FPS** (2-3x faster than Python!)
- Hardware acceleration on Apple Silicon
- Optimized for M1/M2/M3 chips
- Can use GPU + Neural Engine simultaneously

### Native Integration

✅ **Accelerate framework** - SIMD vectorized ops (faster than Go)  
✅ **Metal Performance Shaders** - GPU acceleration  
✅ **Core Image** - Hardware-accelerated image processing  
✅ **Neural Engine** - Dedicated ML hardware (Core ML)  

## Implementation Approaches

### Approach 1: ONNX Runtime (Like Go)

**Pros:**
- ✅ Reuse existing ONNX models
- ✅ No conversion needed
- ✅ Cross-platform consistency

**Cons:**
- ⚠️ Similar speed to Go (~8-10 FPS)
- ⚠️ Doesn't leverage Neural Engine
- ⚠️ C interop complexity

**Estimated performance:** 8-10 FPS

### Approach 2: Core ML (Native Apple)

**Pros:**
- ✅ **2-3x faster** (Neural Engine!)
- ✅ Native Swift integration
- ✅ Better battery life on iOS
- ✅ Optimized for Apple Silicon

**Cons:**
- ⚠️ Requires model conversion (ONNX → Core ML)
- ⚠️ Apple-only (not cross-platform)
- ⚠️ Conversion can be tricky

**Estimated performance:** 20-30 FPS

### Approach 3: Hybrid

**Pros:**
- ✅ Best of both worlds
- ✅ Fallback to ONNX if Core ML fails

**Implementation:**
```swift
if let coreMLModel = try? loadCoreML() {
    // Use Core ML (20-30 FPS)
    return coreMLModel.predict(...)
} else {
    // Fallback to ONNX Runtime (8-10 FPS)
    return onnxModel.predict(...)
}
```

## Recommended: Core ML First

For macOS/iOS, Core ML is the best choice because:

1. **Performance**: 20-30 FPS (2-3x faster!)
2. **Native**: No C bridging complexity
3. **Battery**: More efficient on mobile
4. **Apple Silicon**: Leverages Neural Engine

## Core ML Conversion

### Option A: Use onnx-coreml (Simpler)

```bash
pip install onnx-coreml
python3 -c "
from onnx_coreml import convert

# Convert audio encoder
model_audio = convert(model='model/sanders_full_onnx/models/audio_encoder.onnx')
model_audio.save('swift_inference/AudioEncoder.mlpackage')

# Convert generator
model_gen = convert(model='model/sanders_full_onnx/models/generator.onnx')
model_gen.save('swift_inference/Generator.mlpackage')
"
```

### Option B: Use coremltools (More control)

```python
import coremltools as ct

# For audio encoder
model = ct.convert(
    'audio_encoder.onnx',
    source='onnx',
    minimum_deployment_target=ct.target.macOS13,
    compute_units=ct.ComputeUnit.ALL  # Use CPU + GPU + Neural Engine
)
model.save('AudioEncoder.mlpackage')
```

### Option C: Use Xcode (GUI)

1. Open Xcode
2. Create new Core ML model
3. Import ONNX file
4. Xcode converts automatically

## Swift Implementation

Once we have Core ML models:

```swift
import CoreML
import Accelerate

class FrameGenerator {
    let audioEncoder: AudioEncoder
    let generator: Generator
    
    init() throws {
        // Load Core ML models (compiled .mlmodelc)
        self.audioEncoder = try AudioEncoder(configuration: MLModelConfiguration())
        self.generator = try Generator(configuration: MLModelConfiguration())
    }
    
    func processAudio(_ wavPath: String) -> [MLMultiArray] {
        // Load WAV, process mel spectrograms
        // Run through audioEncoder
        // Returns audio features
    }
    
    func generateFrame(
        roi: CGImage,
        masked: CGImage,
        audioFeatures: MLMultiArray
    ) -> CGImage {
        // Prepare 6-channel input
        // Run through generator (Core ML!)
        // Returns generated frame
    }
}
```

## Expected Performance

### On M1 Pro MacBook:

**Core ML:**
- Audio encoding: ~20s for 1117 frames
- Frame generation: ~1-2s per frame  
- **Total: ~25-30 FPS** 🚀

**Breakdown:**
- Audio: 1117 frames × 0.018s = 20s
- Frames: 250 frames × 0.033s = 8s
- **Total for 250 frames: ~28s** (but 250 frames, not full audio)

### On iPhone 14 Pro:

**Core ML + Neural Engine:**
- Even faster! Neural Engine is optimized for ML
- Expected: **15-25 FPS**

## Why Core ML is Faster

1. **Neural Engine**: Dedicated ML hardware
2. **Metal**: GPU acceleration for operations
3. **Optimizations**: Apple-specific ML optimizations
4. **Accelerate**: Vectorized image processing

## Next Steps

### Step 1: Convert Models (Python - one time only!)

```bash
pip install onnx-coreml
python3 convert_to_coreml.py
```

### Step 2: Create Swift CLI

```bash
cd swift_inference
swift build --configuration release
```

### Step 3: Test Performance

```bash
time swift run swift-infer --frames 250
```

### Step 4: Compare

Expected results:
- Python: 12.6 FPS
- Go: 8.9 FPS
- **Swift + Core ML: 20-30 FPS** 🚀

## Recommendation

**For macOS/iOS production:**
- Use Core ML (not ONNX Runtime)
- Leverage Neural Engine
- Get 2-3x better performance
- Better battery life on mobile

**The conversion is one-time Python usage, then pure Swift!**

---

Want me to implement the Core ML version? It should be significantly faster than both Python and Go!

