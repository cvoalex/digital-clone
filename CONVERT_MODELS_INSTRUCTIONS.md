# Core ML Model Conversion - Quick Instructions

## Models to Convert

You need to convert these 2 ONNX models:

### 1. Audio Encoder
**Path:** `model/sanders_full_onnx/models/audio_encoder.onnx`

**Specs:**
- Input: `mel` - shape [1, 1, 80, 16] - Float32
- Output: `emb` - shape [1, 512] - Float32

**Save as:** `swift_inference/AudioEncoder.mlpackage`

### 2. U-Net Generator  
**Path:** `model/sanders_full_onnx/models/generator.onnx`

**Specs:**
- Input 1: `input` - shape [1, 6, 320, 320] - Float32
- Input 2: `audio` - shape [1, 32, 16, 16] - Float32
- Output: `output` - shape [1, 3, 320, 320] - Float32

**Save as:** `swift_inference/Generator.mlpackage`

## Using Xcode (Easiest)

1. Open Xcode
2. Create new project (or open existing)
3. Drag `audio_encoder.onnx` into project navigator
4. Xcode automatically converts → Right-click → "Show in Finder"
5. Copy the `.mlpackage` to `swift_inference/AudioEncoder.mlpackage`
6. Repeat for `generator.onnx` → `swift_inference/Generator.mlpackage`

## Using coremltools (If You Have Working Version)

```python
import coremltools as ct

# Audio Encoder
audio = ct.convert(
    'model/sanders_full_onnx/models/audio_encoder.onnx',
    minimum_deployment_target=ct.target.macOS13,
    compute_units=ct.ComputeUnit.ALL
)
audio.save('swift_inference/AudioEncoder.mlpackage')

# Generator
generator = ct.convert(
    'model/sanders_full_onnx/models/generator.onnx',
    minimum_deployment_target=ct.target.macOS13,
    compute_units=ct.ComputeUnit.ALL
)
generator.save('swift_inference/Generator.mlpackage')
```

## What to Configure

**Deployment Target:** macOS 13+ (for Neural Engine)

**Compute Units:** ALL (CPU + GPU + Neural Engine)

**Precision:** Float32 (default)

## Verification

After conversion, check:

```bash
ls -lh swift_inference/*.mlpackage
```

Should see:
- `AudioEncoder.mlpackage` (~11 MB)
- `Generator.mlpackage` (~46 MB)

## Once Converted

Let me know and I'll complete the Swift code that uses these Core ML models!

The Swift implementation will then:
- Load .mlpackage files (native Core ML)
- Run on Neural Engine (20-30 FPS expected!)
- Be 100% Python-free
- Leverage Apple Silicon

---

**Please convert the models, then I'll finish the Swift code!** 🚀

