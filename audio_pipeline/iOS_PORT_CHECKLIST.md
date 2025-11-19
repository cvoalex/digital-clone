# iOS Porting Checklist

Use this checklist to track your iOS/CoreML implementation progress.

## Prerequisites

- [x] Reference dataset generated (`test_data/reference_output/`)
- [ ] Xcode project set up
- [ ] CoreML framework integrated
- [ ] Accelerate framework available

## Phase 1: AudioEncoder Model Conversion

### Convert to CoreML

```python
# conversion_script.py
import torch
import coremltools as ct
from audio_pipeline.audio_encoder import AudioEncoder

# Load model
model = AudioEncoder()
checkpoint = torch.load('model/checkpoints/audio_visual_encoder.pth')
# Handle checkpoint format
state_dict = {}
for k, v in checkpoint.items():
    if not k.startswith('audio_encoder.'):
        state_dict[f'audio_encoder.{k}'] = v
    else:
        state_dict[k] = v
model.load_state_dict(state_dict)
model.eval()

# Trace model
example_input = torch.randn(1, 1, 80, 16)
traced_model = torch.jit.trace(model, example_input)

# Convert to CoreML
mlmodel = ct.convert(
    traced_model,
    inputs=[ct.TensorType(
        name="mel_input",
        shape=(1, 1, 80, 16),
        dtype=np.float32
    )],
    outputs=[ct.TensorType(name="features")],
    minimum_deployment_target=ct.target.iOS15
)

# Save
mlmodel.save("AudioEncoder.mlmodel")
print("✓ Model converted to CoreML")
```

**Checklist:**
- [ ] Create conversion script
- [ ] Run conversion successfully
- [ ] `AudioEncoder.mlmodel` file created
- [ ] Model added to Xcode project
- [ ] Test model loads in iOS app

## Phase 2: Mel Spectrogram Processing

### Implement in Swift

```swift
import Accelerate

class MelSpectrogramProcessor {
    // Parameters (match Python exactly)
    let sampleRate: Int = 16000
    let nFFT: Int = 800
    let hopLength: Int = 200
    let winLength: Int = 800
    let nMels: Int = 80
    let fmin: Float = 55.0
    let fmax: Float = 7600.0
    let preemphasisCoef: Float = 0.97
    
    // Mel filter bank (build once)
    private var melBasis: [[Float]]?
    
    init() {
        self.melBasis = buildMelBasis()
    }
    
    func process(audio: [Float]) -> [[Float]] {
        // 1. Pre-emphasis
        let preemphasized = applyPreemphasis(audio)
        
        // 2. STFT
        let stft = computeSTFT(preemphasized)
        
        // 3. Magnitude
        let magnitude = computeMagnitude(stft)
        
        // 4. Mel filterbank
        let melSpec = applyMelFilterbank(magnitude)
        
        // 5. Amplitude to dB
        let melDB = amplitudeToDb(melSpec)
        
        // 6. Normalize to [-4, 4]
        let normalized = normalize(melDB)
        
        return normalized
    }
    
    private func applyPreemphasis(_ audio: [Float]) -> [Float] {
        var output = [Float](repeating: 0, count: audio.count)
        output[0] = audio[0]
        for i in 1..<audio.count {
            output[i] = audio[i] - preemphasisCoef * audio[i-1]
        }
        return output
    }
    
    private func computeSTFT(_ audio: [Float]) -> [DSPSplitComplex] {
        // Use vDSP for FFT
        // Implementation details...
        fatalError("Implement STFT using vDSP")
    }
    
    private func buildMelBasis() -> [[Float]] {
        // Build mel filterbank matrix
        // Implementation details...
        fatalError("Implement mel filterbank")
    }
    
    // ... other methods
}
```

**Checklist:**
- [ ] Create `MelSpectrogramProcessor.swift`
- [ ] Implement pre-emphasis filter
- [ ] Implement STFT using vDSP
- [ ] Build mel filterbank
- [ ] Implement mel conversion
- [ ] Implement amplitude to dB conversion
- [ ] Implement normalization

### Test Mel Processing

```swift
func testMelProcessing() {
    // Load reference audio
    let audioURL = Bundle.main.url(forResource: "reference_audio", withExtension: "wav")!
    let audioData = loadWAV(url: audioURL)
    
    // Process
    let processor = MelSpectrogramProcessor()
    let melSpec = processor.process(audio: audioData)
    
    // Load reference
    let referenceMel = loadNumpyArray("mel_spectrogram.npy")
    
    // Compare
    let maxDiff = computeMaxDifference(melSpec, referenceMel)
    
    assert(maxDiff < 1e-3, "Mel spectrogram validation failed: \(maxDiff)")
    print("✓ Mel spectrogram validation passed")
}
```

**Checklist:**
- [ ] Add reference files to Xcode project
- [ ] Implement NumPy loader for validation
- [ ] Test passes with max diff < 1e-3
- [ ] Visualize mel spectrogram (optional)

## Phase 3: Audio Pipeline Integration

### Implement Complete Pipeline

```swift
class AudioPipeline {
    private let melProcessor: MelSpectrogramProcessor
    private let audioEncoder: AudioEncoderModel
    private let mode: Mode = .ave
    private let fps: Int = 25
    
    enum Mode {
        case ave, hubert, wenet
    }
    
    init() {
        melProcessor = MelSpectrogramProcessor()
        
        // Load CoreML model
        let config = MLModelConfiguration()
        audioEncoder = try! AudioEncoderModel(configuration: config)
    }
    
    func processAudio(_ audio: [Float]) -> ProcessedAudio {
        // 1. Convert to mel spectrogram
        let melSpec = melProcessor.process(audio: audio)
        
        // 2. Extract mel windows
        let melWindows = extractMelWindows(melSpec)
        
        // 3. Process through AudioEncoder
        var audioFeatures: [[Float]] = []
        for window in melWindows {
            let features = inferAudioEncoder(window)
            audioFeatures.append(features)
        }
        
        // 4. Add temporal padding
        let paddedFeatures = addTemporalPadding(audioFeatures)
        
        return ProcessedAudio(features: paddedFeatures)
    }
    
    func getFeaturesForFrame(_ allFeatures: [[Float]], frameIndex: Int) -> MLMultiArray {
        // Extract ±8 frame context
        let contextWindow = extractContextWindow(allFeatures, at: frameIndex)
        
        // Reshape for model
        let reshaped = reshape(contextWindow, mode: mode)
        
        return reshaped
    }
    
    private func inferAudioEncoder(_ melWindow: [[Float]]) -> [Float] {
        // Convert to MLMultiArray
        let input = convertToMLMultiArray(melWindow, shape: [1, 1, 80, 16])
        
        // Run inference
        let prediction = try! audioEncoder.prediction(mel_input: input)
        
        // Extract features
        return extractFeatures(from: prediction.features)
    }
    
    private func extractMelWindows(_ melSpec: [[Float]]) -> [[[Float]]] {
        // Extract 16-frame windows for each video frame
        // Implementation...
        fatalError("Implement window extraction")
    }
    
    private func reshape(_ features: [[Float]], mode: Mode) -> MLMultiArray {
        // Reshape based on mode
        switch mode {
        case .ave:
            return reshapeToAVE(features) // (32, 16, 16)
        case .hubert:
            return reshapeToHubert(features) // (32, 32, 32)
        case .wenet:
            return reshapeToWeNet(features) // (256, 16, 32)
        }
    }
}
```

**Checklist:**
- [ ] Create `AudioPipeline.swift`
- [ ] Implement mel window extraction
- [ ] Integrate CoreML model
- [ ] Implement temporal padding
- [ ] Implement context window extraction
- [ ] Implement reshape methods
- [ ] Add proper error handling

## Phase 4: Validation

### Generate iOS Outputs

```swift
func generateValidationOutputs() {
    let pipeline = AudioPipeline()
    
    // Load reference audio
    let audioURL = Bundle.main.url(forResource: "reference_audio", withExtension: "wav")!
    let audioData = loadWAV(url: audioURL)
    
    // Process
    let processed = pipeline.processAudio(audioData)
    
    // Save outputs for validation
    let outputDir = getDocumentsDirectory().appendingPathComponent("ios_outputs")
    
    // Save mel spectrogram
    let melSpec = pipeline.melProcessor.process(audio: audioData)
    saveAsNumpy(melSpec, to: outputDir.appendingPathComponent("mel_spectrogram.npy"))
    
    // Save audio features
    saveAsNumpy(processed.features, to: outputDir.appendingPathComponent("audio_features_padded.npy"))
    
    // Save per-frame features
    let framesDir = outputDir.appendingPathComponent("frames")
    for (i, _) in processed.features.enumerated() {
        let frameFeatures = pipeline.getFeaturesForFrame(processed.features, frameIndex: i)
        saveAsNumpy(frameFeatures, to: framesDir.appendingPathComponent("frame_\(String(format: "%05d", i))_reshaped.npy"))
    }
    
    print("✓ Validation outputs saved to: \(outputDir.path)")
}
```

### Run Python Validation

```bash
# On your Mac, run:
python3 audio_pipeline/validate_ios_port.py \
    audio_pipeline/test_data/reference_output \
    <path_to_ios_outputs>
```

**Checklist:**
- [ ] Implement NumPy export in Swift
- [ ] Generate iOS outputs
- [ ] Copy outputs to Mac
- [ ] Run validation script
- [ ] All validations pass (max diff < 1e-3)

## Phase 5: Performance Optimization

**Checklist:**
- [ ] Profile mel processing performance
- [ ] Profile CoreML inference performance
- [ ] Optimize critical paths
- [ ] Test on actual device (not simulator)
- [ ] Measure memory usage
- [ ] Test with various audio lengths

## Phase 6: Integration with SyncTalk_2D

**Checklist:**
- [ ] Integrate AudioPipeline into main app
- [ ] Connect to video frame generation
- [ ] Test end-to-end pipeline
- [ ] Verify lip sync quality
- [ ] Test with various audio inputs

## Validation Criteria

### Mel Spectrogram
- [ ] Shape matches: (80, n_frames)
- [ ] Value range: [-4, 4]
- [ ] Max difference < 1e-3

### Audio Features
- [ ] Shape matches: (n_frames, 512)
- [ ] Max difference < 1e-3

### Frame Features
- [ ] Shape matches: (32, 16, 16) for AVE
- [ ] Max difference < 1e-3

## Resources

### Reference Files
- `audio_pipeline/test_data/reference_output/` - All reference outputs
- `audio_pipeline/TESTING_REPORT.md` - Detailed test results
- `audio_pipeline/README.md` - Full documentation
- `audio_pipeline/QUICK_START.md` - Quick start guide

### Python Tools
- `audio_pipeline/validate_ios_port.py` - Validation script
- `audio_pipeline/tests/test_pipeline.py` - Reference implementation

### Key Parameters
```
Sample Rate: 16000 Hz
n_FFT: 800
Hop Length: 200
Mel Bands: 80
Frequency Range: 55-7600 Hz
Pre-emphasis: 0.97
Normalization Range: [-4, 4]
```

## Troubleshooting

### Mel Spectrogram Differences > 1e-3
- Check FFT implementation (window function, padding)
- Verify mel filterbank construction
- Check normalization formula
- Verify pre-emphasis coefficient

### Audio Encoder Differences > 1e-3
- Re-export CoreML model
- Check input tensor format (NCHW vs NHWC)
- Verify batch normalization parameters
- Check for numerical precision issues

### Performance Issues
- Use vDSP for all DSP operations
- Batch mel window processing
- Pre-compute mel filterbank
- Use Metal for GPU acceleration

## Success Criteria

✅ All validation checks pass (max diff < 1e-3)
✅ Real-time performance on target device
✅ Memory usage within acceptable limits
✅ End-to-end integration successful
✅ Lip sync quality matches Python version

---

**Status**: Ready to start iOS implementation
**Reference Dataset**: Available in `audio_pipeline/test_data/reference_output/`
**Validation Tool**: `validate_ios_port.py`

