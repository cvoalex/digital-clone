# Testing Guide - macOS App

## Quick Start

### 1. Build and Run
```bash
# Open in Xcode
cd /Users/alexanderrusich/Projects/digital-clone/audio_pipeline_swift/mels
open mels.xcodeproj

# Then press Cmd+R in Xcode
```

### 2. Test Mel Processor Only (Fast)

1. **Keep toggle OFF** ("Use Full Pipeline" unchecked)
2. Click "Select Audio File"
3. Navigate to: `/Users/alexanderrusich/Projects/digital-clone/audio_pipeline/audio.wav`
4. Click "Process Audio"

**Expected Result:**
```
Status: ✅ Mel processing complete!

Mel Spectrogram Results:
Shape: (80, ~4797)
Range: [-4.000, ~2.024]
```

This should match the Go output!

### 3. Test Full Pipeline (Mel + ONNX)

1. **Turn toggle ON** ("Use Full Pipeline")
2. Click "Select Audio File" (select same audio.wav)
3. Click "Run Full Pipeline"
4. Wait ~45 seconds (processing 1,496 frames)

**Expected Result:**
```
Status: ✅ Full pipeline complete!

Mel Spectrogram Results:
Shape: (80, ~4797)
Range: [-4.000, ~2.024]

Full Pipeline Results:
Frames: ~1496
Audio Features: (1498, 512)
Features Range: [0.000, ~9.722]
```

## What Each Mode Does

### Mel Only (Fast - ~1 second)
- Loads audio
- Computes mel spectrogram using Accelerate
- Shows shape and range
- **Use this to verify mel processor is working**

### Full Pipeline (Slower - ~45 seconds)
- Everything from "Mel Only" PLUS:
- Extracts 16-frame windows
- Processes through ONNX AudioEncoder
- Generates 512-dim features per frame
- Adds temporal padding
- **Use this to get complete audio features for U-Net**

## Validation Against Python/Go

### Expected Outputs:

| Implementation | Mel Shape | Mel Range | Features Shape | Features Range |
|----------------|-----------|-----------|----------------|----------------|
| Python | (80, 4801) | [-4.0, 2.08] | (1499, 512) | [0, 10.17] |
| Go | (80, 4797) | [-4.0, 2.02] | (1498, 512) | [0, 9.72] |
| Swift | (80, ~4797) | [-4.0, ~2.02] | (1498, 512) | [0, ~9.72] |

**Differences < 1% are normal** due to different DSP implementations.

## Troubleshooting

### Error: "ONNX server failed to start"
**Solution**: Make sure `onnx_server.py` and `audio_encoder.onnx` are in the mels directory:
```bash
ls /Users/alexanderrusich/Projects/digital-clone/audio_pipeline_swift/mels/
# Should see: audio_encoder.onnx and onnx_server.py
```

### Error: "Audio file not found"
**Solution**: Use the full file picker, navigate to the audio.wav location manually.

### Full Pipeline is Slow
**Expected**: Processing 1,496 frames through ONNX takes ~45 seconds on CPU.
- This is normal for the Python bridge approach
- Native ONNX Runtime or Core ML would be faster
- For now, validates the pipeline works!

### Mel Processor Crashes
**Solution**: Make sure audio file is valid WAV format:
```bash
ffmpeg -i your_audio.mp3 -ar 16000 -ac 1 output.wav
```

## Console Output

Watch Xcode's console (Cmd+Shift+Y) to see detailed processing logs:
```
MelProcessor initialized
  Sample rate: 16000 Hz
  N-FFT: 800, Hop: 200
  Mel bands: 80
Loaded audio: 960000 samples, 60.0s
Processing audio: 960000 samples
Mel spectrogram shape: (80, 4797)
  Value range: [-4.0, 2.024]
```

## Save Results

To save the Swift output for comparison:

1. Run with full pipeline
2. Check console output for values
3. Compare with Python/Go results manually

Or I can add export functionality if needed!

## Next Steps

Once you verify it works:
1. ✅ Mel processor validated
2. ✅ Full pipeline working
3. 🔄 Optimize ONNX integration (native runtime)
4. 🔄 Port to iOS device
5. 🔄 Integrate with video generation

---

**Status**: Ready to test! Just compile and run in Xcode! 🎯

