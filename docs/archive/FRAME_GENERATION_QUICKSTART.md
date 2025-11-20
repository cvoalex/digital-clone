# Frame Generation Pipeline - Quick Start

Get up and running with the frame generation pipeline in 5 minutes.

## Prerequisites

You should have already completed the audio processing pipeline. You need:
- ✅ Audio features (`.npy` file)
- ✅ Template images and landmarks
- ✅ Trained U-Net checkpoint

## Choose Your Platform

### 🐍 Python (Recommended for First Time)

**Best for:** Research, prototyping, generating reference outputs

```bash
# 1. Navigate to directory
cd frame_generation_pipeline

# 2. Install dependencies (takes 2-3 min)
pip install -r requirements.txt

# 3. Generate a video (takes 5-10 min for 1000 frames)
python -c "
from pipeline import FrameGenerationPipeline

pipeline = FrameGenerationPipeline(
    checkpoint_path='../checkpoint/May/5.pth',
    mode='ave'
)

video = pipeline.generate_video(
    audio_features_path='../audio_pipeline/my_audio_output/audio_features_padded.npy',
    template_dir='../dataset/May',
    audio_path='../demo/talk_hb.wav',
    output_path='../result/my_result.mp4',
    fps=25,
    crf=20
)

print(f'Video saved to: {video}')
"
```

**Done!** Your video is at `result/my_result.mp4`

---

### 🔷 Go (Best Performance)

**Best for:** Production servers, batch processing, no Python dependency

```bash
# 1. Navigate to directory
cd frame_generation_go

# 2. Export model to ONNX (one time)
cd ../frame_generation_pipeline
python export_model.py \
  --checkpoint ../checkpoint/May/5.pth \
  --output ./models/unet_328.onnx \
  --mode ave

# 3. Convert audio features to binary format (one time)
python -c "
import numpy as np
import json

features = np.load('../audio_pipeline/my_audio_output/audio_features_padded.npy')
features.astype(np.float32).tofile('../frame_generation_go/test_data/audio_features.bin')

metadata = {
    'num_frames': features.shape[0],
    'feature_size': int(np.prod(features.shape[1:])),
    'shape': list(features.shape)
}
with open('../frame_generation_go/test_data/audio_features.bin.json', 'w') as f:
    json.dump(metadata, f, indent=2)
"

# 4. Build CLI tool (takes 1-2 min)
cd ../frame_generation_go
go build -o bin/generate ./cmd/generate

# 5. Generate frames (takes 3-7 min for 1000 frames, 2-3x faster than Python)
./bin/generate \
  --model ../frame_generation_pipeline/models/unet_328.onnx \
  --audio ./test_data/audio_features.bin \
  --template ../dataset/May \
  --output ./output/frames \
  --mode ave

# 6. Create video with ffmpeg
ffmpeg -framerate 25 -i ./output/frames/frame_%05d.jpg \
  -i ../demo/talk_hb.wav \
  -c:v libx264 -c:a aac -crf 20 \
  ../result/my_result_go.mp4 -y
```

**Done!** Your video is at `result/my_result_go.mp4`

---

### 🍎 Swift/iOS (Native Apple)

**Best for:** iOS apps, macOS apps, Apple Silicon optimization

```bash
# 1. Navigate to directory
cd frame_generation_swift

# 2. Copy ONNX model (one time)
cp ../frame_generation_pipeline/models/unet_328.onnx ./Models/

# 3. Convert audio features (one time)
python -c "
import numpy as np
import json

features = np.load('../audio_pipeline/my_audio_output/audio_features_padded.npy')
features.astype(np.float32).tofile('./TestData/audio_features.bin')

metadata = {
    'num_frames': features.shape[0],
    'feature_size': int(np.prod(features.shape[1:])),
    'shape': list(features.shape)
}
with open('./TestData/audio_features.bin.json', 'w') as f:
    json.dump(metadata, f, indent=2)
"

# 4. Build and run (if using SPM)
swift build
swift run

# Or open in Xcode
open FrameGenerator.xcodeproj
# Then: Cmd+R to build and run
```

**Done!** Frames generated in the app's output directory.

---

## Troubleshooting

### "Checkpoint not found"

Make sure you have the trained model:
```bash
ls -la checkpoint/May/5.pth
```

If missing, you need to train the model first or get a pre-trained checkpoint.

### "Template directory not found"

Ensure your dataset is set up:
```bash
ls -la dataset/May/full_body_img/
ls -la dataset/May/landmarks/
```

### "ONNX Runtime not found" (Go/Swift)

**macOS:**
```bash
brew install onnxruntime
```

**Linux:**
```bash
# Download from https://github.com/microsoft/onnxruntime/releases
# Extract and set LD_LIBRARY_PATH
```

### "Out of memory"

Reduce batch size or process in chunks:
```python
# Python
for i in range(0, total_frames, 100):
    frames = generate_frames(start=i, end=min(i+100, total_frames))
    save_frames(frames)
```

### "Wrong colors / artifacts"

Check that you're using the correct mode:
```python
mode='ave'  # for audio features from our pipeline
# not 'hubert' or 'wenet'
```

---

## Verifying Output

### Visual Check
```bash
# Open the video
open result/my_result.mp4
```

Look for:
- ✅ Smooth lip movements
- ✅ Synchronized with audio
- ✅ Natural face appearance
- ✅ No artifacts or glitches

### Numerical Check (Compare implementations)
```bash
# Generate with Python
cd frame_generation_pipeline
python generate.py --output ../test_output_python

# Generate with Go
cd ../frame_generation_go
./bin/generate --output ../test_output_go

# Compare
python compare_outputs.py \
  --ref ../test_output_python \
  --test ../test_output_go
```

Should see:
- MSE < 0.01
- PSNR > 40 dB
- SSIM > 0.95

---

## Common Workflows

### 1. Quick Test (1 video, Python)
```bash
cd frame_generation_pipeline
python -m pipeline \
  --checkpoint ../checkpoint/May/5.pth \
  --audio-features ../audio_features.npy \
  --template ../dataset/May \
  --audio ../demo/audio.wav \
  --output ../result/test.mp4
```
**Time:** 5-10 minutes

### 2. Batch Processing (Multiple videos, Go)
```bash
cd frame_generation_go
for audio in ../audio_features/*.bin; do
  name=$(basename $audio .bin)
  ./bin/generate \
    --audio $audio \
    --template ../dataset/May \
    --output ../results/$name
done
```
**Time:** 3-7 minutes per video

### 3. iOS App Development (Swift)
```bash
cd frame_generation_swift
open FrameGenerator.xcodeproj

# In Xcode:
# 1. Set up signing
# 2. Connect iPhone
# 3. Cmd+R to build and run
# 4. Test on device
```
**Time:** 15-20 minutes for first setup

---

## Performance Expectations

### Small video (100 frames, ~4 seconds)
- Python: 5 seconds
- Go: 2 seconds
- Swift: 2 seconds

### Medium video (500 frames, ~20 seconds)
- Python: 25 seconds
- Go: 10 seconds
- Swift: 10 seconds

### Large video (1500 frames, ~60 seconds)
- Python: 75 seconds
- Go: 30 seconds
- Swift: 30 seconds

*Based on M1 Pro / RTX 3090 hardware*

---

## Next Steps

Once you've generated your first video:

1. **Tune parameters:**
   - CRF (18-28) for video quality
   - FPS (20-30) for smoothness
   - Resolution (affect template size)

2. **Try different templates:**
   - Use your own face images
   - Different backgrounds
   - Various expressions

3. **Integrate into your app:**
   - Use the APIs directly
   - Build a web service (Go)
   - Create a mobile app (Swift)

4. **Optimize performance:**
   - GPU acceleration
   - Batch processing
   - Model quantization

---

## Getting Help

1. **Check the documentation:**
   - [Complete Guide](FRAME_GENERATION_GUIDE.md)
   - [Python README](frame_generation_pipeline/README.md)
   - [Go README](frame_generation_go/README.md)
   - [Swift README](frame_generation_swift/README.md)

2. **Run the tests:**
   ```bash
   # Python
   cd frame_generation_pipeline && pytest tests/
   
   # Go
   cd frame_generation_go && go test ./...
   
   # Swift
   cd frame_generation_swift && swift test
   ```

3. **Compare with reference:**
   - Generate reference outputs first (Python)
   - Compare other implementations against it
   - Check MSE/PSNR/SSIM metrics

---

## Summary

You've now:
- ✅ Generated your first video
- ✅ Learned the basic workflow
- ✅ Understood performance characteristics
- ✅ Know how to troubleshoot common issues

**Happy generating! 🎬**

