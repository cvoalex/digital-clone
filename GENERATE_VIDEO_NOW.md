# Generate Video - Quick Guide

## The Easiest Way: Use Python

The Python implementation is the most complete and easiest to use for generating videos.

### Prerequisites

You need:
1. ✅ Trained U-Net checkpoint (e.g., `checkpoint/May/5.pth`)
2. ✅ Audio features from audio pipeline (`.npy` file)
3. ✅ Template images and landmarks (in `dataset/May/`)
4. ✅ Original audio file (`.wav`)

### Step 1: Install Dependencies

```bash
cd frame_generation_pipeline
pip install -r requirements.txt
```

### Step 2: Generate Your Video

```bash
python generate_video.py \
  --checkpoint ../checkpoint/May/5.pth \
  --audio-features ../audio_pipeline/my_audio_output/audio_features_padded.npy \
  --template ../dataset/May \
  --audio ../demo/talk_hb.wav \
  --output ../result/my_video.mp4
```

**That's it!** Your video will be at `result/my_video.mp4`

### Example Output

```
============================================================
Frame Generation Pipeline
============================================================
Checkpoint: ../checkpoint/May/5.pth
Audio features: ../audio_pipeline/my_audio_output/audio_features_padded.npy
Template: ../dataset/May
Audio: ../demo/talk_hb.wav
Output: ../result/my_video.mp4
Mode: ave
FPS: 25
CRF: 20
============================================================

[1/3] Initializing pipeline...
INFO - Initializing U-Net model on cuda with mode=ave
INFO - Loading checkpoint from ../checkpoint/May/5.pth
INFO - U-Net model loaded successfully
INFO - Pipeline initialized successfully

[2/3] Generating video frames...
INFO - Generating video: ../result/my_video.mp4
INFO - Loading audio features from ../audio_pipeline/my_audio_output/audio_features_padded.npy
INFO - Audio features shape: (1497, 512)
INFO - Generating frames...
Generating frames: 100%|███████████████| 1497/1497 [02:30<00:00, 9.95it/s]
INFO - Generated 1497 frames
INFO - Merging video with audio...
INFO - Video saved to ../result/my_video.mp4

[3/3] Complete!
============================================================
✓ Video saved to: ../result/my_video.mp4
✓ File size: 45.32 MB
============================================================

You can now:
  • Open video: open ../result/my_video.mp4
  • Play video: ffplay ../result/my_video.mp4
  • Check video info: ffprobe ../result/my_video.mp4
```

## Using Different Checkpoints/Datasets

### For Different Character

```bash
python generate_video.py \
  --checkpoint ../checkpoint/Alice/10.pth \
  --audio-features ../my_audio_features.npy \
  --template ../dataset/Alice \
  --audio ../my_audio.wav \
  --output ../result/alice_video.mp4
```

### High Quality Video

```bash
python generate_video.py \
  --checkpoint ../checkpoint/May/5.pth \
  --audio-features ../audio_features.npy \
  --template ../dataset/May \
  --audio ../audio.wav \
  --output ../result/hq_video.mp4 \
  --crf 18  # Lower CRF = higher quality (but larger file)
```

### Save Individual Frames

```bash
python generate_video.py \
  --checkpoint ../checkpoint/May/5.pth \
  --audio-features ../audio_features.npy \
  --template ../dataset/May \
  --audio ../audio.wav \
  --output ../result/video.mp4 \
  --save-frames \
  --frames-dir ../result/frames
```

## Options

```
Required:
  --checkpoint          Path to U-Net checkpoint (.pth)
  --audio-features      Path to audio features (.npy)
  --template           Path to template directory
  --audio              Path to audio file (.wav)
  --output             Path to output video (.mp4)

Optional:
  --mode               ave, hubert, or wenet (default: ave)
  --fps                Frames per second (default: 25)
  --crf                Video quality 18-28, lower=better (default: 20)
  --start-frame        Starting frame in template (default: 0)
  --use-parsing        Use parsing masks if available
  --save-frames        Save individual frames
  --frames-dir         Where to save frames
  --device             cpu or cuda (auto-detect by default)
```

## Troubleshooting

### "Checkpoint not found"

Make sure your checkpoint path is correct:
```bash
ls -la checkpoint/May/5.pth
```

### "Audio features not found"

Run the audio pipeline first:
```bash
cd ../audio_pipeline
python pipeline.py --audio ../demo/talk_hb.wav
```

This will generate `audio_features_padded.npy` in the output directory.

### "Template directory not found"

Your template directory should have this structure:
```
dataset/May/
├── full_body_img/
│   ├── 0.jpg
│   ├── 1.jpg
│   └── ...
└── landmarks/
    ├── 0.lms
    ├── 1.lms
    └── ...
```

### "CUDA out of memory"

Use CPU instead:
```bash
python generate_video.py \
  --checkpoint ../checkpoint/May/5.pth \
  --audio-features ../audio_features.npy \
  --template ../dataset/May \
  --audio ../audio.wav \
  --output ../result/video.mp4 \
  --device cpu
```

### "ffmpeg not found"

Install ffmpeg:
```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt-get install ffmpeg

# Windows
# Download from https://ffmpeg.org/
```

## What About Go and Swift?

### Go Implementation

The Go version generates frames but you need to:
1. Export the model to ONNX first
2. Convert audio features to binary format
3. Install GoCV and ONNX Runtime

It's faster but more complex to set up. See `frame_generation_go/README.md`

### Swift/iOS Implementation

The Swift version is for iOS/macOS apps. It requires:
1. Xcode setup
2. ONNX Runtime framework
3. App development workflow

See `frame_generation_swift/README.md`

## For Most Users: Use Python

**The Python implementation is:**
- ✅ Easiest to set up
- ✅ Most complete
- ✅ Best documented
- ✅ Generates videos directly
- ✅ Works out of the box

**Use Go/Swift only if you need:**
- Zero Python dependency (Go)
- iOS/macOS native app (Swift)
- Deployment in Go/Swift environments

## Quick Test

To verify everything works, run with a short audio clip:

```bash
# Use just first 5 seconds
python -c "
import numpy as np
features = np.load('../audio_pipeline/my_audio_output/audio_features_padded.npy')
np.save('../test_features.npy', features[:125])  # 5 seconds at 25fps
"

# Generate short video
python generate_video.py \
  --checkpoint ../checkpoint/May/5.pth \
  --audio-features ../test_features.npy \
  --template ../dataset/May \
  --audio ../demo/talk_hb.wav \
  --output ../result/test_video.mp4
```

This should complete in ~10 seconds.

---

**Need Help?** 

Check the detailed documentation:
- [FRAME_GENERATION_QUICKSTART.md](FRAME_GENERATION_QUICKSTART.md)
- [frame_generation_pipeline/README.md](frame_generation_pipeline/README.md)
- [FRAME_GENERATION_GUIDE.md](FRAME_GENERATION_GUIDE.md)

