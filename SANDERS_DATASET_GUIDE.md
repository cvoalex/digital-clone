# Sanders Dataset - Ready to Use!

## 🎉 You Have Everything!

The `model/sanders_full_onnx/` directory contains a **complete pre-packaged dataset** with everything needed for video generation!

## 📦 What's Included

```
model/sanders_full_onnx/
├── models/
│   ├── generator.onnx           ✅ U-Net model (46 MB) - READY!
│   └── audio_encoder.onnx       ✅ Audio encoder (11 MB) - READY!
├── aud_ave.npy                  ✅ Audio features (522 frames) - READY!
├── aud.wav                      ✅ Original audio - READY!
├── landmarks/                   ✅ 523 landmark files - READY!
│   ├── 0.lms
│   ├── 1.lms
│   └── ... (523 total)
├── full_body_video.mp4          ✅ Template video (2.9 MB)
├── crops_328_video.mp4          ✅ 328x328 crops
├── rois_320_video.mp4           ✅ 320x320 ROIs
└── model_inputs_video.mp4       ✅ Masked inputs
```

## ✅ You're Ready For:

### 1. Go Implementation (Python-free!)

The Go implementation can use this directly:

```bash
cd frame_generation_go

# 1. Copy the ONNX model
cp ../model/sanders_full_onnx/models/generator.onnx ./models/unet_328.onnx

# 2. Convert audio features to binary (one-time)
python -c "
import numpy as np
import json

features = np.load('../model/sanders_full_onnx/aud_ave.npy')
features.astype('float32').tofile('./test_data/sanders_audio.bin')

metadata = {
    'num_frames': features.shape[0],
    'feature_size': int(np.prod(features.shape[1:])),
    'shape': list(features.shape)
}
with open('./test_data/sanders_audio.bin.json', 'w') as f:
    json.dump(metadata, f, indent=2)
print('Audio features converted!')
"

# 3. Extract template frames from video
mkdir -p test_data/sanders/full_body_img
ffmpeg -i ../model/sanders_full_onnx/full_body_video.mp4 \
  test_data/sanders/full_body_img/%d.jpg

# 4. Copy landmarks
cp -r ../model/sanders_full_onnx/landmarks test_data/sanders/

# 5. Build and run
go build -o bin/generate ./cmd/generate

./bin/generate \
  --model ./models/unet_328.onnx \
  --audio ./test_data/sanders_audio.bin \
  --template ./test_data/sanders \
  --output ./output/sanders_frames \
  --mode ave
```

### 2. Swift Implementation (iOS/macOS!)

```bash
cd frame_generation_swift

# 1. Copy ONNX model
cp ../model/sanders_full_onnx/models/generator.onnx ./Models/

# 2. Convert audio features
python -c "
import numpy as np
import json

features = np.load('../model/sanders_full_onnx/aud_ave.npy')
features.astype('float32').tofile('./TestData/sanders_audio.bin')

metadata = {
    'num_frames': features.shape[0],
    'feature_size': int(np.prod(features.shape[1:])),
    'shape': list(features.shape)
}
with open('./TestData/sanders_audio.bin.json', 'w') as f:
    json.dump(metadata, f, indent=2)
"

# 3. Extract frames
mkdir -p TestData/sanders/full_body_img
ffmpeg -i ../model/sanders_full_onnx/full_body_video.mp4 \
  TestData/sanders/full_body_img/%d.jpg

# 4. Copy landmarks
cp -r ../model/sanders_full_onnx/landmarks TestData/sanders/

# 5. Build and test
swift build && swift test
```

### 3. Python Implementation (For Testing/Reference)

The Python implementation needs a PyTorch checkpoint, but you can use the ONNX model with ONNXRuntime:

```bash
cd frame_generation_pipeline

# Extract frames
mkdir -p ../temp_sanders/full_body_img
ffmpeg -i ../model/sanders_full_onnx/full_body_video.mp4 \
  ../temp_sanders/full_body_img/%d.jpg

# Copy landmarks
cp -r ../model/sanders_full_onnx/landmarks ../temp_sanders/

# If you have onnxruntime:
pip install onnxruntime

# Then you can modify the pipeline to use ONNX directly
```

## 🚀 Quick Start (Extract Everything)

Run this script to extract and prepare everything:

```bash
python generate_video_sanders.py --output result/sanders_video.mp4
```

Or manually extract:

```bash
# Create working directory
mkdir -p dataset/sanders

# Extract template frames
ffmpeg -i model/sanders_full_onnx/full_body_video.mp4 \
  dataset/sanders/full_body_img/%d.jpg

# Copy landmarks
cp -r model/sanders_full_onnx/landmarks dataset/sanders/

# Now you have a standard dataset structure!
```

## 📊 Dataset Details

**Sanders Character:**
- **Frames**: 523 (about 21 seconds at 25 FPS)
- **Resolution**: Full body images
- **Audio**: "ave" mode features (522 frames)
- **Landmarks**: 523 facial landmark files (.lms format)

**ONNX Models:**
- **Generator**: `generator.onnx` (46 MB) - The U-Net model
  - Input: Image (1, 6, 320, 320) + Audio (1, 32, 16, 16)
  - Output: Generated region (1, 3, 320, 320)
  - Validated with max error: 5.9e-7 (excellent!)

- **Audio Encoder**: `audio_encoder.onnx` (11 MB)
  - Already have features, so this is optional

## 🎯 What To Do Next

### Option 1: Use Go (Recommended - Python-free!)

1. Copy ONNX model to Go project
2. Convert audio features to binary (one-time Python)
3. Extract frames from video
4. Run Go binary - **No Python needed!**

### Option 2: Use Swift (For iOS/macOS)

1. Copy ONNX model to Swift project
2. Convert audio features to binary (one-time Python)
3. Extract frames from video
4. Build Swift app - **No Python needed!**

### Option 3: Test with Python First

1. Extract frames
2. Use existing pipeline with ONNX Runtime
3. Validate outputs
4. Then switch to Go/Swift

## 💡 Key Insight

You have the **ONNX model** which is what Go and Swift need!

The videos are just **pre-packaged template frames** - you need to extract them as individual images for the pipeline to use.

## 🔧 Complete Setup Script

```bash
#!/bin/bash
# setup_sanders.sh - Complete setup for Sanders dataset

echo "Setting up Sanders dataset for all platforms..."

# Base paths
SANDERS_DIR="model/sanders_full_onnx"
DATASET_DIR="dataset/sanders"

# Create dataset directory
mkdir -p $DATASET_DIR/full_body_img
mkdir -p $DATASET_DIR/landmarks

# Extract frames
echo "Extracting template frames..."
ffmpeg -i $SANDERS_DIR/full_body_video.mp4 \
  $DATASET_DIR/full_body_img/%d.jpg -y

# Copy landmarks
echo "Copying landmarks..."
cp -r $SANDERS_DIR/landmarks/* $DATASET_DIR/landmarks/

# For Go
echo "Setting up for Go..."
mkdir -p frame_generation_go/models
mkdir -p frame_generation_go/test_data
cp $SANDERS_DIR/models/generator.onnx \
  frame_generation_go/models/unet_328.onnx

# Convert audio features
python3 -c "
import numpy as np
import json

features = np.load('$SANDERS_DIR/aud_ave.npy')
features.astype('float32').tofile('frame_generation_go/test_data/sanders_audio.bin')

metadata = {
    'num_frames': features.shape[0],
    'feature_size': int(np.prod(features.shape[1:])),
    'shape': list(features.shape)
}
with open('frame_generation_go/test_data/sanders_audio.bin.json', 'w') as f:
    json.dump(metadata, f, indent=2)
"

# For Swift
echo "Setting up for Swift..."
mkdir -p frame_generation_swift/Models
mkdir -p frame_generation_swift/TestData
cp $SANDERS_DIR/models/generator.onnx \
  frame_generation_swift/Models/

python3 -c "
import numpy as np
import json

features = np.load('$SANDERS_DIR/aud_ave.npy')
features.astype('float32').tofile('frame_generation_swift/TestData/sanders_audio.bin')

metadata = {
    'num_frames': features.shape[0],
    'feature_size': int(np.prod(features.shape[1:])),
    'shape': list(features.shape)
}
with open('frame_generation_swift/TestData/sanders_audio.bin.json', 'w') as f:
    json.dump(metadata, f, indent=2)
"

echo "✓ Setup complete!"
echo ""
echo "Now you can:"
echo "  • Generate with Go: cd frame_generation_go && go run cmd/generate/main.go ..."
echo "  • Generate with Swift: cd frame_generation_swift && swift build ..."
echo ""
echo "All platforms are ready! 🚀"
```

Save as `setup_sanders.sh` and run:
```bash
chmod +x setup_sanders.sh
./setup_sanders.sh
```

## ✨ Summary

You have **EVERYTHING** needed:

- ✅ ONNX U-Net model (`generator.onnx`) - 46 MB
- ✅ Audio features (`aud_ave.npy`) - 522 frames
- ✅ Template frames (in video - need to extract)
- ✅ Landmarks (523 .lms files)
- ✅ Original audio (`aud.wav`)

**Next step:** Extract the frames from the video and you're ready to generate with Go or Swift - **completely Python-free**! 🎉

