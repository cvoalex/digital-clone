#!/bin/bash
# setup_sanders.sh - Complete setup for Sanders dataset

set -e  # Exit on error

echo "============================================================"
echo "Setting up Sanders dataset for all platforms"
echo "============================================================"

# Base paths
SANDERS_DIR="model/sanders_full_onnx"
DATASET_DIR="dataset/sanders"

# Validate source files exist
if [ ! -d "$SANDERS_DIR" ]; then
    echo "Error: Sanders directory not found: $SANDERS_DIR"
    exit 1
fi

if [ ! -f "$SANDERS_DIR/models/generator.onnx" ]; then
    echo "Error: Generator ONNX model not found"
    exit 1
fi

echo ""
echo "[1/5] Creating dataset directory structure..."
mkdir -p $DATASET_DIR/full_body_img
mkdir -p $DATASET_DIR/landmarks
echo "  ✓ Directories created"

echo ""
echo "[2/5] Extracting template frames from video..."
ffmpeg -i $SANDERS_DIR/full_body_video.mp4 \
  $DATASET_DIR/full_body_img/%d.jpg \
  -loglevel error -y
FRAME_COUNT=$(ls $DATASET_DIR/full_body_img/*.jpg | wc -l)
echo "  ✓ Extracted $FRAME_COUNT frames"

echo ""
echo "[3/5] Copying landmarks..."
cp $SANDERS_DIR/landmarks/* $DATASET_DIR/landmarks/
LMS_COUNT=$(ls $DATASET_DIR/landmarks/*.lms | wc -l)
echo "  ✓ Copied $LMS_COUNT landmark files"

echo ""
echo "[4/5] Setting up Go implementation..."
mkdir -p frame_generation_go/models
mkdir -p frame_generation_go/test_data/sanders/full_body_img
mkdir -p frame_generation_go/test_data/sanders/landmarks

cp $SANDERS_DIR/models/generator.onnx frame_generation_go/models/unet_328.onnx
cp $DATASET_DIR/full_body_img/*.jpg frame_generation_go/test_data/sanders/full_body_img/
cp $DATASET_DIR/landmarks/*.lms frame_generation_go/test_data/sanders/landmarks/

# Convert audio features to binary
python3 -c "
import numpy as np
import json

features = np.load('$SANDERS_DIR/aud_ave.npy')
features.astype('float32').tofile('frame_generation_go/test_data/sanders_audio.bin')

metadata = {
    'num_frames': int(features.shape[0]),
    'feature_size': int(np.prod(features.shape[1:])),
    'shape': [int(x) for x in features.shape]
}
with open('frame_generation_go/test_data/sanders_audio.bin.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print(f'Converted audio features: {features.shape}')
"
echo "  ✓ Go setup complete"

echo ""
echo "[5/5] Setting up Swift implementation..."
mkdir -p frame_generation_swift/Models
mkdir -p frame_generation_swift/TestData/sanders/full_body_img
mkdir -p frame_generation_swift/TestData/sanders/landmarks

cp $SANDERS_DIR/models/generator.onnx frame_generation_swift/Models/
cp $DATASET_DIR/full_body_img/*.jpg frame_generation_swift/TestData/sanders/full_body_img/
cp $DATASET_DIR/landmarks/*.lms frame_generation_swift/TestData/sanders/landmarks/

python3 -c "
import numpy as np
import json

features = np.load('$SANDERS_DIR/aud_ave.npy')
features.astype('float32').tofile('frame_generation_swift/TestData/sanders_audio.bin')

metadata = {
    'num_frames': int(features.shape[0]),
    'feature_size': int(np.prod(features.shape[1:])),
    'shape': [int(x) for x in features.shape]
}
with open('frame_generation_swift/TestData/sanders_audio.bin.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print(f'Converted audio features: {features.shape}')
"
echo "  ✓ Swift setup complete"

echo ""
echo "============================================================"
echo "Setup Complete! ✅"
echo "============================================================"
echo ""
echo "What's ready:"
echo "  ✓ Dataset extracted: $DATASET_DIR/"
echo "  ✓ Go ready: frame_generation_go/"
echo "  ✓ Swift ready: frame_generation_swift/"
echo ""
echo "Next steps:"
echo ""
echo "  🔷 Go (Python-free):"
echo "     cd frame_generation_go"
echo "     go build -o bin/generate ./cmd/generate"
echo "     ./bin/generate \\"
echo "       --model ./models/unet_328.onnx \\"
echo "       --audio ./test_data/sanders_audio.bin \\"
echo "       --template ./test_data/sanders \\"
echo "       --output ./output/sanders_frames"
echo ""
echo "  🍎 Swift (iOS/macOS):"
echo "     cd frame_generation_swift"
echo "     swift build"
echo "     # Or open in Xcode"
echo ""
echo "Both implementations are now Python-free! 🚀"
echo "============================================================"

