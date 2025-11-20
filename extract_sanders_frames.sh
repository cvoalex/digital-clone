#!/bin/bash
# Extract all pre-cut frames from Sanders videos
# NO PYTHON - just ffmpeg!

set -e

SANDERS_DIR="model/sanders_full_onnx"

echo "============================================================"
echo "Extracting Pre-Cut Sanders Frames"
echo "============================================================"

cd "$SANDERS_DIR"

echo ""
echo "[1/4] Extracting 320x320 ROI frames (model inputs)..."
mkdir -p rois_320
ffmpeg -i rois_320_video.mp4 rois_320/%d.jpg -loglevel error -y
FRAME_COUNT=$(ls rois_320/*.jpg | wc -l)
echo "  ✓ Extracted $FRAME_COUNT ROI frames (320x320)"

echo ""
echo "[2/4] Extracting masked input frames..."
mkdir -p model_inputs
ffmpeg -i model_inputs_video.mp4 model_inputs/%d.jpg -loglevel error -y
FRAME_COUNT=$(ls model_inputs/*.jpg | wc -l)
echo "  ✓ Extracted $FRAME_COUNT masked frames"

echo ""
echo "[3/4] Extracting full body frames (for final output)..."
mkdir -p full_body_img
ffmpeg -i full_body_video.mp4 full_body_img/%d.jpg -loglevel error -y
FRAME_COUNT=$(ls full_body_img/*.jpg | wc -l)
echo "  ✓ Extracted $FRAME_COUNT full body frames"

echo ""
echo "[4/4] Extracting 328x328 crop frames..."
mkdir -p crops_328
ffmpeg -i crops_328_video.mp4 crops_328/%d.jpg -loglevel error -y
FRAME_COUNT=$(ls crops_328/*.jpg | wc -l)
echo "  ✓ Extracted $FRAME_COUNT crop frames (328x328)"

echo ""
echo "============================================================"
echo "Extraction Complete! ✅"
echo "============================================================"
echo ""
echo "Directory structure:"
echo "  model/sanders_full_onnx/"
echo "  ├── rois_320/           ← 320x320 frames (model input)"
echo "  ├── model_inputs/       ← Masked frames"
echo "  ├── full_body_img/      ← Original frames"
echo "  ├── crops_328/          ← 328x328 crops"
echo "  ├── models/"
echo "  │   └── generator.onnx  ← ONNX model"
echo "  ├── aud_ave.npy         ← Audio features"
echo "  └── cache/"
echo "      └── crop_rectangles.json ← Paste coordinates"
echo ""
echo "Next: Go/Swift just need to:"
echo "  1. Load frames from rois_320/"
echo "  2. Run ONNX inference"
echo "  3. Use crop_rectangles.json to paste back"
echo ""
echo "NO image processing needed - everything pre-cut! 🎉"
echo "============================================================"

