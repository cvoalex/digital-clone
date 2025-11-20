#!/bin/bash
# Run comparison between Python and Go implementations

set -e

NUM_FRAMES=${1:-10}
AUDIO=${2:-demo/talk_hb.wav}

echo "============================================================"
echo "Running Python vs Go Comparison"
echo "============================================================"
echo "Frames: $NUM_FRAMES"
echo "Audio: $AUDIO"
echo "============================================================"

# Clean previous results
echo ""
echo "Cleaning previous results..."
rm -rf comparison_results/python_output/frames
rm -rf comparison_results/go_output/frames
mkdir -p comparison_results/python_output/frames
mkdir -p comparison_results/go_output/frames

# Run Python
echo ""
echo "=========================================="
echo "Running Python ONNX Pipeline"
echo "=========================================="
cd python_inference
python3 generate_frames.py \
  --sanders ../model/sanders_full_onnx \
  --audio ../$AUDIO \
  --frames $NUM_FRAMES

# Run Go
echo ""
echo "=========================================="
echo "Running Go ONNX Pipeline"
echo "=========================================="
cd ../simple_inference_go
DYLD_LIBRARY_PATH=/opt/homebrew/lib ./bin/infer \
  --sanders ../model/sanders_full_onnx \
  --audio ../$AUDIO \
  --frames $NUM_FRAMES

# Create videos
cd ..
echo ""
echo "=========================================="
echo "Creating Comparison Videos"
echo "=========================================="

echo "Creating Python video..."
ffmpeg -framerate 25 -i comparison_results/python_output/frames/frame_%05d.jpg \
  -i $AUDIO \
  -c:v libx264 -c:a aac -crf 20 \
  comparison_results/python_output/video.mp4 -y -loglevel error
echo "✓ comparison_results/python_output/video.mp4"

echo "Creating Go video..."
ffmpeg -framerate 25 -i comparison_results/go_output/frames/frame_%05d.jpg \
  -i $AUDIO \
  -c:v libx264 -c:a aac -crf 20 \
  comparison_results/go_output/video.mp4 -y -loglevel error
echo "✓ comparison_results/go_output/video.mp4"

echo "Creating side-by-side comparison..."
ffmpeg -i comparison_results/python_output/video.mp4 \
  -i comparison_results/go_output/video.mp4 \
  -filter_complex "[0:v]scale=640:360,drawtext=text='Python':x=10:y=10:fontsize=24:fontcolor=yellow:box=1:boxcolor=black@0.5[left];[1:v]scale=640:360,drawtext=text='Go':x=10:y=10:fontsize=24:fontcolor=yellow:box=1:boxcolor=black@0.5[right];[left][right]hstack" \
  -c:v libx264 -crf 20 \
  comparison_results/comparison.mp4 -y -loglevel error
echo "✓ comparison_results/comparison.mp4"

# Compare pixels
echo ""
echo "=========================================="
echo "Pixel Comparison"
echo "=========================================="
python3 -c "
import numpy as np
from PIL import Image

print('Frame-by-frame comparison:')
print()

for i in range(1, $NUM_FRAMES + 1):
    py_img = np.array(Image.open(f'comparison_results/python_output/frames/frame_{i:05d}.jpg'))
    go_img = np.array(Image.open(f'comparison_results/go_output/frames/frame_{i:05d}.jpg'))
    
    diff = np.abs(py_img.astype(float) - go_img.astype(float))
    
    print(f'Frame {i:2d}: Max={diff.max():5.1f}, Mean={diff.mean():6.3f}, Identical={np.sum(diff==0)/diff.size*100:5.1f}%')

print()
print('✅ Comparison complete!')
"

echo ""
echo "============================================================"
echo "Results Ready!"
echo "============================================================"
echo ""
echo "Directory: comparison_results/"
echo ""
echo "Python outputs:"
echo "  • Frames: comparison_results/python_output/frames/"
echo "  • Video:  comparison_results/python_output/video.mp4"
echo ""
echo "Go outputs:"
echo "  • Frames: comparison_results/go_output/frames/"
echo "  • Video:  comparison_results/go_output/video.mp4"
echo ""
echo "Comparison:"
echo "  • Video:  comparison_results/comparison.mp4 ⭐"
echo ""
echo "To view:"
echo "  open comparison_results/comparison.mp4"
echo ""
echo "============================================================"

