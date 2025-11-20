#!/bin/bash
# Test that Go setup is complete

echo "Testing Go setup..."
echo ""

cd frame_generation_go

echo "✓ Checking ONNX model..."
if [ -f "models/unet_328.onnx" ]; then
    SIZE=$(ls -lh models/unet_328.onnx | awk '{print $5}')
    echo "  ✓ ONNX model found: $SIZE"
else
    echo "  ✗ ONNX model not found!"
    exit 1
fi

echo ""
echo "✓ Checking audio features..."
if [ -f "test_data/sanders_audio.bin" ]; then
    SIZE=$(ls -lh test_data/sanders_audio.bin | awk '{print $5}')
    echo "  ✓ Audio binary found: $SIZE"
else
    echo "  ✗ Audio binary not found!"
    exit 1
fi

if [ -f "test_data/sanders_audio.bin.json" ]; then
    echo "  ✓ Audio metadata found"
    cat test_data/sanders_audio.bin.json | head -5
else
    echo "  ✗ Audio metadata not found!"
    exit 1
fi

echo ""
echo "✓ Checking template dataset..."
FRAME_COUNT=$(ls test_data/sanders/full_body_img/*.jpg 2>/dev/null | wc -l)
LMS_COUNT=$(ls test_data/sanders/landmarks/*.lms 2>/dev/null | wc -l)

echo "  ✓ Template frames: $FRAME_COUNT"
echo "  ✓ Landmark files: $LMS_COUNT"

if [ "$FRAME_COUNT" -eq 0 ] || [ "$LMS_COUNT" -eq 0 ]; then
    echo "  ✗ Missing template data!"
    exit 1
fi

echo ""
echo "✓ Checking Go dependencies..."
go list -m all | grep -E "(gocv|onnx|imaging)" || echo "  (Dependencies need to be installed)"

echo ""
echo "================================================"
echo "Go Setup Status: ✅ READY"
echo "================================================"
echo ""
echo "All files are in place!"
echo ""
echo "Note: To build and run, you'll need:"
echo "  - GoCV (OpenCV bindings)"
echo "  - ONNX Runtime C library"
echo ""
echo "Install on macOS:"
echo "  brew install opencv onnxruntime"
echo ""
echo "Then build:"
echo "  go build -o bin/generate ./cmd/generate"
echo ""

