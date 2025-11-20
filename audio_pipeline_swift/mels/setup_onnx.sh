#!/bin/bash

# Setup script to add ONNX Runtime to the macOS app

echo "=========================================="
echo "Setting up ONNX Runtime for macOS"
echo "=========================================="

# Download ONNX Runtime for macOS if not already present
ONNX_VERSION="1.16.3"
ONNX_DIR="onnxruntime-osx-universal2-${ONNX_VERSION}"
ONNX_URL="https://github.com/microsoft/onnxruntime/releases/download/v${ONNX_VERSION}/onnxruntime-osx-universal2-${ONNX_VERSION}.tgz"

if [ ! -d "Frameworks/${ONNX_DIR}" ]; then
    echo "Downloading ONNX Runtime..."
    cd Frameworks 2>/dev/null || mkdir Frameworks && cd Frameworks
    
    if [ ! -f "${ONNX_DIR}.tgz" ]; then
        curl -L -o "${ONNX_DIR}.tgz" "$ONNX_URL"
    fi
    
    echo "Extracting..."
    tar -xzf "${ONNX_DIR}.tgz"
    
    echo "✓ ONNX Runtime downloaded and extracted"
    cd ..
else
    echo "✓ ONNX Runtime already present"
fi

# Copy ONNX model
echo ""
echo "Copying ONNX model..."
mkdir -p Resources
cp ../../audio_pipeline_go/models/audio_encoder.onnx Resources/
echo "✓ Model copied to Resources/"

echo ""
echo "=========================================="
echo "✅ Setup complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Add onnxruntime.framework to your Xcode project:"
echo "   - Select project in Xcode"
echo "   - Go to Target → General → Frameworks and Libraries"
echo "   - Click + and add Frameworks/${ONNX_DIR}/lib/onnxruntime.framework"
echo ""
echo "2. Add audio_encoder.onnx to your Xcode project:"
echo "   - Drag Resources/audio_encoder.onnx into Xcode"
echo "   - Make sure 'Copy items if needed' is checked"
echo ""
echo "Or use CocoaPods (easier):"
echo "   pod init"
echo "   # Add 'pod \"onnxruntime-objc\"' to Podfile"
echo "   pod install"
echo ""

