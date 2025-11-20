# Core ML Conversion Guide for Swift

## Problem

The coremltools Python library has compatibility issues with Python 3.13 and the ONNX models.

## Solutions

### Option 1: Use Xcode (Recommended - GUI)

1. **Open Xcode**
2. **Create new macOS Command Line Tool project**
3. **Add ML Model:**
   - Right-click project → Add Files
   - Select `model/sanders_full_onnx/models/audio_encoder.onnx`
   - Xcode will automatically convert to Core ML!
4. **Repeat for generator.onnx**

Xcode handles the conversion automatically and creates `.mlmodel` or `.mlpackage` files.

### Option 2: Use Python 3.11 (not 3.13)

The issue is Python 3.13 compatibility. Try with Python 3.11:

```bash
# Install Python 3.11
brew install python@3.11

# Create virtual environment
python3.11 -m venv venv_coreml
source venv_coreml/bin/activate

# Install coremltools
pip install coremltools==7.2 onnx

# Convert
python convert_to_coreml_simple.py
```

### Option 3: Use Existing ONNX Runtime in Swift

Since you have ONNX Runtime working (from audio pipeline), just use that!

**Pros:**
- ✅ Already working
- ✅ No conversion needed
- ✅ Proven approach (like Go)

**Cons:**
- ⚠️ Won't use Neural Engine
- ⚠️ Performance similar to Go (~8-10 FPS)

### Option 4: Manual Core ML Model Creation

Create Core ML models programmatically in Swift using Core ML Builder APIs.

## My Recommendation

### For Speed (Get Working Today):

**Use ONNX Runtime in Swift** (copy Go approach):
- Working approach (validated in Go)
- Uses existing ONNXWrapper.swift
- Expected: 8-10 FPS
- **Time: 2-3 hours**

### For Performance (Best Long-term):

**Convert via Xcode:**
1. Open Xcode
2. Drag ONNX files into project
3. Xcode converts automatically
4. Use Core ML APIs
5. Get 20-30 FPS with Neural Engine
- **Time: 4-6 hours total**

## What I Recommend

Since you want the Swift implementation done:

**Let's use ONNX Runtime** (like Go) to get it working quickly, then you can convert to Core ML later for the performance boost.

This way:
1. Swift working today (~2-3 hours)
2. Core ML upgrade later (when you have time for conversion)

The Go implementation proves ONNX Runtime works well - Swift will be similar!

Want me to complete Swift with ONNX Runtime (matching Go's approach)?

