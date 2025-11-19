# Adding Native ONNX Runtime to Xcode Project

## Steps to Add ONNX Runtime (NO PYTHON)

### Option 1: Manual Framework (Recommended)

1. **Download ONNX Runtime**
```bash
cd /Users/alexanderrusich/Projects/digital-clone/audio_pipeline_swift/mels
curl -L -o onnxruntime.tgz https://github.com/microsoft/onnxruntime/releases/download/v1.16.3/onnxruntime-osx-universal2-1.16.3.tgz
tar -xzf onnxruntime.tgz
```

2. **In Xcode:**
   - Select the `mels` project in the left sidebar
   - Select the `mels` target
   - Go to "General" tab
   - Scroll to "Frameworks, Libraries, and Embedded Content"
   - Click the "+" button
   - Click "Add Other..." → "Add Files..."
   - Navigate to: `onnxruntime-osx-universal2-1.16.3/lib/onnxruntime.framework`
   - Select it and click "Open"
   - Make sure "Embed & Sign" is selected

3. **Add C header bridge:**
   - File → New → File
   - Choose "Header File"
   - Name it `mels-Bridging-Header.h`
   - Add the import

### Option 2: Using the C API directly (What I'll do now)

Let me create a native Swift wrapper using the ONNX Runtime C API.

