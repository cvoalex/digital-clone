# Simple Inference Go - Sanders Frame Generation

**Simplified Go implementation that uses pre-cut frames from Sanders dataset.**

## What Makes This Simple

- ✅ **No image processing** - uses pre-cut frames
- ✅ **No OpenCV needed** - just standard Go libraries
- ✅ **Just ONNX Runtime** - for inference only
- ✅ **~200 lines of code** - minimal implementation

## Prerequisites

1. **ONNX Runtime C library**:
```bash
# macOS
brew install onnxruntime

# Linux
wget https://github.com/microsoft/onnxruntime/releases/download/v1.16.3/onnxruntime-linux-x64-1.16.3.tgz
tar -xzf onnxruntime-linux-x64-1.16.3.tgz
sudo cp -r onnxruntime-linux-x64-1.16.3/lib/* /usr/local/lib/
sudo ldconfig
```

2. **Frames extracted** (already done!):
```bash
# If not done yet, from project root:
./extract_sanders_frames.sh
```

3. **Audio features in binary format**:
```bash
# Convert numpy to binary (one-time)
python3 -c "
import numpy as np
audio = np.load('model/sanders_full_onnx/aud_ave.npy')
audio.astype('float32').tofile('model/sanders_full_onnx/aud_ave.bin')
print(f'Converted {audio.shape} to binary')
"
```

## Build

```bash
cd simple_inference_go
go mod download
go build -o bin/infer ./cmd/infer
```

## Run

```bash
./bin/infer --sanders ../../model/sanders_full_onnx --output ./output
```

## How It Works

### 1. Loads Pre-Cut Frames
```
rois_320/{i}.jpg       → 320x320 image (already perfect size!)
model_inputs/{i}.jpg   → 320x320 masked (already masked!)
full_body_img/{i}.jpg  → Original frame
```

### 2. Runs ONNX Inference
```
input = concatenate(rois_320, model_inputs)  // 6 channels
audio = reshape(aud_ave[i])                  // (1, 32, 16, 16)
output = model.run(input, audio)             // (1, 3, 320, 320)
```

### 3. Composites Back
```
rect = crop_rectangles[i]
paste(output, full_body_img[i], rect)
save(output/{i}.jpg)
```

## Directory Structure

```
simple_inference_go/
├── cmd/infer/           # Main CLI
│   └── main.go
├── pkg/
│   ├── loader/          # Image loading & tensor conversion
│   │   └── loader.go
│   ├── onnx/            # ONNX inference
│   │   └── inference.go
│   └── compositor/      # Frame generation logic
│       └── compositor.go
├── go.mod
└── README.md
```

## Dependencies

Only 2 dependencies:
1. Standard Go libraries (`image/jpeg`, `encoding/json`)
2. ONNX Runtime Go bindings (`github.com/yalue/onnxruntime_go`)

**NO OpenCV, NO complex image processing!**

## Create Video

After generating frames:

```bash
ffmpeg -framerate 25 -i output/frame_%05d.jpg \
  -i ../../model/sanders_full_onnx/aud.wav \
  -c:v libx264 -c:a aac -crf 20 \
  sanders_video.mp4 -y
```

## Troubleshooting

### "ONNX Runtime not found"

Make sure ONNX Runtime is installed:
```bash
# macOS
brew install onnxruntime

# Check
ls /opt/homebrew/lib/libonnxruntime*
```

### "Audio features not found"

Convert numpy to binary:
```bash
python3 -c "
import numpy as np
audio = np.load('../../model/sanders_full_onnx/aud_ave.npy')
audio.astype('float32').tofile('../../model/sanders_full_onnx/aud_ave.bin')
"
```

### "Frames not found"

Extract frames:
```bash
cd ../..
./extract_sanders_frames.sh
```

## Performance

Expected: ~0.1s per frame on modern CPU

- 523 frames total
- ~52 seconds to generate all frames
- Plus video encoding time

## Comparison

| Feature | This Implementation | Full Pipeline |
|---------|-------------------|---------------|
| Lines of Code | ~200 | ~3,000 |
| Dependencies | 1 (ONNX) | 3 (ONNX, OpenCV, imaging) |
| Image Processing | None | Complex |
| Setup Complexity | Simple | Complex |
| Performance | Fast | Fast |

**This uses pre-cut frames, so it's much simpler!**

---

**Status: Ready to run!** 🚀

Just need to:
1. Install ONNX Runtime: `brew install onnxruntime`
2. Convert audio: (see above)
3. Build: `go build -o bin/infer ./cmd/infer`
4. Run: `./bin/infer`

