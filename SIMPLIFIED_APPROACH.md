# 🎉 Even Simpler! Everything is Pre-Cut!

## You're Right!

The Sanders package has **EVERYTHING already pre-processed**:

```
model/sanders_full_onnx/
├── full_body_video.mp4         ← Original frames
├── crops_328_video.mp4         ← Already cropped to 328x328!
├── rois_320_video.mp4          ← Already cropped to 320x320! (model input size)
├── model_inputs_video.mp4      ← Already masked!
├── models/generator.onnx       ← ONNX model
├── aud_ave.npy                 ← Audio features
└── cache/
    └── crop_rectangles.json    ← Crop coordinates for pasting back
```

## What This Means

**No image processing needed!**

Just:
1. ✅ Extract frames from `rois_320_video.mp4` (already 320x320, ready for model)
2. ✅ Load ONNX model
3. ✅ Run inference
4. ✅ Use `crop_rectangles.json` to paste back

## Minimal Dependencies

We only need:
- ✅ ONNX Runtime (to run the model)
- ✅ Basic image loading (to read extracted frames)
- ✅ ffmpeg (to extract frames and create final video)

**No OpenCV needed!** Just ffmpeg + ONNX Runtime!

## Super Simple Workflow

### 1. Extract Pre-Cut Frames (ffmpeg only!)

```bash
cd model/sanders_full_onnx

# Extract 320x320 ROIs (model inputs)
mkdir -p rois_320
ffmpeg -i rois_320_video.mp4 rois_320/%d.jpg

# Extract masked inputs
mkdir -p model_inputs
ffmpeg -i model_inputs_video.mp4 model_inputs/%d.jpg

# Extract full body (for final compositing)
mkdir -p full_body_img  
ffmpeg -i full_body_video.mp4 full_body_img/%d.jpg

# Extract 328x328 crops
mkdir -p crops_328
ffmpeg -i crops_328_video.mp4 crops_328/%d.jpg
```

### 2. Run ONNX Model

Load the 320x320 ROI frames, run through ONNX model with audio features, get output.

### 3. Composite Back

Use `crop_rectangles.json` to know where to paste the generated region.

## Even Simpler: Python Script (No Training Required)

Since everything is pre-cut, you can use a simple Python script with just:
- `onnxruntime` (pure Python, no PyTorch!)
- `numpy`
- `PIL` or basic image library
- `ffmpeg`

**No training frameworks, no complex dependencies!**

```python
import onnxruntime as ort
import numpy as np
from PIL import Image

# Load ONNX model
session = ort.InferenceSession("models/generator.onnx")

# Load audio features
audio_feats = np.load("aud_ave.npy")

# For each frame
for i in range(522):
    # Load pre-cut 320x320 ROI
    roi = load_image(f"rois_320/{i}.jpg")
    
    # Load pre-masked input
    masked = load_image(f"model_inputs/{i}.jpg")
    
    # Prepare 6-channel input
    input_tensor = concatenate([roi, masked])  # (1, 6, 320, 320)
    
    # Get audio window
    audio = get_audio_window(audio_feats, i)  # (1, 32, 16, 16)
    
    # Run model
    output = session.run(None, {"image": input_tensor, "audio": audio})
    
    # Paste back using crop_rectangles.json
    composite_frame(output, i)
```

## Go/Swift Even Simpler Too

With pre-cut videos, Go/Swift just needs:
- ONNX Runtime (C library)
- Basic image loading
- JSON parsing for crop rectangles

No complex image processing!

## Action Plan

Want me to create:

**Option A**: Simple Python script using ONNX Runtime (not PyTorch)?
- Uses pre-cut frames
- Just ONNX inference
- Very minimal dependencies

**Option B**: Simplified Go that just reads pre-cut frames?
- Uses pre-cut videos
- Minimal image processing
- Just ONNX + basic file I/O

**Option C**: Use the pre-cut frames directly with existing code?
- Point the pipeline at the pre-extracted frames
- Skip all the cropping/masking steps

What would you prefer?

---

**Bottom line: You're right - they gave you everything pre-processed! We can make this MUCH simpler!** 🎉

