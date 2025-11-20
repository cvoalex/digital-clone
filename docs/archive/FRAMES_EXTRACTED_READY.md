# ✅ Frames Extracted - Ready for Inference!

## What's Done

All pre-cut frames have been extracted from videos:

```
model/sanders_full_onnx/
├── rois_320/              ✅ 523 frames (320x320) - MODEL INPUT SIZE
├── model_inputs/          ✅ 523 frames - PRE-MASKED
├── full_body_img/         ✅ 523 frames - ORIGINAL  
├── crops_328/             ✅ 523 frames (328x328) - CROPS
├── models/generator.onnx  ✅ U-Net model (46 MB)
├── aud_ave.npy           ✅ Audio features (522 frames)
├── landmarks/            ✅ 523 .lms files
└── cache/
    └── crop_rectangles.json ✅ Paste coordinates
```

**All extracted with ffmpeg - NO PYTHON! ✅**

## What Go/Swift Need To Do Now

### Super Simple 3-Step Process:

#### 1. Load Pre-Cut Data
```
For each frame i (0-522):
  - Load image: rois_320/{i}.jpg  (already 320x320!)
  - Load masked: model_inputs/{i}.jpg (already masked!)
  - Load audio: aud_ave.npy[i] (already extracted!)
```

#### 2. Run ONNX Inference
```
input_tensor = concatenate(rois_320[i], model_inputs[i])  // 6 channels
audio_tensor = reshape(aud_ave[i], [1, 32, 16, 16])

output = onnx_model.run(input_tensor, audio_tensor)  // (1, 3, 320, 320)
```

#### 3. Paste Back Using JSON
```
rect = crop_rectangles[i]["rect"]  // [x1, y1, x2, y2]
full_frame = load(full_body_img/{i}.jpg)
paste(output, full_frame, rect)
save(full_frame, output/{i}.jpg)
```

## Crop Rectangles Format

```json
{
  "0": {
    "rect": [532, 210, 714, 392],  // [x1, y1, x2, y2]
    ...
  },
  "1": {
    "rect": [532, 211, 714, 393],
    ...
  }
}
```

Where to paste the 320x320 generated output back into the full frame.

## What This Simplifies

### Before (Complex):
1. ❌ Load full image
2. ❌ Load landmarks
3. ❌ Calculate crop region
4. ❌ Resize to 328x328
5. ❌ Extract 320x320
6. ❌ Create mask
7. ❌ Run inference
8. ❌ Resize back
9. ❌ Paste into crop
10. ❌ Paste crop into frame

### Now (Simple):
1. ✅ Load pre-cut 320x320 (already done!)
2. ✅ Run inference
3. ✅ Paste using JSON coordinates

**90% of complexity eliminated!**

## Go Implementation Needs

**Just 3 things:**
1. ONNX Runtime (for inference)
2. Basic image I/O (load/save JPEGs)
3. JSON parsing (for rectangles)

**NO OpenCV needed!** ✨

Can use:
- `image/jpeg` (standard Go library)
- `encoding/json` (standard Go library)  
- `github.com/yalue/onnxruntime_go` (ONNX inference)

## Swift Implementation Needs

**Just 3 things:**
1. ONNX Runtime (for inference)
2. UIImage/NSImage (load/save JPEGs)
3. JSON Decoder (for rectangles)

**All built into Swift!** ✨

Can use:
- `UIKit/AppKit` (image I/O)
- `Foundation` (JSON)
- ONNX Runtime C API

## File Sizes

```
rois_320/      : 523 JPEGs × ~5KB  = ~2.6 MB
model_inputs/  : 523 JPEGs × ~2KB  = ~1 MB
full_body_img/ : 523 JPEGs × ~50KB = ~26 MB
aud_ave.npy    : 522 × 512 floats  = ~1 MB
generator.onnx : ONNX model        = 46 MB

Total: ~77 MB (everything needed!)
```

## Next Steps

### For Go:
1. Create simple loader for JPEGs
2. Add ONNX Runtime inference
3. Parse crop_rectangles.json
4. Paste and save

### For Swift:
1. Load JPEGs with UIImage
2. Add ONNX Runtime inference
3. Parse JSON with Codable
4. Paste and save

**Both can be done with MINIMAL code - maybe 200 lines total!**

## Test First Frame

Quick test to verify everything works:

```bash
# Check files exist
ls model/sanders_full_onnx/rois_320/1.jpg
ls model/sanders_full_onnx/model_inputs/1.jpg
ls model/sanders_full_onnx/full_body_img/1.jpg

# Check they're the right size
file model/sanders_full_onnx/rois_320/1.jpg
# Should show: 320x320

file model/sanders_full_onnx/full_body_img/1.jpg
# Should show: larger (e.g. 1920x1080)
```

---

## Summary

✅ **Extraction**: Complete (ffmpeg only, no Python!)  
✅ **Data Ready**: All 523 frames pre-processed  
✅ **ONNX Model**: Ready (46 MB)  
✅ **Audio Features**: Ready (522 frames)  
✅ **Coordinates**: Ready (crop_rectangles.json)  

**Next**: Simple Go/Swift code to:
1. Load images ✓
2. Run ONNX ✓  
3. Paste back ✓

**No complex image processing. No Python. Just inference!** 🚀

