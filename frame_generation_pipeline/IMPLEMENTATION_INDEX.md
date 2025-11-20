# Frame Generation Pipeline - Implementation Index

Complete reference guide for all three implementations of the frame generation pipeline.

## 📚 Documentation Index

### Getting Started
1. **[Quick Start Guide](../FRAME_GENERATION_QUICKSTART.md)** - Get running in 5 minutes
2. **[Complete Guide](../FRAME_GENERATION_GUIDE.md)** - Comprehensive overview
3. **[Implementation Summary](../FRAME_GENERATION_SUMMARY.md)** - What was built

### Implementation Guides
1. **[Python README](README.md)** - Python reference implementation
2. **[Go README](../frame_generation_go/README.md)** - Go validation implementation
3. **[Swift README](../frame_generation_swift/README.md)** - Swift/iOS production implementation

## 🗂️ Directory Structure

```
frame_generation_pipeline/        # Python (THIS DIRECTORY)
├── __init__.py                   # Package initialization
├── unet_model.py                 # U-Net model wrapper (200 lines)
├── image_processor.py            # Image operations (300 lines)
├── frame_generator.py            # Frame generation (250 lines)
├── pipeline.py                   # Complete pipeline (350 lines)
├── export_model.py               # ONNX export script (100 lines)
├── requirements.txt              # Python dependencies
├── tests/                        # Test suite
│   ├── __init__.py
│   ├── test_image_processor.py  # 150 lines
│   ├── test_frame_generator.py
│   └── test_pipeline.py
├── models/                       # ONNX models
│   └── unet_328.onnx            # Exported model
├── test_data/                    # Test data
│   └── reference_outputs/        # Reference frames for validation
└── README.md                     # This implementation's docs

../frame_generation_go/           # Go implementation
../frame_generation_swift/        # Swift implementation
```

## 🔧 Module Reference

### Python Modules

#### `unet_model.py`
**Purpose:** U-Net model wrapper for PyTorch

**Classes:**
- `UNetModel` - Main model wrapper

**Key Methods:**
- `__init__(checkpoint_path, mode, device)` - Initialize model
- `predict(image_tensor, audio_features)` - Run inference
- `export_to_onnx(output_path)` - Export to ONNX format
- `get_input_shapes()` - Get expected input shapes

**Example:**
```python
model = UNetModel("./checkpoint/May/5.pth", mode="ave")
output = model.predict(image_tensor, audio_features)
model.export_to_onnx("./models/unet.onnx")
```

#### `image_processor.py`
**Purpose:** Image processing operations

**Classes:**
- `ImageProcessor` - Image operations handler

**Key Methods:**
- `load_image(path)` - Load image from disk
- `load_landmarks(path)` - Load facial landmarks
- `crop_face_region(image, landmarks)` - Crop face
- `resize_image(image, target_size)` - Resize with cubic interpolation
- `create_masked_region(image)` - Create masked version
- `prepare_input_tensors(image)` - Prepare for U-Net
- `paste_generated_region(...)` - Paste back into frame

**Example:**
```python
processor = ImageProcessor()
img = processor.load_image("./template/0.jpg")
lms = processor.load_landmarks("./template/0.lms")
crop, coords = processor.crop_face_region(img, lms)
```

#### `frame_generator.py`
**Purpose:** Frame-by-frame generation logic

**Classes:**
- `FrameGenerator` - Frame generation coordinator

**Key Methods:**
- `__init__(unet_model, mode)` - Initialize generator
- `generate_frame(template, landmarks, audio)` - Single frame
- `generate_frames_from_template_sequence(...)` - Batch frames
- `save_frames(frames, output_dir)` - Save to disk

**Example:**
```python
generator = FrameGenerator(unet_model, mode="ave")
frames = generator.generate_frames_from_template_sequence(
    img_dir="./templates/full_body_img",
    lms_dir="./templates/landmarks",
    audio_features=audio_feats
)
```

#### `pipeline.py`
**Purpose:** Complete end-to-end pipeline

**Classes:**
- `FrameGenerationPipeline` - High-level API

**Key Methods:**
- `__init__(checkpoint_path, mode, device)` - Initialize
- `generate_video(...)` - Generate complete video
- `generate_frames_only(...)` - Frames without video
- `export_model_to_onnx(path)` - Export model
- `generate_reference_outputs(...)` - Create validation data

**Example:**
```python
pipeline = FrameGenerationPipeline(
    checkpoint_path="./checkpoint/May/5.pth",
    mode="ave"
)

video = pipeline.generate_video(
    audio_features_path="./audio_features.npy",
    template_dir="./dataset/May",
    audio_path="./audio.wav",
    output_path="./result.mp4"
)
```

## 🚀 Usage Patterns

### Pattern 1: Quick Video Generation
```python
from frame_generation_pipeline import FrameGenerationPipeline

pipeline = FrameGenerationPipeline("./checkpoint/May/5.pth")
pipeline.generate_video(
    audio_features_path="./audio_features.npy",
    template_dir="./dataset/May",
    audio_path="./audio.wav",
    output_path="./result.mp4"
)
```

### Pattern 2: Custom Processing
```python
from frame_generation_pipeline import (
    UNetModel, 
    ImageProcessor, 
    FrameGenerator
)

# Initialize components
model = UNetModel("./checkpoint/May/5.pth")
processor = ImageProcessor()
generator = FrameGenerator(model)

# Custom processing loop
for i, audio_feat in enumerate(audio_features):
    template = processor.load_image(f"./templates/{i}.jpg")
    landmarks = processor.load_landmarks(f"./templates/{i}.lms")
    
    frame = generator.generate_frame(template, landmarks, audio_feat)
    
    # Custom post-processing
    frame = apply_custom_filter(frame)
    save_frame(frame, f"./output/{i}.jpg")
```

### Pattern 3: Batch Processing with Progress
```python
from tqdm import tqdm

pipeline = FrameGenerationPipeline("./checkpoint/May/5.pth")

for audio_file in tqdm(audio_files):
    pipeline.generate_video(
        audio_features_path=audio_file,
        template_dir="./dataset/May",
        audio_path=audio_file.replace('.npy', '.wav'),
        output_path=audio_file.replace('.npy', '.mp4')
    )
```

### Pattern 4: Generate Reference Outputs
```python
pipeline = FrameGenerationPipeline("./checkpoint/May/5.pth")

metadata = pipeline.generate_reference_outputs(
    audio_features_path="./audio_features.npy",
    template_dir="./dataset/May",
    output_dir="./test_data/reference",
    num_sample_frames=10
)

print(f"Generated {len(metadata['reference_frames'])} reference frames")
```

## 🔬 Testing

### Run All Tests
```bash
pytest tests/ -v
```

### Run Specific Test
```bash
pytest tests/test_image_processor.py -v
```

### Run with Coverage
```bash
pytest tests/ --cov=. --cov-report=html
```

### Test Structure
```
tests/
├── test_image_processor.py    # Image processing tests
├── test_frame_generator.py    # Frame generation tests
├── test_pipeline.py            # Pipeline integration tests
└── conftest.py                 # Shared fixtures
```

## 📊 Performance Tuning

### GPU vs CPU
```python
# Force CPU
pipeline = FrameGenerationPipeline(
    checkpoint_path="./checkpoint/May/5.pth",
    device="cpu"
)

# Force GPU
pipeline = FrameGenerationPipeline(
    checkpoint_path="./checkpoint/May/5.pth",
    device="cuda"
)

# Auto-detect (default)
pipeline = FrameGenerationPipeline(
    checkpoint_path="./checkpoint/May/5.pth"
)
```

### Memory Management
```python
# Process in batches to save memory
batch_size = 100
for i in range(0, total_frames, batch_size):
    frames = pipeline.generate_frames_only(
        audio_features_path=f"./batch_{i}.npy",
        template_dir="./dataset/May",
        output_dir=f"./frames/batch_{i}"
    )
    del frames  # Free memory
    torch.cuda.empty_cache()  # Clear GPU cache
```

### Video Quality
```python
# High quality (larger file)
pipeline.generate_video(..., crf=18)

# Balanced (default)
pipeline.generate_video(..., crf=20)

# Lower quality (smaller file)
pipeline.generate_video(..., crf=28)
```

## 🐛 Debugging

### Enable Debug Logging
```python
import logging

logging.basicConfig(level=logging.DEBUG)

pipeline = FrameGenerationPipeline("./checkpoint/May/5.pth")
```

### Save Intermediate Results
```python
pipeline.generate_video(
    ...,
    save_frames=True,
    frames_dir="./debug_frames"
)
```

### Check Model Output
```python
model = UNetModel("./checkpoint/May/5.pth")

# Get shapes
image_shape, audio_shape = model.get_input_shapes()
print(f"Image shape: {image_shape}")
print(f"Audio shape: {audio_shape}")

# Test with dummy data
import torch
dummy_img = torch.randn(*image_shape)
dummy_audio = torch.randn(*audio_shape)

output = model.predict(dummy_img, dummy_audio)
print(f"Output shape: {output.shape}")
```

## 🔄 Cross-Platform Integration

### Python → Go
```python
# 1. Export model
pipeline.export_model_to_onnx("./models/unet_328.onnx")

# 2. Convert audio features
import numpy as np
import json

features = np.load("./audio_features.npy")
features.astype(np.float32).tofile("./audio_features.bin")

metadata = {
    'num_frames': features.shape[0],
    'feature_size': int(np.prod(features.shape[1:])),
    'shape': list(features.shape)
}
with open("./audio_features.bin.json", "w") as f:
    json.dump(metadata, f)
```

### Python → Swift
```python
# Same as Go, but copy files to Swift project
import shutil

shutil.copy("./models/unet_328.onnx", "../frame_generation_swift/Models/")
shutil.copy("./audio_features.bin", "../frame_generation_swift/TestData/")
shutil.copy("./audio_features.bin.json", "../frame_generation_swift/TestData/")
```

## 📖 API Reference

### Complete API Documentation

See individual module docstrings for detailed API documentation:

```python
# View module help
import frame_generation_pipeline
help(frame_generation_pipeline)

# View class help
from frame_generation_pipeline import FrameGenerationPipeline
help(FrameGenerationPipeline)

# View method help
help(FrameGenerationPipeline.generate_video)
```

## 🎯 Common Tasks

### Task: Export Model for Deployment
```bash
python export_model.py \
  --checkpoint ../checkpoint/May/5.pth \
  --output ./models/unet_328.onnx \
  --mode ave
```

### Task: Generate Single Frame
```python
from frame_generation_pipeline import UNetModel, ImageProcessor

model = UNetModel("./checkpoint/May/5.pth")
processor = ImageProcessor()

img = processor.load_image("./template/0.jpg")
lms = processor.load_landmarks("./template/0.lms")

# Process image
crop, coords = processor.crop_face_region(img, lms)
crop_328 = processor.resize_image(crop, (328, 328))
inner = crop_328[4:324, 4:324]

# Prepare input
img_tensor, _ = processor.prepare_input_tensors(inner)

# Generate
output = model.predict(img_tensor, audio_features)

# Paste back
result = processor.paste_generated_region(img, output, coords, (h, w))
```

### Task: Benchmark Performance
```python
import time

start = time.time()
pipeline.generate_video(...)
end = time.time()

print(f"Time: {end - start:.2f}s")
print(f"FPS: {num_frames / (end - start):.2f}")
```

## 📝 Notes

- All paths can be absolute or relative
- Mode must match audio feature extraction method
- Template directory must have `full_body_img/` and `landmarks/` subdirectories
- ONNX export requires `onnx` package
- Video assembly requires `ffmpeg` installed

## 🔗 Related Documentation

- [INFERENCE_PIPELINE_PROMPT.md](../INFERENCE_PIPELINE_PROMPT.md) - Original specification
- [audio_pipeline/README.md](../audio_pipeline/README.md) - Audio feature extraction
- [docs/model-details.md](../docs/model-details.md) - Model architecture details

## ✅ Checklist for New Users

- [ ] Install dependencies (`pip install -r requirements.txt`)
- [ ] Have trained checkpoint or use provided one
- [ ] Have template images and landmarks
- [ ] Have audio features from audio pipeline
- [ ] Try quick start example
- [ ] Generate reference outputs
- [ ] Run tests to verify installation
- [ ] Export model to ONNX (if using Go/Swift)

---

**Last Updated:** November 19, 2025

**Version:** 1.0.0

**Status:** Production Ready ✅

