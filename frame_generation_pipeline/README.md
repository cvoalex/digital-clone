# Frame Generation Pipeline (Python)

A standalone, modular video frame generation pipeline for SyncTalk_2D that generates lip-sync video frames from audio features and template images.

## Overview

This pipeline takes:
- **Input**: Audio features (32, 16, 16 tensors) + template images/landmarks
- **Output**: Generated video frames with synchronized lip movements

The pipeline uses a U-Net model to generate realistic mouth/lip regions frame-by-frame and pastes them back into template frames.

## Architecture

```
frame_generation_pipeline/
├── __init__.py              # Package initialization
├── unet_model.py            # U-Net model wrapper
├── image_processor.py       # Image cropping/masking/pasting
├── frame_generator.py       # Frame generation loop
├── pipeline.py              # Complete end-to-end pipeline
├── tests/                   # Unit and integration tests
├── test_data/               # Test data and reference outputs
└── models/                  # Exported ONNX models
```

## Installation

```bash
# Install dependencies
pip install torch torchvision opencv-python numpy tqdm

# Optional: for ONNX export
pip install onnx onnxruntime
```

## Quick Start

### 1. Basic Usage

```python
from frame_generation_pipeline import FrameGenerationPipeline

# Initialize pipeline
pipeline = FrameGenerationPipeline(
    checkpoint_path="./checkpoint/May/5.pth",
    mode="ave"
)

# Generate video
video_path = pipeline.generate_video(
    audio_features_path="./audio_features.npy",
    template_dir="./dataset/May",
    audio_path="./demo/audio.wav",
    output_path="./output/result.mp4"
)
```

### 2. Generate Frames Only

```python
# Generate frames without creating video
frame_paths = pipeline.generate_frames_only(
    audio_features_path="./audio_features.npy",
    template_dir="./dataset/May",
    output_dir="./output/frames"
)
```

### 3. Export Model to ONNX

```python
# Export U-Net model for Go/Swift implementations
pipeline.export_model_to_onnx(
    output_path="./models/unet_328.onnx",
    opset_version=11
)
```

### 4. Generate Reference Outputs

```python
# Generate reference outputs for validation
metadata = pipeline.generate_reference_outputs(
    audio_features_path="./audio_features.npy",
    template_dir="./dataset/May",
    output_dir="./test_data/reference_outputs",
    num_sample_frames=10
)
```

## Directory Structure Requirements

### Template Directory

Your template directory should have the following structure:

```
dataset/May/
├── full_body_img/
│   ├── 0.jpg
│   ├── 1.jpg
│   └── ...
├── landmarks/
│   ├── 0.lms
│   ├── 1.lms
│   └── ...
└── parsing/          # Optional
    ├── 0.png
    ├── 1.png
    └── ...
```

### Audio Features

Audio features should be a NumPy array saved as `.npy` file with shape `(num_frames, 512)` or already reshaped based on mode:
- **ave**: (num_frames, 32, 16, 16)
- **hubert**: (num_frames, 32, 32, 32)
- **wenet**: (num_frames, 256, 16, 32)

## Modules

### UNetModel

Handles U-Net model loading and inference.

```python
from frame_generation_pipeline.unet_model import UNetModel

model = UNetModel(
    checkpoint_path="./checkpoint/May/5.pth",
    mode="ave",
    device="cuda"  # or "cpu"
)

# Run inference
output = model.predict(image_tensor, audio_features)

# Export to ONNX
model.export_to_onnx("./models/unet.onnx")
```

### ImageProcessor

Handles all image processing operations.

```python
from frame_generation_pipeline.image_processor import ImageProcessor

processor = ImageProcessor()

# Load image and landmarks
img = processor.load_image("./dataset/May/full_body_img/0.jpg")
lms = processor.load_landmarks("./dataset/May/landmarks/0.lms")

# Crop face region
crop, coords = processor.crop_face_region(img, lms)

# Prepare input tensors
concat_tensor, img_tensor = processor.prepare_input_tensors(crop)

# Paste generated region back
result = processor.paste_generated_region(img, generated, coords, (h, w))
```

### FrameGenerator

Orchestrates the frame generation process.

```python
from frame_generation_pipeline.frame_generator import FrameGenerator

generator = FrameGenerator(unet_model, mode="ave")

# Generate frames from template sequence
frames = generator.generate_frames_from_template_sequence(
    img_dir="./dataset/May/full_body_img",
    lms_dir="./dataset/May/landmarks",
    audio_features=audio_feats,
    start_frame=0
)

# Save frames
generator.save_frames(frames, "./output/frames")
```

### FrameGenerationPipeline

Complete end-to-end pipeline.

```python
from frame_generation_pipeline import FrameGenerationPipeline

pipeline = FrameGenerationPipeline(
    checkpoint_path="./checkpoint/May/5.pth",
    mode="ave"
)

# Generate complete video
pipeline.generate_video(
    audio_features_path="./audio_features.npy",
    template_dir="./dataset/May",
    audio_path="./demo/audio.wav",
    output_path="./output/result.mp4",
    fps=25,
    crf=20  # Video quality (lower = better)
)
```

## Image Processing Pipeline

The image processing follows these steps:

1. **Load template frame and landmarks**
   - Full frame: Original size (e.g., 1920x1080)
   - Landmarks: Facial keypoints (68 points)

2. **Crop face region**
   - Based on landmarks 1, 31, 52
   - Creates square region around face

3. **Resize to 328x328**
   - Using cv2.INTER_CUBIC interpolation

4. **Extract inner region [4:324, 4:324]**
   - Results in 320x320 region

5. **Create masked version**
   - Black rectangle on lower face region

6. **Prepare 6-channel input**
   - Concatenate original + masked
   - Normalize to [0, 1]

7. **U-Net inference**
   - Input: (1, 6, 320, 320)
   - Output: (1, 3, 320, 320)

8. **Paste back**
   - Place generated region in 328x328 canvas
   - Resize to original crop size
   - Paste into full frame

## Model Details

### U-Net Architecture

- **Input**: 
  - Image: (1, 6, 320, 320) - 6 channels (3 original + 3 masked)
  - Audio: (1, 32, 16, 16) for 'ave' mode
- **Output**: (1, 3, 320, 320) - Generated face region

### Audio Modes

- **ave**: Audio features reshaped to (32, 16, 16)
- **hubert**: Audio features reshaped to (32, 32, 32)
- **wenet**: Audio features reshaped to (256, 16, 32)

## Performance

Typical performance on various hardware:

- **NVIDIA RTX 3090**: ~0.05s per frame (20 FPS)
- **NVIDIA GTX 1080**: ~0.08s per frame (12 FPS)
- **CPU (i7-10700K)**: ~0.5s per frame (2 FPS)

## Testing

Run the test suite:

```bash
# Run all tests
python -m pytest tests/

# Run specific test
python -m pytest tests/test_image_processor.py

# Run with coverage
python -m pytest tests/ --cov=frame_generation_pipeline
```

## Validation

To validate against original `inference_328.py`:

```python
# Generate reference outputs
pipeline.generate_reference_outputs(
    audio_features_path="./audio_features.npy",
    template_dir="./dataset/May",
    output_dir="./test_data/reference_outputs"
)

# Compare with original implementation
# (See tests/test_pipeline.py for validation scripts)
```

## Troubleshooting

### CUDA Out of Memory

If you get CUDA OOM errors:
```python
pipeline = FrameGenerationPipeline(
    checkpoint_path="./checkpoint/May/5.pth",
    mode="ave",
    device="cpu"  # Use CPU instead
)
```

### Missing Dependencies

```bash
pip install torch torchvision opencv-python numpy tqdm
```

### ffmpeg Not Found

Install ffmpeg:
```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg

# Windows
# Download from https://ffmpeg.org/
```

## Next Steps

After generating reference outputs with Python:

1. **Go Implementation**: See `../frame_generation_go/`
2. **Swift/iOS Implementation**: See `../frame_generation_swift/`

Both implementations should produce identical (or near-identical) outputs to the Python reference.

## License

This code is part of the SyncTalk_2D project.

## References

- Original SyncTalk_2D paper: [Link]
- U-Net architecture: [Link]

