# Comprehensive Prompt: Video Frame Generation Pipeline

## Project Goal

Create a standalone, cross-platform video frame generation pipeline for SyncTalk_2D that can run independently in Python, Go, and iOS/macOS - with NO Python dependencies at runtime for Go/Swift implementations.

## Context

The SyncTalk_2D system generates lip-sync videos by:
1. Processing audio → audio features (DONE - we have this working!)
2. **Taking those audio features + a template image → generating video frames** (THIS IS THE NEXT TASK)

The current implementation is in `inference_328.py` which:
- Loads a template video/images
- Loads facial landmarks
- Takes audio features (the tensors we just created)
- Feeds them through a U-Net model
- Generates mouth/lip regions frame-by-frame
- Pastes generated regions back into template frames
- Outputs a complete video

## What We Need

Build three implementations of the **frame generation pipeline**:

### 1. Python Implementation (Reference)
- Extract the frame generation code from `inference_328.py`
- Create a standalone module in `frame_generation_pipeline/`
- Input: Audio features (32, 16, 16) tensors + template images/landmarks
- Output: Generated video frames
- Include:
  - U-Net model loading and inference
  - Template image/landmark loading
  - Face region cropping and masking
  - Frame generation loop
  - Paste-back logic
  - Video assembly with ffmpeg
- Generate reference outputs for validation
- Comprehensive tests
- Full documentation

### 2. Go Implementation
- Export U-Net model to ONNX format
- Implement in Go using ONNX Runtime
- Replicate exact image processing (cropping, resizing, masking)
- Match Python output for validation
- Build CLI tool
- NO Python at runtime

### 3. Swift/iOS/macOS Implementation
- Port image processing to Swift (use Core Image/Accelerate)
- Integrate U-Net model via ONNX Runtime or Core ML
- Create macOS test app with UI
- Validate against Python/Go outputs
- Pure Swift/native libraries only
- Ready for iOS deployment

## Technical Requirements

### Models to Export
- **U-Net model** (`unet_328.py`) → ONNX format
- Input: 6-channel image (328x328) + audio features (32, 16, 16)
- Output: Generated face region (320x320)

### Image Processing Steps
1. Load template frame and landmarks
2. Crop face region based on landmarks
3. Resize to 328x328
4. Create masked version (lower face blacked out)
5. Concatenate original + masked (6 channels)
6. Feed to U-Net with audio features
7. Get generated region (320x320)
8. Paste back into 328x328 crop
9. Resize and paste into full frame
10. Save frame

### Validation Criteria
- Frame-by-frame pixel comparison
- Max difference < 1e-3 (or 1e-2 for mobile)
- Video quality metrics
- Lip-sync accuracy
- Processing speed benchmarks

## Directory Structure

```
frame_generation_pipeline/          # Python implementation
├── __init__.py
├── unet_model.py                   # U-Net model wrapper
├── image_processor.py              # Image cropping/masking/pasting
├── frame_generator.py              # Main generation loop
├── pipeline.py                     # Complete pipeline
├── tests/
│   ├── test_image_processor.py
│   ├── test_frame_generator.py
│   └── test_pipeline.py
├── test_data/
│   └── reference_outputs/          # Generated reference frames
└── README.md

frame_generation_go/                # Go implementation
├── pkg/
│   ├── unet/                       # ONNX U-Net wrapper
│   ├── imageproc/                  # Image processing
│   └── generator/                  # Frame generation
├── cmd/generate/                   # CLI tool
├── models/
│   └── unet_328.onnx              # Exported model
└── README.md

frame_generation_swift/             # Swift/macOS/iOS
├── FrameGenerator/                 # Xcode project
│   ├── UNetModel.swift            # Model wrapper
│   ├── ImageProcessor.swift       # Core Image/vImage
│   ├── FrameGenerator.swift       # Generation logic
│   └── ContentView.swift          # UI
└── README.md
```

## Step-by-Step Process

### Phase 1: Python (Reference)
1. Read and understand `inference_328.py` 
2. Extract frame generation logic into modular components
3. Export U-Net model to ONNX
4. Create standalone pipeline that takes:
   - Audio features (from previous pipeline)
   - Template images/landmarks
   - Outputs: Video frames
5. Test with known inputs
6. Generate reference outputs
7. Document all parameters and transformations

### Phase 2: Go (Validation)
1. Implement image processing in Go:
   - Image loading/saving
   - Resizing (matching cv2.resize with INTER_CUBIC)
   - Cropping
   - Masking
   - Channel concatenation
2. Integrate ONNX U-Net model
3. Replicate frame generation loop
4. Validate outputs match Python pixel-by-pixel
5. Build CLI tool
6. Performance benchmarks

### Phase 3: Swift/iOS (Production)
1. Implement image processing with Core Image/Accelerate
2. Convert U-Net to Core ML OR use ONNX Runtime
3. Build macOS test app
4. Validate against Python/Go
5. Optimize for iOS
6. Create example iOS app
7. Test on actual device

## Key Considerations

### Image Processing
- **Resizing**: Must match cv2.INTER_CUBIC exactly
- **Color space**: BGR (OpenCV) vs RGB (Swift) - handle conversions
- **Data types**: uint8 vs float32 conversions
- **Normalization**: /255.0 for model input

### Model Integration
- **ONNX format**: Universal across platforms
- **Input shapes**: Validate exact dimensions
- **Output processing**: Handle model outputs correctly
- **Batch processing**: May need to process one frame at a time

### Performance
- **Python**: Baseline performance
- **Go**: Should match or exceed Python
- **Swift/iOS**: Optimize for mobile (Metal/Neural Engine)

### Testing Strategy
1. Test with single frame first
2. Compare intermediate outputs (cropped regions, masks, etc.)
3. Validate model outputs
4. Test complete video generation
5. Compare final videos visually and numerically

## Deliverables

### For Each Implementation:

1. **Source Code**
   - Modular, well-documented
   - Type hints/annotations
   - Error handling

2. **Tests**
   - Unit tests for each component
   - Integration tests for full pipeline
   - Validation against reference

3. **Documentation**
   - README with usage instructions
   - API documentation
   - Architecture overview
   - Performance benchmarks

4. **Reference Outputs**
   - Sample frames at key indices
   - Complete metadata
   - Validation instructions

## Success Criteria

✅ All three implementations produce identical (or near-identical) frames  
✅ Go implementation has NO Python runtime dependency  
✅ Swift implementation has NO Python runtime dependency  
✅ Performance is real-time capable on target platforms  
✅ Complete documentation for each implementation  
✅ Validated against original `inference_328.py` behavior  
✅ Ready for production deployment  

## Expected Timeline

- **Python extraction & modularization**: 2-3 hours
- **Go implementation**: 3-4 hours  
- **Swift/macOS implementation**: 4-5 hours
- **Validation & documentation**: 2 hours
- **Total**: ~12-15 hours of focused work

## Starting Point

Begin by reading:
1. `inference_328.py` - Current implementation
2. `unet_328.py` - U-Net model architecture
3. `utils.py` - Helper functions
4. Our completed audio pipeline implementations as reference

The goal is to replicate what we just did for audio processing, but for video frame generation - creating standalone, testable, cross-platform implementations with complete validation.

---

**Use this prompt to guide the complete implementation of the frame generation pipeline across all three platforms, ensuring consistency, validation, and zero Python dependencies for Go/Swift versions.**


