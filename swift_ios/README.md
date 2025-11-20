# Frame Generator - iOS Implementation

**Native iOS app for lip-sync video generation using Core ML + Neural Engine**

## Overview

This is the iOS version of the frame generation pipeline, optimized for iPhone and iPad.

**Key differences from macOS:**
- Uses `UIImage` instead of `NSImage`
- SwiftUI-based interface
- Models bundled in app
- Optimized for mobile (Neural Engine + GPU)
- Battery-efficient

## Expected Performance

**On iPhone 14 Pro:**
- ~30-40 FPS with Neural Engine
- Real-time capable
- Low battery usage

**On iPad Pro (M1/M2):**
- ~40-50 FPS (similar to Mac)
- Desktop-class performance

## Setup

### 1. Create Xcode Project

1. Open Xcode
2. Create new iOS App
3. Add files from `swift_ios/` to project
4. Add Core ML models to project

### 2. Add Core ML Models

Copy these to your Xcode project:
- `../swift_inference/AudioEncoder.mlmodelc`
- `../swift_inference/Generator.mlmodelc`

**Important:** Add to app bundle (check "Copy Bundle Resources")

### 3. Configure

In Xcode project settings:
- **Deployment Target:** iOS 16.0+
- **Frameworks:** CoreML, Metal, Accelerate (auto-linked)
- **Capabilities:** None required

## Architecture

```
FrameGeneratorApp.swift      App entry point
ContentView.swift             SwiftUI interface
FrameGeneratorIOS.swift       Core ML inference engine
  ├── Audio encoder (Core ML)
  ├── Generator (Core ML)  
  ├── Metal GPU operations
  └── Neural Engine acceleration
```

## Usage

### In App:

1. Tap "Generate Demo Frames"
2. App loads Core ML models
3. Generates frames using Neural Engine
4. Displays results

### As Framework:

```swift
let generator = try FrameGeneratorIOS()

let frame = try generator.generateFrame(
    roiImage: roiImage,
    maskedImage: maskedImage,
    audioFeatures: audioFeatures
)
```

## Key iOS Optimizations

### 1. Neural Engine
- Automatic with `computeUnits = .all`
- 16-core ML accelerator on A15+
- 4-6 TOPs performance

### 2. Metal GPU
- Parallel tensor operations
- Zero-copy when possible
- Shared memory buffers

### 3. Battery Efficiency
- Neural Engine is power-efficient
- Batch processing reduces overhead
- Async/await for responsiveness

## Differences from macOS Version

| Feature | macOS | iOS |
|---------|-------|-----|
| Image type | NSImage | UIImage |
| Interface | Command-line | SwiftUI |
| Bundle | Separate | App bundle |
| Performance | 47 FPS | 30-40 FPS |
| Neural Engine | M1/M2 | A15+ |

## File Structure

```
swift_ios/
├── FrameGeneratorApp.swift       SwiftUI app
├── ContentView.swift              UI
├── FrameGeneratorIOS.swift        Generator (UIImage-based)
└── README.md                      This file
```

## Building

### For Simulator:
```bash
# From Xcode: Product → Destination → iPhone Simulator
# Build: Cmd+B
```

### For Device:
```bash
# From Xcode: Product → Destination → Your iPhone
# Requires: Apple Developer account + signing
# Build: Cmd+B
```

## Testing

1. Build and run in Xcode
2. Tap "Generate Demo Frames"
3. Check FPS reported
4. Verify frames display

## Production Integration

To use in production app:

1. **Bundle models** - Add .mlmodelc to app
2. **Add assets** - Template images, audio files
3. **Integrate generator** - Call from your code
4. **Handle errors** - Graceful fallbacks
5. **Test on device** - Verify performance

## Expected App Size

- Base app: ~5 MB
- Core ML models: ~30 MB (audio + generator)
- **Total: ~35 MB**

## Performance Tips

### For Best Performance:
- Use `.all` compute units (Neural Engine + GPU)
- Pre-load models at app launch
- Batch process when possible
- Use Metal for array operations

### For Best Battery:
- Use `.cpuAndNeuralEngine` (skip GPU)
- Process in background
- Lower frame rate if not real-time

## Limitations

- Requires iOS 16.0+ (for Neural Engine)
- ~35 MB app size (models)
- Needs ~200 MB RAM during processing

## Next Steps

1. Create Xcode project
2. Add Swift files
3. Bundle Core ML models
4. Add demo assets
5. Test on device
6. Optimize for your use case

---

**This iOS version brings desktop-class lip-sync generation to iPhone/iPad!** 📱🚀

