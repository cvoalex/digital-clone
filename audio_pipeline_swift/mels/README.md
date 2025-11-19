# Mels - Audio Pipeline macOS App

A macOS application for testing the SyncTalk_2D audio pipeline mel spectrogram processor.

## Features

✅ **Mel Spectrogram Processing**
- Pure Swift implementation using Accelerate framework
- Compatible with Python/Go implementations
- Real-time processing feedback

✅ **User Interface**
- Drag & drop or file picker for audio files
- Real-time processing status
- Results display with shape and value range
- Test button for demo audio

## Usage

### 1. Build and Run

Open `mels.xcodeproj` in Xcode and run the app (Cmd+R)

### 2. Process Audio

**Option A: Use Test Audio**
- Click "Test with Demo Audio" to process reference audio from Python tests

**Option B: Select Your Own**
1. Click "Select Audio File"
2. Choose a WAV file (16kHz recommended)
3. Click "Process Audio"

### 3. View Results

The app displays:
- Processing status
- Mel spectrogram shape (80, n_frames)
- Value range (should be around [-4, 4])

## Files

- **`MelProcessor.swift`**: Complete mel spectrogram processor
  - Pre-emphasis filter
  - STFT using vDSP
  - Mel filterbank
  - dB conversion and normalization
  
- **`ContentView.swift`**: User interface
  - Audio file selection
  - Processing pipeline
  - Results display

## Technical Details

### Mel Processor Parameters

```swift
Sample Rate: 16,000 Hz
N-FFT: 800
Hop Length: 200 (12.5ms)
Window Length: 800 (50ms)
Mel Bands: 80
Frequency Range: 55-7600 Hz
Pre-emphasis: 0.97
Normalization Range: [-4, 4]
```

### Processing Pipeline

```
Audio File (WAV)
    ↓
AVFoundation (load & resample to 16kHz)
    ↓
Pre-emphasis Filter
    ↓
STFT (vDSP FFT)
    ↓
Mel Filterbank
    ↓
Amplitude to dB
    ↓
Normalize to [-4, 4]
    ↓
Mel Spectrogram (80, n_frames)
```

## Validation

Compare Swift output with Python/Go:

Expected results for 2-second test audio:
- Shape: approximately (80, 161)
- Range: [-4.0, ~2.0]

### Python Comparison

```bash
# Process same audio in Python
cd ../../audio_pipeline
python3 -c "
from mel_processor import MelSpectrogramProcessor
proc = MelSpectrogramProcessor()
mel = proc.process_file('test_data/reference_audio.wav')
print(f'Shape: {mel.shape}')
print(f'Range: [{mel.min():.3f}, {mel.max():.3f}]')
"
```

Compare the shape and range values!

## Next Steps

1. ✅ Mel processor working
2. 🔄 Add ONNX Runtime for AudioEncoder
3. 🔄 Complete pipeline integration
4. 🔄 Validate against Python outputs
5. 🔄 Port to iOS

## Requirements

- macOS 13.0+
- Xcode 15.0+
- Audio files: WAV format, 16kHz mono recommended

## Troubleshooting

**"Audio format error"**
- Ensure audio is WAV format
- Try converting: `ffmpeg -i input.mp3 -ar 16000 -ac 1 output.wav`

**"Processing takes long time"**
- Normal for large files (60s audio ~ 2-3 seconds)
- Progress shown in status display

**"Values out of range"**
- Should be approximately [-4, 4]
- Slight variations are normal

## Credits

Based on SyncTalk_2D audio processing pipeline with implementations in Python and Go.

---

**Status**: ✅ Mel Processor Complete  
**Next**: ONNX Runtime Integration

