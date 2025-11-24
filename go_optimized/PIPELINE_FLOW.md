# Go Optimized Pipeline: Detailed Execution Flow

This document details the exact step-by-step execution flow of the `go_optimized` frame generation pipeline, from raw input to final video frame.

## 1. Initialization Phase

Before any processing begins, the system sets up parallel resources.

1. **Resource Allocation**
   - Detects CPU cores (e.g., 8 cores on M1 Pro).
   - Sets `GOMAXPROCS` to utilize all cores.

2. **ONNX Session Pooling (The Parallel Engine)**
   - **Audio Encoder Pool:** Creates 1 ONNX session (sequential processing is sufficient).
   - **Generator Pool:** Creates **N** ONNX sessions (where N = CPU cores).
   - *Why:* ONNX Runtime is thread-safe but has internal locks. Separate sessions allow true parallel inference.

3. **Memory Pooling (Zero-Allocation Strategy)**
   - Pre-allocates `sync.Pool` for:
     - `[]float32` tensors (6-channel input, 3-channel output).
     - `*image.RGBA` buffers (for loading and resizing).
   - *Benefit:* Eliminates Garbage Collector (GC) pauses during high-speed generation.

4. **Data Loading**
   - Loads `crop_rectangles.json` (coordinates for where the face is in each frame).
   - Initializes `TensorCache` (disk-based cache for converted image tensors).

---

## 2. Audio Processing Stage

**Input:** Raw WAV file (e.g., `audio.wav`)  
**Output:** Audio Feature Tensors `[N, 512]`

1. **Load WAV**
   - Decodes WAV header.
   - Reads 16-bit PCM integer samples.
   - Converts to `float64` normalized to `[-1.0, 1.0]`.

2. **Mel Spectrogram Generation (`pkg/mel`)**
   - **Pre-emphasis:** Applies filter `y[t] = x[t] - 0.97 * x[t-1]` (boosts high frequencies).
   - **STFT:** Short-Time Fourier Transform using Hann window (800 size, 200 hop).
   - **Mel Filterbank:** Maps linear frequencies to 80 Mel bands (55Hz - 7600Hz).
   - **Log Amplitude:** Converts to decibels (dB).
   - **Normalization:** Clips and scales values to range `[-4.0, 4.0]`.

3. **Feature Encoding (AudioEncoder Model)**
   - **Windowing:** Extracts 16-frame overlapping windows from the Mel spectrogram.
   - **Tensor Creation:** Creates ONNX tensor shape `[1, 1, 80, 16]`.
   - **Inference:** Runs `audio_encoder.onnx` to produce a 512-dimensional feature vector.
   - **Result:** A sequence of 512-float vectors, one per video frame.

---

## 3. Frame Generation Stage (Parallel)

**Input:** Audio Features, Template Images  
**Output:** Final Video Frame (1280x720)

This stage runs in **parallel batches** (default batch size: 15 frames).

### For Each Frame (running on a worker thread):

1. **Select Template Frame**
   - Determines which template frame (1-250) corresponds to the current video timestamp.
   - Paths:
     - `rois_320/{i}.jpg` (Cropped face, 320x320)
     - `model_inputs/{i}.jpg` (Masked face, 320x320, lower half black)
     - `full_body_img/{i}.jpg` (Full 1280x720 frame)

2. **Prepare Visual Input (Tensor Construction)**
   - **Check Cache:** Does `cache/go_tensors/{i}.tensor` exist?
     - **YES:** Load binary float32 data directly (Instant).
     - **NO:** Load JPEG → Decode to RGB → Convert to BGR Planar Tensor → Save to Cache.
   - **Tensor Structure:** 
     - Shape: `[1, 6, 320, 320]`
     - First 3 channels: ROI image (BGR, 0-1 normalized)
     - Next 3 channels: Masked image (BGR, 0-1 normalized)

3. **Prepare Audio Input**
   - Takes the 512-dimension vector for the current frame.
   - **Tiling:** Repeats the vector 16 times to fill 8192 elements (`512 * 16`).
   - **Reshape:** `[1, 32, 16, 16]` tensor.

4. **U-Net Inference (The Core Generation)**
   - **Acquire Session:** Grabs a free generator session from the pool.
   - **Run Model:** `generator.onnx` takes Visual Tensor + Audio Tensor.
   - **Output:** `[1, 3, 320, 320]` tensor (Generated mouth region).
   - **Release Session:** Returns session to pool for other workers.

5. **Post-Processing**
   - **Tensor to Image:**
     - Converts Planar BGR floats (`0.0-1.0`) back to Interleaved RGB bytes (`0-255`).
     - Creates a 320x320 `*image.RGBA`.

6. **Compositing (The "Paste" Step)**
   - Loads the full body frame (1280x720).
   - lookups up crop coordinates `[x1, y1, x2, y2]` from JSON.
   - **Bilinear Resizing (Crucial Step):**
     - The target face region on the full body might be e.g., `182x182` (not exactly 320x320).
     - Resizes the generated 320x320 mouth image to fit the target rect using **Bilinear Interpolation**.
     - *Note:* Originally used Nearest Neighbor, which caused pixelation.
   - Pastes the resized mouth onto the full body frame.

7. **Save Output**
   - Encodes final image to JPEG.
   - Saves to output directory (e.g., `frame_00001.jpg`).

---

## Data Flow Summary

```mermaid
graph TD
    A[Audio WAV] -->|Mel Processor| B[Mel Spectrogram]
    B -->|Windowing| C[ONNX Audio Encoder]
    C -->|512 Features| D[Audio Tensor 32x16x16]
    
    E[Template ROI] -->|Cache/Convert| F[Visual Tensor 6x320x320]
    G[Template Masked] -->|Cache/Convert| F
    
    D --> H{ONNX Generator U-Net}
    F --> H
    
    H -->|Output Tensor| I[Generated Mouth 320x320]
    I -->|Bilinear Resize| J[Resized Mouth]
    
    K[Full Body Frame] --> L[Compositor]
    J --> L
    L --> M[Final JPEG]
```

