# Manual ONNX Runtime Setup - Complete Instructions

## ✅ Files Already Created:

1. `ONNXWrapper.swift` - Native ONNX C API wrapper
2. `AudioEncoder.swift` - Uses native ONNX (NO PYTHON!)
3. `mels-Bridging-Header.h` - C API bridge
4. `audio_encoder.onnx` - Model file
5. `onnxruntime-osx-universal2-1.16.3/` - ONNX Runtime library

## 🔧 Steps to Complete in Xcode (5 minutes):

### Step 1: Add ONNXWrapper.swift to Project

1. In Xcode, **Right-click on "mels" folder** (yellow folder icon)
2. Choose **"Add Files to mels..."**
3. Navigate to and select: `ONNXWrapper.swift`
4. Make sure **"Copy items if needed"** is CHECKED
5. Click **"Add"**

### Step 2: Add audio_encoder.onnx to Project

1. **Right-click on "mels" folder** again
2. Choose **"Add Files to mels..."**
3. Select: `audio_encoder.onnx`
4. Make sure **"Copy items if needed"** is CHECKED
5. Click **"Add"**

### Step 3: Configure Build Settings

**In Xcode:**

1. Click on the **blue "mels" project** icon (top of left sidebar)

2. Select the **"mels" target** (under TARGETS)

3. Click **"Build Settings"** tab

4. In the search box, type: **"Header Search"**

5. Double-click on **"Header Search Paths"**

6. Click **"+"**

7. Paste this path:
   ```
   $(PROJECT_DIR)/onnxruntime-osx-universal2-1.16.3/include
   ```

8. Press Enter

### Step 4: Add Library Search Path

1. Still in **"Build Settings"**

2. Clear search box, type: **"Library Search"**

3. Double-click on **"Library Search Paths"**

4. Click **"+"**

5. Paste:
   ```
   $(PROJECT_DIR)/onnxruntime-osx-universal2-1.16.3/lib
   ```

6. Press Enter

### Step 5: Configure Bridging Header

1. Still in **"Build Settings"**

2. Clear search, type: **"Bridging"**

3. Find **"Objective-C Bridging Header"**

4. Double-click the value field

5. Enter:
   ```
   mels/mels-Bridging-Header.h
   ```

6. Press Enter

### Step 6: Link the ONNX Runtime Library

1. Click on **"Build Phases"** tab

2. Expand **"Link Binary With Libraries"**

3. Click the **"+"** button

4. Click **"Add Other..."** → **"Add Files..."**

5. Navigate to:
   ```
   onnxruntime-osx-universal2-1.16.3/lib/
   ```

6. Select **`libonnxruntime.1.16.3.dylib`**

7. Click **"Open"**

### Step 7: Embed the Library

1. Still in **"Build Phases"**

2. Click **"+"** at the top left (next to filter box)

3. Choose **"New Copy Files Phase"**

4. In the dropdown, select **"Frameworks"**

5. Click **"+"** in this new section

6. Find and add **`libonnxruntime.1.16.3.dylib`**

7. Make sure **"Code Sign On Copy"** is CHECKED

### Step 8: Build!

Press **Cmd+R** to build and run!

## 🎯 What Will Happen:

When toggle is **OFF**:
- Pure Swift mel processor (already works!)
- Shows mel spectrogram shape & range

When toggle is **ON**:
- Complete pipeline with native ONNX
- Mel → ONNX AudioEncoder → Audio Features
- Shows both mel and feature results
- **NO PYTHON SUBPROCESS!**

## 📊 Expected Results:

**Mel Only:**
```
Shape: (80, 4797)
Range: [-4.000, 2.024]
```

**Full Pipeline:**
```
Mel: (80, 4797)
Features: (1498, 512)
Features Range: [0.000, 9.722]
```

## ❌ If Build Fails:

Check:
1. Bridging header path is correct
2. Header search paths include onnxruntime/include
3. Library is linked in Build Phases
4. Library is embedded in Copy Files Phase

## Alternative: Test Mel Processor First

You can test the mel processor RIGHT NOW (toggle OFF) without ONNX Runtime. Once that works, then add ONNX.

---

Follow these steps carefully and the complete native solution will work!

