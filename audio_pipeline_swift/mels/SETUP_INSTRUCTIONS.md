# Complete Setup Instructions - Native ONNX Runtime

## 🎯 Goal: Get native ONNX Runtime working (NO PYTHON!)

The ONNX Runtime library has been downloaded. Now we need to add it to your Xcode project.

## Step-by-Step Instructions

### 1. Add ONNX Runtime Framework to Xcode

**In Xcode:**

1. Click on the **mels** project in the left sidebar (the blue icon at the top)

2. Select the **mels** target (under "Targets")

3. Go to the **"General"** tab

4. Scroll down to **"Frameworks, Libraries, and Embedded Content"**

5. Click the **"+"** button

6. Click **"Add Other..."** → **"Add Files..."**

7. Navigate to:
   ```
   /Users/alexanderrusich/Projects/digital-clone/audio_pipeline_swift/mels/onnxruntime-osx-universal2-1.16.3/lib/
   ```

8. You'll see `libonnxruntime.1.16.3.dylib` - **Select it**

9. Click **"Open"**

10. In the dropdown, change from "Do Not Embed" to **"Embed & Sign"**

### 2. Add Header Search Path

**In Xcode:**

1. Still in the **mels** target

2. Go to **"Build Settings"** tab

3. Search for: **"Header Search Paths"**

4. Double-click on the value field

5. Click **"+"**

6. Add:
   ```
   /Users/alexanderrusich/Projects/digital-clone/audio_pipeline_swift/mels/onnxruntime-osx-universal2-1.16.3/include
   ```

7. Make sure it says **"recursive"**

### 3. Add Bridging Header

**In Xcode:**

1. File → New → File

2. Choose **"Header File"**

3. Name it: **`mels-Bridging-Header.h`**

4. Click "Create"

5. Replace its contents with:
   ```objc
   #import <onnxruntime_c_api.h>
   ```

6. Save the file

### 4. Configure Bridging Header

**In Xcode:**

1. Select the **mels** project

2. Select the **mels** target

3. Go to **"Build Settings"**

4. Search for: **"Objective-C Bridging Header"**

5. Set the value to:
   ```
   mels/mels-Bridging-Header.h
   ```

### 5. Add Library Search Path

**In Xcode:**

1. Still in **"Build Settings"**

2. Search for: **"Library Search Paths"**

3. Add:
   ```
   /Users/alexanderrusich/Projects/digital-clone/audio_pipeline_swift/mels/onnxruntime-osx-universal2-1.16.3/lib
   ```

### 6. Link the Library

**In Xcode:**

1. Go to **"Build Phases"** tab

2. Expand **"Link Binary With Libraries"**

3. Click **"+"**

4. Click **"Add Other..."** → **"Add Files..."**

5. Navigate to the onnxruntime lib folder and add `libonnxruntime.1.16.3.dylib`

### 7. Build and Test

Press **Cmd+R** to build and run!

## ✅ What This Gives You

- ✅ Native ONNX Runtime (NO Python subprocess!)
- ✅ Full pipeline: Mel → ONNX → Features
- ✅ Fast inference on Mac
- ✅ Ready to validate against Python/Go

## Alternative: Simpler Approach

If the above seems complex, I can create a version that uses the system Python ONE TIME to pre-compute the audio features, then the Swift app can load them directly. This validates the mel processor works correctly.

Let me know if you need help with any of these steps!

