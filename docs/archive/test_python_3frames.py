#!/usr/bin/env python3
"""
Test Python inference on same 3 frames as Go for comparison
Uses ONNX Runtime (not PyTorch) for fair comparison
"""

import os
import numpy as np
import onnxruntime as ort
from PIL import Image
import json

def load_image_as_tensor(path, normalize=True):
    """Load image and convert to CHW RGB tensor"""
    img = Image.open(path).convert('RGB')
    arr = np.array(img, dtype=np.float32)
    
    # HWC to CHW
    tensor = arr.transpose(2, 0, 1)
    
    if normalize:
        tensor = tensor / 255.0
    
    return tensor

def tensor_to_image(tensor):
    """Convert CHW tensor back to image"""
    # CHW to HWC
    arr = tensor.transpose(1, 2, 0)
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)

def paste_into_frame(full_frame_path, generated, rect):
    """Paste generated region into full frame"""
    full = Image.open(full_frame_path)
    full_arr = np.array(full)
    
    x1, y1, x2, y2 = rect
    
    # Resize generated to fit rect
    gen_resized = generated.resize((x2-x1, y2-y1), Image.BICUBIC)
    gen_arr = np.array(gen_resized)
    
    # Paste
    full_arr[y1:y2, x1:x2] = gen_arr
    
    return Image.fromarray(full_arr)

def main():
    print("=" * 60)
    print("Python ONNX Inference - 3 Frame Test")
    print("=" * 60)
    
    sanders_dir = "model/sanders_full_onnx"
    
    # Load ONNX model
    print("\n[1/3] Loading ONNX model...")
    model_path = f"{sanders_dir}/models/generator.onnx"
    session = ort.InferenceSession(model_path)
    print("✓ Model loaded")
    
    # Load crop rectangles
    with open(f"{sanders_dir}/cache/crop_rectangles.json") as f:
        crop_rects = json.load(f)
    
    # Load audio features
    print("\n[2/3] Loading audio features...")
    audio_feats = np.load(f"{sanders_dir}/aud_ave.npy")
    print(f"✓ Audio features loaded: {audio_feats.shape}")
    
    # Create output directory
    output_dir = "python_test_output"
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n[3/3] Generating 3 frames...")
    
    # Process frames 1, 2, 3
    for i in range(1, 4):
        print(f"Processing frame {i}/3...")
        
        # Load pre-cut frames
        roi_path = f"{sanders_dir}/rois_320/{i}.jpg"
        masked_path = f"{sanders_dir}/model_inputs/{i}.jpg"
        full_path = f"{sanders_dir}/full_body_img/{i}.jpg"
        
        # Load and convert to tensors
        roi_tensor = load_image_as_tensor(roi_path, normalize=True)
        masked_tensor = load_image_as_tensor(masked_path, normalize=True)
        
        # Concatenate into 6-channel input
        input_tensor = np.concatenate([roi_tensor, masked_tensor], axis=0)
        input_tensor = input_tensor[np.newaxis, ...]  # Add batch dim
        
        # Get audio features for this frame
        audio_idx = i - 1
        audio_feat = audio_feats[audio_idx]
        
        # Reshape audio to (1, 32, 16, 16) = 8192 values
        # But we only have 512, so we need to upsample/repeat
        audio_reshaped = np.tile(audio_feat, 16)[:8192].reshape(1, 32, 16, 16).astype(np.float32)
        
        # Run inference
        outputs = session.run(None, {
            'input': input_tensor.astype(np.float32),
            'audio': audio_reshaped
        })
        
        # Get output (1, 3, 320, 320)
        output_tensor = outputs[0][0] * 255.0  # Scale to 0-255
        
        # Convert to image
        generated = tensor_to_image(output_tensor)
        
        # Get crop rectangle
        rect_key = str(i-1)  # 0-indexed
        rect = crop_rects[rect_key]['rect']
        
        # Paste into full frame
        final_frame = paste_into_frame(full_path, generated, rect)
        
        # Save
        output_path = f"{output_dir}/frame_{i:05d}.jpg"
        final_frame.save(output_path, quality=95)
        print(f"  ✓ Saved {output_path}")
    
    print("\n" + "=" * 60)
    print("✓ Python inference complete!")
    print("=" * 60)
    print(f"\nOutput: {output_dir}/")
    print("\nTo compare with Go output:")
    print("  diff -r python_test_output/ simple_inference_go/output_test/")
    print("\nTo create video:")
    print(f"  ffmpeg -framerate 25 -i {output_dir}/frame_%05d.jpg \\")
    print(f"    -i {sanders_dir}/aud.wav -t 0.12 \\")
    print(f"    -c:v libx264 -c:a aac -crf 20 \\")
    print(f"    {output_dir}/test_video.mp4 -y")

if __name__ == '__main__':
    main()

