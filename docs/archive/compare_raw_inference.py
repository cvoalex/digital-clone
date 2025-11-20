#!/usr/bin/env python3
"""
Compare raw ONNX inference outputs between Python and what Go should get
This tests just the model inference, not the image processing
"""

import numpy as np
import onnxruntime as ort
from PIL import Image

def load_image_as_tensor(path, normalize=True):
    """Load image and convert to CHW RGB tensor"""
    img = Image.open(path).convert('RGB')
    arr = np.array(img, dtype=np.float32)
    
    # HWC to CHW
    tensor = arr.transpose(2, 0, 1)
    
    if normalize:
        tensor = tensor / 255.0
    
    return tensor

print("=" * 60)
print("Testing Raw ONNX Inference (Model Output Only)")
print("=" * 60)

sanders_dir = "model/sanders_full_onnx"

# Load ONNX model
print("\nLoading ONNX model...")
model_path = f"{sanders_dir}/models/generator.onnx"
session = ort.InferenceSession(model_path)
print("✓ Model loaded")

# Load audio features
audio_feats = np.load(f"{sanders_dir}/aud_ave.npy")

# Test frame 1
print("\n" + "=" * 60)
print("Testing Frame 1")
print("=" * 60)

# Load pre-cut frames
roi_path = f"{sanders_dir}/rois_320/1.jpg"
masked_path = f"{sanders_dir}/model_inputs/1.jpg"

print(f"\nLoading: {roi_path}")
roi_img = Image.open(roi_path)
print(f"  ROI size: {roi_img.size}")

print(f"Loading: {masked_path}")
masked_img = Image.open(masked_path)
print(f"  Masked size: {masked_img.size}")

# Convert to tensors
roi_tensor = load_image_as_tensor(roi_path, normalize=True)
masked_tensor = load_image_as_tensor(masked_path, normalize=True)

print(f"\nTensor shapes:")
print(f"  ROI: {roi_tensor.shape}")
print(f"  Masked: {masked_tensor.shape}")

# Concatenate
input_tensor = np.concatenate([roi_tensor, masked_tensor], axis=0)
input_tensor = input_tensor[np.newaxis, ...]  # Add batch dim

print(f"  Input: {input_tensor.shape}")
print(f"  Input range: [{input_tensor.min():.3f}, {input_tensor.max():.3f}]")

# Prepare audio
audio_feat = audio_feats[0]  # Frame 1 (0-indexed)
print(f"\nAudio features:")
print(f"  Shape: {audio_feat.shape}")
print(f"  Range: [{audio_feat.min():.3f}, {audio_feat.max():.3f}]")

# Reshape audio - need to match (1, 32, 16, 16) = 8192
# We have 512 values
audio_reshaped = np.tile(audio_feat, 16)[:8192].reshape(1, 32, 16, 16).astype(np.float32)
print(f"  Reshaped: {audio_reshaped.shape}")
print(f"  Range: [{audio_reshaped.min():.3f}, {audio_reshaped.max():.3f}]")

# Run inference
print("\nRunning ONNX inference...")
outputs = session.run(None, {
    'input': input_tensor.astype(np.float32),
    'audio': audio_reshaped
})

output_tensor = outputs[0][0]  # Remove batch dim
print(f"✓ Inference complete")
print(f"  Output shape: {output_tensor.shape}")
print(f"  Output range: [{output_tensor.min():.3f}, {output_tensor.max():.3f}]")

# Scale to 0-255
output_scaled = output_tensor * 255.0
print(f"  Scaled range: [{output_scaled.min():.2f}, {output_scaled.max():.2f}]")

# Save raw output for inspection
print("\nSaving raw model output...")
output_img = output_tensor.transpose(1, 2, 0)  # CHW to HWC
output_img = np.clip(output_img * 255, 0, 255).astype(np.uint8)
Image.fromarray(output_img).save("raw_output_frame1.jpg")
print("✓ Saved raw_output_frame1.jpg")

print("\n" + "=" * 60)
print("This is what the model outputs directly (320x320)")
print("Before pasting into full frame")
print("=" * 60)
print("\nGo should produce the same raw output if using:")
print("  - Same input tensors")
print("  - Same audio reshaping")
print("  - Same ONNX model")

