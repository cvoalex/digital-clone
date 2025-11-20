#!/usr/bin/env python3
"""
Debug audio tensor differences between implementations
"""

import numpy as np
import struct

def load_swift_audio_features():
    """Load audio features from Swift's processing"""
    # Swift should be saving these to the audio encoder output
    # For now, let's check what Go is using
    go_audio = np.load('model/sanders_full_onnx/aud_ave.npy')
    print(f"Go/Python audio features:")
    print(f"  Shape: {go_audio.shape}")
    print(f"  Range: [{go_audio.min():.3f}, {go_audio.max():.3f}]")
    print(f"  Mean: {go_audio.mean():.3f}")
    print(f"  Std: {go_audio.std():.3f}")
    print()
    
    # Check first few frames
    print("First 3 frames (first 10 values):")
    for i in range(3):
        print(f"  Frame {i}: {go_audio[i][:10]}")
    print()
    
    return go_audio

def check_reshaping():
    """Check how audio is reshaped to (32, 16, 16)"""
    features = np.load('model/sanders_full_onnx/aud_ave.npy')
    
    print("Checking reshape from (512) to (32, 16, 16):")
    print()
    
    # This is what Go/Python do - tile/repeat to fill 8192
    frame_feat = features[0]  # (512,)
    print(f"Original frame features: {frame_feat.shape}")
    print(f"  First 10: {frame_feat[:10]}")
    print()
    
    # Reshape by tiling
    target_size = 32 * 16 * 16  # 8192
    tiled = np.tile(frame_feat, (target_size // 512) + 1)[:target_size]
    reshaped = tiled.reshape(1, 32, 16, 16)
    
    print(f"After tiling and reshape: {reshaped.shape}")
    print(f"  Range: [{reshaped.min():.3f}, {reshaped.max():.3f}]")
    print(f"  First 10 of flattened: {reshaped.flatten()[:10]}")
    print()
    
    # Check if all non-zero
    non_zero = np.count_nonzero(reshaped)
    print(f"Non-zero values: {non_zero} / {reshaped.size}")
    print(f"All zeros would mean no audio signal!")
    print()
    
    return reshaped

def main():
    print("=" * 60)
    print("Audio Tensor Debugging")
    print("=" * 60)
    print()
    
    go_audio = load_swift_audio_features()
    reshaped = check_reshaping()
    
    print("=" * 60)
    print("What to check in Swift:")
    print("=" * 60)
    print()
    print("1. Are audio features loaded correctly?")
    print("   - Should have ~1042-1117 frames")
    print("   - Each frame should be (512,) floats")
    print("   - Range should be [0, ~10]")
    print()
    print("2. Is reshaping correct?")
    print("   - Should tile 512 → 8192 values")
    print("   - Then reshape to (1, 32, 16, 16)")
    print("   - Values should match the tiled pattern")
    print()
    print("3. Is the audio tensor being passed to Core ML?")
    print("   - Input name: 'audio_input'")
    print("   - Shape: [1, 32, 16, 16]")
    print("   - Check it's not all zeros!")
    print()
    print("4. Add logging in Swift:")
    print("   - Print audio features count")
    print("   - Print first frame features (first 10 values)")
    print("   - Print reshaped tensor (first 10 values)")
    print("   - Check if values match Go/Python")

if __name__ == '__main__':
    main()

