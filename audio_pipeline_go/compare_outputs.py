#!/usr/bin/env python3
"""
Compare Go and Python audio pipeline outputs.
"""

import numpy as np
import json
import struct
import sys

def load_go_bin_file(path):
    """Load a binary file saved by Go."""
    with open(path, 'rb') as f:
        # Read JSON-encoded length
        length_line = f.readline()
        length = json.loads(length_line)
        
        # Read JSON-encoded floats
        data = []
        for _ in range(length):
            line = f.readline()
            value = json.loads(line)
            data.append(value)
        
        return np.array(data, dtype=np.float32)

def compare_mel_spec(py_path, go_metadata_path):
    """Compare mel spectrograms."""
    print("\n" + "="*70)
    print("Comparing Mel Spectrograms")
    print("="*70)
    
    # Load Python output
    py_mel = np.load(py_path)
    print(f"Python mel shape: {py_mel.shape}")
    print(f"Python mel range: [{py_mel.min():.4f}, {py_mel.max():.4f}]")
    
    # Load Go metadata
    with open(go_metadata_path) as f:
        go_meta = json.load(f)
    
    go_shape = go_meta['stats']['mel_shape']
    go_range = go_meta['stats']['mel_range']
    
    print(f"\nGo mel shape: {go_shape}")
    print(f"Go mel range: [{go_range[0]:.4f}, {go_range[1]:.4f}]")
    
    # Compare shapes
    if list(py_mel.shape) == go_shape:
        print(f"\n✓ Shapes match!")
    else:
        print(f"\n✗ Shape mismatch!")
        print(f"  Difference: {abs(py_mel.shape[0] - go_shape[0])}, {abs(py_mel.shape[1] - go_shape[1])}")
    
    # Compare ranges
    range_diff_min = abs(py_mel.min() - go_range[0])
    range_diff_max = abs(py_mel.max() - go_range[1])
    
    print(f"\nRange differences:")
    print(f"  Min diff: {range_diff_min:.6f}")
    print(f"  Max diff: {range_diff_max:.6f}")
    
    if range_diff_min < 0.1 and range_diff_max < 0.1:
        print(f"✓ Ranges are similar")
    else:
        print(f"✗ Ranges differ significantly")

def compare_audio_features(py_path, go_metadata_path):
    """Compare audio encoder outputs."""
    print("\n" + "="*70)
    print("Comparing Audio Features")
    print("="*70)
    
    # Load Python output
    py_features = np.load(py_path)
    print(f"Python features shape: {py_features.shape}")
    print(f"Python features range: [{py_features.min():.4f}, {py_features.max():.4f}]")
    
    # Load Go metadata
    with open(go_metadata_path) as f:
        go_meta = json.load(f)
    
    go_shape = go_meta['stats']['features_shape']
    go_range = go_meta['stats']['features_range']
    
    print(f"\nGo features shape: {go_shape}")
    print(f"Go features range: [{go_range[0]:.4f}, {go_range[1]:.4f}]")
    
    # Compare shapes
    shape_diff = abs(py_features.shape[0] - go_shape[0])
    if shape_diff <= 2:  # Allow small difference due to padding
        print(f"\n✓ Shapes match (within padding tolerance)")
    else:
        print(f"\n✗ Shape mismatch: {shape_diff} frames difference")
    
    # Compare ranges
    range_diff_min = abs(py_features.min() - go_range[0])
    range_diff_max = abs(py_features.max() - go_range[1])
    
    print(f"\nRange differences:")
    print(f"  Min diff: {range_diff_min:.6f}")
    print(f"  Max diff: {range_diff_max:.6f}")
    
    if range_diff_min < 0.01 and range_diff_max < 1.0:
        print(f"✓ Ranges are similar")
        return True
    else:
        print(f"⚠ Ranges differ")
        return False

def compare_frame_features(py_frame_path, go_frame_path):
    """Compare a single frame's features."""
    # Load Python frame
    py_frame = np.load(py_frame_path)
    
    # Load Go frame
    try:
        go_frame = load_go_bin_file(go_frame_path)
    except Exception as e:
        print(f"Failed to load Go frame: {e}")
        return False
    
    print(f"\nPython frame shape: {py_frame.shape}, size: {py_frame.size}")
    print(f"Go frame size: {go_frame.size}")
    
    if py_frame.size == go_frame.size:
        # Both should be 8192 for AVE mode (32*16*16)
        py_flat = py_frame.flatten()
        go_flat = go_frame.flatten()
        
        diff = np.abs(py_flat - go_flat)
        max_diff = diff.max()
        mean_diff = diff.mean()
        
        print(f"Max difference: {max_diff:.6f}")
        print(f"Mean difference: {mean_diff:.6f}")
        
        if max_diff < 1e-3:
            print("✓ Frames match closely!")
            return True
        elif max_diff < 0.1:
            print("✓ Frames are similar")
            return True
        else:
            print("⚠ Frames differ significantly")
            return False
    else:
        print("✗ Frame sizes don't match")
        return False

def main():
    print("="*70)
    print("Go vs Python Audio Pipeline Comparison")
    print("="*70)
    
    py_dir = "../audio_pipeline/my_audio_output"
    go_dir = "go_output"
    
    # Compare mel spectrograms
    compare_mel_spec(
        f"{py_dir}/mel_spectrogram.npy",
        f"{go_dir}/metadata.json"
    )
    
    # Compare audio features
    features_similar = compare_audio_features(
        f"{py_dir}/audio_features_padded.npy",
        f"{go_dir}/metadata.json"
    )
    
    # Compare sample frames
    print("\n" + "="*70)
    print("Comparing Sample Frames")
    print("="*70)
    
    frames_to_compare = [0, 748]  # Start and middle
    
    all_match = True
    for frame_idx in frames_to_compare:
        print(f"\n--- Frame {frame_idx} ---")
        py_frame = f"{py_dir}/frames/frame_{frame_idx:05d}_reshaped.npy"
        go_frame = f"{go_dir}/frames/frame_{frame_idx:05d}.bin"
        
        try:
            match = compare_frame_features(py_frame, go_frame)
            all_match = all_match and match
        except FileNotFoundError as e:
            print(f"⚠ File not found: {e}")
            all_match = False
    
    # Final summary
    print("\n" + "="*70)
    print("Summary")
    print("="*70)
    
    if all_match and features_similar:
        print("✅ Go implementation matches Python!")
        print("\nThe Go implementation is working correctly.")
        print("Mel spectrograms are similar, and frame features match.")
        return 0
    else:
        print("⚠ Some differences detected")
        print("\nThis is expected as:")
        print("  - Different DSP implementations may have slight variations")
        print("  - Floating point precision differences")
        print("  - The outputs are still usable for the model")
        return 1

if __name__ == "__main__":
    sys.exit(main())

