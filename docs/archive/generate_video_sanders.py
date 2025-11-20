#!/usr/bin/env python3
"""
Generate video using the Sanders pre-packaged ONNX dataset.

This uses the complete pre-packaged dataset at model/sanders_full_onnx/ which includes:
- ONNX model (generator.onnx)
- Audio features (aud_ave.npy) 
- Template videos (full_body_video.mp4, crops_328_video.mp4, etc.)
- Landmarks (523 .lms files)
- Audio (aud.wav)

Usage:
    python generate_video_sanders.py --output result/sanders_video.mp4
"""

import argparse
import sys
import os
import cv2
import numpy as np
import subprocess
from pathlib import Path

# Add paths
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def extract_frames_from_video(video_path, output_dir, prefix="frame"):
    """Extract frames from video."""
    print(f"Extracting frames from {video_path}...")
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break
        output_path = os.path.join(output_dir, f"{i}.jpg")
        cv2.imwrite(output_path, frame)
    
    cap.release()
    print(f"  Extracted {frame_count} frames to {output_dir}")
    return frame_count

def main():
    parser = argparse.ArgumentParser(
        description='Generate video using Sanders pre-packaged dataset',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='result/sanders_video.mp4',
        help='Path to save output video'
    )
    parser.add_argument(
        '--fps',
        type=int,
        default=25,
        help='Frames per second'
    )
    parser.add_argument(
        '--crf',
        type=int,
        default=20,
        help='Video quality (18-28, lower is better)'
    )
    
    args = parser.parse_args()
    
    # Paths to pre-packaged data
    base_dir = "model/sanders_full_onnx"
    onnx_model = f"{base_dir}/models/generator.onnx"
    audio_features = f"{base_dir}/aud_ave.npy"
    audio_file = f"{base_dir}/aud.wav"
    full_body_video = f"{base_dir}/full_body_video.mp4"
    landmarks_dir = f"{base_dir}/landmarks"
    
    # Validate files exist
    required_files = [onnx_model, audio_features, audio_file, full_body_video]
    for f in required_files:
        if not os.path.exists(f):
            print(f"Error: Required file not found: {f}")
            return 1
    
    if not os.path.exists(landmarks_dir):
        print(f"Error: Landmarks directory not found: {landmarks_dir}")
        return 1
    
    print("=" * 60)
    print("Sanders Video Generation (Pre-packaged ONNX)")
    print("=" * 60)
    print(f"Model: {onnx_model}")
    print(f"Audio features: {audio_features}")
    print(f"Audio: {audio_file}")
    print(f"Template video: {full_body_video}")
    print(f"Output: {args.output}")
    print("=" * 60)
    
    # Create temporary directory for extracted frames
    temp_dir = "temp_sanders_frames"
    os.makedirs(temp_dir, exist_ok=True)
    
    print("\n[1/4] Extracting template frames...")
    num_frames = extract_frames_from_video(full_body_video, temp_dir)
    
    print("\n[2/4] Loading audio features...")
    audio_feats = np.load(audio_features)
    print(f"  Audio features shape: {audio_feats.shape}")
    print(f"  Expected frames: {audio_feats.shape[0]}")
    
    # Now use the frame generation pipeline
    print("\n[3/4] Setting up frame generation pipeline...")
    
    try:
        # Try using ONNX directly
        import onnxruntime as ort
        
        print(f"  Loading ONNX model: {onnx_model}")
        session = ort.InferenceSession(onnx_model)
        
        print("  Model inputs:")
        for input in session.get_inputs():
            print(f"    {input.name}: {input.shape}")
        
        print("\n  ✓ ONNX model loaded successfully!")
        print("\n  Note: Full frame generation implementation would go here.")
        print("  For now, using the existing Python pipeline...")
        
    except ImportError:
        print("  onnxruntime not installed, will use PyTorch...")
    
    # For now, let's just use the Python pipeline with the extracted data
    print("\n[4/4] Would generate video here...")
    print(f"\n  To complete this, you can use:")
    print(f"  python frame_generation_pipeline/generate_video.py \\")
    print(f"    --checkpoint [pytorch_checkpoint] \\")
    print(f"    --audio-features {audio_features} \\")
    print(f"    --template {temp_dir} \\")  
    print(f"    --audio {audio_file} \\")
    print(f"    --output {args.output}")
    
    print("\n" + "=" * 60)
    print("Setup complete!")
    print("=" * 60)
    print(f"\nYou have:")
    print(f"  ✓ ONNX model: {onnx_model}")
    print(f"  ✓ Audio features: {audio_features}")
    print(f"  ✓ Template frames: {temp_dir}/*.jpg ({num_frames} frames)")
    print(f"  ✓ Landmarks: {landmarks_dir}/*.lms (523 files)")
    print(f"  ✓ Audio: {audio_file}")
    print(f"\nReady for frame generation with Go or Swift!")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())

