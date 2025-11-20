#!/usr/bin/env python3
"""
Simple script to generate a video from audio features and templates.

Usage:
    python generate_video.py \\
        --checkpoint ../checkpoint/May/5.pth \\
        --audio-features ../audio_pipeline/my_audio_output/audio_features_padded.npy \\
        --template ../dataset/May \\
        --audio ../demo/talk_hb.wav \\
        --output ../result/my_video.mp4
"""

import argparse
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from frame_generation_pipeline import FrameGenerationPipeline
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def main():
    parser = argparse.ArgumentParser(
        description='Generate video from audio features and templates',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to U-Net model checkpoint (.pth file)'
    )
    parser.add_argument(
        '--audio-features',
        type=str,
        required=True,
        help='Path to audio features (.npy file)'
    )
    parser.add_argument(
        '--template',
        type=str,
        required=True,
        help='Path to template directory (containing full_body_img/ and landmarks/)'
    )
    parser.add_argument(
        '--audio',
        type=str,
        required=True,
        help='Path to audio file (.wav)'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Path to save output video (.mp4)'
    )
    parser.add_argument(
        '--mode',
        type=str,
        default='ave',
        choices=['ave', 'hubert', 'wenet'],
        help='Audio feature mode'
    )
    parser.add_argument(
        '--fps',
        type=int,
        default=25,
        help='Frames per second (25 for ave/hubert, 20 for wenet)'
    )
    parser.add_argument(
        '--crf',
        type=int,
        default=20,
        help='Video quality (18-28, lower is better quality)'
    )
    parser.add_argument(
        '--start-frame',
        type=int,
        default=0,
        help='Starting frame index in template sequence'
    )
    parser.add_argument(
        '--use-parsing',
        action='store_true',
        help='Use parsing masks if available'
    )
    parser.add_argument(
        '--save-frames',
        action='store_true',
        help='Save individual frames'
    )
    parser.add_argument(
        '--frames-dir',
        type=str,
        default=None,
        help='Directory to save frames (if --save-frames)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        choices=['cpu', 'cuda'],
        help='Device to use (auto-detect if not specified)'
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        return 1
    
    if not os.path.exists(args.audio_features):
        print(f"Error: Audio features not found: {args.audio_features}")
        return 1
    
    if not os.path.exists(args.template):
        print(f"Error: Template directory not found: {args.template}")
        return 1
    
    if not os.path.exists(args.audio):
        print(f"Error: Audio file not found: {args.audio}")
        return 1
    
    # Create output directory
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    print("=" * 60)
    print("Frame Generation Pipeline")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Audio features: {args.audio_features}")
    print(f"Template: {args.template}")
    print(f"Audio: {args.audio}")
    print(f"Output: {args.output}")
    print(f"Mode: {args.mode}")
    print(f"FPS: {args.fps}")
    print(f"CRF: {args.crf}")
    print("=" * 60)
    
    # Initialize pipeline
    print("\n[1/3] Initializing pipeline...")
    pipeline = FrameGenerationPipeline(
        checkpoint_path=args.checkpoint,
        mode=args.mode,
        device=args.device
    )
    
    # Generate video
    print("\n[2/3] Generating video frames...")
    try:
        video_path = pipeline.generate_video(
            audio_features_path=args.audio_features,
            template_dir=args.template,
            audio_path=args.audio,
            output_path=args.output,
            start_frame=args.start_frame,
            fps=args.fps,
            use_parsing=args.use_parsing,
            crf=args.crf,
            save_frames=args.save_frames,
            frames_dir=args.frames_dir
        )
    except Exception as e:
        print(f"\nError generating video: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print("\n[3/3] Complete!")
    print("=" * 60)
    print(f"✓ Video saved to: {video_path}")
    
    # Get file size
    size_mb = os.path.getsize(video_path) / (1024 * 1024)
    print(f"✓ File size: {size_mb:.2f} MB")
    
    print("=" * 60)
    print("\nYou can now:")
    print(f"  • Open video: open {video_path}")
    print(f"  • Play video: ffplay {video_path}")
    print(f"  • Check video info: ffprobe {video_path}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

