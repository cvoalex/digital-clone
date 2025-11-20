"""
Frame Generation Pipeline

This module provides the complete end-to-end pipeline for generating
video frames from audio features and template images.
"""

import os
import cv2
import numpy as np
import subprocess
from typing import Optional, Dict, List
import logging
import json
from pathlib import Path

from .unet_model import UNetModel
from .frame_generator import FrameGenerator

logger = logging.getLogger(__name__)


class FrameGenerationPipeline:
    """
    Complete pipeline for generating video frames.
    
    This class provides a high-level interface for:
    1. Loading models and templates
    2. Processing audio features
    3. Generating frames
    4. Assembling video with audio
    
    Example usage:
        pipeline = FrameGenerationPipeline(
            checkpoint_path="./checkpoint/May/5.pth",
            mode="ave"
        )
        
        video_path = pipeline.generate_video(
            audio_features_path="./audio_features.npy",
            template_dir="./dataset/May",
            audio_path="./demo/audio.wav",
            output_path="./output/result.mp4"
        )
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        mode: str = 'ave',
        device: Optional[str] = None
    ):
        """
        Initialize the pipeline.
        
        Args:
            checkpoint_path: Path to U-Net model checkpoint
            mode: Audio feature mode ('ave', 'hubert', 'wenet')
            device: Device to run on (None = auto-detect)
        """
        self.mode = mode
        
        logger.info(f"Initializing FrameGenerationPipeline with mode={mode}")
        
        # Initialize U-Net model
        self.unet_model = UNetModel(checkpoint_path, mode=mode, device=device)
        
        # Initialize frame generator
        self.frame_generator = FrameGenerator(self.unet_model, mode=mode)
        
        logger.info("Pipeline initialized successfully")
    
    def generate_video(
        self,
        audio_features_path: str,
        template_dir: str,
        audio_path: str,
        output_path: str,
        start_frame: int = 0,
        fps: int = 25,
        use_parsing: bool = False,
        crf: int = 20,
        save_frames: bool = False,
        frames_dir: Optional[str] = None
    ) -> str:
        """
        Generate a complete video from audio features and templates.
        
        Args:
            audio_features_path: Path to .npy file with audio features
            template_dir: Directory containing template images and landmarks
                         Expected structure:
                         - template_dir/full_body_img/*.jpg
                         - template_dir/landmarks/*.lms
                         - template_dir/parsing/*.png (optional)
            audio_path: Path to original audio file
            output_path: Path to save output video
            start_frame: Starting frame index in template sequence
            fps: Frames per second (25 for ave/hubert, 20 for wenet)
            use_parsing: Whether to use parsing masks
            crf: Video quality (lower = better quality, 18-28 recommended)
            save_frames: Whether to save intermediate frames
            frames_dir: Directory to save frames (if save_frames=True)
            
        Returns:
            str: Path to generated video
        """
        logger.info(f"Generating video: {output_path}")
        
        # Load audio features
        logger.info(f"Loading audio features from {audio_features_path}")
        audio_features = np.load(audio_features_path)
        logger.info(f"Audio features shape: {audio_features.shape}")
        
        # Set up template directories
        img_dir = os.path.join(template_dir, "full_body_img")
        lms_dir = os.path.join(template_dir, "landmarks")
        parsing_dir = os.path.join(template_dir, "parsing") if use_parsing else None
        
        # Validate directories
        if not os.path.exists(img_dir):
            raise FileNotFoundError(f"Template image directory not found: {img_dir}")
        if not os.path.exists(lms_dir):
            raise FileNotFoundError(f"Landmarks directory not found: {lms_dir}")
        
        # Adjust FPS based on mode
        if self.mode == "wenet":
            fps = 20
        
        # Generate frames
        logger.info("Generating frames...")
        frames = self.frame_generator.generate_frames_from_template_sequence(
            img_dir=img_dir,
            lms_dir=lms_dir,
            audio_features=audio_features,
            start_frame=start_frame,
            parsing_dir=parsing_dir,
            fps=fps
        )
        
        logger.info(f"Generated {len(frames)} frames")
        
        # Save frames if requested
        if save_frames:
            if frames_dir is None:
                frames_dir = output_path.replace('.mp4', '_frames')
            logger.info(f"Saving frames to {frames_dir}")
            self.frame_generator.save_frames(frames, frames_dir)
        
        # Create temporary video without audio
        temp_video_path = output_path.replace('.mp4', '_temp.mp4')
        self._write_video(frames, temp_video_path, fps)
        
        # Merge with audio using ffmpeg
        logger.info("Merging video with audio...")
        self._merge_audio(temp_video_path, audio_path, output_path, crf)
        
        # Clean up temp video
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
        
        logger.info(f"Video saved to {output_path}")
        return output_path
    
    def generate_frames_only(
        self,
        audio_features_path: str,
        template_dir: str,
        output_dir: str,
        start_frame: int = 0,
        fps: int = 25,
        use_parsing: bool = False
    ) -> List[str]:
        """
        Generate frames without creating a video file.
        
        Args:
            audio_features_path: Path to audio features
            template_dir: Directory with template images/landmarks
            output_dir: Directory to save frames
            start_frame: Starting frame index
            fps: Frames per second
            use_parsing: Whether to use parsing masks
            
        Returns:
            List of paths to saved frames
        """
        logger.info(f"Generating frames to {output_dir}")
        
        # Load audio features
        audio_features = np.load(audio_features_path)
        
        # Set up directories
        img_dir = os.path.join(template_dir, "full_body_img")
        lms_dir = os.path.join(template_dir, "landmarks")
        parsing_dir = os.path.join(template_dir, "parsing") if use_parsing else None
        
        # Generate frames
        frames = self.frame_generator.generate_frames_from_template_sequence(
            img_dir=img_dir,
            lms_dir=lms_dir,
            audio_features=audio_features,
            start_frame=start_frame,
            parsing_dir=parsing_dir,
            fps=fps
        )
        
        # Save frames
        saved_paths = self.frame_generator.save_frames(frames, output_dir)
        
        return saved_paths
    
    def _write_video(
        self,
        frames: List[np.ndarray],
        output_path: str,
        fps: int
    ) -> None:
        """
        Write frames to a video file.
        
        Args:
            frames: List of frame arrays
            output_path: Path to save video
            fps: Frames per second
        """
        if not frames:
            raise ValueError("No frames to write")
        
        # Get frame dimensions
        h, w = frames[0].shape[:2]
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        
        # Write frames
        for frame in frames:
            video_writer.write(frame)
        
        video_writer.release()
        logger.info(f"Wrote {len(frames)} frames to {output_path}")
    
    def _merge_audio(
        self,
        video_path: str,
        audio_path: str,
        output_path: str,
        crf: int = 20
    ) -> None:
        """
        Merge video with audio using ffmpeg.
        
        Args:
            video_path: Path to video file
            audio_path: Path to audio file
            output_path: Path to save merged video
            crf: Video quality (lower = better)
        """
        cmd = [
            'ffmpeg',
            '-i', video_path,
            '-i', audio_path,
            '-c:v', 'libx264',
            '-c:a', 'aac',
            '-crf', str(crf),
            output_path,
            '-y'
        ]
        
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        if result.returncode != 0:
            logger.error(f"ffmpeg error: {result.stderr}")
            raise RuntimeError(f"ffmpeg failed: {result.stderr}")
    
    def export_model_to_onnx(
        self,
        output_path: str,
        opset_version: int = 11
    ) -> None:
        """
        Export the U-Net model to ONNX format.
        
        Args:
            output_path: Path to save ONNX model
            opset_version: ONNX opset version
        """
        logger.info(f"Exporting model to ONNX: {output_path}")
        self.unet_model.export_to_onnx(output_path, opset_version)
    
    def generate_reference_outputs(
        self,
        audio_features_path: str,
        template_dir: str,
        output_dir: str,
        num_sample_frames: int = 5
    ) -> Dict:
        """
        Generate reference outputs for validation.
        
        This creates a set of sample frames at different indices
        to use as reference for validating Go/Swift implementations.
        
        Args:
            audio_features_path: Path to audio features
            template_dir: Directory with templates
            output_dir: Directory to save reference outputs
            num_sample_frames: Number of sample frames to generate
            
        Returns:
            Dict with metadata about the reference outputs
        """
        logger.info(f"Generating reference outputs to {output_dir}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Load audio features
        audio_features = np.load(audio_features_path)
        num_frames = audio_features.shape[0]
        
        # Select sample frame indices
        sample_indices = np.linspace(0, num_frames - 1, num_sample_frames, dtype=int)
        
        # Set up directories
        img_dir = os.path.join(template_dir, "full_body_img")
        lms_dir = os.path.join(template_dir, "landmarks")
        
        # Generate all frames
        frames = self.frame_generator.generate_frames_from_template_sequence(
            img_dir=img_dir,
            lms_dir=lms_dir,
            audio_features=audio_features,
            start_frame=0,
            fps=25
        )
        
        # Save sample frames
        reference_frames = {}
        for idx in sample_indices:
            frame = frames[idx]
            output_path = os.path.join(output_dir, f"frame_{idx:05d}.jpg")
            cv2.imwrite(output_path, frame)
            reference_frames[int(idx)] = output_path
        
        # Create metadata
        metadata = {
            "num_total_frames": num_frames,
            "sample_indices": sample_indices.tolist(),
            "reference_frames": reference_frames,
            "audio_features_shape": list(audio_features.shape),
            "mode": self.mode,
            "frame_size": [frames[0].shape[1], frames[0].shape[0]],
        }
        
        # Save metadata
        metadata_path = os.path.join(output_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Reference outputs saved to {output_dir}")
        return metadata


if __name__ == '__main__':
    # Test the pipeline
    logging.basicConfig(level=logging.INFO)
    
    print("FrameGenerationPipeline module loaded successfully")
    print("\nExample usage:")
    print("""
    pipeline = FrameGenerationPipeline(
        checkpoint_path="./checkpoint/May/5.pth",
        mode="ave"
    )
    
    video_path = pipeline.generate_video(
        audio_features_path="./audio_features.npy",
        template_dir="./dataset/May",
        audio_path="./demo/audio.wav",
        output_path="./output/result.mp4"
    )
    """)

