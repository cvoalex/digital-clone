"""
Complete Audio Processing Pipeline

This module ties together the mel spectrogram processing and audio encoding
into a single pipeline that matches the behavior of inference_328.py.
"""

import numpy as np
from typing import Optional, Tuple
import logging
from pathlib import Path

from .mel_processor import MelSpectrogramProcessor
from .audio_encoder import AudioEncoderWrapper

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AudioPipeline:
    """
    Complete audio processing pipeline that converts audio files to
    model-ready feature tensors.
    
    This pipeline matches the exact behavior of the audio processing in
    inference_328.py and provides reference outputs for iOS porting.
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        device: Optional[str] = None,
        mode: str = "ave",
        fps: int = 25
    ):
        """
        Initialize the complete audio pipeline.
        
        Args:
            checkpoint_path: Path to the AudioEncoder checkpoint
            device: Device to run inference on (cuda/cpu)
            mode: Audio feature mode ('ave', 'hubert', or 'wenet')
            fps: Target video frame rate
        """
        self.mode = mode
        self.fps = fps
        
        # Initialize mel processor
        self.mel_processor = MelSpectrogramProcessor()
        
        # Initialize audio encoder
        self.audio_encoder = AudioEncoderWrapper(
            checkpoint_path=checkpoint_path,
            device=device,
            mode=mode
        )
        
        logger.info("AudioPipeline initialized")
        logger.info(f"  Mode: {mode}, FPS: {fps}")
    
    def process_audio_file(
        self,
        audio_path: str,
        save_intermediates: bool = False,
        output_dir: Optional[str] = None
    ) -> Tuple[np.ndarray, dict]:
        """
        Process an audio file through the complete pipeline.
        
        Steps:
        1. Load audio and convert to mel spectrogram
        2. Extract mel windows for each video frame
        3. Process through AudioEncoder to get 512-dim features
        4. Add temporal padding
        5. Prepare frame-by-frame features
        
        Args:
            audio_path: Path to audio file
            save_intermediates: Whether to save intermediate outputs
            output_dir: Directory to save intermediate outputs
            
        Returns:
            Tuple of:
            - audio_features: Array with shape (n_frames, 512)
            - metadata: Dictionary with processing information
        """
        logger.info(f"Processing audio file: {audio_path}")
        
        if save_intermediates and output_dir is None:
            raise ValueError("output_dir must be provided when save_intermediates=True")
        
        if save_intermediates:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # Step 1: Convert to mel spectrogram
        logger.info("Step 1: Converting to mel spectrogram...")
        mel_spec = self.mel_processor.process_file(audio_path)
        
        if save_intermediates:
            mel_path = output_dir / "mel_spectrogram.npy"
            self.mel_processor.save_mel_spectrogram(mel_spec, str(mel_path))
        
        # Step 2: Get frame count and extract windows
        logger.info("Step 2: Extracting mel windows for each frame...")
        n_frames = self.mel_processor.get_frame_count(mel_spec, self.fps)
        logger.info(f"  Total frames: {n_frames}")
        
        mel_windows = []
        for i in range(n_frames):
            window = self.mel_processor.crop_audio_window(mel_spec, i, self.fps)
            mel_windows.append(window)
        mel_windows = np.array(mel_windows)
        
        logger.info(f"  Mel windows shape: {mel_windows.shape}")
        
        if save_intermediates:
            windows_path = output_dir / "mel_windows.npy"
            np.save(windows_path, mel_windows)
            logger.info(f"  Saved mel windows to: {windows_path}")
        
        # Step 3: Process through AudioEncoder
        logger.info("Step 3: Extracting deep features with AudioEncoder...")
        audio_features = self.audio_encoder.process_mel_windows(mel_windows)
        
        if save_intermediates:
            features_path = output_dir / "audio_features_raw.npy"
            self.audio_encoder.save_features(audio_features, str(features_path))
        
        # Step 4: Add temporal padding
        logger.info("Step 4: Adding temporal padding...")
        audio_features_padded = self.audio_encoder.add_temporal_padding(audio_features)
        
        if save_intermediates:
            padded_path = output_dir / "audio_features_padded.npy"
            self.audio_encoder.save_features(audio_features_padded, str(padded_path))
        
        # Prepare metadata
        metadata = {
            'audio_path': audio_path,
            'mode': self.mode,
            'fps': self.fps,
            'n_frames': n_frames,
            'mel_shape': mel_spec.shape,
            'mel_windows_shape': mel_windows.shape,
            'features_shape': audio_features.shape,
            'features_padded_shape': audio_features_padded.shape,
            'mel_value_range': (float(mel_spec.min()), float(mel_spec.max())),
            'features_value_range': (float(audio_features_padded.min()), float(audio_features_padded.max())),
        }
        
        logger.info("Pipeline processing complete!")
        logger.info(f"  Output shape: {audio_features_padded.shape}")
        
        return audio_features_padded, metadata
    
    def get_frame_features(
        self,
        audio_features: np.ndarray,
        frame_idx: int,
        reshape: bool = True
    ) -> np.ndarray:
        """
        Get model-ready features for a specific frame.
        
        This extracts the temporal context window and optionally reshapes
        it for the U-Net model.
        
        Args:
            audio_features: Padded audio features with shape (n_frames, 512)
            frame_idx: Target frame index
            reshape: Whether to reshape for model input
            
        Returns:
            Feature tensor ready for model input:
            - If reshape=False: (16, 512)
            - If reshape=True: Mode-specific shape
                - AVE: (32, 16, 16)
                - Hubert: (32, 32, 32)
                - WeNet: (256, 16, 32)
        """
        # Extract temporal context window
        features = self.audio_encoder.get_audio_features_for_frame(
            audio_features, frame_idx
        )
        
        # Optionally reshape
        if reshape:
            features = self.audio_encoder.reshape_for_model(features)
        
        return features
    
    def process_and_save_all_frames(
        self,
        audio_path: str,
        output_dir: str
    ):
        """
        Process an audio file and save features for all frames.
        
        This creates a complete reference dataset for iOS validation.
        
        Args:
            audio_path: Path to audio file
            output_dir: Directory to save outputs
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Process audio
        audio_features, metadata = self.process_audio_file(
            audio_path,
            save_intermediates=True,
            output_dir=output_dir
        )
        
        # Save metadata
        import json
        metadata_path = output_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved metadata to: {metadata_path}")
        
        # Save per-frame features
        frames_dir = output_dir / "frames"
        frames_dir.mkdir(exist_ok=True)
        
        n_frames = metadata['n_frames']
        logger.info(f"Saving features for {n_frames} frames...")
        
        for frame_idx in range(n_frames):
            # Get unreshaped features (temporal window)
            features_window = self.get_frame_features(
                audio_features, frame_idx, reshape=False
            )
            
            # Get reshaped features (model input)
            features_reshaped = self.get_frame_features(
                audio_features, frame_idx, reshape=True
            )
            
            # Save both
            np.save(frames_dir / f"frame_{frame_idx:05d}_window.npy", features_window)
            np.save(frames_dir / f"frame_{frame_idx:05d}_reshaped.npy", features_reshaped)
        
        logger.info(f"Saved all frame features to: {frames_dir}")
        
        # Create a summary file
        summary = {
            'total_frames': n_frames,
            'mode': self.mode,
            'fps': self.fps,
            'window_shape': (16, 512),
            'reshaped_shape': self._get_reshaped_shape(),
        }
        
        summary_path = output_dir / "summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info("Complete reference dataset saved!")
        return output_dir
    
    def _get_reshaped_shape(self) -> Tuple[int, ...]:
        """Get the expected reshaped dimensions for the current mode."""
        if self.mode == "ave":
            return (32, 16, 16)
        elif self.mode == "hubert":
            return (32, 32, 32)
        elif self.mode == "wenet":
            return (256, 16, 32)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

