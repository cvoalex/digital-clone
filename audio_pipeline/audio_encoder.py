"""
Audio Encoder Model

This module provides a wrapper around the AudioEncoder neural network
that extracts deep features from mel spectrograms.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Union, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Conv2d(nn.Module):
    """
    Custom 2D convolutional layer with optional residual connections.
    
    Combines convolution, batch normalization, and activation.
    """
    
    def __init__(
        self,
        cin: int,
        cout: int,
        kernel_size: Union[int, tuple],
        stride: Union[int, tuple],
        padding: Union[int, tuple],
        residual: bool = False,
        leakyReLU: bool = False
    ):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(cin, cout, kernel_size, stride, padding),
            nn.BatchNorm2d(cout)
        )
        self.act = nn.LeakyReLU(0.02) if leakyReLU else nn.ReLU()
        self.residual = residual

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv_block(x)
        if self.residual:
            out += x
        return self.act(out)


class AudioEncoder(nn.Module):
    """
    Audio encoder neural network for extracting features from mel spectrograms.
    
    Input: [batch_size, 1, n_mels, n_frames] - typically [batch, 1, 80, 16]
    Output: [batch_size, 512] - 512-dimensional feature vector
    
    Architecture:
    - Series of convolutional blocks with increasing channels
    - Residual connections for stable training
    - Progressive downsampling to compress temporal/frequency information
    """
    
    def __init__(self):
        super(AudioEncoder, self).__init__()
        
        self.audio_encoder = nn.Sequential(
            # Block 1: 1 -> 32 channels
            Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            
            # Block 2: 32 -> 64 channels, downsample frequency
            Conv2d(32, 64, kernel_size=3, stride=(3, 1), padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            
            # Block 3: 64 -> 128 channels, downsample both dims
            Conv2d(64, 128, kernel_size=3, stride=3, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            
            # Block 4: 128 -> 256 channels
            Conv2d(128, 256, kernel_size=3, stride=(3, 2), padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            
            # Block 5: 256 -> 512 channels, final compression
            Conv2d(256, 512, kernel_size=3, stride=1, padding=0),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0),
        )
        
        logger.info("AudioEncoder initialized")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the audio encoder.
        
        Args:
            x: Input tensor [batch_size, 1, n_mels, n_frames]
            
        Returns:
            Feature tensor [batch_size, 512]
        """
        out = self.audio_encoder(x)
        out = out.squeeze(2).squeeze(2)  # Remove spatial dimensions
        return out


class AudioEncoderWrapper:
    """
    Wrapper class for the AudioEncoder that handles:
    1. Model loading from checkpoint
    2. Batch processing
    3. Feature extraction with temporal context
    4. Mode-specific reshaping (AVE, Hubert, WeNet)
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        device: Optional[Union[str, torch.device]] = None,
        mode: str = "ave"
    ):
        """
        Initialize the AudioEncoder wrapper.
        
        Args:
            checkpoint_path: Path to the pretrained model checkpoint
            device: Device to run inference on (cuda/cpu)
            mode: Audio feature mode ('ave', 'hubert', or 'wenet')
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.mode = mode
        
        # Initialize model
        self.model = AudioEncoder().to(self.device)
        
        # Load checkpoint
        logger.info(f"Loading checkpoint from: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        
        # Handle checkpoint format (may have 'audio_encoder.' prefix)
        state_dict = {}
        for k, v in ckpt.items():
            if k.startswith('audio_encoder.'):
                state_dict[k] = v
            else:
                state_dict[f'audio_encoder.{k}'] = v
        
        self.model.load_state_dict(state_dict)
        self.model.eval()
        
        logger.info(f"AudioEncoder loaded on device: {self.device}")
        logger.info(f"Mode: {mode}")
    
    def process_mel_windows(
        self,
        mel_windows: np.ndarray,
        batch_size: int = 64
    ) -> np.ndarray:
        """
        Process mel spectrogram windows through the audio encoder.
        
        Args:
            mel_windows: Array of mel windows with shape (n_frames, 16, n_mels)
            batch_size: Batch size for processing
            
        Returns:
            Audio features with shape (n_frames, 512)
        """
        n_frames = mel_windows.shape[0]
        all_features = []
        
        logger.info(f"Processing {n_frames} mel windows in batches of {batch_size}")
        
        for i in range(0, n_frames, batch_size):
            batch = mel_windows[i:i+batch_size]
            
            # Convert to tensor: (batch, 16, n_mels) -> (batch, 1, n_mels, 16)
            batch_tensor = torch.FloatTensor(batch)
            batch_tensor = batch_tensor.transpose(1, 2)  # (batch, n_mels, 16)
            batch_tensor = batch_tensor.unsqueeze(1)  # (batch, 1, n_mels, 16)
            batch_tensor = batch_tensor.to(self.device)
            
            # Extract features
            with torch.no_grad():
                features = self.model(batch_tensor)
            
            all_features.append(features.cpu().numpy())
        
        # Concatenate all batches
        audio_features = np.concatenate(all_features, axis=0)
        
        logger.info(f"Extracted audio features with shape: {audio_features.shape}")
        
        return audio_features
    
    def add_temporal_padding(self, features: np.ndarray) -> np.ndarray:
        """
        Add temporal padding by repeating first and last frames.
        
        This matches the behavior in inference_328.py:
        audio_feats = torch.cat([first_frame.repeat(1, 1), outputs, last_frame.repeat(1, 1)], dim=0)
        
        Args:
            features: Audio features with shape (n_frames, 512)
            
        Returns:
            Padded features with shape (n_frames + 2, 512)
        """
        first_frame = features[:1]
        last_frame = features[-1:]
        padded = np.concatenate([first_frame, features, last_frame], axis=0)
        
        logger.info(f"Added temporal padding: {features.shape} -> {padded.shape}")
        
        return padded
    
    def get_audio_features_for_frame(
        self,
        all_features: np.ndarray,
        frame_idx: int,
        context_size: int = 8
    ) -> np.ndarray:
        """
        Extract audio features for a specific frame with temporal context.
        
        This matches get_audio_features() from utils.py, which extracts
        a window of ±8 frames around the target frame.
        
        Args:
            all_features: All audio features with shape (n_frames, 512)
            frame_idx: Target frame index
            context_size: Number of frames before and after (default: 8)
            
        Returns:
            Feature window with shape (16, 512) for context_size=8
        """
        left = frame_idx - context_size
        right = frame_idx + context_size
        pad_left = 0
        pad_right = 0
        
        # Handle boundaries
        if left < 0:
            pad_left = -left
            left = 0
        if right > all_features.shape[0]:
            pad_right = right - all_features.shape[0]
            right = all_features.shape[0]
        
        # Extract window
        window = all_features[left:right]
        
        # Pad if necessary
        if pad_left > 0:
            padding = np.zeros((pad_left, window.shape[1]), dtype=window.dtype)
            window = np.concatenate([padding, window], axis=0)
        if pad_right > 0:
            padding = np.zeros((pad_right, window.shape[1]), dtype=window.dtype)
            window = np.concatenate([window, padding], axis=0)
        
        return window
    
    def reshape_for_model(self, features: np.ndarray) -> np.ndarray:
        """
        Reshape audio features based on the mode.
        
        The feature window has 16 frames × 512 features = 8192 values total.
        These are reshaped to match the expected input dimensions for each mode.
        
        Args:
            features: Feature window with shape (16, 512)
            
        Returns:
            Reshaped features:
            - AVE: (32, 16, 16) = 8,192
            - Hubert: (32, 32, 32) = 32,768 (requires padding/interpolation)
            - WeNet: (256, 16, 32) = 131,072 (requires padding/interpolation)
        """
        flat = features.flatten()
        
        if self.mode == "ave":
            # 16 * 512 = 8192 = 32 * 16 * 16
            reshaped = flat.reshape(32, 16, 16)
        elif self.mode == "hubert":
            # Need to expand to 32 * 32 * 32 = 32,768
            # Pad with zeros
            target_size = 32 * 32 * 32
            padded = np.zeros(target_size, dtype=flat.dtype)
            padded[:flat.shape[0]] = flat
            reshaped = padded.reshape(32, 32, 32)
        elif self.mode == "wenet":
            # Need to expand to 256 * 16 * 32 = 131,072
            # Pad with zeros
            target_size = 256 * 16 * 32
            padded = np.zeros(target_size, dtype=flat.dtype)
            padded[:flat.shape[0]] = flat
            reshaped = padded.reshape(256, 16, 32)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
        
        return reshaped
    
    def save_features(self, features: np.ndarray, output_path: str):
        """Save audio features to a .npy file."""
        np.save(output_path, features)
        logger.info(f"Saved audio features to: {output_path}")
    
    def load_features(self, input_path: str) -> np.ndarray:
        """Load audio features from a .npy file."""
        features = np.load(input_path)
        logger.info(f"Loaded audio features from: {input_path}")
        return features

