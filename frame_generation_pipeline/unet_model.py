"""
U-Net Model Wrapper

This module provides a wrapper for the U-Net model used in frame generation.
It handles model loading, inference, and ONNX export.
"""

import os
import torch
import torch.nn as nn
import numpy as np
from typing import Union, Tuple, Optional
import logging

# Import the model architecture from the parent directory
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from unet_328 import Model

logger = logging.getLogger(__name__)


class UNetModel:
    """
    Wrapper for the U-Net model used in frame generation.
    
    This class handles:
    - Loading the model from checkpoint
    - Running inference on image + audio features
    - Exporting to ONNX format
    - Managing device (CPU/CUDA)
    
    Attributes:
        model: The U-Net neural network model
        device: Device to run the model on (cuda or cpu)
        mode: Audio feature mode ('ave', 'hubert', or 'wenet')
    """
    
    def __init__(
        self, 
        checkpoint_path: str,
        mode: str = 'ave',
        device: Optional[Union[str, torch.device]] = None
    ):
        """
        Initialize the U-Net model.
        
        Args:
            checkpoint_path: Path to the model checkpoint file
            mode: Audio feature mode ('ave', 'hubert', or 'wenet')
            device: Device to run on (defaults to CUDA if available)
        """
        self.mode = mode
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        logger.info(f"Initializing U-Net model on {self.device} with mode={mode}")
        
        # Load model architecture
        # Input: 6 channels (3 original + 3 masked)
        self.model = Model(n_channels=6, mode=mode)
        
        # Load checkpoint
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint)
        
        # Set to evaluation mode and move to device
        self.model = self.model.to(self.device)
        self.model.eval()
        
        logger.info("U-Net model loaded successfully")
    
    def predict(
        self,
        image_tensor: torch.Tensor,
        audio_features: torch.Tensor
    ) -> np.ndarray:
        """
        Run inference on an image with audio features.
        
        Args:
            image_tensor: Input image tensor of shape (1, 6, 320, 320)
                         6 channels = 3 original + 3 masked
            audio_features: Audio feature tensor of shape (1, 32, 16, 16) for 'ave'
                           or (1, 32, 32, 32) for 'hubert'
                           or (1, 256, 16, 32) for 'wenet'
        
        Returns:
            np.ndarray: Generated image as numpy array (320, 320, 3) in uint8 BGR format
        """
        # Ensure tensors are on the correct device
        image_tensor = image_tensor.to(self.device)
        audio_features = audio_features.to(self.device)
        
        # Run inference
        with torch.no_grad():
            output = self.model(image_tensor, audio_features)
        
        # Convert output to numpy
        # Output shape: (1, 3, 320, 320)
        # Convert to (320, 320, 3) and scale to 0-255
        output = output[0].cpu().numpy()  # (3, 320, 320)
        output = output.transpose(1, 2, 0)  # (320, 320, 3)
        output = (output * 255).astype(np.uint8)
        
        return output
    
    def export_to_onnx(
        self,
        output_path: str,
        opset_version: int = 11
    ) -> None:
        """
        Export the model to ONNX format.
        
        Args:
            output_path: Path to save the ONNX model
            opset_version: ONNX opset version to use
        """
        logger.info(f"Exporting model to ONNX: {output_path}")
        
        # Create dummy inputs based on mode
        dummy_image = torch.zeros([1, 6, 320, 320], device=self.device)
        
        if self.mode == 'ave':
            dummy_audio = torch.zeros([1, 32, 16, 16], device=self.device)
        elif self.mode == 'hubert':
            dummy_audio = torch.zeros([1, 32, 32, 32], device=self.device)
        elif self.mode == 'wenet':
            dummy_audio = torch.zeros([1, 256, 16, 32], device=self.device)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
        
        # Export to ONNX
        with torch.no_grad():
            torch.onnx.export(
                self.model,
                (dummy_image, dummy_audio),
                output_path,
                input_names=['image', 'audio'],
                output_names=['output'],
                opset_version=opset_version,
                export_params=True,
                do_constant_folding=True
            )
        
        logger.info(f"Model exported successfully to {output_path}")
        
        # Verify the export
        try:
            import onnx
            onnx_model = onnx.load(output_path)
            onnx.checker.check_model(onnx_model)
            logger.info("ONNX model verification passed")
        except ImportError:
            logger.warning("onnx package not available, skipping verification")
        except Exception as e:
            logger.error(f"ONNX model verification failed: {e}")
    
    def get_input_shapes(self) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
        """
        Get the expected input shapes for image and audio.
        
        Returns:
            Tuple of (image_shape, audio_shape)
        """
        if self.mode == 'ave':
            audio_shape = (1, 32, 16, 16)
        elif self.mode == 'hubert':
            audio_shape = (1, 32, 32, 32)
        elif self.mode == 'wenet':
            audio_shape = (1, 256, 16, 32)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
        
        image_shape = (1, 6, 320, 320)
        return image_shape, audio_shape


if __name__ == '__main__':
    # Test the model wrapper
    logging.basicConfig(level=logging.INFO)
    
    # Example usage
    checkpoint_path = "../checkpoint/May/5.pth"  # Update with actual path
    
    if os.path.exists(checkpoint_path):
        model = UNetModel(checkpoint_path, mode='ave')
        
        # Test with dummy data
        dummy_image = torch.randn(1, 6, 320, 320)
        dummy_audio = torch.randn(1, 32, 16, 16)
        
        output = model.predict(dummy_image, dummy_audio)
        print(f"Output shape: {output.shape}")
        print(f"Output dtype: {output.dtype}")
        print(f"Output range: [{output.min()}, {output.max()}]")
        
        # Export to ONNX
        onnx_path = "../frame_generation_pipeline/models/unet_328.onnx"
        os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
        model.export_to_onnx(onnx_path)
    else:
        print(f"Checkpoint not found: {checkpoint_path}")
        print("Update the path and try again")

