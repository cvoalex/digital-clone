"""
Frame Generation Pipeline

A standalone, cross-platform video frame generation pipeline for SyncTalk_2D
that generates lip-sync video frames from audio features and template images.

This package provides modular components for:
- U-Net model loading and inference
- Image processing (cropping, masking, resizing)
- Frame generation loop
- Video assembly

The pipeline takes audio features (32, 16, 16 tensors) and template images/landmarks
as input and generates complete video frames with synchronized lip movements.
"""

__version__ = "1.0.0"

from .unet_model import UNetModel
from .image_processor import ImageProcessor
from .frame_generator import FrameGenerator
from .pipeline import FrameGenerationPipeline

__all__ = [
    "UNetModel",
    "ImageProcessor", 
    "FrameGenerator",
    "FrameGenerationPipeline"
]

