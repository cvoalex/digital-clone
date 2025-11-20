"""
Frame Generator

This module handles the frame-by-frame generation logic.
It coordinates the image processor and U-Net model to generate video frames.
"""

import os
import cv2
import torch
import numpy as np
from typing import List, Tuple, Optional, Dict
import logging
from tqdm import tqdm

from .unet_model import UNetModel
from .image_processor import ImageProcessor

logger = logging.getLogger(__name__)


class FrameGenerator:
    """
    Generates video frames from audio features and template images.
    
    This class orchestrates the frame generation process:
    1. Load template images and landmarks
    2. For each audio feature frame:
       - Crop face region
       - Create masked version
       - Run U-Net inference
       - Paste result back into full frame
    3. Save generated frames
    
    Attributes:
        unet_model: U-Net model for generating face regions
        image_processor: Image processor for cropping/pasting operations
        mode: Audio feature mode ('ave', 'hubert', 'wenet')
    """
    
    def __init__(
        self,
        unet_model: UNetModel,
        mode: str = 'ave'
    ):
        """
        Initialize the frame generator.
        
        Args:
            unet_model: Initialized U-Net model
            mode: Audio feature mode
        """
        self.unet_model = unet_model
        self.image_processor = ImageProcessor()
        self.mode = mode
        
        logger.info(f"FrameGenerator initialized with mode={mode}")
    
    def generate_frame(
        self,
        template_image: np.ndarray,
        landmarks: np.ndarray,
        audio_features: torch.Tensor,
        parsing_image: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Generate a single frame from template image and audio features.
        
        Args:
            template_image: Full template image
            landmarks: Facial landmarks for the template
            audio_features: Audio features for this frame (reshaped appropriately)
            parsing_image: Optional parsing mask
            
        Returns:
            np.ndarray: Generated frame
        """
        # Crop face region
        crop_img, crop_coords = self.image_processor.crop_face_region(
            template_image, landmarks
        )
        
        # Get original crop size
        h, w = crop_img.shape[:2]
        original_crop_size = (h, w)
        
        # Resize to 328x328
        crop_328 = self.image_processor.resize_image(crop_img, (328, 328))
        
        # Extract inner region [4:324, 4:324] -> 320x320
        inner_crop = crop_328[4:324, 4:324].copy()
        
        # Prepare input tensors
        concat_tensor, _ = self.image_processor.prepare_input_tensors(inner_crop)
        
        # Run U-Net inference
        generated_region = self.unet_model.predict(concat_tensor, audio_features)
        
        # Paste back into full frame
        if parsing_image is not None:
            output_frame = self.image_processor.process_frame_with_parsing(
                template_image,
                generated_region,
                crop_coords,
                original_crop_size,
                parsing_image
            )
        else:
            output_frame = self.image_processor.paste_generated_region(
                template_image,
                generated_region,
                crop_coords,
                original_crop_size
            )
        
        return output_frame
    
    def generate_frames_from_template_sequence(
        self,
        img_dir: str,
        lms_dir: str,
        audio_features: np.ndarray,
        start_frame: int = 0,
        parsing_dir: Optional[str] = None,
        fps: int = 25
    ) -> List[np.ndarray]:
        """
        Generate frames using a template image sequence.
        
        This follows the "ping-pong" pattern from inference_328.py where
        the frame index bounces back and forth between first and last frame.
        
        Args:
            img_dir: Directory containing template images (0.jpg, 1.jpg, ...)
            lms_dir: Directory containing landmark files (0.lms, 1.lms, ...)
            audio_features: Audio features array (num_frames, 512) or reshaped
            start_frame: Starting frame index in template sequence
            parsing_dir: Optional directory with parsing masks
            fps: Frames per second (affects audio feature extraction)
            
        Returns:
            List of generated frames
        """
        num_frames = audio_features.shape[0]
        
        # Get number of template images
        img_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg')])
        len_img = len(img_files) - 1  # Max index
        
        logger.info(f"Generating {num_frames} frames from {len(img_files)} template images")
        
        generated_frames = []
        
        # Initialize ping-pong motion
        step_stride = 0
        img_idx = 0
        
        for i in tqdm(range(num_frames), desc="Generating frames"):
            # Ping-pong logic
            if img_idx > len_img - 1:
                step_stride = -1
            if img_idx < 1:
                step_stride = 1
            img_idx += step_stride
            
            # Load template image and landmarks
            img_path = os.path.join(img_dir, f"{img_idx + start_frame}.jpg")
            lms_path = os.path.join(lms_dir, f"{img_idx + start_frame}.lms")
            
            template_img = self.image_processor.load_image(img_path)
            landmarks = self.image_processor.load_landmarks(lms_path)
            
            # Load parsing if provided
            parsing_img = None
            if parsing_dir is not None:
                parsing_path = os.path.join(parsing_dir, f"{img_idx + start_frame}.png")
                if os.path.exists(parsing_path):
                    parsing_img = cv2.imread(parsing_path)
            
            # Get audio features for this frame
            audio_feat = self._get_audio_features_for_frame(audio_features, i)
            
            # Generate frame
            frame = self.generate_frame(
                template_img,
                landmarks,
                audio_feat,
                parsing_img
            )
            
            generated_frames.append(frame)
        
        return generated_frames
    
    def _get_audio_features_for_frame(
        self,
        audio_features: np.ndarray,
        frame_idx: int
    ) -> torch.Tensor:
        """
        Extract audio features for a specific frame and reshape appropriately.
        
        This replicates the get_audio_features logic from utils.py and reshapes
        according to the mode.
        
        Args:
            audio_features: Full audio features array
            frame_idx: Frame index
            
        Returns:
            torch.Tensor: Reshaped audio features ready for U-Net
        """
        # Extract window around frame
        left = frame_idx - 8
        right = frame_idx + 8
        pad_left = 0
        pad_right = 0
        
        if left < 0:
            pad_left = -left
            left = 0
        if right > audio_features.shape[0]:
            pad_right = right - audio_features.shape[0]
            right = audio_features.shape[0]
        
        auds = torch.from_numpy(audio_features[left:right])
        
        if pad_left > 0:
            auds = torch.cat([torch.zeros_like(auds[:pad_left]), auds], dim=0)
        if pad_right > 0:
            auds = torch.cat([auds, torch.zeros_like(auds[:pad_right])], dim=0)
        
        # Reshape based on mode
        if self.mode == "hubert":
            audio_feat = auds.reshape(32, 32, 32)
        elif self.mode == "wenet":
            audio_feat = auds.reshape(256, 16, 32)
        elif self.mode == "ave":
            audio_feat = auds.reshape(32, 16, 16)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
        
        # Add batch dimension
        audio_feat = audio_feat[None]
        
        return audio_feat
    
    def save_frames(
        self,
        frames: List[np.ndarray],
        output_dir: str,
        prefix: str = "frame"
    ) -> List[str]:
        """
        Save frames to disk.
        
        Args:
            frames: List of frame arrays
            output_dir: Directory to save frames
            prefix: Filename prefix
            
        Returns:
            List of saved file paths
        """
        os.makedirs(output_dir, exist_ok=True)
        
        saved_paths = []
        for i, frame in enumerate(frames):
            output_path = os.path.join(output_dir, f"{prefix}_{i:05d}.jpg")
            cv2.imwrite(output_path, frame)
            saved_paths.append(output_path)
        
        logger.info(f"Saved {len(frames)} frames to {output_dir}")
        return saved_paths


if __name__ == '__main__':
    # Test the frame generator
    logging.basicConfig(level=logging.INFO)
    
    print("FrameGenerator module loaded successfully")
    print("This module coordinates U-Net model and image processor")
    print("to generate video frames from audio features and templates.")

