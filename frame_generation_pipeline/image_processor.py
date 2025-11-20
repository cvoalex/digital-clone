"""
Image Processor

This module handles all image processing operations for frame generation:
- Loading images and landmarks
- Cropping face regions based on landmarks
- Resizing images
- Creating masked versions
- Pasting generated regions back into frames
"""

import cv2
import numpy as np
import torch
from typing import Tuple, Optional, List
import logging

logger = logging.getLogger(__name__)


class ImageProcessor:
    """
    Handles all image processing operations for frame generation.
    
    This class provides methods for:
    - Loading and parsing landmark files
    - Cropping face regions based on landmarks
    - Resizing images with cv2.INTER_CUBIC
    - Creating masked versions (lower face blacked out)
    - Preparing tensors for U-Net input
    - Pasting generated regions back into full frames
    """
    
    def __init__(self):
        """Initialize the image processor."""
        pass
    
    @staticmethod
    def load_image(image_path: str) -> np.ndarray:
        """
        Load an image from disk.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            np.ndarray: Loaded image in BGR format
        """
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Could not load image: {image_path}")
        return img
    
    @staticmethod
    def load_landmarks(lms_path: str) -> np.ndarray:
        """
        Load facial landmarks from a .lms file.
        
        The .lms file format has one landmark per line with x and y coordinates
        separated by space.
        
        Args:
            lms_path: Path to the landmarks file
            
        Returns:
            np.ndarray: Array of landmarks with shape (num_landmarks, 2)
        """
        lms_list = []
        with open(lms_path, "r") as f:
            lines = f.read().splitlines()
            for line in lines:
                arr = line.split(" ")
                arr = np.array(arr, dtype=np.float32)
                lms_list.append(arr)
        lms = np.array(lms_list, dtype=np.int32)
        return lms
    
    @staticmethod
    def get_crop_region(landmarks: np.ndarray) -> Tuple[int, int, int, int]:
        """
        Calculate the crop region based on facial landmarks.
        
        This method follows the same logic as inference_328.py:
        - xmin = landmark 1 (left face edge)
        - xmax = landmark 31 (right face edge)
        - ymin = landmark 52 (top of face region)
        - ymax = ymin + width (to create a square region)
        
        Args:
            landmarks: Array of facial landmarks (num_landmarks, 2)
            
        Returns:
            Tuple of (xmin, ymin, xmax, ymax)
        """
        xmin = landmarks[1][0]
        ymin = landmarks[52][1]
        xmax = landmarks[31][0]
        width = xmax - xmin
        ymax = ymin + width
        
        return xmin, ymin, xmax, ymax
    
    @staticmethod
    def crop_face_region(
        image: np.ndarray,
        landmarks: np.ndarray
    ) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """
        Crop the face region from an image based on landmarks.
        
        Args:
            image: Full image
            landmarks: Facial landmarks
            
        Returns:
            Tuple of (cropped_image, crop_coords)
            where crop_coords is (xmin, ymin, xmax, ymax)
        """
        xmin, ymin, xmax, ymax = ImageProcessor.get_crop_region(landmarks)
        crop_img = image[ymin:ymax, xmin:xmax]
        return crop_img, (xmin, ymin, xmax, ymax)
    
    @staticmethod
    def resize_image(
        image: np.ndarray,
        target_size: Tuple[int, int],
        interpolation: int = cv2.INTER_CUBIC
    ) -> np.ndarray:
        """
        Resize an image using the specified interpolation method.
        
        Args:
            image: Input image
            target_size: Target size as (width, height)
            interpolation: cv2 interpolation method (default: INTER_CUBIC)
            
        Returns:
            np.ndarray: Resized image
        """
        return cv2.resize(image, target_size, interpolation=interpolation)
    
    @staticmethod
    def create_masked_region(
        image: np.ndarray,
        inner_crop: bool = True
    ) -> np.ndarray:
        """
        Create a masked version of the image with lower face region blacked out.
        
        Following inference_328.py logic:
        - Crop inner region [4:324, 4:324] from 328x328 image
        - Draw black rectangle on lower face region
        
        Args:
            image: Input image (should be 320x320 after inner crop)
            inner_crop: Whether the image is already the inner crop
            
        Returns:
            np.ndarray: Masked image with lower face blacked out
        """
        masked = image.copy()
        # Black out the lower face region
        # Rectangle coordinates: (x, y, width, height)
        # This matches: cv2.rectangle(img_masked,(5,5,310,305),(0,0,0),-1)
        masked = cv2.rectangle(masked, (5, 5, 310, 305), (0, 0, 0), -1)
        return masked
    
    @staticmethod
    def prepare_input_tensors(
        image: np.ndarray,
        normalize: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare input tensors for the U-Net model.
        
        This creates:
        1. Original image tensor
        2. Masked image tensor
        3. Concatenated tensor (6 channels)
        
        Args:
            image: Input image (320x320x3 BGR)
            normalize: Whether to normalize to [0, 1] range
            
        Returns:
            Tuple of (concatenated_tensor, original_tensor)
            concatenated_tensor shape: (1, 6, 320, 320)
        """
        # Create masked version
        masked = ImageProcessor.create_masked_region(image)
        
        # Convert to CHW format and float32
        img_chw = image.transpose(2, 0, 1).astype(np.float32)
        masked_chw = masked.transpose(2, 0, 1).astype(np.float32)
        
        # Normalize if requested
        if normalize:
            img_chw = img_chw / 255.0
            masked_chw = masked_chw / 255.0
        
        # Convert to tensors
        img_tensor = torch.from_numpy(img_chw)
        masked_tensor = torch.from_numpy(masked_chw)
        
        # Concatenate along channel dimension
        concat_tensor = torch.cat([img_tensor, masked_tensor], dim=0)[None]
        
        return concat_tensor, img_tensor
    
    @staticmethod
    def paste_generated_region(
        full_frame: np.ndarray,
        generated_region: np.ndarray,
        crop_coords: Tuple[int, int, int, int],
        original_crop_size: Tuple[int, int]
    ) -> np.ndarray:
        """
        Paste the generated face region back into the full frame.
        
        This follows the logic from inference_328.py:
        1. Paste generated region into center of 328x328 crop
        2. Resize back to original crop size
        3. Paste back into full frame at crop coordinates
        
        Args:
            full_frame: Original full frame
            generated_region: Generated face region (320x320)
            crop_coords: Coordinates where to paste (xmin, ymin, xmax, ymax)
            original_crop_size: Original size of the crop before resizing to 328x328
            
        Returns:
            np.ndarray: Full frame with generated region pasted in
        """
        xmin, ymin, xmax, ymax = crop_coords
        h, w = original_crop_size
        
        # Create 328x328 canvas
        canvas = np.zeros((328, 328, 3), dtype=np.uint8)
        
        # Paste generated region in center [4:324, 4:324]
        canvas[4:324, 4:324] = generated_region
        
        # Resize back to original crop size
        resized = cv2.resize(canvas, (w, h), interpolation=cv2.INTER_CUBIC)
        
        # Create output frame
        output_frame = full_frame.copy()
        
        # Paste back into full frame
        output_frame[ymin:ymax, xmin:xmax] = resized
        
        return output_frame
    
    @staticmethod
    def process_frame_with_parsing(
        full_frame: np.ndarray,
        generated_region: np.ndarray,
        crop_coords: Tuple[int, int, int, int],
        original_crop_size: Tuple[int, int],
        parsing_image: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Process frame with optional parsing mask.
        
        If parsing image is provided, certain regions are kept from the original
        instead of using the generated region.
        
        Args:
            full_frame: Original full frame
            generated_region: Generated face region (320x320)
            crop_coords: Coordinates where to paste
            original_crop_size: Original size of the crop
            parsing_image: Optional parsing/segmentation mask
            
        Returns:
            np.ndarray: Processed frame
        """
        xmin, ymin, xmax, ymax = crop_coords
        h, w = original_crop_size
        
        # Create 328x328 canvas
        canvas = np.zeros((328, 328, 3), dtype=np.uint8)
        
        # Paste generated region in center
        canvas[4:324, 4:324] = generated_region
        
        # Resize back to original crop size
        resized = cv2.resize(canvas, (w, h), interpolation=cv2.INTER_CUBIC)
        
        # Apply parsing mask if provided
        if parsing_image is not None:
            crop_parsing = parsing_image[ymin:ymax, xmin:xmax]
            # Keep regions with [0, 0, 255] or [255, 255, 255]
            parsing_mask = (
                (crop_parsing == [0, 0, 255]).all(axis=2) | 
                (crop_parsing == [255, 255, 255]).all(axis=2)
            )
            original_crop = full_frame[ymin:ymax, xmin:xmax]
            resized[parsing_mask] = original_crop[parsing_mask]
        
        # Create output frame
        output_frame = full_frame.copy()
        output_frame[ymin:ymax, xmin:xmax] = resized
        
        return output_frame


if __name__ == '__main__':
    # Test the image processor
    logging.basicConfig(level=logging.INFO)
    
    print("ImageProcessor module loaded successfully")
    print("Available methods:")
    print("  - load_image()")
    print("  - load_landmarks()")
    print("  - crop_face_region()")
    print("  - resize_image()")
    print("  - create_masked_region()")
    print("  - prepare_input_tensors()")
    print("  - paste_generated_region()")

