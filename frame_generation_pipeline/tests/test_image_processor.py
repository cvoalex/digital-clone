"""
Tests for ImageProcessor module.
"""

import os
import sys
import pytest
import numpy as np
import cv2
import torch

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from image_processor import ImageProcessor


class TestImageProcessor:
    """Test suite for ImageProcessor class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.processor = ImageProcessor()
        
        # Create dummy image
        self.test_image = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        
        # Create dummy landmarks (68 landmarks)
        self.test_landmarks = np.array([
            [100 + i * 10, 200 + i * 5] for i in range(68)
        ], dtype=np.int32)
    
    def test_load_image(self, tmp_path):
        """Test image loading."""
        # Save test image
        img_path = tmp_path / "test.jpg"
        cv2.imwrite(str(img_path), self.test_image)
        
        # Load image
        loaded = self.processor.load_image(str(img_path))
        
        assert loaded is not None
        assert loaded.shape == self.test_image.shape
    
    def test_load_image_not_found(self):
        """Test loading non-existent image."""
        with pytest.raises(FileNotFoundError):
            self.processor.load_image("nonexistent.jpg")
    
    def test_load_landmarks(self, tmp_path):
        """Test landmark loading."""
        # Create test landmarks file
        lms_path = tmp_path / "test.lms"
        with open(lms_path, "w") as f:
            for lm in self.test_landmarks:
                f.write(f"{lm[0]} {lm[1]}\n")
        
        # Load landmarks
        loaded = self.processor.load_landmarks(str(lms_path))
        
        assert loaded is not None
        assert loaded.shape == self.test_landmarks.shape
        np.testing.assert_array_equal(loaded, self.test_landmarks)
    
    def test_get_crop_region(self):
        """Test crop region calculation."""
        xmin, ymin, xmax, ymax = self.processor.get_crop_region(self.test_landmarks)
        
        # Check that region is valid
        assert xmin < xmax
        assert ymin < ymax
        
        # Check square region
        width = xmax - xmin
        height = ymax - ymin
        assert width == height
    
    def test_crop_face_region(self):
        """Test face region cropping."""
        crop, coords = self.processor.crop_face_region(
            self.test_image,
            self.test_landmarks
        )
        
        assert crop is not None
        assert len(coords) == 4
        
        xmin, ymin, xmax, ymax = coords
        assert crop.shape[0] == ymax - ymin
        assert crop.shape[1] == xmax - xmin
    
    def test_resize_image(self):
        """Test image resizing."""
        resized = self.processor.resize_image(
            self.test_image,
            (328, 328)
        )
        
        assert resized.shape == (328, 328, 3)
    
    def test_create_masked_region(self):
        """Test masked region creation."""
        # Create 320x320 test image
        test_img = np.ones((320, 320, 3), dtype=np.uint8) * 255
        
        masked = self.processor.create_masked_region(test_img)
        
        # Check that masking was applied
        assert masked.shape == test_img.shape
        
        # Check that some region is black
        black_pixels = np.sum(masked == 0)
        assert black_pixels > 0
    
    def test_prepare_input_tensors(self):
        """Test input tensor preparation."""
        # Create 320x320 test image
        test_img = np.random.randint(0, 255, (320, 320, 3), dtype=np.uint8)
        
        concat_tensor, img_tensor = self.processor.prepare_input_tensors(test_img)
        
        # Check shapes
        assert concat_tensor.shape == (1, 6, 320, 320)
        assert img_tensor.shape == (3, 320, 320)
        
        # Check normalization
        assert concat_tensor.max() <= 1.0
        assert concat_tensor.min() >= 0.0
    
    def test_prepare_input_tensors_no_normalize(self):
        """Test input tensor preparation without normalization."""
        test_img = np.random.randint(0, 255, (320, 320, 3), dtype=np.uint8)
        
        concat_tensor, _ = self.processor.prepare_input_tensors(
            test_img,
            normalize=False
        )
        
        # Check that values are not normalized
        assert concat_tensor.max() > 1.0
    
    def test_paste_generated_region(self):
        """Test pasting generated region back."""
        # Create test images
        full_frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        generated = np.random.randint(0, 255, (320, 320, 3), dtype=np.uint8)
        
        crop_coords = (100, 200, 500, 600)  # Square region
        original_crop_size = (400, 400)
        
        result = self.processor.paste_generated_region(
            full_frame,
            generated,
            crop_coords,
            original_crop_size
        )
        
        # Check that result has same shape as input
        assert result.shape == full_frame.shape
        
        # Check that something was pasted
        xmin, ymin, xmax, ymax = crop_coords
        assert not np.array_equal(
            result[ymin:ymax, xmin:xmax],
            full_frame[ymin:ymax, xmin:xmax]
        )
    
    def test_process_frame_with_parsing(self):
        """Test frame processing with parsing mask."""
        full_frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        generated = np.random.randint(0, 255, (320, 320, 3), dtype=np.uint8)
        parsing = np.zeros((1080, 1920, 3), dtype=np.uint8)
        
        crop_coords = (100, 200, 500, 600)
        original_crop_size = (400, 400)
        
        result = self.processor.process_frame_with_parsing(
            full_frame,
            generated,
            crop_coords,
            original_crop_size,
            parsing
        )
        
        assert result.shape == full_frame.shape


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

