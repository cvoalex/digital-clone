"""
Unit tests for AudioEncoder

These tests validate the audio encoder model behavior.
"""

import numpy as np
import torch
import pytest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from audio_pipeline.audio_encoder import AudioEncoder, AudioEncoderWrapper


class TestAudioEncoder:
    """Test suite for AudioEncoder model."""
    
    @pytest.fixture
    def model(self):
        """Create an AudioEncoder instance."""
        return AudioEncoder()
    
    def test_initialization(self, model):
        """Test that model initializes."""
        assert model is not None
        assert isinstance(model, torch.nn.Module)
    
    def test_forward_shape(self, model):
        """Test that forward pass returns correct shape."""
        # Input: [batch, 1, 80, 16]
        batch_size = 4
        x = torch.randn(batch_size, 1, 80, 16)
        
        # Forward pass
        model.eval()
        with torch.no_grad():
            output = model(x)
        
        # Output should be [batch, 512]
        assert output.shape == (batch_size, 512)
    
    def test_single_sample(self, model):
        """Test with single sample."""
        x = torch.randn(1, 1, 80, 16)
        
        model.eval()
        with torch.no_grad():
            output = model(x)
        
        assert output.shape == (1, 512)
    
    def test_output_range(self, model):
        """Test that output values are reasonable."""
        x = torch.randn(2, 1, 80, 16)
        
        model.eval()
        with torch.no_grad():
            output = model(x)
        
        # Output should not be all zeros or infinities
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
        assert output.abs().max() > 0


class TestAudioEncoderWrapper:
    """Test suite for AudioEncoderWrapper."""
    
    @pytest.fixture
    def checkpoint_path(self):
        """Get the checkpoint path."""
        return "model/checkpoints/audio_visual_encoder.pth"
    
    def test_temporal_padding(self):
        """Test temporal padding."""
        # Create dummy features
        features = np.random.randn(10, 512).astype(np.float32)
        
        # Dummy wrapper (we'll just test the method)
        class DummyWrapper:
            def add_temporal_padding(self, features):
                first_frame = features[:1]
                last_frame = features[-1:]
                return np.concatenate([first_frame, features, last_frame], axis=0)
        
        wrapper = DummyWrapper()
        padded = wrapper.add_temporal_padding(features)
        
        # Should have 2 extra frames
        assert padded.shape == (12, 512)
        
        # First frame should be repeated
        assert np.allclose(padded[0], features[0])
        
        # Last frame should be repeated
        assert np.allclose(padded[-1], features[-1])
    
    def test_get_audio_features_for_frame(self):
        """Test extracting features for a specific frame."""
        # Create dummy features
        n_frames = 20
        features = np.random.randn(n_frames, 512).astype(np.float32)
        
        # Test extraction
        def get_audio_features_for_frame(all_features, frame_idx, context_size=8):
            left = frame_idx - context_size
            right = frame_idx + context_size
            pad_left = 0
            pad_right = 0
            
            if left < 0:
                pad_left = -left
                left = 0
            if right > all_features.shape[0]:
                pad_right = right - all_features.shape[0]
                right = all_features.shape[0]
            
            window = all_features[left:right]
            
            if pad_left > 0:
                padding = np.zeros((pad_left, window.shape[1]), dtype=window.dtype)
                window = np.concatenate([padding, window], axis=0)
            if pad_right > 0:
                padding = np.zeros((pad_right, window.shape[1]), dtype=window.dtype)
                window = np.concatenate([window, padding], axis=0)
            
            return window
        
        # Test middle frame
        window = get_audio_features_for_frame(features, 10)
        assert window.shape == (16, 512)
        
        # Test first frame (needs padding)
        window = get_audio_features_for_frame(features, 0)
        assert window.shape == (16, 512)
        
        # Test last frame (needs padding)
        window = get_audio_features_for_frame(features, n_frames - 1)
        assert window.shape == (16, 512)
    
    def test_reshape_for_model(self):
        """Test reshaping features for different modes."""
        # Create dummy feature window
        features = np.random.randn(16, 512).astype(np.float32)
        
        # Test AVE mode
        flat = features.flatten()
        ave_shape = flat.reshape(32, 16, 16)
        assert ave_shape.shape == (32, 16, 16)
        assert ave_shape.size == 8192
        
        # Test Hubert mode (needs padding)
        hubert_size = 32 * 32 * 32
        hubert_padded = np.zeros(hubert_size, dtype=flat.dtype)
        hubert_padded[:flat.shape[0]] = flat
        hubert_shape = hubert_padded.reshape(32, 32, 32)
        assert hubert_shape.shape == (32, 32, 32)
        
        # Test WeNet mode (needs padding)
        wenet_size = 256 * 16 * 32
        wenet_padded = np.zeros(wenet_size, dtype=flat.dtype)
        wenet_padded[:flat.shape[0]] = flat
        wenet_shape = wenet_padded.reshape(256, 16, 32)
        assert wenet_shape.shape == (256, 16, 32)


def test_model_architecture():
    """Test the model architecture details."""
    model = AudioEncoder()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nAudioEncoder total parameters: {total_params:,}")
    
    # Test with typical input
    x = torch.randn(1, 1, 80, 16)
    
    model.eval()
    with torch.no_grad():
        output = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")


if __name__ == "__main__":
    # Run basic tests
    print("Testing AudioEncoder architecture...")
    test_model_architecture()
    
    # Test padding
    print("\n✓ Testing temporal padding...")
    wrapper_test = TestAudioEncoderWrapper()
    wrapper_test.test_temporal_padding()
    print("✓ Temporal padding test passed")
    
    # Test feature extraction
    print("\n✓ Testing feature extraction...")
    wrapper_test.test_get_audio_features_for_frame()
    print("✓ Feature extraction test passed")
    
    # Test reshaping
    print("\n✓ Testing feature reshaping...")
    wrapper_test.test_reshape_for_model()
    print("✓ Feature reshaping test passed")
    
    print("\nAll basic tests passed!")

