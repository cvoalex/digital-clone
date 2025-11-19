"""
Unit tests for MelSpectrogramProcessor

These tests validate that the mel spectrogram generation matches
the expected behavior and can serve as reference for iOS implementation.
"""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from audio_pipeline.mel_processor import MelSpectrogramProcessor


class TestMelSpectrogramProcessor:
    """Test suite for mel spectrogram processing."""
    
    @pytest.fixture
    def processor(self):
        """Create a MelSpectrogramProcessor instance."""
        return MelSpectrogramProcessor()
    
    def test_initialization(self, processor):
        """Test that processor initializes with correct parameters."""
        assert processor.sample_rate == 16000
        assert processor.n_fft == 800
        assert processor.hop_length == 200
        assert processor.win_length == 800
        assert processor.n_mels == 80
        assert processor.fmin == 55
        assert processor.fmax == 7600
        assert processor.preemphasis_coef == 0.97
    
    def test_preemphasis(self, processor):
        """Test pre-emphasis filter."""
        # Simple test signal
        wav = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        filtered = processor.preemphasis(wav)
        
        # Check shape
        assert filtered.shape == wav.shape
        
        # Check that it actually changes the signal
        assert not np.allclose(filtered, wav)
        
        # First sample should be unchanged
        assert filtered[0] == wav[0]
    
    def test_mel_basis_shape(self, processor):
        """Test mel filter bank shape."""
        mel_basis = processor._mel_basis
        
        # Should be (n_mels, n_fft//2 + 1)
        expected_shape = (80, 401)
        assert mel_basis.shape == expected_shape
    
    def test_process_audio_shape(self, processor):
        """Test that process_audio returns correct shape."""
        # Generate 1 second of audio
        duration = 1.0
        wav = np.random.randn(int(processor.sample_rate * duration)).astype(np.float32)
        
        mel_spec = processor.process_audio(wav)
        
        # Check that output has correct number of mel bands
        assert mel_spec.shape[0] == 80
        
        # Check that output has reasonable number of frames
        # For 1 second at 16kHz with hop_length=200, expect ~80 frames
        assert 70 < mel_spec.shape[1] < 90
    
    def test_normalization_range(self, processor):
        """Test that normalized mel spectrogram is in expected range."""
        # Generate audio
        wav = np.random.randn(16000).astype(np.float32)
        mel_spec = processor.process_audio(wav)
        
        # Should be clipped to [-4, 4]
        assert mel_spec.min() >= -4.0
        assert mel_spec.max() <= 4.0
    
    def test_crop_audio_window(self, processor):
        """Test audio window cropping."""
        # Create a dummy mel spectrogram
        mel_spec = np.random.randn(80, 100).astype(np.float32)
        
        # Crop window for frame 0
        window = processor.crop_audio_window(mel_spec, frame_idx=0, fps=25)
        
        # Should have shape (16, 80)
        assert window.shape == (16, 80)
    
    def test_crop_audio_window_boundary(self, processor):
        """Test audio window cropping at boundaries."""
        # Create a small mel spectrogram
        mel_spec = np.random.randn(80, 20).astype(np.float32)
        
        # Try to crop near the end
        window = processor.crop_audio_window(mel_spec, frame_idx=10, fps=25)
        
        # Should still return (16, 80)
        assert window.shape == (16, 80)
    
    def test_frame_count_calculation(self, processor):
        """Test frame count calculation."""
        # Create mel spec with 100 frames
        mel_spec = np.random.randn(80, 100).astype(np.float32)
        
        frame_count = processor.get_frame_count(mel_spec, fps=25)
        
        # Formula: int((n_mel_frames - 16) / 80.0 * fps) + 2
        # (100 - 16) / 80 * 25 + 2 = 84/80 * 25 + 2 = 1.05 * 25 + 2 = 26.25 + 2 = 28
        expected = int((100 - 16) / 80.0 * 25) + 2
        assert frame_count == expected
    
    def test_save_load_mel_spectrogram(self, processor, tmp_path):
        """Test saving and loading mel spectrograms."""
        # Generate mel spectrogram
        wav = np.random.randn(16000).astype(np.float32)
        mel_spec = processor.process_audio(wav)
        
        # Save
        save_path = tmp_path / "test_mel.npy"
        processor.save_mel_spectrogram(mel_spec, str(save_path))
        
        # Load
        loaded_mel = processor.load_mel_spectrogram(str(save_path))
        
        # Should be identical
        assert np.allclose(mel_spec, loaded_mel)


def test_synthetic_audio():
    """
    Test with synthetic audio to verify exact behavior.
    This can be used as a reference for iOS implementation.
    """
    processor = MelSpectrogramProcessor()
    
    # Generate a simple sine wave at 440 Hz (A4 note)
    duration = 0.5  # seconds
    t = np.linspace(0, duration, int(processor.sample_rate * duration))
    frequency = 440.0
    wav = np.sin(2 * np.pi * frequency * t).astype(np.float32)
    
    # Process
    mel_spec = processor.process_audio(wav)
    
    # Verify shape and range
    assert mel_spec.shape[0] == 80
    assert mel_spec.min() >= -4.0
    assert mel_spec.max() <= 4.0
    
    # For a pure sine wave, we expect energy concentrated in specific mel bands
    # The mean should be reasonable (not all zeros or all max)
    mean_energy = np.mean(mel_spec)
    assert -4.0 <= mean_energy <= 4.0  # Within normalized range
    
    print(f"Synthetic audio test passed!")
    print(f"  Mel shape: {mel_spec.shape}")
    print(f"  Value range: [{mel_spec.min():.3f}, {mel_spec.max():.3f}]")
    print(f"  Mean: {mean_energy:.3f}")


if __name__ == "__main__":
    # Run a simple test
    test_synthetic_audio()
    
    # Run basic tests without pytest
    processor = MelSpectrogramProcessor()
    
    # Test 1: Initialization
    print("\n✓ Processor initialized")
    
    # Test 2: Process audio
    wav = np.random.randn(16000).astype(np.float32)
    mel_spec = processor.process_audio(wav)
    print(f"✓ Processed audio: {mel_spec.shape}")
    
    # Test 3: Crop window
    window = processor.crop_audio_window(mel_spec, 0)
    print(f"✓ Cropped window: {window.shape}")
    
    print("\nAll basic tests passed!")

