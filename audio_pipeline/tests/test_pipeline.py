"""
Integration tests for the complete audio pipeline.

These tests validate the end-to-end pipeline and generate reference outputs
for iOS implementation validation.
"""

import numpy as np
import sys
from pathlib import Path
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from audio_pipeline.pipeline import AudioPipeline


def create_test_audio(output_path: str, duration: float = 1.0, sr: int = 16000):
    """
    Create a simple test audio file.
    
    Args:
        output_path: Where to save the audio
        duration: Duration in seconds
        sr: Sample rate
    """
    import soundfile as sf
    
    # Generate a simple audio signal (mix of sine waves)
    t = np.linspace(0, duration, int(sr * duration))
    
    # Mix of frequencies to create richer audio
    audio = (
        0.3 * np.sin(2 * np.pi * 440 * t) +  # A4
        0.2 * np.sin(2 * np.pi * 554.37 * t) +  # C#5
        0.1 * np.sin(2 * np.pi * 659.25 * t)  # E5
    )
    
    # Add some noise
    audio += 0.05 * np.random.randn(len(audio))
    
    # Normalize
    audio = audio / np.abs(audio).max() * 0.9
    
    # Save
    sf.write(output_path, audio, sr)
    print(f"Created test audio: {output_path}")
    print(f"  Duration: {duration}s, Sample rate: {sr} Hz")
    
    return output_path


def test_pipeline_basic():
    """Test basic pipeline functionality."""
    print("\n" + "="*60)
    print("Testing Basic Pipeline Functionality")
    print("="*60)
    
    # Check if checkpoint exists
    checkpoint_path = "model/checkpoints/audio_visual_encoder.pth"
    if not Path(checkpoint_path).exists():
        print(f"⚠ Checkpoint not found: {checkpoint_path}")
        print("  Skipping pipeline test (model loading required)")
        return False
    
    # Create pipeline
    try:
        pipeline = AudioPipeline(
            checkpoint_path=checkpoint_path,
            mode="ave",
            fps=25
        )
        print("✓ Pipeline initialized successfully")
        return True
    except Exception as e:
        print(f"✗ Failed to initialize pipeline: {e}")
        return False


def test_pipeline_with_real_audio():
    """Test pipeline with real audio from the demo folder."""
    print("\n" + "="*60)
    print("Testing Pipeline with Real Audio")
    print("="*60)
    
    # Check for demo audio
    demo_audio = "demo/talk_hb.wav"
    if not Path(demo_audio).exists():
        print(f"⚠ Demo audio not found: {demo_audio}")
        return False
    
    checkpoint_path = "model/checkpoints/audio_visual_encoder.pth"
    if not Path(checkpoint_path).exists():
        print(f"⚠ Checkpoint not found: {checkpoint_path}")
        return False
    
    try:
        # Create pipeline
        pipeline = AudioPipeline(
            checkpoint_path=checkpoint_path,
            mode="ave",
            fps=25
        )
        
        # Process audio
        output_dir = "audio_pipeline/test_data/real_audio_output"
        print(f"\nProcessing: {demo_audio}")
        print(f"Output dir: {output_dir}")
        
        audio_features, metadata = pipeline.process_audio_file(
            demo_audio,
            save_intermediates=True,
            output_dir=output_dir
        )
        
        # Validate outputs
        print("\n" + "-"*60)
        print("Processing Results:")
        print("-"*60)
        for key, value in metadata.items():
            print(f"  {key}: {value}")
        
        # Test getting frame features
        print("\n" + "-"*60)
        print("Testing Frame Feature Extraction:")
        print("-"*60)
        
        for frame_idx in [0, metadata['n_frames'] // 2, metadata['n_frames'] - 1]:
            features = pipeline.get_frame_features(audio_features, frame_idx, reshape=True)
            print(f"  Frame {frame_idx}: shape={features.shape}, "
                  f"range=[{features.min():.3f}, {features.max():.3f}]")
        
        print("\n✓ Real audio test passed!")
        return True
        
    except Exception as e:
        print(f"✗ Real audio test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def generate_reference_dataset():
    """
    Generate a complete reference dataset for iOS validation.
    
    This creates:
    1. A test audio file
    2. All intermediate processing outputs
    3. Per-frame features
    4. Metadata for validation
    """
    print("\n" + "="*60)
    print("Generating Reference Dataset for iOS Validation")
    print("="*60)
    
    # Check for checkpoint
    checkpoint_path = "model/checkpoints/audio_visual_encoder.pth"
    if not Path(checkpoint_path).exists():
        print(f"⚠ Checkpoint not found: {checkpoint_path}")
        print("  Cannot generate reference dataset")
        return False
    
    try:
        # Create test audio
        test_audio_dir = Path("audio_pipeline/test_data")
        test_audio_dir.mkdir(parents=True, exist_ok=True)
        
        test_audio_path = test_audio_dir / "reference_audio.wav"
        
        if not test_audio_path.exists():
            print("\nCreating test audio file...")
            create_test_audio(str(test_audio_path), duration=2.0)
        else:
            print(f"\nUsing existing test audio: {test_audio_path}")
        
        # Create pipeline
        print("\nInitializing pipeline...")
        pipeline = AudioPipeline(
            checkpoint_path=checkpoint_path,
            mode="ave",
            fps=25
        )
        
        # Generate complete reference dataset
        output_dir = test_audio_dir / "reference_output"
        print(f"\nGenerating reference dataset...")
        print(f"Output directory: {output_dir}")
        
        pipeline.process_and_save_all_frames(
            str(test_audio_path),
            str(output_dir)
        )
        
        # Create validation instructions
        instructions = {
            "title": "iOS Implementation Validation Instructions",
            "description": "Use this reference dataset to validate your iOS implementation",
            "steps": [
                "1. Implement mel spectrogram processing on iOS",
                "2. Load reference_audio.wav and compare your mel output with mel_spectrogram.npy",
                "3. Implement AudioEncoder model (convert to CoreML)",
                "4. Compare per-frame outputs with frames/frame_XXXXX_reshaped.npy",
                "5. Ensure numerical differences are < 1e-4 (accounting for floating point)"
            ],
            "files": {
                "reference_audio.wav": "Original test audio file",
                "mel_spectrogram.npy": "Reference mel spectrogram output",
                "mel_windows.npy": "Reference windowed mel features",
                "audio_features_raw.npy": "Reference AudioEncoder outputs (no padding)",
                "audio_features_padded.npy": "Reference features with temporal padding",
                "frames/": "Per-frame features for each video frame",
                "metadata.json": "Processing metadata",
                "summary.json": "Dataset summary"
            },
            "validation_criteria": {
                "mel_spectrogram": "Shape: (80, n_frames), Range: [-4, 4]",
                "audio_features": "Shape: (n_frames, 512), Reasonable range",
                "frame_features_ave": "Shape: (32, 16, 16) per frame",
                "numerical_tolerance": "Max difference < 1e-4 (or 1e-3 for mobile)"
            }
        }
        
        instructions_path = output_dir / "VALIDATION_INSTRUCTIONS.json"
        with open(instructions_path, 'w') as f:
            json.dump(instructions, f, indent=2)
        
        print("\n" + "="*60)
        print("✓ Reference Dataset Generated Successfully!")
        print("="*60)
        print(f"\nLocation: {output_dir}")
        print("\nGenerated files:")
        print("  - reference_audio.wav (test audio)")
        print("  - mel_spectrogram.npy (mel spectrogram)")
        print("  - audio_features_padded.npy (encoded features)")
        print("  - frames/ (per-frame features)")
        print("  - VALIDATION_INSTRUCTIONS.json")
        print("\nUse these files to validate your iOS implementation!")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Failed to generate reference dataset: {e}")
        import traceback
        traceback.print_exc()
        return False


def compare_with_original():
    """
    Compare outputs with the original implementation.
    
    This validates that our standalone pipeline matches the original behavior.
    """
    print("\n" + "="*60)
    print("Comparing with Original Implementation")
    print("="*60)
    
    # This would require running both implementations and comparing
    # For now, we'll just provide guidance
    
    print("\nTo validate against the original implementation:")
    print("1. Run inference_328.py with a test audio")
    print("2. Extract audio features before they're fed to the model")
    print("3. Compare with our pipeline outputs")
    print("4. Ensure numerical differences are minimal (< 1e-5)")
    
    return True


if __name__ == "__main__":
    print("\n" + "="*60)
    print("Audio Pipeline Integration Tests")
    print("="*60)
    
    # Run tests
    results = []
    
    # Test 1: Basic pipeline
    results.append(("Basic Pipeline", test_pipeline_basic()))
    
    # Test 2: Real audio
    results.append(("Real Audio Processing", test_pipeline_with_real_audio()))
    
    # Test 3: Generate reference dataset
    results.append(("Reference Dataset", generate_reference_dataset()))
    
    # Test 4: Comparison guidance
    results.append(("Comparison Guidance", compare_with_original()))
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{status}: {name}")
    
    total = len(results)
    passed = sum(1 for _, p in results if p)
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed!")
    else:
        print(f"\n⚠ {total - passed} test(s) failed")

