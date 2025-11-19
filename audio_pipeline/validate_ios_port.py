#!/usr/bin/env python3
"""
iOS Port Validation Script

This script helps validate that your iOS implementation produces outputs
that match the Python reference implementation.

Usage:
    python validate_ios_port.py <reference_dir> <ios_output_dir>
"""

import numpy as np
import sys
import json
from pathlib import Path
from typing import Tuple, Dict


def load_numpy_array(path: str) -> np.ndarray:
    """Load a numpy array from file."""
    return np.load(path)


def compute_difference(reference: np.ndarray, test: np.ndarray) -> Dict[str, float]:
    """
    Compute various difference metrics between two arrays.
    
    Returns:
        Dictionary with difference metrics
    """
    # Ensure same shape
    if reference.shape != test.shape:
        return {
            'error': f'Shape mismatch: reference={reference.shape}, test={test.shape}'
        }
    
    # Compute metrics
    abs_diff = np.abs(reference - test)
    
    metrics = {
        'max_abs_diff': float(np.max(abs_diff)),
        'mean_abs_diff': float(np.mean(abs_diff)),
        'median_abs_diff': float(np.median(abs_diff)),
        'std_abs_diff': float(np.std(abs_diff)),
        'max_rel_diff': float(np.max(abs_diff / (np.abs(reference) + 1e-8))),
        'mean_rel_diff': float(np.mean(abs_diff / (np.abs(reference) + 1e-8))),
    }
    
    return metrics


def validate_mel_spectrogram(ref_dir: Path, test_dir: Path) -> Tuple[bool, Dict]:
    """Validate mel spectrogram output."""
    print("\nValidating Mel Spectrogram...")
    print("-" * 60)
    
    ref_path = ref_dir / "mel_spectrogram.npy"
    test_path = test_dir / "mel_spectrogram.npy"
    
    if not ref_path.exists():
        return False, {'error': f'Reference file not found: {ref_path}'}
    
    if not test_path.exists():
        return False, {'error': f'Test file not found: {test_path}'}
    
    reference = load_numpy_array(str(ref_path))
    test = load_numpy_array(str(test_path))
    
    print(f"Reference shape: {reference.shape}")
    print(f"Test shape:      {test.shape}")
    print(f"Reference range: [{reference.min():.3f}, {reference.max():.3f}]")
    print(f"Test range:      [{test.min():.3f}, {test.max():.3f}]")
    
    metrics = compute_difference(reference, test)
    
    if 'error' in metrics:
        print(f"✗ {metrics['error']}")
        return False, metrics
    
    print(f"\nDifference Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.6f}")
    
    # Check tolerance
    tolerance = 1e-3  # More lenient for mobile implementations
    passed = metrics['max_abs_diff'] < tolerance
    
    if passed:
        print(f"\n✓ PASSED (max diff {metrics['max_abs_diff']:.6f} < {tolerance})")
    else:
        print(f"\n✗ FAILED (max diff {metrics['max_abs_diff']:.6f} >= {tolerance})")
    
    return passed, metrics


def validate_audio_features(ref_dir: Path, test_dir: Path) -> Tuple[bool, Dict]:
    """Validate audio encoder output."""
    print("\nValidating Audio Features...")
    print("-" * 60)
    
    ref_path = ref_dir / "audio_features_padded.npy"
    test_path = test_dir / "audio_features_padded.npy"
    
    if not ref_path.exists():
        return False, {'error': f'Reference file not found: {ref_path}'}
    
    if not test_path.exists():
        return False, {'error': f'Test file not found: {test_path}'}
    
    reference = load_numpy_array(str(ref_path))
    test = load_numpy_array(str(test_path))
    
    print(f"Reference shape: {reference.shape}")
    print(f"Test shape:      {test.shape}")
    print(f"Reference range: [{reference.min():.3f}, {reference.max():.3f}]")
    print(f"Test range:      [{test.min():.3f}, {test.max():.3f}]")
    
    metrics = compute_difference(reference, test)
    
    if 'error' in metrics:
        print(f"✗ {metrics['error']}")
        return False, metrics
    
    print(f"\nDifference Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.6f}")
    
    # Check tolerance
    tolerance = 1e-3
    passed = metrics['max_abs_diff'] < tolerance
    
    if passed:
        print(f"\n✓ PASSED (max diff {metrics['max_abs_diff']:.6f} < {tolerance})")
    else:
        print(f"\n✗ FAILED (max diff {metrics['max_abs_diff']:.6f} >= {tolerance})")
    
    return passed, metrics


def validate_frame_features(ref_dir: Path, test_dir: Path, num_frames: int = 10) -> Tuple[bool, Dict]:
    """Validate per-frame features."""
    print(f"\nValidating Frame Features (sampling {num_frames} frames)...")
    print("-" * 60)
    
    ref_frames_dir = ref_dir / "frames"
    test_frames_dir = test_dir / "frames"
    
    if not ref_frames_dir.exists():
        return False, {'error': f'Reference frames directory not found: {ref_frames_dir}'}
    
    if not test_frames_dir.exists():
        return False, {'error': f'Test frames directory not found: {test_frames_dir}'}
    
    # Get list of frame files
    ref_frame_files = sorted(ref_frames_dir.glob("frame_*_reshaped.npy"))
    
    if not ref_frame_files:
        return False, {'error': 'No reference frame files found'}
    
    # Sample frames evenly
    total_frames = len(ref_frame_files)
    sample_indices = np.linspace(0, total_frames - 1, min(num_frames, total_frames), dtype=int)
    
    all_metrics = []
    failed_frames = []
    
    for idx in sample_indices:
        ref_file = ref_frame_files[idx]
        test_file = test_frames_dir / ref_file.name
        
        if not test_file.exists():
            print(f"✗ Frame {idx}: Test file not found")
            failed_frames.append(idx)
            continue
        
        reference = load_numpy_array(str(ref_file))
        test = load_numpy_array(str(test_file))
        
        metrics = compute_difference(reference, test)
        
        if 'error' in metrics:
            print(f"✗ Frame {idx}: {metrics['error']}")
            failed_frames.append(idx)
        else:
            all_metrics.append(metrics)
            max_diff = metrics['max_abs_diff']
            status = "✓" if max_diff < 1e-3 else "✗"
            print(f"{status} Frame {idx}: max_diff={max_diff:.6f}")
    
    if not all_metrics:
        return False, {'error': 'No frames could be validated'}
    
    # Compute aggregate metrics
    aggregate = {
        'max_abs_diff': max(m['max_abs_diff'] for m in all_metrics),
        'mean_abs_diff': np.mean([m['mean_abs_diff'] for m in all_metrics]),
        'frames_tested': len(all_metrics),
        'frames_failed': len(failed_frames),
    }
    
    print(f"\nAggregate Metrics:")
    for key, value in aggregate.items():
        print(f"  {key}: {value}")
    
    tolerance = 1e-3
    passed = aggregate['max_abs_diff'] < tolerance and aggregate['frames_failed'] == 0
    
    if passed:
        print(f"\n✓ PASSED")
    else:
        print(f"\n✗ FAILED")
    
    return passed, aggregate


def main():
    if len(sys.argv) != 3:
        print("Usage: python validate_ios_port.py <reference_dir> <ios_output_dir>")
        print("\nExample:")
        print("  python validate_ios_port.py \\")
        print("    audio_pipeline/test_data/reference_output \\")
        print("    ios_outputs")
        sys.exit(1)
    
    ref_dir = Path(sys.argv[1])
    test_dir = Path(sys.argv[2])
    
    if not ref_dir.exists():
        print(f"Error: Reference directory not found: {ref_dir}")
        sys.exit(1)
    
    if not test_dir.exists():
        print(f"Error: Test directory not found: {test_dir}")
        sys.exit(1)
    
    print("="*70)
    print("iOS Port Validation")
    print("="*70)
    print(f"Reference: {ref_dir}")
    print(f"Test:      {test_dir}")
    
    results = {}
    
    # Validate mel spectrogram
    passed, metrics = validate_mel_spectrogram(ref_dir, test_dir)
    results['mel_spectrogram'] = {'passed': passed, 'metrics': metrics}
    
    # Validate audio features
    passed, metrics = validate_audio_features(ref_dir, test_dir)
    results['audio_features'] = {'passed': passed, 'metrics': metrics}
    
    # Validate frame features
    passed, metrics = validate_frame_features(ref_dir, test_dir)
    results['frame_features'] = {'passed': passed, 'metrics': metrics}
    
    # Summary
    print("\n" + "="*70)
    print("Validation Summary")
    print("="*70)
    
    all_passed = True
    for name, result in results.items():
        status = "✓ PASSED" if result['passed'] else "✗ FAILED"
        print(f"{status}: {name}")
        if not result['passed']:
            all_passed = False
    
    # Save results
    results_file = test_dir / "validation_results.json"
    with open(results_file, 'w') as f:
        # Convert numpy types to Python types for JSON serialization
        def convert(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        serializable_results = json.loads(
            json.dumps(results, default=convert)
        )
        json.dump(serializable_results, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    
    if all_passed:
        print("\n🎉 All validations passed! Your iOS implementation matches the reference.")
        return 0
    else:
        print("\n⚠ Some validations failed. Please review the metrics above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

