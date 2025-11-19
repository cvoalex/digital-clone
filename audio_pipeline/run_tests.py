#!/usr/bin/env python3
"""
Convenience script to run all tests and generate reference outputs.

Usage:
    python run_tests.py                    # Run all tests
    python run_tests.py --generate-ref     # Generate reference dataset
    python run_tests.py --test-only        # Run tests without generation
"""

import sys
import argparse
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def run_unit_tests():
    """Run unit tests."""
    print("\n" + "="*70)
    print("Running Unit Tests")
    print("="*70)
    
    try:
        import pytest
        # Run pytest programmatically
        exit_code = pytest.main([
            'tests/',
            '-v',
            '--tb=short',
            '--color=yes'
        ])
        return exit_code == 0
    except ImportError:
        print("pytest not installed, running tests manually...")
        
        # Run tests manually
        from tests import test_mel_processor, test_audio_encoder
        
        try:
            test_mel_processor.test_synthetic_audio()
            print("✓ Mel processor tests passed")
            
            test_audio_encoder.test_model_architecture()
            print("✓ Audio encoder tests passed")
            
            return True
        except Exception as e:
            print(f"✗ Tests failed: {e}")
            return False


def run_integration_tests():
    """Run integration tests."""
    print("\n" + "="*70)
    print("Running Integration Tests")
    print("="*70)
    
    from tests.test_pipeline import (
        test_pipeline_basic,
        test_pipeline_with_real_audio
    )
    
    results = []
    results.append(("Basic Pipeline", test_pipeline_basic()))
    results.append(("Real Audio", test_pipeline_with_real_audio()))
    
    return all(result for _, result in results)


def generate_reference_dataset():
    """Generate reference dataset for iOS validation."""
    print("\n" + "="*70)
    print("Generating Reference Dataset")
    print("="*70)
    
    from tests.test_pipeline import generate_reference_dataset
    
    return generate_reference_dataset()


def print_summary(results):
    """Print test summary."""
    print("\n" + "="*70)
    print("Test Summary")
    print("="*70)
    
    for name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{status}: {name}")
    
    total = len(results)
    passed = sum(1 for p in results.values() if p)
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed!")
        return True
    else:
        print(f"\n⚠ {total - passed} test(s) failed")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run audio pipeline tests")
    parser.add_argument(
        '--generate-ref',
        action='store_true',
        help='Generate reference dataset for iOS validation'
    )
    parser.add_argument(
        '--test-only',
        action='store_true',
        help='Run tests without generating reference dataset'
    )
    parser.add_argument(
        '--unit-only',
        action='store_true',
        help='Run only unit tests'
    )
    
    args = parser.parse_args()
    
    results = {}
    
    # Run unit tests
    if not args.generate_ref:
        results['Unit Tests'] = run_unit_tests()
        
        if not args.unit_only:
            results['Integration Tests'] = run_integration_tests()
    
    # Generate reference dataset
    if args.generate_ref or (not args.test_only and not args.unit_only):
        results['Reference Dataset Generation'] = generate_reference_dataset()
    
    # Print summary
    success = print_summary(results)
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())

