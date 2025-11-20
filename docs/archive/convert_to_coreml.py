#!/usr/bin/env python3
"""
Convert ONNX models to Core ML for Swift/macOS implementation
"""

import coremltools as ct
import sys

def convert_model(onnx_path, output_path, model_name):
    """Convert ONNX model to Core ML"""
    print(f"Converting {onnx_path} to Core ML...")
    
    # Load ONNX model
    model = ct.converters.onnx.convert(
        model=onnx_path,
        minimum_deployment_target=ct.target.macOS13,
    )
    
    # Save Core ML model
    model.save(output_path)
    print(f"✓ Saved to {output_path}")
    
    return model

def main():
    sanders_dir = "model/sanders_full_onnx"
    output_dir = "swift_inference"
    
    print("=" * 60)
    print("Converting ONNX Models to Core ML")
    print("=" * 60)
    
    # Convert audio encoder
    print("\n[1/2] Audio Encoder...")
    audio_encoder = convert_model(
        f"{sanders_dir}/models/audio_encoder.onnx",
        f"{output_dir}/AudioEncoder.mlpackage",
        "AudioEncoder"
    )
    
    print("\nAudio Encoder Info:")
    print(f"  Inputs: {audio_encoder.input_description}")
    print(f"  Outputs: {audio_encoder.output_description}")
    
    # Convert U-Net generator
    print("\n[2/2] U-Net Generator...")
    generator = convert_model(
        f"{sanders_dir}/models/generator.onnx",
        f"{output_dir}/Generator.mlpackage",
        "Generator"
    )
    
    print("\nGenerator Info:")
    print(f"  Inputs: {generator.input_description}")
    print(f"  Outputs: {generator.output_description}")
    
    print("\n" + "=" * 60)
    print("✓ Conversion complete!")
    print("=" * 60)
    print("\nCore ML models created:")
    print(f"  • {output_dir}/AudioEncoder.mlpackage")
    print(f"  • {output_dir}/Generator.mlpackage")
    print("\nThese can now be used in Swift with Core ML!")

if __name__ == '__main__':
    try:
        main()
    except ImportError as e:
        print("Error: coremltools not installed")
        print("Install with: pip install coremltools")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

