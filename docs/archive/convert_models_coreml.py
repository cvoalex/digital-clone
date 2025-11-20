#!/usr/bin/env python3
"""
Convert ONNX models to Core ML using coremltools
"""

import coremltools as ct
from coremltools.converters.mil import Builder as mb

print(f"coremltools version: {ct.__version__}")
print()

def convert_audio_encoder():
    """
    Convert audio encoder ONNX to Core ML
    Input: mel (1, 1, 80, 16)
    Output: emb (1, 512)
    """
    print("=" * 60)
    print("Converting Audio Encoder")
    print("=" * 60)
    
    onnx_path = 'model/sanders_full_onnx/models/audio_encoder.onnx'
    output_path = 'swift_inference/AudioEncoder.mlpackage'
    
    print(f"Input: {onnx_path}")
    print(f"Output: {output_path}")
    print()
    
    # Convert with fixed input shape
    mlmodel = ct.converters.onnx.convert(
        model=onnx_path,
        minimum_deployment_target=ct.target.macOS13,
        inputs=[ct.TensorType(name='mel', shape=(1, 1, 80, 16))],
        compute_units=ct.ComputeUnit.ALL  # Use CPU + GPU + Neural Engine
    )
    
    # Save
    mlmodel.save(output_path)
    print(f"✓ Saved to {output_path}")
    print(f"  Compute units: ALL (CPU + GPU + Neural Engine)")
    print()
    
    return mlmodel

def convert_generator():
    """
    Convert U-Net generator ONNX to Core ML
    Input: input (1, 6, 320, 320), audio (1, 32, 16, 16)
    Output: output (1, 3, 320, 320)
    """
    print("=" * 60)
    print("Converting U-Net Generator")
    print("=" * 60)
    
    onnx_path = 'model/sanders_full_onnx/models/generator.onnx'
    output_path = 'swift_inference/Generator.mlpackage'
    
    print(f"Input: {onnx_path}")
    print(f"Output: {output_path}")
    print()
    
    # Convert with fixed input shapes
    mlmodel = ct.converters.onnx.convert(
        model=onnx_path,
        minimum_deployment_target=ct.target.macOS13,
        inputs=[
            ct.TensorType(name='input', shape=(1, 6, 320, 320)),
            ct.TensorType(name='audio', shape=(1, 32, 16, 16))
        ],
        compute_units=ct.ComputeUnit.ALL  # Use CPU + GPU + Neural Engine
    )
    
    # Save
    mlmodel.save(output_path)
    print(f"✓ Saved to {output_path}")
    print(f"  Compute units: ALL (CPU + GPU + Neural Engine)")
    print()
    
    return mlmodel

def main():
    print("=" * 60)
    print("ONNX to Core ML Conversion")
    print("=" * 60)
    print()
    
    try:
        # Convert audio encoder
        audio_model = convert_audio_encoder()
        
        # Convert generator
        gen_model = convert_generator()
        
        print("=" * 60)
        print("✅ SUCCESS - Both models converted!")
        print("=" * 60)
        print()
        print("Core ML models created:")
        print("  • swift_inference/AudioEncoder.mlpackage")
        print("  • swift_inference/Generator.mlpackage")
        print()
        print("These models will use:")
        print("  ✓ Neural Engine (Apple Silicon)")
        print("  ✓ GPU (Metal)")
        print("  ✓ CPU (fallback)")
        print()
        print("Expected performance: 20-30 FPS on M1/M2! 🚀")
        print()
        print("Next:")
        print("  cd swift_inference")
        print("  swift build --configuration release")
        print("  time .build/release/swift-infer --frames 250")
        
    except Exception as e:
        print(f"Error: {e}")
        print()
        print("Troubleshooting:")
        print("  - Make sure coremltools is installed: pip install coremltools")
        print("  - Try with Python 3.9-3.11 if using 3.13")
        print("  - Or use Xcode to convert (drag ONNX files into project)")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == '__main__':
    import sys
    sys.exit(main())

