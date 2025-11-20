#!/usr/bin/env python3
"""
Convert ONNX models to Core ML
Using our knowledge of exact input/output shapes
"""

import coremltools as ct
import onnx
import numpy as np

def convert_audio_encoder():
    """
    Convert audio encoder
    Input: mel (1, 1, 80, 16)
    Output: emb (1, 512)
    """
    print("=" * 60)
    print("Converting Audio Encoder to Core ML")
    print("=" * 60)
    
    onnx_path = "model/sanders_full_onnx/models/audio_encoder.onnx"
    output_path = "swift_inference/AudioEncoder.mlpackage"
    
    # Load and check ONNX model
    onnx_model = onnx.load(onnx_path)
    print(f"\nONNX Model: {onnx_path}")
    print(f"Inputs: {[(i.name, [d.dim_value for d in i.type.tensor_type.shape.dim]) for i in onnx_model.graph.input]}")
    print(f"Outputs: {[(o.name, [d.dim_value for d in o.type.tensor_type.shape.dim]) for o in onnx_model.graph.output]}")
    
    # Convert
    print("\nConverting to Core ML...")
    model = ct.convert(
        onnx_path,
        source='onnx',
        minimum_deployment_target=ct.target.macOS13,
        compute_units=ct.ComputeUnit.ALL,  # Use CPU + GPU + Neural Engine
        convert_to='mlprogram'
    )
    
    # Save
    model.save(output_path)
    print(f"✓ Saved to {output_path}")
    
    return model

def convert_generator():
    """
    Convert U-Net generator
    Input: input (1, 6, 320, 320), audio (1, 32, 16, 16)
    Output: output (1, 3, 320, 320)
    """
    print("\n" + "=" * 60)
    print("Converting U-Net Generator to Core ML")
    print("=" * 60)
    
    onnx_path = "model/sanders_full_onnx/models/generator.onnx"
    output_path = "swift_inference/Generator.mlpackage"
    
    # Load and check ONNX model
    onnx_model = onnx.load(onnx_path)
    print(f"\nONNX Model: {onnx_path}")
    print(f"Inputs: {[(i.name, [d.dim_value for d in i.type.tensor_type.shape.dim]) for i in onnx_model.graph.input]}")
    print(f"Outputs: {[(o.name, [d.dim_value for d in o.type.tensor_type.shape.dim]) for o in onnx_model.graph.output]}")
    
    # Convert
    print("\nConverting to Core ML...")
    model = ct.convert(
        onnx_path,
        source='onnx',
        minimum_deployment_target=ct.target.macOS13,
        compute_units=ct.ComputeUnit.ALL,  # Use CPU + GPU + Neural Engine
        convert_to='mlprogram'
    )
    
    # Save
    model.save(output_path)
    print(f"✓ Saved to {output_path}")
    
    return model

def main():
    print("\n" + "=" * 60)
    print("ONNX to Core ML Conversion")
    print("=" * 60)
    print("\nKnown tensor shapes:")
    print("  Audio Encoder:")
    print("    Input: mel (1, 1, 80, 16)")
    print("    Output: emb (1, 512)")
    print("  Generator:")
    print("    Input: input (1, 6, 320, 320), audio (1, 32, 16, 16)")
    print("    Output: output (1, 3, 320, 320)")
    print()
    
    try:
        # Convert audio encoder
        audio_model = convert_audio_encoder()
        
        # Convert generator
        gen_model = convert_generator()
        
        print("\n" + "=" * 60)
        print("✓ Conversion Complete!")
        print("=" * 60)
        print("\nCore ML models created:")
        print("  • swift_inference/AudioEncoder.mlpackage")
        print("  • swift_inference/Generator.mlpackage")
        print("\nThese will use:")
        print("  ✓ Neural Engine (Apple Silicon)")
        print("  ✓ GPU acceleration")
        print("  ✓ CPU fallback")
        print("\nExpected performance: 20-30 FPS on M1 Pro!")
        
    except Exception as e:
        print(f"\nError during conversion: {e}")
        print("\nTroubleshooting:")
        print("  1. Check coremltools version: pip install coremltools==7.0")
        print("  2. Try: pip uninstall coremltools && pip install coremltools==7.0")
        print("  3. Make sure Python 3.9-3.11 (not 3.13)")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == '__main__':
    import sys
    sys.exit(main())

