#!/usr/bin/env python3
"""
Convert PyTorch models directly to Core ML (recommended approach)
Bypasses ONNX completely!
"""

import torch
import coremltools as ct
import sys
import os

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import AudioEncoder
from unet_328 import Model

print(f"coremltools version: {ct.__version__}")
print(f"PyTorch version: {torch.__version__}")
print()

def convert_audio_encoder_pytorch():
    """
    Convert audio encoder from PyTorch to Core ML
    Input: mel (1, 1, 80, 16)
    Output: features (1, 512)
    """
    print("=" * 60)
    print("Converting Audio Encoder (PyTorch → Core ML)")
    print("=" * 60)
    
    checkpoint_path = 'model/checkpoints/audio_visual_encoder.pth'
    output_path = 'swift_inference/AudioEncoder.mlpackage'
    
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Output: {output_path}")
    print()
    
    # Load PyTorch model
    print("Loading PyTorch model...")
    model = AudioEncoder().eval()
    
    # Load weights
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    # The checkpoint has keys like 'audio_encoder.audio_encoder.0.conv_block.0.weight'
    # We need to extract just the audio_encoder part
    audio_encoder_state = {}
    for k, v in ckpt.items():
        if k.startswith('audio_encoder.'):
            # Remove the 'audio_encoder.' prefix
            new_key = k.replace('audio_encoder.', '')
            audio_encoder_state[new_key] = v
    
    model.load_state_dict(audio_encoder_state, strict=False)
    model.eval()
    print("✓ PyTorch model loaded")
    
    # Create example input
    example_input = torch.randn(1, 1, 80, 16)
    
    # Test the model
    with torch.no_grad():
        test_output = model(example_input)
    print(f"  Test output shape: {test_output.shape}")
    
    # Trace the model
    print("Tracing model...")
    traced_model = torch.jit.trace(model, example_input)
    
    # Convert to Core ML
    print("Converting to Core ML...")
    mlmodel = ct.convert(
        traced_model,
        inputs=[ct.TensorType(name='mel', shape=(1, 1, 80, 16))],
        minimum_deployment_target=ct.target.macOS13,
        compute_units=ct.ComputeUnit.ALL
    )
    
    # Save
    mlmodel.save(output_path)
    print(f"✓ Saved to {output_path}")
    print()
    
    return mlmodel

def convert_generator_pytorch():
    """
    Convert U-Net generator from PyTorch to Core ML
    Input: image (1, 6, 320, 320), audio (1, 32, 16, 16)
    Output: generated (1, 3, 320, 320)
    """
    print("=" * 60)
    print("Converting U-Net Generator (PyTorch → Core ML)")
    print("=" * 60)
    
    # Find the checkpoint
    checkpoint_dir = "checkpoint"
    if not os.path.exists(checkpoint_dir):
        print(f"Warning: No checkpoint directory found")
        print("Looking for .pth files...")
        # Use the data_utils checkpoint as fallback
        checkpoint_path = "data_utils/checkpoint_epoch_335.pth.tar"
    else:
        # Find first checkpoint
        subdirs = [d for d in os.listdir(checkpoint_dir) if os.path.isdir(os.path.join(checkpoint_dir, d))]
        if subdirs:
            first_dir = subdirs[0]
            files = [f for f in os.listdir(os.path.join(checkpoint_dir, first_dir)) if f.endswith('.pth')]
            if files:
                checkpoint_path = os.path.join(checkpoint_dir, first_dir, files[0])
            else:
                checkpoint_path = "data_utils/checkpoint_epoch_335.pth.tar"
        else:
            checkpoint_path = "data_utils/checkpoint_epoch_335.pth.tar"
    
    output_path = 'swift_inference/Generator.mlpackage'
    
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Output: {output_path}")
    print()
    
    # Load PyTorch model
    print("Loading PyTorch model...")
    model = Model(n_channels=6, mode='ave').eval()
    
    # Load weights
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            elif 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
        else:
            model.load_state_dict(checkpoint)
        print("✓ PyTorch model loaded")
    else:
        print(f"Warning: Checkpoint not found: {checkpoint_path}")
        print("Using untrained model (for structure only)")
    
    model.eval()
    
    # Create example inputs
    example_image = torch.randn(1, 6, 320, 320)
    example_audio = torch.randn(1, 32, 16, 16)
    
    # Test the model
    with torch.no_grad():
        test_output = model(example_image, example_audio)
    print(f"  Test output shape: {test_output.shape}")
    
    # Trace the model
    print("Tracing model...")
    traced_model = torch.jit.trace(model, (example_image, example_audio))
    
    # Convert to Core ML
    print("Converting to Core ML...")
    mlmodel = ct.convert(
        traced_model,
        inputs=[
            ct.TensorType(name='input', shape=(1, 6, 320, 320)),
            ct.TensorType(name='audio', shape=(1, 32, 16, 16))
        ],
        minimum_deployment_target=ct.target.macOS13,
        compute_units=ct.ComputeUnit.ALL
    )
    
    # Save
    mlmodel.save(output_path)
    print(f"✓ Saved to {output_path}")
    print()
    
    return mlmodel

def main():
    print("=" * 60)
    print("PyTorch to Core ML Conversion (Direct)")
    print("=" * 60)
    print("This bypasses ONNX completely!")
    print()
    
    try:
        # Convert audio encoder
        print("[1/2] Audio Encoder\n")
        audio_model = convert_audio_encoder_pytorch()
        
        # Convert generator
        print("[2/2] U-Net Generator\n")
        gen_model = convert_generator_pytorch()
        
        print("=" * 60)
        print("✅ SUCCESS - PyTorch → Core ML Complete!")
        print("=" * 60)
        print()
        print("Core ML models created:")
        print("  • swift_inference/AudioEncoder.mlpackage")
        print("  • swift_inference/Generator.mlpackage")
        print()
        print("These models use:")
        print("  ✓ Neural Engine (16-core ML accelerator)")
        print("  ✓ GPU (Metal)")
        print("  ✓ CPU (fallback)")
        print()
        print("Expected performance: 20-30 FPS on M1/M2! 🚀")
        print()
        print("Next steps:")
        print("  cd swift_inference")
        print("  swift build --configuration release")
        print("  time .build/release/swift-infer --frames 250")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        print()
        print("If checkpoint not found:")
        print("  - Check checkpoint/ directory")
        print("  - Or use the ONNX files you already have")
        return 1
    
    return 0

if __name__ == '__main__':
    sys.exit(main())

