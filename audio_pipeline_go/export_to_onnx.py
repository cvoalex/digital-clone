#!/usr/bin/env python3
"""
Export AudioEncoder model to ONNX format for Go implementation.
"""

import torch
import sys
import os

# Add parent directory to path to import audio_pipeline
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from audio_pipeline.audio_encoder import AudioEncoder


def export_audio_encoder_to_onnx(
    checkpoint_path: str,
    output_path: str,
    opset_version: int = 12
):
    """
    Export AudioEncoder model to ONNX format.
    
    Args:
        checkpoint_path: Path to PyTorch checkpoint
        output_path: Where to save ONNX model
        opset_version: ONNX opset version (12 is widely supported)
    """
    print("=" * 70)
    print("Exporting AudioEncoder to ONNX")
    print("=" * 70)
    
    # Load model
    print(f"\n1. Loading PyTorch model from: {checkpoint_path}")
    model = AudioEncoder()
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Handle checkpoint format
    state_dict = {}
    for k, v in checkpoint.items():
        if k.startswith('audio_encoder.'):
            state_dict[k] = v
        else:
            state_dict[f'audio_encoder.{k}'] = v
    
    model.load_state_dict(state_dict)
    model.eval()
    
    print(f"   ✓ Model loaded successfully")
    print(f"   ✓ Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create example input
    print(f"\n2. Creating example input")
    batch_size = 1
    channels = 1
    n_mels = 80
    n_frames = 16
    
    example_input = torch.randn(batch_size, channels, n_mels, n_frames)
    print(f"   ✓ Input shape: {example_input.shape}")
    
    # Test forward pass
    print(f"\n3. Testing forward pass")
    with torch.no_grad():
        output = model(example_input)
    print(f"   ✓ Output shape: {output.shape}")
    print(f"   ✓ Output range: [{output.min():.4f}, {output.max():.4f}]")
    
    # Export to ONNX
    print(f"\n4. Exporting to ONNX format")
    print(f"   Output: {output_path}")
    print(f"   Opset version: {opset_version}")
    
    torch.onnx.export(
        model,
        example_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['mel_input'],
        output_names=['features'],
        dynamic_axes={
            'mel_input': {0: 'batch_size'},
            'features': {0: 'batch_size'}
        }
    )
    
    print(f"   ✓ ONNX export successful!")
    
    # Verify the ONNX model
    print(f"\n5. Verifying ONNX model")
    import onnx
    
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print(f"   ✓ ONNX model verification passed")
    
    # Print model info
    print(f"\n6. ONNX Model Information")
    print(f"   IR version: {onnx_model.ir_version}")
    print(f"   Producer: {onnx_model.producer_name}")
    print(f"   Opset: {onnx_model.opset_import[0].version}")
    
    print(f"\n   Inputs:")
    for input_tensor in onnx_model.graph.input:
        print(f"     - {input_tensor.name}")
        shape = [dim.dim_value if dim.dim_value > 0 else 'dynamic' 
                 for dim in input_tensor.type.tensor_type.shape.dim]
        print(f"       Shape: {shape}")
    
    print(f"\n   Outputs:")
    for output_tensor in onnx_model.graph.output:
        print(f"     - {output_tensor.name}")
        shape = [dim.dim_value if dim.dim_value > 0 else 'dynamic' 
                 for dim in output_tensor.type.tensor_type.shape.dim]
        print(f"       Shape: {shape}")
    
    # Test ONNX Runtime inference
    print(f"\n7. Testing ONNX Runtime inference")
    import onnxruntime as ort
    
    session = ort.InferenceSession(output_path)
    
    # Run inference
    ort_inputs = {session.get_inputs()[0].name: example_input.numpy()}
    ort_outputs = session.run(None, ort_inputs)
    
    print(f"   ✓ ONNX Runtime inference successful")
    print(f"   ✓ Output shape: {ort_outputs[0].shape}")
    print(f"   ✓ Output range: [{ort_outputs[0].min():.4f}, {ort_outputs[0].max():.4f}]")
    
    # Compare PyTorch vs ONNX outputs
    print(f"\n8. Comparing PyTorch vs ONNX outputs")
    diff = torch.abs(output - torch.from_numpy(ort_outputs[0]))
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    print(f"   Max difference: {max_diff:.6f}")
    print(f"   Mean difference: {mean_diff:.6f}")
    
    if max_diff < 1e-5:
        print(f"   ✓ Outputs match perfectly!")
    elif max_diff < 1e-3:
        print(f"   ✓ Outputs match within acceptable tolerance")
    else:
        print(f"   ⚠ Warning: Larger than expected difference")
    
    print("\n" + "=" * 70)
    print("✅ Export Complete!")
    print("=" * 70)
    print(f"\nONNX model saved to: {output_path}")
    print(f"Ready for use in Go with ONNX Runtime")
    
    return output_path


if __name__ == "__main__":
    checkpoint_path = "../model/checkpoints/audio_visual_encoder.pth"
    output_path = "models/audio_encoder.onnx"
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        sys.exit(1)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    export_audio_encoder_to_onnx(checkpoint_path, output_path)

