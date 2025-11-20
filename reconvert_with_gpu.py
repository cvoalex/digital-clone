#!/usr/bin/env python3
"""
Reconvert models with GPU/Neural Engine support enabled
"""

import torch
import coremltools as ct
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import AudioEncoder
from unet_328 import Model

print("Reconverting models with GPU/Neural Engine support...")
print()

# Audio Encoder
print("[1/2] Audio Encoder...")
audio_model = AudioEncoder().eval()

# Load weights
ckpt = torch.load('model/checkpoints/audio_visual_encoder.pth', map_location='cpu')
audio_state = {k.replace('audio_encoder.', ''): v for k, v in ckpt.items() if k.startswith('audio_encoder.')}
audio_model.load_state_dict(audio_state, strict=False)

example_input = torch.randn(1, 1, 80, 16)
traced = torch.jit.trace(audio_model, example_input)

# Convert with ALL compute units
mlmodel = ct.convert(
    traced,
    inputs=[ct.TensorType(shape=(1, 1, 80, 16), name='mel_spectrogram')],
    minimum_deployment_target=ct.target.macOS13,
    compute_units=ct.ComputeUnit.ALL  # ← THIS IS THE KEY!
)

mlmodel.save('swift_inference/AudioEncoder.mlpackage')
print("✓ Audio encoder saved with ALL compute units")
print()

# Generator
print("[2/2] Generator...")
gen_model = Model(6, mode='ave').eval()

# Find checkpoint
checkpoint_path = 'convertmodeltocoreml/best_trainloss.pth'
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get('state_dict', checkpoint)
    else:
        state_dict = checkpoint
    gen_model.load_state_dict(state_dict, strict=False)
    print(f"✓ Loaded weights from {checkpoint_path}")
else:
    print("⚠️ No checkpoint found, using random weights")

visual_input = torch.randn(1, 6, 320, 320)
audio_input = torch.randn(1, 32, 16, 16)
traced = torch.jit.trace(gen_model, (visual_input, audio_input))

# Convert with ALL compute units
mlmodel = ct.convert(
    traced,
    inputs=[
        ct.TensorType(shape=(1, 6, 320, 320), name='visual_input'),
        ct.TensorType(shape=(1, 32, 16, 16), name='audio_input')
    ],
    minimum_deployment_target=ct.target.macOS13,
    compute_units=ct.ComputeUnit.ALL  # ← THIS IS THE KEY!
)

mlmodel.save('swift_inference/Generator.mlpackage')
print("✓ Generator saved with ALL compute units")
print()

print("=" * 60)
print("✅ Models reconverted with GPU/Neural Engine support!")
print("=" * 60)
print()
print("Now recompile:")
print("  cd swift_inference")
print("  rm -rf *.mlmodelc")
print("  xcrun coremlcompiler compile AudioEncoder.mlpackage .")
print("  xcrun coremlcompiler compile Generator.mlpackage .")
print("  swift build --configuration release")
print()
print("This should now use GPU + Neural Engine! 🚀")

