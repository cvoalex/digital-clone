#!/usr/bin/env python3
"""
Python frame generation using ONNX (full pipeline with audio processing)
Matches the Go implementation exactly
"""

import os
import sys
import numpy as np
import onnxruntime as ort
from PIL import Image
import json
import librosa

# Add parent to path for utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import melspectrogram, load_wav

def load_image_as_tensor(path, normalize=True):
    """Load image and convert to CHW RGB tensor"""
    img = Image.open(path).convert('RGB')
    arr = np.array(img, dtype=np.float32)
    
    # HWC to CHW
    tensor = arr.transpose(2, 0, 1)
    
    if normalize:
        tensor = tensor / 255.0
    
    return tensor

def tensor_to_image(tensor):
    """Convert CHW tensor back to image"""
    # CHW to HWC
    arr = tensor.transpose(1, 2, 0)
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)

def paste_into_frame(full_frame_path, generated, rect):
    """Paste generated region into full frame"""
    full = Image.open(full_frame_path)
    full_arr = np.array(full)
    
    x1, y1, x2, y2 = rect
    
    # Resize generated to fit rect
    gen_resized = generated.resize((x2-x1, y2-y1), Image.BICUBIC)
    gen_arr = np.array(gen_resized)
    
    # Paste
    full_arr[y1:y2, x1:x2] = gen_arr
    
    return Image.fromarray(full_arr)

def process_audio(audio_path, audio_encoder_session):
    """Process audio file through encoder"""
    print(f"Processing audio: {audio_path}")
    
    # Load audio
    wav = load_wav(audio_path, 16000)
    print(f"  Loaded audio: {len(wav)} samples")
    
    # Generate mel spectrogram
    mel_spec = melspectrogram(wav).T  # (time, 80)
    print(f"  Generated mel spectrogram: {mel_spec.shape}")
    
    # Calculate number of frames
    data_len = int((mel_spec.shape[0] - 16) / 80. * float(25)) + 2
    print(f"  Number of frames: {data_len}")
    
    # Process each frame through audio encoder
    audio_features = []
    
    for idx in range(data_len):
        # Crop audio window
        start_idx = int(80. * (idx / float(25)))
        end_idx = start_idx + 16
        
        if end_idx > mel_spec.shape[0]:
            end_idx = mel_spec.shape[0]
            start_idx = end_idx - 16
        
        mel_window = mel_spec[start_idx:end_idx, :]  # (16, 80)
        
        # Prepare for encoder: (1, 1, 80, 16)
        mel_tensor = mel_window.T[np.newaxis, np.newaxis, ...]  # (1, 1, 80, 16)
        
        # Run audio encoder
        output = audio_encoder_session.run(None, {'mel': mel_tensor.astype(np.float32)})
        features = output[0][0]  # (512,)
        
        audio_features.append(features)
        
        if (idx + 1) % 100 == 0:
            print(f"  Encoded {idx + 1}/{data_len} frames")
    
    print(f"✓ Generated {len(audio_features)} audio feature frames")
    return np.array(audio_features)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate frames with Python ONNX')
    parser.add_argument('--sanders', default='../model/sanders_full_onnx', help='Sanders directory')
    parser.add_argument('--audio', default='../demo/talk_hb.wav', help='Audio WAV file')
    parser.add_argument('--output', default='../comparison_results/python_output/frames', help='Output directory')
    parser.add_argument('--frames', type=int, default=10, help='Number of frames to generate')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Python ONNX Inference - Full Pipeline")
    print("=" * 60)
    print(f"Sanders directory: {args.sanders}")
    print(f"Audio file: {args.audio}")
    print(f"Output directory: {args.output}")
    print(f"Number of frames: {args.frames}")
    print("=" * 60)
    
    # Load models
    print("\n[1/4] Loading models...")
    generator_path = f"{args.sanders}/models/generator.onnx"
    audio_encoder_path = f"{args.sanders}/models/audio_encoder.onnx"
    
    generator_session = ort.InferenceSession(generator_path)
    audio_encoder_session = ort.InferenceSession(audio_encoder_path)
    print("✓ Models loaded")
    
    # Load crop rectangles
    with open(f"{args.sanders}/cache/crop_rectangles.json") as f:
        crop_rects = json.load(f)
    
    # Process audio
    print("\n[2/4] Processing audio...")
    audio_features = process_audio(args.audio, audio_encoder_session)
    
    # Limit frames
    num_frames = min(args.frames, len(audio_features))
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    print(f"\n[3/4] Generating {num_frames} video frames...")
    
    # Generate frames
    for i in range(1, num_frames + 1):
        if i % 5 == 1:
            print(f"Processing frame {i}/{num_frames}...")
        
        # Load pre-cut frames
        roi_path = f"{args.sanders}/rois_320/{i}.jpg"
        masked_path = f"{args.sanders}/model_inputs/{i}.jpg"
        full_path = f"{args.sanders}/full_body_img/{i}.jpg"
        
        # Load and convert to tensors
        roi_tensor = load_image_as_tensor(roi_path, normalize=True)
        masked_tensor = load_image_as_tensor(masked_path, normalize=True)
        
        # Concatenate into 6-channel input
        input_tensor = np.concatenate([roi_tensor, masked_tensor], axis=0)
        input_tensor = input_tensor[np.newaxis, ...]  # Add batch dim
        
        # Get audio features for this frame
        audio_idx = i - 1
        audio_feat = audio_features[audio_idx]
        
        # Reshape audio to (1, 32, 16, 16) = 8192 values
        audio_reshaped = np.tile(audio_feat, 16)[:8192].reshape(1, 32, 16, 16).astype(np.float32)
        
        # Run generator
        outputs = generator_session.run(None, {
            'input': input_tensor.astype(np.float32),
            'audio': audio_reshaped
        })
        
        # Get output (1, 3, 320, 320)
        output_tensor = outputs[0][0] * 255.0
        
        # Convert to image
        generated = tensor_to_image(output_tensor)
        
        # Get crop rectangle
        rect_key = str(i-1)
        rect = crop_rects[rect_key]['rect']
        
        # Paste into full frame
        final_frame = paste_into_frame(full_path, generated, rect)
        
        # Save
        output_path = f"{args.output}/frame_{i:05d}.jpg"
        final_frame.save(output_path, quality=95)
    
    print(f"✓ Generated {num_frames} frames")
    
    print("\n[4/4] Video assembly...")
    print("To create video, run:")
    print(f"  ffmpeg -framerate 25 -i {args.output}/frame_%05d.jpg \\")
    print(f"    -i {args.audio} \\")
    print(f"    -c:v libx264 -c:a aac -crf 20 \\")
    print(f"    output_video.mp4 -y")
    
    print("\n" + "=" * 60)
    print("✓ Frame generation complete!")
    print("=" * 60)

if __name__ == '__main__':
    main()

