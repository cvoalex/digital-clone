#!/usr/bin/env python3
"""
Export U-Net model to ONNX format

This script exports the trained U-Net model to ONNX format for use in
Go and Swift implementations.

Usage:
    python export_model.py --checkpoint ./checkpoint/May/5.pth --output ./models/unet_328.onnx --mode ave
"""

import argparse
import os
import sys
import logging

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from frame_generation_pipeline.unet_model import UNetModel

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Export U-Net model to ONNX')
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='./frame_generation_pipeline/models/unet_328.onnx',
        help='Path to save ONNX model'
    )
    parser.add_argument(
        '--mode',
        type=str,
        default='ave',
        choices=['ave', 'hubert', 'wenet'],
        help='Audio feature mode'
    )
    parser.add_argument(
        '--opset',
        type=int,
        default=11,
        help='ONNX opset version'
    )
    
    args = parser.parse_args()
    
    # Check if checkpoint exists
    if not os.path.exists(args.checkpoint):
        logger.error(f"Checkpoint not found: {args.checkpoint}")
        return 1
    
    # Create output directory
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Load model
    logger.info(f"Loading model from {args.checkpoint}")
    model = UNetModel(
        checkpoint_path=args.checkpoint,
        mode=args.mode
    )
    
    # Export to ONNX
    logger.info(f"Exporting to {args.output}")
    model.export_to_onnx(args.output, opset_version=args.opset)
    
    # Print input/output shapes
    image_shape, audio_shape = model.get_input_shapes()
    logger.info(f"Model exported successfully!")
    logger.info(f"Input shapes:")
    logger.info(f"  Image: {image_shape}")
    logger.info(f"  Audio: {audio_shape}")
    logger.info(f"Output shape: (1, 3, 320, 320)")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

