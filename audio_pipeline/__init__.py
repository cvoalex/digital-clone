"""
Audio Processing Pipeline for SyncTalk_2D

This module provides a standalone, testable implementation of the audio processing
pipeline used in SyncTalk_2D. It's designed to be:
1. Portable to iOS/CoreML
2. Testable with reference outputs
3. Independent of the main codebase
"""

from .mel_processor import MelSpectrogramProcessor
from .audio_encoder import AudioEncoderWrapper
from .pipeline import AudioPipeline

__version__ = "1.0.0"
__all__ = ["MelSpectrogramProcessor", "AudioEncoderWrapper", "AudioPipeline"]

