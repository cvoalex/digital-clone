"""
Mel Spectrogram Processor

This module handles the conversion of audio waveforms to mel spectrograms.
It replicates the exact behavior of the main SyncTalk_2D audio processing.
"""

import numpy as np
import librosa
import librosa.filters
from scipy import signal
from typing import Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MelSpectrogramProcessor:
    """
    Processes audio waveforms into mel spectrograms using the exact parameters
    from SyncTalk_2D.
    
    Parameters match the hparams configuration:
    - Sample rate: 16000 Hz
    - n_fft: 800
    - hop_length: 200 (12.5ms)
    - win_length: 800 (50ms)
    - n_mels: 80
    - fmin: 55 Hz
    - fmax: 7600 Hz
    - Pre-emphasis: 0.97
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 800,
        hop_length: int = 200,
        win_length: int = 800,
        n_mels: int = 80,
        fmin: int = 55,
        fmax: int = 7600,
        preemphasis_coef: float = 0.97,
        ref_level_db: float = 20.0,
        min_level_db: float = -100.0,
        max_abs_value: float = 4.0
    ):
        """
        Initialize the mel spectrogram processor.
        
        Args:
            sample_rate: Audio sample rate in Hz
            n_fft: FFT window size
            hop_length: Number of samples between successive frames
            win_length: Window length
            n_mels: Number of mel filterbanks
            fmin: Minimum frequency
            fmax: Maximum frequency
            preemphasis_coef: Pre-emphasis filter coefficient
            ref_level_db: Reference level in dB for normalization
            min_level_db: Minimum level in dB for clipping
            max_abs_value: Maximum absolute value for normalization
        """
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax
        self.preemphasis_coef = preemphasis_coef
        self.ref_level_db = ref_level_db
        self.min_level_db = min_level_db
        self.max_abs_value = max_abs_value
        
        # Build mel filterbank
        self._mel_basis = self._build_mel_basis()
        
        logger.info(f"MelSpectrogramProcessor initialized with:")
        logger.info(f"  Sample rate: {sample_rate} Hz")
        logger.info(f"  n_fft: {n_fft}, hop: {hop_length}, win: {win_length}")
        logger.info(f"  Mel bands: {n_mels}, freq range: {fmin}-{fmax} Hz")
    
    def _build_mel_basis(self) -> np.ndarray:
        """Build mel filter bank matrix."""
        return librosa.filters.mel(
            sr=self.sample_rate,
            n_fft=self.n_fft,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax
        )
    
    def load_wav(self, path: str) -> np.ndarray:
        """
        Load a WAV file with the target sample rate.
        
        Args:
            path: Path to the WAV file
            
        Returns:
            Audio waveform as numpy array
        """
        wav, sr = librosa.core.load(path, sr=self.sample_rate)
        logger.info(f"Loaded audio: {path}")
        logger.info(f"  Duration: {len(wav)/self.sample_rate:.2f}s, Shape: {wav.shape}")
        return wav
    
    def preemphasis(self, wav: np.ndarray) -> np.ndarray:
        """
        Apply pre-emphasis filter to the audio waveform.
        
        Pre-emphasis amplifies high frequencies to balance the frequency spectrum.
        
        Args:
            wav: Audio waveform
            
        Returns:
            Filtered audio waveform
        """
        return signal.lfilter([1, -self.preemphasis_coef], [1], wav)
    
    def _stft(self, y: np.ndarray) -> np.ndarray:
        """
        Apply Short-Time Fourier Transform.
        
        Args:
            y: Audio waveform
            
        Returns:
            STFT result (complex-valued)
        """
        return librosa.stft(
            y=y,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length
        )
    
    def _linear_to_mel(self, spectrogram: np.ndarray) -> np.ndarray:
        """
        Convert linear spectrogram to mel scale.
        
        Args:
            spectrogram: Linear spectrogram
            
        Returns:
            Mel spectrogram
        """
        return np.dot(self._mel_basis, spectrogram)
    
    def _amp_to_db(self, x: np.ndarray) -> np.ndarray:
        """
        Convert amplitude to decibels.
        
        Args:
            x: Input amplitude
            
        Returns:
            Amplitude in decibels
        """
        min_level = np.exp(-5 * np.log(10))
        return 20 * np.log10(np.maximum(min_level, x))
    
    def _normalize(self, S: np.ndarray) -> np.ndarray:
        """
        Normalize the spectrogram to the range [-4, 4].
        
        This matches the exact normalization used in utils.py.
        
        Args:
            S: Input spectrogram in dB
            
        Returns:
            Normalized spectrogram
        """
        # Original formula: np.clip((2 * 4.) * ((S - -100) / (--100)) - 4., -4., 4.)
        # Simplified: np.clip(8.0 * ((S + 100) / 100) - 4.0, -4.0, 4.0)
        normalized = (2 * self.max_abs_value) * ((S - self.min_level_db) / (-self.min_level_db)) - self.max_abs_value
        return np.clip(normalized, -self.max_abs_value, self.max_abs_value)
    
    def process_audio(self, wav: np.ndarray) -> np.ndarray:
        """
        Convert audio waveform to mel spectrogram.
        
        This is the main processing function that applies all steps:
        1. Pre-emphasis
        2. STFT
        3. Linear to mel conversion
        4. Amplitude to dB
        5. Normalization
        
        Args:
            wav: Audio waveform
            
        Returns:
            Mel spectrogram with shape (n_mels, n_frames)
        """
        # Apply pre-emphasis
        wav_preemph = self.preemphasis(wav)
        
        # STFT
        D = self._stft(wav_preemph)
        
        # Linear to mel
        mel_spec = self._linear_to_mel(np.abs(D))
        
        # Amplitude to dB
        mel_db = self._amp_to_db(mel_spec)
        
        # Apply reference level
        mel_db = mel_db - self.ref_level_db
        
        # Normalize
        mel_normalized = self._normalize(mel_db)
        
        logger.info(f"Mel spectrogram shape: {mel_normalized.shape}")
        logger.info(f"  Value range: [{mel_normalized.min():.3f}, {mel_normalized.max():.3f}]")
        
        return mel_normalized
    
    def process_file(self, audio_path: str) -> np.ndarray:
        """
        Process an audio file directly.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Mel spectrogram with shape (n_mels, n_frames)
        """
        wav = self.load_wav(audio_path)
        return self.process_audio(wav)
    
    def get_frame_count(self, mel_spec: np.ndarray, fps: int = 25) -> int:
        """
        Calculate the number of video frames for a mel spectrogram.
        
        This matches the calculation in AudDataset.__init__
        
        Args:
            mel_spec: Mel spectrogram with shape (n_mels, n_frames)
            fps: Target video frame rate
            
        Returns:
            Number of frames
        """
        n_mel_frames = mel_spec.shape[1]
        frame_count = int((n_mel_frames - 16) / 80.0 * float(fps)) + 2
        return frame_count
    
    def crop_audio_window(
        self,
        mel_spec: np.ndarray,
        frame_idx: int,
        fps: int = 25
    ) -> np.ndarray:
        """
        Crop a 16-frame window from the mel spectrogram for a specific video frame.
        
        This matches the AudDataset.crop_audio_window method.
        
        Args:
            mel_spec: Mel spectrogram with shape (n_mels, n_frames)
            frame_idx: Video frame index
            fps: Video frame rate
            
        Returns:
            Cropped mel spectrogram with shape (16, n_mels)
        """
        # Calculate mel frame index
        start_idx = int(80.0 * (frame_idx / float(fps)))
        end_idx = start_idx + 16
        
        # Handle boundary conditions
        if end_idx > mel_spec.shape[1]:
            end_idx = mel_spec.shape[1]
            start_idx = end_idx - 16
        
        # Transpose to match expected shape: (16, n_mels)
        window = mel_spec[:, start_idx:end_idx].T
        
        if window.shape[0] != 16:
            raise ValueError(f"Window shape is {window.shape}, expected (16, {self.n_mels})")
        
        return window
    
    def save_mel_spectrogram(self, mel_spec: np.ndarray, output_path: str):
        """
        Save mel spectrogram to a .npy file.
        
        Args:
            mel_spec: Mel spectrogram
            output_path: Output file path
        """
        np.save(output_path, mel_spec)
        logger.info(f"Saved mel spectrogram to: {output_path}")
    
    def load_mel_spectrogram(self, input_path: str) -> np.ndarray:
        """
        Load mel spectrogram from a .npy file.
        
        Args:
            input_path: Input file path
            
        Returns:
            Mel spectrogram
        """
        mel_spec = np.load(input_path)
        logger.info(f"Loaded mel spectrogram from: {input_path}")
        return mel_spec

