"""
ClipSense Prosody Extractor (W2)

Extracts measurable physical acoustic signals from audio/video:
- Fundamental frequency (F0 / pitch) contour and statistics
- Root-Mean-Square (RMS) loudness energy and dynamic range
- Voiced speech fraction
- Speaking rate estimates over continuous temporal windows

Strict Constraints:
- NO emotion recognition.
- NO highlight classification.
- NO hard-coded highlight thresholds.
- Only physical acoustic signal measurements.
"""

import json
import logging
import math
import os
import pathlib
import subprocess
from typing import List, Optional
import numpy as np
import scipy.io.wavfile as wavfile

from app.config import config
from core.schemas import ProsodyWindow, ProsodyData, TranscriptData

logger = logging.getLogger("clipsense.extraction.prosody")


class ProsodyExtractor:
    """Extracts measurable physical prosodic features over sliding time windows."""

    def __init__(
        self,
        window_length_sec: Optional[float] = None,
        hop_length_sec: Optional[float] = None,
        sample_rate: Optional[int] = None,
        f0_min_hz: Optional[float] = None,
        f0_max_hz: Optional[float] = None,
    ):
        self.window_length_sec = window_length_sec or config.prosody.window_length_sec
        self.hop_length_sec = hop_length_sec or config.prosody.hop_length_sec
        self.sample_rate = sample_rate or config.prosody.sample_rate
        self.f0_min_hz = f0_min_hz or config.prosody.f0_min_hz
        self.f0_max_hz = f0_max_hz or config.prosody.f0_max_hz

    @staticmethod
    def ensure_wav_16k(media_path: str, target_wav_path: str) -> str:
        """Ensure audio is formatted as 16kHz 16-bit mono WAV."""
        target_p = pathlib.Path(target_wav_path)
        target_p.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            "ffmpeg",
            "-y",
            "-i", str(media_path),
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            str(target_p),
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"FFmpeg prosody audio extraction failed: {res.stderr}")
        return str(target_p)

    def compute_f0_autocorrelation(self, signal: np.ndarray, sr: int) -> float:
        """
        Estimate fundamental frequency F0 (in Hz) using normalized autocorrelation.
        Restricted to human vocal fundamental range [f0_min_hz, f0_max_hz].
        Returns 0.0 if unvoiced.
        """
        if len(signal) < 2:
            return 0.0

        min_lag = int(sr / self.f0_max_hz)
        max_lag = int(sr / self.f0_min_hz)

        if max_lag >= len(signal):
            max_lag = len(signal) - 1
        if min_lag >= max_lag:
            return 0.0

        # Center signal
        centered = signal - np.mean(signal)
        energy = np.sum(centered ** 2)
        if energy < 1e-6:
            return 0.0

        # Autocorrelation via FFT for speed
        n = len(centered)
        fft_size = 2 ** math.ceil(math.log2(2 * n - 1))
        fft_data = np.fft.fft(centered, n=fft_size)
        autocorr = np.fft.ifft(fft_data * np.conj(fft_data)).real[:n]
        autocorr = autocorr / (autocorr[0] + 1e-9)

        # Search peak within plausible pitch lags
        valid_range = autocorr[min_lag:max_lag]
        if len(valid_range) == 0:
            return 0.0

        best_lag_offset = int(np.argmax(valid_range))
        best_peak = valid_range[best_lag_offset]
        best_lag = min_lag + best_lag_offset

        # Threshold for voiced frame
        if best_peak > 0.35 and best_lag > 0:
            return float(sr / best_lag)
        return 0.0

    def extract(
        self,
        media_path: str,
        output_json_path: Optional[str] = None,
        transcript_data: Optional[TranscriptData] = None,
        total_duration: Optional[float] = None,
        force_rerun: bool = False,
    ) -> ProsodyData:
        """
        Extract windowed prosodic features from audio/video.
        Uses caching if output_json_path exists and force_rerun is False.
        """
        # Cache check
        if output_json_path and os.path.exists(output_json_path) and not force_rerun:
            logger.info(f"Loading cached prosody data from {output_json_path}")
            with open(output_json_path, "r", encoding="utf-8") as f:
                data_dict = json.load(f)
            return ProsodyData.model_validate(data_dict)

        media_p = pathlib.Path(media_path)
        if not media_p.exists():
            raise FileNotFoundError(f"Media file not found: {media_path}")

        temp_wav_dir = media_p.parent / "temp_prosody"
        temp_wav_file = temp_wav_dir / f"{media_p.stem}_prosody_16k.wav"
        wav_path = self.ensure_wav_16k(str(media_p), str(temp_wav_file))

        try:
            sr, raw_audio = wavfile.read(wav_path)
            if raw_audio.dtype == np.int16:
                audio = raw_audio.astype(np.float32) / 32768.0
            else:
                audio = raw_audio.astype(np.float32)

            total_samples = len(audio)
            audio_duration = float(total_samples / sr) if sr > 0 else 0.0
            # Clamp against actual video duration if provided
            effective_duration = float(total_duration) if (total_duration is not None and total_duration > 0) else audio_duration

            window_samples = int(self.window_length_sec * sr)
            hop_samples = int(self.hop_length_sec * sr)
            subframe_samples = int(0.04 * sr)  # 40ms subframes for short-term pitch & voicing

            windows: List[ProsodyWindow] = []

            # Slide window over audio
            for start_idx in range(0, total_samples, hop_samples):
                end_idx = min(start_idx + window_samples, total_samples)
                curr_window_audio = audio[start_idx:end_idx]

                if len(curr_window_audio) < subframe_samples:
                    break

                win_start_sec = min(float(start_idx / sr), effective_duration)
                win_end_sec = min(float(end_idx / sr), effective_duration)
                if win_start_sec >= effective_duration or win_end_sec <= win_start_sec:
                    break

                win_duration = float(win_end_sec - win_start_sec)

                # 1. RMS Energy
                rms_energy = float(np.sqrt(np.mean(curr_window_audio ** 2)))

                # Subframe RMS for dynamic range
                sub_rms_list = []
                f0_list = []
                voiced_count = 0
                total_subframes = 0

                for sub_start in range(0, len(curr_window_audio) - subframe_samples + 1, subframe_samples):
                    sub = curr_window_audio[sub_start : sub_start + subframe_samples]
                    sub_rms = float(np.sqrt(np.mean(sub ** 2)))
                    sub_rms_list.append(sub_rms)

                    sub_f0 = self.compute_f0_autocorrelation(sub, sr)
                    total_subframes += 1
                    if sub_f0 > 0:
                        voiced_count += 1
                        f0_list.append(sub_f0)

                voicing_fraction = float(voiced_count / total_subframes) if total_subframes > 0 else 0.0
                rms_dynamic_range = float(max(sub_rms_list) - min(sub_rms_list)) if sub_rms_list else 0.0

                if f0_list:
                    mean_f0 = float(np.mean(f0_list))
                    peak_f0 = float(np.max(f0_list))
                    f0_std = float(np.std(f0_list))
                else:
                    mean_f0 = 0.0
                    peak_f0 = 0.0
                    f0_std = 0.0

                # 2. Speaking Rate (Words per second)
                if transcript_data and transcript_data.words:
                    # Count words active within this window
                    words_in_window = [
                        w for w in transcript_data.words
                        if (w.start < win_end_sec and w.end > win_start_sec)
                    ]
                    speech_rate_wps = float(len(words_in_window) / win_duration) if win_duration > 0 else 0.0
                else:
                    # Syllable approximation based on energy bursts
                    energy_peaks = sum(
                        1 for r in sub_rms_list if r > (rms_energy * 1.2 and r > 0.02)
                    )
                    speech_rate_wps = float(energy_peaks / win_duration) if win_duration > 0 else 0.0

                windows.append(
                    ProsodyWindow(
                        start_time=win_start_sec,
                        end_time=win_end_sec,
                        duration=win_duration,
                        mean_f0=mean_f0,
                        peak_f0=peak_f0,
                        f0_std=f0_std,
                        rms_energy=rms_energy,
                        rms_dynamic_range=rms_dynamic_range,
                        voicing_fraction=voicing_fraction,
                        speech_rate_wps=speech_rate_wps,
                    )
                )

            prosody_data = ProsodyData(
                duration=effective_duration,
                sample_rate=sr,
                window_length_sec=self.window_length_sec,
                hop_length_sec=self.hop_length_sec,
                windows=windows,
            )

            # Save artifact
            if output_json_path:
                out_p = pathlib.Path(output_json_path)
                out_p.parent.mkdir(parents=True, exist_ok=True)
                with open(out_p, "w", encoding="utf-8") as f:
                    f.write(prosody_data.model_dump_json(indent=2))
                logger.info(f"Saved prosody data artifact to {output_json_path}")

            return prosody_data

        finally:
            if temp_wav_file.exists():
                try:
                    os.remove(temp_wav_file)
                except OSError:
                    pass
