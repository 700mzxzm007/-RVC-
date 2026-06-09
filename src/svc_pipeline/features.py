from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np

from svc_pipeline.audio import AudioClip
from svc_pipeline.config import FeatureConfig


@dataclass(frozen=True)
class SingingFeatures:
    f0_hz: np.ndarray
    voiced: np.ndarray
    rms: np.ndarray
    hop_length: int


class FeatureExtractor:
    def __init__(self, config: FeatureConfig) -> None:
        self.config = config

    def extract(self, clip: AudioClip) -> SingingFeatures:
        if self.config.f0_method != "pyin":
            raise ValueError(f"Unsupported f0 method: {self.config.f0_method}")

        f0, voiced_flag, _ = librosa.pyin(
            clip.samples,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
            sr=clip.sample_rate,
            hop_length=self.config.hop_length,
        )
        rms = librosa.feature.rms(y=clip.samples, hop_length=self.config.hop_length)[0]
        return SingingFeatures(
            f0_hz=np.nan_to_num(f0, nan=0.0).astype(np.float32),
            voiced=voiced_flag.astype(bool),
            rms=rms.astype(np.float32),
            hop_length=self.config.hop_length,
        )
