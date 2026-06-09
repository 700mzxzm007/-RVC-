from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


@dataclass(frozen=True)
class AudioClip:
    samples: np.ndarray
    sample_rate: int


def load_audio(
    path: Path,
    sample_rate: int,
    mono: bool = True,
    offset: float = 0.0,
    duration: float | None = None,
) -> AudioClip:
    samples, sr = librosa.load(path, sr=sample_rate, mono=mono, offset=offset, duration=duration)
    return AudioClip(samples=samples.astype(np.float32), sample_rate=sr)


def save_audio(path: Path, clip: AudioClip) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, clip.samples, clip.sample_rate)
