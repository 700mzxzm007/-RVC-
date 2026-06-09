from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

from svc_pipeline.audio import AudioClip, load_audio, save_audio
from svc_pipeline.config import SeparationConfig


class VocalSeparator(ABC):
    @abstractmethod
    def separate_vocals(self, clip: AudioClip) -> AudioClip:
        """Return vocal stem from a full mix or dry vocal input."""


class PassthroughSeparator(VocalSeparator):
    def separate_vocals(self, clip: AudioClip) -> AudioClip:
        return clip


class DemucsSeparator(VocalSeparator):
    def __init__(self, config: SeparationConfig) -> None:
        self.config = config

    def separate_vocals(self, clip: AudioClip) -> AudioClip:
        with tempfile.TemporaryDirectory(prefix="svc-demucs-") as tmp_dir_name:
            tmp_dir = Path(tmp_dir_name)
            input_path = tmp_dir / "mix.wav"
            output_dir = tmp_dir / "separated"
            demucs_input = clip
            if clip.samples.ndim == 1:
                demucs_input = AudioClip(
                    samples=np.column_stack([clip.samples, clip.samples]),
                    sample_rate=clip.sample_rate,
                )
            save_audio(input_path, demucs_input)

            command = [
                sys.executable,
                "-m",
                "demucs",
                "--two-stems",
                "vocals",
                "--name",
                self.config.model,
                "--shifts",
                str(self.config.shifts),
                "--overlap",
                str(self.config.overlap),
                "--out",
                str(output_dir),
            ]
            if self.config.device:
                command.extend(["--device", self.config.device])
            command.append(str(input_path))

            completed = subprocess.run(command, capture_output=True, text=True, check=False)
            if completed.returncode != 0:
                message = completed.stderr.strip() or completed.stdout.strip()
                raise RuntimeError(f"Demucs failed: {message}")

            vocals_path = output_dir / self.config.model / input_path.stem / "vocals.wav"
            if not vocals_path.exists():
                raise RuntimeError(f"Demucs did not write expected vocal stem: {vocals_path}")

            return load_audio(vocals_path, clip.sample_rate, mono=True)


def build_separator(config: SeparationConfig) -> VocalSeparator:
    if config.backend == "passthrough":
        return PassthroughSeparator()
    if config.backend == "demucs":
        return DemucsSeparator(config)
    raise ValueError(f"Unsupported separation backend: {config.backend}")
