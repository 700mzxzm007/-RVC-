from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import csv
import shutil
from pathlib import Path

import librosa
import numpy as np

from svc_pipeline.audio import AudioClip, load_audio, save_audio
from svc_pipeline.config import PipelineConfig
from svc_pipeline.separation import build_separator

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg"}


@dataclass(frozen=True)
class PreparedClip:
    source_path: Path
    output_path: Path
    duration: float
    peak: float
    rms: float


@dataclass(frozen=True)
class DatasetSummary:
    output_dir: Path
    clips: list[PreparedClip]
    skipped_files: list[Path]


def discover_audio_files(input_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
    )


class DatasetPreparer:
    def __init__(
        self,
        config: PipelineConfig,
        separate_vocals: bool = False,
        progress_callback: Callable[[str], None] | None = None,
    ) -> None:
        self.config = config
        self.separate_vocals = separate_vocals
        self.separator = build_separator(config.separation) if separate_vocals else None
        self.progress_callback = progress_callback

    def prepare(
        self,
        input_dir: Path,
        output_dir: Path,
        speaker: str,
        min_duration: float = 3.0,
        max_duration: float = 12.0,
        top_db: int = 35,
        peak_target: float = 0.95,
        max_files: int | None = None,
    ) -> DatasetSummary:
        audio_files = discover_audio_files(input_dir)
        if max_files is not None:
            audio_files = audio_files[:max_files]

        wav_dir = output_dir / speaker / "wavs"
        metadata_path = output_dir / speaker / "metadata.csv"
        wav_dir.mkdir(parents=True, exist_ok=True)

        clips: list[PreparedClip] = []
        skipped_files: list[Path] = []

        total_files = len(audio_files)
        for file_index, source_path in enumerate(audio_files, start=1):
            try:
                self._report(f"[{file_index}/{total_files}] loading {source_path.name}")
                source_clip = load_audio(
                    source_path,
                    sample_rate=self.config.audio.sample_rate,
                    mono=True,
                )
                if self.separator:
                    self._report(f"[{file_index}/{total_files}] separating vocals {source_path.name}")
                clip = self.separator.separate_vocals(source_clip) if self.separator else source_clip
                source_clips = self._write_segments(
                    clip=clip,
                    source_path=source_path,
                    wav_dir=wav_dir,
                    existing_count=len(clips),
                    min_duration=min_duration,
                    max_duration=max_duration,
                    top_db=top_db,
                    peak_target=peak_target,
                )
                clips.extend(source_clips)
                self._report(
                    f"[{file_index}/{total_files}] wrote {len(source_clips)} clips "
                    f"from {source_path.name}"
                )
            except Exception as error:
                skipped_files.append(source_path)
                self._report(
                    f"[{file_index}/{total_files}] skipped {source_path.name}: "
                    f"{type(error).__name__}: {error}"
                )

        self._write_metadata(metadata_path, clips)
        return DatasetSummary(output_dir=output_dir / speaker, clips=clips, skipped_files=skipped_files)

    def _report(self, message: str) -> None:
        if self.progress_callback:
            self.progress_callback(message)

    def _write_segments(
        self,
        clip: AudioClip,
        source_path: Path,
        wav_dir: Path,
        existing_count: int,
        min_duration: float,
        max_duration: float,
        top_db: int,
        peak_target: float,
    ) -> list[PreparedClip]:
        intervals = librosa.effects.split(clip.samples, top_db=top_db)
        written: list[PreparedClip] = []
        min_samples = int(min_duration * clip.sample_rate)
        max_samples = int(max_duration * clip.sample_rate)

        for start, end in intervals:
            segment = clip.samples[start:end]
            cursor = 0
            while cursor < len(segment):
                chunk = segment[cursor : cursor + max_samples]
                cursor += max_samples
                if len(chunk) < min_samples:
                    continue

                normalized, peak, rms = normalize_peak(chunk, peak_target)
                if peak <= 1e-4 or rms <= 1e-4:
                    continue

                clip_index = existing_count + len(written) + 1
                output_path = wav_dir / f"{clip_index:05d}_{source_path.stem[:40]}.wav"
                save_audio(output_path, AudioClip(samples=normalized, sample_rate=clip.sample_rate))
                written.append(
                    PreparedClip(
                        source_path=source_path,
                        output_path=output_path,
                        duration=len(normalized) / clip.sample_rate,
                        peak=peak,
                        rms=rms,
                    )
                )

        return written

    def _write_metadata(self, metadata_path: Path, clips: list[PreparedClip]) -> None:
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        with metadata_path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["path", "source", "duration", "peak", "rms"])
            for clip in clips:
                writer.writerow(
                    [
                        clip.output_path,
                        clip.source_path,
                        f"{clip.duration:.3f}",
                        f"{clip.peak:.6f}",
                        f"{clip.rms:.6f}",
                    ]
                )


def normalize_peak(samples: np.ndarray, peak_target: float) -> tuple[np.ndarray, float, float]:
    samples = samples.astype(np.float32)
    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    rms = float(np.sqrt(np.mean(samples**2))) if samples.size else 0.0
    if peak <= 1e-8:
        return samples, peak, rms
    return (samples / peak * peak_target).astype(np.float32), peak, rms


def filter_dataset(
    input_dataset_dir: Path,
    output_dataset_dir: Path,
    min_duration: float = 3.0,
    max_duration: float = 12.0,
    min_rms: float = 0.005,
) -> DatasetSummary:
    metadata_path = input_dataset_dir / "metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata: {metadata_path}")

    wav_dir = output_dataset_dir / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)
    output_metadata_path = output_dataset_dir / "metadata.csv"

    kept: list[PreparedClip] = []
    skipped: list[Path] = []
    with metadata_path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    for row in rows:
        source_clip_path = Path(row["path"])
        duration = float(row["duration"])
        peak = float(row["peak"])
        rms = float(row["rms"])
        if not source_clip_path.exists():
            skipped.append(source_clip_path)
            continue
        if duration < min_duration or duration > max_duration or rms < min_rms:
            skipped.append(source_clip_path)
            continue

        output_path = wav_dir / source_clip_path.name
        shutil.copy2(source_clip_path, output_path)
        kept.append(
            PreparedClip(
                source_path=Path(row["source"]),
                output_path=output_path,
                duration=duration,
                peak=peak,
                rms=rms,
            )
        )

    DatasetPreparer(PipelineConfig())._write_metadata(output_metadata_path, kept)
    return DatasetSummary(output_dir=output_dataset_dir, clips=kept, skipped_files=skipped)
