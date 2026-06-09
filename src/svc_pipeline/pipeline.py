from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from svc_pipeline.audio import load_audio, save_audio
from svc_pipeline.config import PipelineConfig
from svc_pipeline.features import FeatureExtractor
from svc_pipeline.separation import build_separator
from svc_pipeline.vc_backend import build_vc_backend


@dataclass(frozen=True)
class ConversionResult:
    output_path: Path
    f0_frames: int
    voiced_frames: int
    sample_rate: int


class SingingVoiceConversionPipeline:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.separator = build_separator(config.separation)
        self.feature_extractor = FeatureExtractor(config.features)
        self.vc_backend = build_vc_backend(config.conversion)

    def run(
        self,
        input_path: Path,
        output_path: Path,
        target_speaker: str | None = None,
        start: float = 0.0,
        duration: float | None = None,
        vocal_output_path: Path | None = None,
    ) -> ConversionResult:
        clip = load_audio(
            input_path,
            self.config.audio.sample_rate,
            self.config.audio.mono,
            offset=start,
            duration=duration,
        )
        vocal = self.separator.separate_vocals(clip)
        if vocal_output_path is not None:
            save_audio(vocal_output_path, vocal)
        features = self.feature_extractor.extract(vocal)
        converted = self.vc_backend.convert(
            vocal=vocal,
            features=features,
            target_speaker=target_speaker or self.config.conversion.target_speaker,
        )
        save_audio(output_path, converted)
        return ConversionResult(
            output_path=output_path,
            f0_frames=len(features.f0_hz),
            voiced_frames=int(features.voiced.sum()),
            sample_rate=converted.sample_rate,
        )
