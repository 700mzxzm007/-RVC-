from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class AudioConfig(BaseModel):
    sample_rate: int = 44100
    mono: bool = True


class SeparationConfig(BaseModel):
    backend: str = "passthrough"
    model: str = "htdemucs"
    device: str | None = None
    shifts: int = 1
    overlap: float = 0.25


class FeatureConfig(BaseModel):
    f0_method: str = "pyin"
    hop_length: int = 512


class ConversionConfig(BaseModel):
    backend: str = "identity"
    model_path: Path | None = None
    target_speaker: str = "demo"


class PipelineConfig(BaseModel):
    audio: AudioConfig = Field(default_factory=AudioConfig)
    separation: SeparationConfig = Field(default_factory=SeparationConfig)
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    conversion: ConversionConfig = Field(default_factory=ConversionConfig)


def load_config(path: Path | None) -> PipelineConfig:
    if path is None:
        return PipelineConfig()
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    return PipelineConfig.model_validate(data)
