from __future__ import annotations

from abc import ABC, abstractmethod

from svc_pipeline.audio import AudioClip
from svc_pipeline.config import ConversionConfig
from svc_pipeline.features import SingingFeatures


class VoiceConversionBackend(ABC):
    @abstractmethod
    def convert(
        self,
        vocal: AudioClip,
        features: SingingFeatures,
        target_speaker: str,
    ) -> AudioClip:
        """Convert the vocal stem into the authorized target speaker's timbre."""


class IdentityBackend(VoiceConversionBackend):
    def convert(
        self,
        vocal: AudioClip,
        features: SingingFeatures,
        target_speaker: str,
    ) -> AudioClip:
        _ = features, target_speaker
        return vocal


def build_vc_backend(config: ConversionConfig) -> VoiceConversionBackend:
    if config.backend == "identity":
        return IdentityBackend()
    raise ValueError(f"Unsupported conversion backend: {config.backend}")
