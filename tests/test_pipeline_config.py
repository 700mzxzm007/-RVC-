import numpy as np

from svc_pipeline.config import PipelineConfig
from svc_pipeline.dataset import discover_audio_files, normalize_peak
from svc_pipeline.separation import DemucsSeparator, build_separator


def test_default_config_uses_safe_placeholder_backends() -> None:
    config = PipelineConfig()

    assert config.separation.backend == "passthrough"
    assert config.conversion.backend == "identity"


def test_demucs_separator_can_be_selected() -> None:
    config = PipelineConfig.model_validate({"separation": {"backend": "demucs"}})

    assert isinstance(build_separator(config.separation), DemucsSeparator)


def test_normalize_peak_scales_audio() -> None:
    samples = np.array([0.0, 0.5, -0.25], dtype="float32")

    normalized, peak, rms = normalize_peak(samples, peak_target=1.0)

    assert peak == 0.5
    assert rms > 0
    assert normalized.max() == 1.0


def test_discover_audio_files_ignores_non_audio(tmp_path) -> None:
    (tmp_path / "a.wav").write_bytes(b"")
    (tmp_path / "b.txt").write_text("nope")

    assert discover_audio_files(tmp_path) == [tmp_path / "a.wav"]
