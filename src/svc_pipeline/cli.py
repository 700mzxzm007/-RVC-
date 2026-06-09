from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from svc_pipeline.config import load_config
from svc_pipeline.pipeline import SingingVoiceConversionPipeline

app = typer.Typer(help="Authorized singing voice conversion tools.")
console = Console()


@app.command()
def convert(
    input_path: Path = typer.Argument(..., exists=True, readable=True),
    output_path: Path = typer.Argument(...),
    config_path: Path | None = typer.Option(None, "--config", "-c", exists=True, readable=True),
    target_speaker: str | None = typer.Option(None, "--target-speaker", "-s"),
    start: float = typer.Option(0.0, "--start", min=0.0, help="Start time in seconds."),
    duration: float | None = typer.Option(None, "--duration", min=0.1, help="Duration in seconds."),
    vocal_output: Path | None = typer.Option(
        None,
        "--vocal-output",
        help="Optional path for the separated vocal wav.",
    ),
) -> None:
    """Run the singing voice conversion pipeline."""
    config = load_config(config_path)
    result = SingingVoiceConversionPipeline(config).run(
        input_path,
        output_path,
        target_speaker,
        start=start,
        duration=duration,
        vocal_output_path=vocal_output,
    )
    console.print(f"[green]Wrote[/green] {result.output_path}")
    console.print(
        f"sample_rate={result.sample_rate}, f0_frames={result.f0_frames}, "
        f"voiced_frames={result.voiced_frames}"
    )


if __name__ == "__main__":
    app()
