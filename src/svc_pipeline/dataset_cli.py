from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from svc_pipeline.config import load_config
from svc_pipeline.dataset import DatasetPreparer, discover_audio_files, filter_dataset

app = typer.Typer(help="Prepare authorized voice datasets for training.")
console = Console()


@app.command()
def scan(
    input_dir: Path = typer.Argument(..., exists=True, file_okay=False, readable=True),
) -> None:
    """List supported audio files in a dataset directory."""
    files = discover_audio_files(input_dir)
    console.print(f"Found {len(files)} audio files")
    for path in files[:30]:
        console.print(path)
    if len(files) > 30:
        console.print(f"... and {len(files) - 30} more")


@app.command()
def prepare(
    input_dir: Path = typer.Argument(..., exists=True, file_okay=False, readable=True),
    output_dir: Path = typer.Argument(...),
    speaker: str = typer.Option("target", "--speaker", "-s"),
    config_path: Path | None = typer.Option(None, "--config", "-c", exists=True, readable=True),
    min_duration: float = typer.Option(3.0, "--min-duration", min=0.5),
    max_duration: float = typer.Option(12.0, "--max-duration", min=1.0),
    top_db: int = typer.Option(35, "--top-db", min=5, max=80),
    max_files: int | None = typer.Option(None, "--max-files", min=1),
    separate_vocals: bool = typer.Option(
        False,
        "--separate-vocals",
        help="Run the configured separator before slicing. Use only with authorized material.",
    ),
) -> None:
    """Slice and normalize authorized source audio into trainable wav clips."""
    config = load_config(config_path)
    summary = DatasetPreparer(
        config,
        separate_vocals=separate_vocals,
        progress_callback=console.print,
    ).prepare(
        input_dir=input_dir,
        output_dir=output_dir,
        speaker=speaker,
        min_duration=min_duration,
        max_duration=max_duration,
        top_db=top_db,
        max_files=max_files,
    )
    console.print(f"[green]Prepared[/green] {len(summary.clips)} clips in {summary.output_dir}")
    console.print(f"metadata={summary.output_dir / 'metadata.csv'}")
    if summary.skipped_files:
        console.print(f"[yellow]Skipped[/yellow] {len(summary.skipped_files)} files")


@app.command()
def filter(
    input_dataset_dir: Path = typer.Argument(..., exists=True, file_okay=False, readable=True),
    output_dataset_dir: Path = typer.Argument(...),
    min_duration: float = typer.Option(3.0, "--min-duration", min=0.5),
    max_duration: float = typer.Option(12.0, "--max-duration", min=1.0),
    min_rms: float = typer.Option(0.005, "--min-rms", min=0.0),
) -> None:
    """Filter a prepared dataset by duration and source RMS."""
    summary = filter_dataset(
        input_dataset_dir=input_dataset_dir,
        output_dataset_dir=output_dataset_dir,
        min_duration=min_duration,
        max_duration=max_duration,
        min_rms=min_rms,
    )
    console.print(f"[green]Kept[/green] {len(summary.clips)} clips in {summary.output_dir}")
    console.print(f"[yellow]Filtered[/yellow] {len(summary.skipped_files)} clips")
    console.print(f"metadata={summary.output_dir / 'metadata.csv'}")


if __name__ == "__main__":
    app()
