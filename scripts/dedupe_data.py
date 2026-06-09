#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import re
import shutil
import unicodedata
from dataclasses import dataclass
from pathlib import Path


AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg"}


@dataclass(frozen=True)
class AudioFile:
    path: Path
    key: str
    size: int
    digest: str


def normalize_title(path: Path) -> str:
    title = unicodedata.normalize("NFKC", path.stem)
    title = re.sub(r"^\._", "", title)
    title = re.sub(r"^(林俊杰|JJ Lin|JJ)\s*[-_ ]\s*", "", title, flags=re.IGNORECASE)
    title = re.sub(r"\s*[\(（][0-9]+[\)）]\s*$", "", title)
    title = re.sub(r"\s+", "", title)
    title = re.sub(r"[-—_·,，.。:：!！?？\[\]【】()（）《》〈〉\"“”']", "", title)
    return title.lower()


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def unique_destination(path: Path, archive_dir: Path) -> Path:
    target = archive_dir / path.name
    if not target.exists():
        return target
    for index in range(1, 10000):
        candidate = archive_dir / f"{path.stem}__dup{index}{path.suffix}"
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not create unique archive path for {path}")


def discover_files(data_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in data_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
    )


def pick_keep_file(files: list[AudioFile]) -> AudioFile:
    def score(item: AudioFile) -> tuple[int, int, str]:
        clean_name_bonus = 1 if not re.search(r"[\(（][0-9]+[\)）]$", item.path.stem) else 0
        return (item.size, clean_name_bonus, item.path.name)

    return max(files, key=score)


def dedupe(data_dir: Path, archive_dir: Path, dry_run: bool) -> tuple[int, int, int]:
    archive_dir.mkdir(parents=True, exist_ok=True)
    report_path = archive_dir / "dedupe_report.csv"

    audio_paths = discover_files(data_dir)
    appledouble = [path for path in audio_paths if path.name.startswith("._")]
    real_audio_paths = [path for path in audio_paths if not path.name.startswith("._")]

    grouped: dict[str, list[AudioFile]] = {}
    for path in real_audio_paths:
        audio = AudioFile(
            path=path,
            key=normalize_title(path),
            size=path.stat().st_size,
            digest=file_digest(path),
        )
        grouped.setdefault(audio.key, []).append(audio)

    duplicate_paths: list[tuple[Path, str, str, Path | None]] = []
    for path in appledouble:
        duplicate_paths.append((path, "appledouble", "macOS resource fork file", None))

    for key, files in grouped.items():
        if len(files) <= 1:
            continue
        keep = pick_keep_file(files)
        for item in files:
            if item.path == keep.path:
                continue
            reason = "same_title"
            if item.digest == keep.digest:
                reason = "same_hash_and_title"
            duplicate_paths.append((item.path, reason, key, keep.path))

    with report_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["action", "reason", "key", "path", "kept_path", "size"])
        kept_paths = {pick_keep_file(files).path for files in grouped.values()}
        for path in sorted(kept_paths):
            writer.writerow(["keep", "", normalize_title(path), path, "", path.stat().st_size])
        for path, reason, key, keep_path in sorted(duplicate_paths):
            writer.writerow(
                [
                    "move",
                    reason,
                    key,
                    path,
                    keep_path or "",
                    path.stat().st_size,
                ]
            )

    if not dry_run:
        for path, _reason, _key, _keep_path in duplicate_paths:
            destination = unique_destination(path, archive_dir)
            shutil.move(str(path), str(destination))

    return len(audio_paths), len(real_audio_paths), len(duplicate_paths)


def main() -> None:
    parser = argparse.ArgumentParser(description="Dedupe audio files in the training data folder.")
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("archive_dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    total, real_count, moved_count = dedupe(args.data_dir, args.archive_dir, args.dry_run)
    action = "would move" if args.dry_run else "moved"
    print(f"Scanned {total} audio-like files, {real_count} real audio files, {action} {moved_count}.")
    print(f"Report: {args.archive_dir / 'dedupe_report.csv'}")


if __name__ == "__main__":
    main()
