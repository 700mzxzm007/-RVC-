#!/usr/bin/env bash
set -euo pipefail

ROOT="${RVC_ROOT:-/root/workspace/singing-voice-conversion/external/Retrieval-based-Voice-Conversion-WebUI}"
DATA_ROOT="${DATA_ROOT:-/data/rvc_datasets}"
EXP_NAME="${EXP_NAME:-jj_v2_48k}"

mkdir -p "$DATA_ROOT"

move_and_link() {
  local name="$1"
  local src="$ROOT/logs/$name"
  local dst="$DATA_ROOT/$name"

  if [ -L "$src" ]; then
    echo "$src is already a symlink -> $(readlink "$src")"
    return
  fi

  if [ ! -e "$src" ]; then
    echo "Missing source: $src" >&2
    exit 1
  fi

  if [ -e "$dst" ]; then
    echo "Destination already exists: $dst" >&2
    echo "Move it away or set DATA_ROOT to another directory." >&2
    exit 1
  fi

  mv "$src" "$dst"
  ln -s "$dst" "$src"
  echo "Moved $src -> $dst"
}

move_and_link "$EXP_NAME"

if [ -e "$ROOT/logs/mute" ] || [ -L "$ROOT/logs/mute" ]; then
  move_and_link "mute"
fi

echo "Done."
echo "Dataset root: $DATA_ROOT"
echo "Experiment symlink: $ROOT/logs/$EXP_NAME -> $(readlink "$ROOT/logs/$EXP_NAME")"
