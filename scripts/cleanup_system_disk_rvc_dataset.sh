#!/usr/bin/env bash
set -euo pipefail

ROOT="${RVC_ROOT:-/workspace/singing-voice-conversion/external/Retrieval-based-Voice-Conversion-WebUI}"
DATA_ROOT="${DATA_ROOT:-/data/rvc_datasets}"
EXP_NAME="${EXP_NAME:-jj_v2_48k}"

cleanup_one() {
  local name="$1"
  local src="$ROOT/logs/$name"
  local dst="$DATA_ROOT/$name"

  if [ ! -d "$dst" ]; then
    echo "Missing /data copy: $dst" >&2
    echo "Refusing to delete system-disk copy." >&2
    exit 1
  fi

  if [ -L "$src" ]; then
    echo "$src is already a symlink -> $(readlink "$src")"
    return
  fi

  if [ -e "$src" ]; then
    rm -rf "$src"
  fi

  ln -s "$dst" "$src"
  echo "Deleted system-disk copy and linked $src -> $dst"
}

cleanup_one "$EXP_NAME"

if [ -d "$DATA_ROOT/mute" ]; then
  cleanup_one "mute"
fi

echo "Done."
