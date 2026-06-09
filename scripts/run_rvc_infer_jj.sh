#!/usr/bin/env bash
set -euo pipefail

INPUT_WAV="$1"
OUTPUT_WAV="$2"
F0_UP_KEY="${3:-0}"

ROOT="${RVC_ROOT:-/root/workspace/singing-voice-conversion/external/Retrieval-based-Voice-Conversion-WebUI}"
PY="${PYTHON:-python}"
MODEL_NAME="${MODEL_NAME:-jj_v2_48k.pth}"
INDEX_PATH="${INDEX_PATH-}"
F0_METHOD="${F0_METHOD:-rmvpe}"
INDEX_RATE="${INDEX_RATE:-0.75}"
PROTECT="${PROTECT:-0.55}"
FILTER_RADIUS="${FILTER_RADIUS:-5}"
RMS_MIX_RATE="${RMS_MIX_RATE:-0.85}"
DEVICE="${DEVICE:-cuda:0}"

mkdir -p "$(dirname "$OUTPUT_WAV")"
cd "$ROOT"

"$PY" tools/infer_cli.py \
  --model_name "$MODEL_NAME" \
  --input_path "$INPUT_WAV" \
  --opt_path "$OUTPUT_WAV" \
  --index_path "$INDEX_PATH" \
  --f0up_key "$F0_UP_KEY" \
  --f0method "$F0_METHOD" \
  --index_rate "$INDEX_RATE" \
  --filter_radius "$FILTER_RADIUS" \
  --resample_sr 0 \
  --rms_mix_rate "$RMS_MIX_RATE" \
  --protect "$PROTECT" \
  --device "$DEVICE" \
  --is_half False