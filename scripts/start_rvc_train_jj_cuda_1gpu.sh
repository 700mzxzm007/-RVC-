#!/usr/bin/env bash
set -euo pipefail

ROOT="${RVC_ROOT:-/workspace/singing-voice-conversion/external/Retrieval-based-Voice-Conversion-WebUI}"
PY="${PYTHON:-python}"

cd "$ROOT"

"$PY" infer/modules/train/train.py \
  -e jj_v2_48k \
  -sr 48k \
  -f0 1 \
  -bs "${BATCH_SIZE:-8}" \
  -g 0 \
  -te "${TOTAL_EPOCH:-100}" \
  -se "${SAVE_EVERY_EPOCH:-10}" \
  -pg assets/pretrained_v2/f0G48k.pth \
  -pd assets/pretrained_v2/f0D48k.pth \
  -l 1 \
  -c 0 \
  -sw 1 \
  -v v2
