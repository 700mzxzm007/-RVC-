#!/usr/bin/env bash
set -euo pipefail

ROOT="/Users/mzx/PycharmProjects/singing-voice-conversion/external/Retrieval-based-Voice-Conversion-WebUI"
PY="/opt/anaconda3/envs/RVCTrain/bin/python"

cd "$ROOT"

"$PY" infer/modules/train/train.py \
  -e jj_v2_48k \
  -sr 48k \
  -f0 1 \
  -bs 1 \
  -te 1 \
  -se 1 \
  -pg assets/pretrained_v2/f0G48k.pth \
  -pd assets/pretrained_v2/f0D48k.pth \
  -l 1 \
  -c 0 \
  -sw 0 \
  -v v2
