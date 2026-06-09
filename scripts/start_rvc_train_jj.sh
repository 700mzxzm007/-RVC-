#!/usr/bin/env bash
set -euo pipefail

ROOT="/Users/mzx/PycharmProjects/singing-voice-conversion/external/Retrieval-based-Voice-Conversion-WebUI"
PY="/opt/anaconda3/envs/RVCTrain/bin/python"

cd "$ROOT"

"$PY" infer/modules/train/train.py \
  -e jj_v2_48k \
  -sr 48k \
  -f0 1 \
  -bs 4 \
  -g 0 \
  -te 100 \
  -se 10 \
  -pg assets/pretrained_v2/f0G48k.pth \
  -pd assets/pretrained_v2/f0D48k.pth \
  -l 1 \
  -c 0 \
  -sw 1 \
  -v v2
