#!/usr/bin/env bash
# Continues the demo pipeline after the already-running SFT job (pid $1) finishes.
set -euo pipefail
SFT_PID=$1
echo "waiting for SFT (pid $SFT_PID)..."
while kill -0 "$SFT_PID" 2>/dev/null; do sleep 30; done
test -f outputs/sft-adapter/adapter_config.json || { echo "SFT adapter missing; aborting"; exit 1; }

python training_models_v1.py dpo
python training_models_v1.py grpo
python training_models_v1.py eval --regen
python training_models_v1.py report
python scripts/make_results_md.py
HF_REPO_PRIVATE=false python train_and_upload.py --upload-only
echo "FULL DEMO PIPELINE COMPLETE"
