#!/usr/bin/env bash
# End-to-end run on a single training node (CUDA GPU, Apple Silicon, or CPU).
#   pip install -r requirements.txt
#   export HF_TOKEN=...        # required for the gated Llama 3.1 repo
# Device is auto-detected; force with e.g. `PIPELINE_ARGS="--device mps" ./run_all.sh`
set -euo pipefail

PY=${PY:-python3}
ARGS=${PIPELINE_ARGS:-}

$PY training_models_v1.py download $ARGS
$PY training_models_v1.py build    $ARGS
$PY training_models_v1.py sft      $ARGS
$PY training_models_v1.py dpo      $ARGS
# $PY training_models_v1.py grpo   $ARGS   # optional: ~3-4x SFT cost
$PY training_models_v1.py eval     $ARGS
$PY training_models_v1.py report   $ARGS   # -> results/model_report.pdf
