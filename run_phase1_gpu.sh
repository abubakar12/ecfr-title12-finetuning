#!/usr/bin/env bash
# Phase-1 CPT on a CUDA GPU (>= 40 GB VRAM; 80 GB recommended for batch_size 2 x 4096).
# The frozen corpus + benchmark in experiments/title12-chapter-I-govinfo-gpu-v1 must be
# present (copy it with the repo). Requires HF access to meta-llama/Meta-Llama-3.1-8B-Instruct.
#   export HF_TOKEN=hf_...            # or: hf auth login
#   python3 -m venv .venv && . .venv/bin/activate && pip install -r requirements-phase1.txt
#   nohup ./run_phase1_gpu.sh > phase1_gpu.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")"

PY=${PY:-python}
CFG=${CFG:-phase1.gpu.json}
EXP=$($PY -c "import json;print(json.load(open('$CFG'))['experiment_dir'])")
export HF_HUB_DISABLE_XET=${HF_HUB_DISABLE_XET:-1}
export TOKENIZERS_PARALLELISM=false

run() { echo "== $* ($(date -u +%FT%TZ))"; "$@"; }

run $PY training_models_v1.py check-runtime --config "$CFG"
run $PY training_models_v1.py check-hub     --config "$CFG"
run $PY training_models_v1.py audit-corpus  --config "$CFG"
[ -f "$EXP/smoke/completed.json" ] || run $PY training_models_v1.py pretrain --smoke --config "$CFG"
[ -f "$EXP/training/completed.json" ] || run $PY training_models_v1.py pretrain --config "$CFG"
[ -f "$EXP/evaluation/generation.lock.json" ] || run $PY training_models_v1.py eval-cpt --config "$CFG"
run $PY training_models_v1.py auto-score-cpt --config "$CFG"
echo "== done: $EXP/evaluation/results.json"
cat "$EXP/evaluation/results.json"
