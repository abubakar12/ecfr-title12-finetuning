#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

GATEWAY_DEVICE=${REGBENCH_GATEWAY_DEVICE:-cuda}

if [ "$GATEWAY_DEVICE" != "cpu" ] && pgrep -f "training_models_v1.py (cpt|sft|dpo|grpo)" >/dev/null; then
  echo "Training is still using the GPU. Start the gateway after run_all.sh completes." >&2
  exit 2
fi

if [ -f .env ]; then
  set -a
  . ./.env
  set +a
fi

if [ "$GATEWAY_DEVICE" = "cpu" ]; then
  export CUDA_VISIBLE_DEVICES=""
fi

exec python3 model_gateway.py --host "${REGBENCH_GATEWAY_HOST:-127.0.0.1}" --port "${REGBENCH_GATEWAY_PORT:-8090}"
