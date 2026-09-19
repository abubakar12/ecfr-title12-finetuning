#!/usr/bin/env bash
# End-to-end run on a single training node (CUDA GPU, Apple Silicon, or CPU).
#   pip install -r requirements.txt
#   export HF_TOKEN=...        # required for the gated Llama 3.1 repo
# Device is auto-detected; force with e.g. `PIPELINE_ARGS="--device mps" ./run_all.sh`
set -euo pipefail

PY=${PY:-python3}
ARGS=${PIPELINE_ARGS:-}

# Re-read .env before each stage so HF_TOKEN / GITHUB_TOKEN can be added mid-run.
step() { [ -f .env ] && set -a && . ./.env && set +a; echo "== $* ($(date -u +%FT%TZ))"; $PY training_models_v1.py "$@" $ARGS; }

step download
step build
step cpt       # continued pretraining on regulation text -> publishes CPT adapter to HF (needs HF_TOKEN)
step sft       # continues the CPT adapter on Q&A            -> publishes SFT adapter
step dpo       # preference tuning on top of SFT             -> publishes DPO adapter
step grpo      # RL with verifiable rewards on top of SFT    -> publishes GRPO adapter
step eval      # base vs cpt vs sft vs dpo vs grpo on the held-out set
step report    # -> results/model_report.pdf + results/examples_all.md
$PY scripts/make_results_md.py    || echo "[warn] RESULTS.md not regenerated"
$PY scripts/make_readme_assets.py || echo "[warn] README assets not regenerated"

# Publish data, results and report to GitHub (weights go to HF, see .gitignore).
# Needs GITHUB_TOKEN (repo scope) in the environment or a configured credential helper.
if [ "${GIT_PUSH:-1}" = "1" ]; then
  [ -f .env ] && set -a && . ./.env && set +a   # GITHUB_TOKEN may be added here while the run is in progress
  git config user.name  >/dev/null || git config user.name  "${GIT_AUTHOR_NAME:-ecfr-pipeline}"
  git config user.email >/dev/null || git config user.email "${GIT_AUTHOR_EMAIL:-ecfr-pipeline@users.noreply.github.com}"
  git add -A config.yaml data results assets RESULTS.md 2>/dev/null || git add -A config.yaml data results
  git commit -m "Full-title 12 CFR run: CPT/SFT/DPO/GRPO datasets, eval results, report ($(date -u +%F))" || echo "nothing to commit"
  BRANCH=$(git rev-parse --abbrev-ref HEAD)
  if [ -n "${GITHUB_TOKEN:-}" ]; then
    URL=$(git remote get-url origin | sed -E "s#https://(.*@)?#https://x-access-token:${GITHUB_TOKEN}@#")
    git push "$URL" "HEAD:$BRANCH"
  else
    git push origin "HEAD:$BRANCH"
  fi
fi
