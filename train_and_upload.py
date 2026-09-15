#!/usr/bin/env python3
"""Train every eCFR adapter and publish each one to Hugging Face Hub."""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

STAGES = ("sft", "dpo", "grpo")


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _repo_id(namespace: str, prefix: str, stage: str) -> str:
    name = f"{prefix}-{stage}".strip("-")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}", name):
        raise SystemExit(f"Invalid Hugging Face repository name: {name!r}")
    return f"{namespace}/{name}"


def _metrics_section(cfg: dict, stage: str) -> str:
    results_path = Path(cfg["paths"]["results_dir"]) / "eval_results.json"
    if not results_path.exists():
        return ""
    payload = json.loads(results_path.read_text())
    if payload.get("smoke") or stage not in payload.get("checkpoints", {}):
        return ""
    cols = ("citation_accuracy", "wrong_citation_rate", "token_f1", "rouge_l")
    lines = [
        "",
        f"## Evaluation ({payload['n_eval']} held-out examples, greedy decoding)",
        "",
        "| checkpoint | " + " | ".join(cols) + " |",
        "|---|" + "---|" * len(cols),
    ]
    for name in ("base", stage):
        overall = payload["checkpoints"].get(name, {}).get("overall")
        if overall:
            lines.append(
                f"| {name} | " + " | ".join(f"{overall[c]:.4f}" for c in cols) + " |"
            )
    lines += [
        "",
        "`wrong_citation_rate` is lower-is-better; the other metrics are higher-is-better.",
        "Full protocol and per-question-type results are in the training repository.",
    ]
    return "\n".join(lines) + "\n"


def _model_card(cfg: dict, stage: str, repo_id: str, adapter_dir: Path) -> str:
    adapter_cfg = json.loads((adapter_dir / "adapter_config.json").read_text())
    base_model = adapter_cfg.get("base_model_name_or_path", cfg["model"]["id"])
    snapshot = cfg["ecfr"]["date"]
    chapter = cfg["ecfr"]["chapter"]
    return f"""---
base_model: {base_model}
library_name: peft
pipeline_tag: text-generation
tags:
- legal
- finance
- ecfr
- lora
---

# {repo_id}

This repository contains the **{stage.upper()} PEFT adapter** produced by the
eCFR Title 12 training pipeline. It was trained on a point-in-time eCFR
snapshot dated {snapshot}, Title {cfg['ecfr']['title']}, Chapter {chapter}.

## Base model

`{base_model}`

## Intended use

The adapter is intended for research on grounded banking-regulation question
answering. Generated answers should be checked against the current eCFR and
must not be treated as legal advice.
{_metrics_section(cfg, stage)}"""


def _upload(api, cfg: dict, stage: str, adapter_dir: Path, repo_id: str, private: bool) -> None:
    manifest = Path(cfg["paths"]["outputs_dir"]) / "manifests" / f"{stage}_manifest.json"
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(adapter_dir),
        commit_message=f"Upload {stage.upper()} adapter",
    )
    api.upload_file(
        repo_id=repo_id,
        repo_type="model",
        path_in_repo="README.md",
        path_or_fileobj=_model_card(cfg, stage, repo_id, adapter_dir).encode(),
        commit_message="Add model card",
    )
    if manifest.exists():
        api.upload_file(
            repo_id=repo_id,
            repo_type="model",
            path_in_repo="training_manifest.json",
            path_or_fileobj=str(manifest),
            commit_message="Add training manifest",
        )
    print(f"[hub] published {stage.upper()} adapter -> https://huggingface.co/{repo_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--model", help="override the configured base model")
    parser.add_argument("--smoke", action="store_true", help="run short training jobs")
    parser.add_argument("--skip-data", action="store_true", help="reuse existing datasets")
    parser.add_argument(
        "--upload-only",
        action="store_true",
        help="publish existing adapters in outputs/ without training",
    )
    args = parser.parse_args()

    from dotenv import load_dotenv

    load_dotenv(args.env_file)
    token = os.getenv("HF_TOKEN")
    namespace = os.getenv("HF_REPO_NAMESPACE") or os.getenv("HF_USERNAME")
    prefix = os.getenv("HF_REPO_PREFIX", "ecfr-title12-llama31-8b")
    private = _env_bool("HF_REPO_PRIVATE", True)
    if not token:
        raise SystemExit(f"HF_TOKEN is required in {args.env_file} or the environment")
    if not namespace:
        raise SystemExit(
            f"HF_REPO_NAMESPACE (or HF_USERNAME) is required in {args.env_file} or the environment"
        )

    from huggingface_hub import HfApi
    from ecfr_pipeline import common

    cfg = common.load_config(args.config)
    if args.device:
        cfg.setdefault("runtime", {})["device"] = args.device
    common.ensure_dirs(cfg)

    api = HfApi(token=token)
    account = api.whoami()
    print(f"[hub] authenticated as {account['name']}; repositories are {'private' if private else 'public'}")

    if args.upload_only:
        out_root = Path(cfg["paths"]["outputs_dir"])
        for stage in STAGES:
            adapter_dir = out_root / f"{stage}-adapter"
            if not (adapter_dir / "adapter_config.json").exists():
                print(f"[hub] skipping {stage}: no adapter at {adapter_dir}")
                continue
            _upload(api, cfg, stage, adapter_dir, _repo_id(namespace, prefix, stage), private)
        return

    if not args.skip_data:
        from ecfr_pipeline import build_dataset, download_ecfr

        download_ecfr.run(cfg, smoke=args.smoke, model_override=args.model)
        build_dataset.run(cfg, smoke=args.smoke, model_override=args.model)

    from ecfr_pipeline import train_dpo, train_grpo, train_sft

    trainers = {"sft": train_sft.run, "dpo": train_dpo.run, "grpo": train_grpo.run}
    for stage in STAGES:
        adapter_dir = trainers[stage](cfg, smoke=args.smoke, model_override=args.model)
        repo_id = _repo_id(namespace, prefix, stage)
        _upload(api, cfg, stage, adapter_dir, repo_id, private)


if __name__ == "__main__":
    main()