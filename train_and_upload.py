#!/usr/bin/env python3
"""Train every eCFR adapter and publish each one to Hugging Face Hub."""
from __future__ import annotations

import argparse
from pathlib import Path

STAGES = ("sft", "dpo", "grpo")


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
    parser.add_argument("--stage", choices=STAGES, help="Train or retry a single stage")
    args = parser.parse_args()

    from dotenv import load_dotenv

    load_dotenv(args.env_file)
    from ecfr_pipeline import common, hub

    cfg = common.load_config(args.config)
    if args.device:
        cfg.setdefault("runtime", {})["device"] = args.device
    common.ensure_dirs(cfg)

    cfg.setdefault("hub", {})["enabled"] = True
    if not args.smoke:
        hub.check(cfg)

    if args.upload_only:
        if args.smoke:
            parser.error("Smoke models cannot be published")
        out_root = Path(cfg["paths"]["outputs_dir"])
        for stage in ([args.stage] if args.stage else STAGES):
            adapter_dir = out_root / f"{stage}-adapter"
            if not (adapter_dir / "adapter_config.json").exists():
                print(f"[hub] skipping {stage}: no adapter at {adapter_dir}")
                continue
            hub.publish(cfg, stage)
        return

    if not args.skip_data:
        from ecfr_pipeline import build_dataset, download_ecfr

        download_ecfr.run(cfg, smoke=args.smoke, model_override=args.model)
        build_dataset.run(cfg, smoke=args.smoke, model_override=args.model)

    from ecfr_pipeline import train_dpo, train_grpo, train_sft

    trainers = {"sft": train_sft.run, "dpo": train_dpo.run, "grpo": train_grpo.run}
    for stage in ([args.stage] if args.stage else STAGES):
        trainers[stage](cfg, smoke=args.smoke, model_override=args.model)


if __name__ == "__main__":
    main()