#!/usr/bin/env python3
"""CLI for the eCFR Title 12 LoRA fine-tuning pipeline.

Stages: download -> build -> sft -> dpo [-> grpo] -> eval
Runs on CUDA GPUs (QLoRA/LoRA), Apple Silicon GPUs via MPS (M1-M5), or CPU.
"""
import argparse
import os

# must be set before torch initializes; harmless on non-Mac platforms
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

STAGES = ["download", "build", "sft", "dpo", "grpo", "eval", "report"]


def main() -> None:
    # Phase 1 uses an isolated JSON configuration and stdlib-only data path.
    import sys
    from ecfr_pipeline.phase1 import COMMANDS, main as phase1_main
    if len(sys.argv) > 1 and sys.argv[1] in COMMANDS:
        phase1_main(sys.argv[1:])
        return
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    ap = argparse.ArgumentParser(
        description="eCFR Title 12 fine-tuning pipeline (SFT / DPO / GRPO / eval)",
        epilog="Isolated Phase 1 commands: " + ", ".join(COMMANDS) + ". Use COMMAND --help for their options.",
    )
    ap.add_argument(
        "command",
        choices=STAGES + ["all"],
        help="'all' runs download,build,sft,dpo,eval (grpo is opt-in: run it explicitly)",
    )
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument(
        "--device",
        choices=["auto", "cuda", "mps", "cpu"],
        help="override runtime.device (mps = Apple Silicon GPU, M1-M5)",
    )
    ap.add_argument("--seed", type=int, help="override seed")
    ap.add_argument("--model", help="override model id (e.g. a local path)")
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="tiny model + a few steps to validate the full loop cheaply",
    )
    ap.add_argument(
        "--regen",
        action="store_true",
        help="eval only: ignore cached generations and regenerate",
    )
    args = ap.parse_args()

    from ecfr_pipeline import common

    cfg = common.load_config(args.config)
    if args.device:
        cfg.setdefault("runtime", {})["device"] = args.device
    if args.seed is not None:
        cfg["seed"] = args.seed
    common.set_global_seed(cfg["seed"])
    common.ensure_dirs(cfg)

    steps = (
        ["download", "build", "sft", "dpo", "eval", "report"]
        if args.command == "all"
        else [args.command]
    )
    for step in steps:
        if step == "download":
            from ecfr_pipeline import download_ecfr

            download_ecfr.run(cfg, smoke=args.smoke, model_override=args.model)
        elif step == "build":
            from ecfr_pipeline import build_dataset

            build_dataset.run(cfg, smoke=args.smoke, model_override=args.model)
        elif step == "sft":
            from ecfr_pipeline import train_sft

            train_sft.run(cfg, smoke=args.smoke, model_override=args.model)
        elif step == "dpo":
            from ecfr_pipeline import train_dpo

            train_dpo.run(cfg, smoke=args.smoke, model_override=args.model)
        elif step == "grpo":
            from ecfr_pipeline import train_grpo

            train_grpo.run(cfg, smoke=args.smoke, model_override=args.model)
        elif step == "eval":
            from ecfr_pipeline import evaluate

            evaluate.run(cfg, smoke=args.smoke, model_override=args.model, regen=args.regen)
        elif step == "report":
            from ecfr_pipeline import make_report

            make_report.run(cfg, smoke=args.smoke, model_override=args.model)


if __name__ == "__main__":
    main()
