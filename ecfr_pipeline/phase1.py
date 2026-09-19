"""CLI for the isolated phase-one experiment; data commands need only Python."""
import argparse
from pathlib import Path

from . import corpus


COMMANDS = ["build-corpus", "audit-corpus", "fetch-assets", "draft-benchmark", "tokenize-corpus", "freeze-benchmark", "pretrain", "eval-cpt", "score-cpt", "auto-score-cpt", "check-runtime", "check-hub", "push-model"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=COMMANDS)
    parser.add_argument("--config", default="phase1.json")
    parser.add_argument("--refresh", action="store_true", help="Requires a new experiment directory")
    parser.add_argument("--rebuild-corpus", action="store_true", help="Archive and rebuild derived corpus before benchmark freeze")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--precision", choices=["auto", "bf16", "fp32"])
    parser.add_argument("--resume", help="Explicit Trainer checkpoint directory")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--inventory-only", action="store_true", help="Inventory graphic URLs without downloading")
    parser.add_argument("--input", help="Reviewed benchmark or completed blind reviews JSONL")
    parser.add_argument("--output", help="auto-score-cpt: where to write the automatic review (default evaluation/auto_review.jsonl)")
    args = parser.parse_args(argv)
    cfg = corpus.read_json(Path(args.config))
    for key in ("device", "precision"):
        if getattr(args, key) is not None:
            cfg.setdefault("runtime", {})[key] = getattr(args, key)
    if cfg["ecfr"]["title"] != 12:
        parser.error("This experiment is restricted to Title 12")
    if cfg["pretrain"]["max_length"] <= 0 or cfg["pretrain"]["overlap"] < 0:
        parser.error("Invalid sequence/overlap configuration")
    if args.command in {"check-hub", "push-model"}:
        from . import hub
        if args.smoke:
            parser.error("Smoke models cannot be published")
        hub.check(cfg) if args.command == "check-hub" else hub.publish(cfg)
    elif args.command == "check-runtime":
        import json
        from .phase1_runtime import resolve
        print(json.dumps(resolve(cfg, args.smoke), indent=2))
    elif args.command == "build-corpus":
        result = corpus.build(cfg, args.refresh, args.rebuild_corpus)
        if not result["passed"]:
            raise SystemExit("Corpus audit failed; inspect audit.json")
    elif args.command == "audit-corpus":
        corpus.require_audit(cfg)
    elif args.command == "fetch-assets":
        corpus.fetch_assets(cfg, download=not args.inventory_only)
    elif args.command == "draft-benchmark":
        from .benchmark_seed import draft
        draft(cfg)
    elif args.command == "tokenize-corpus":
        from .cpt import prepare
        prepare(cfg, args.smoke)
    elif args.command == "pretrain":
        from .cpt import train
        train(cfg, args.smoke, args.resume, args.preflight_only)
    elif args.command == "auto-score-cpt":
        from . import autoscore
        autoscore.run(cfg, args.output)
    else:
        from . import benchmark
        if args.command in {"freeze-benchmark", "score-cpt"} and not args.input:
            parser.error("--input is required")
        if args.command == "freeze-benchmark":
            benchmark.freeze(cfg, args.input)
        elif args.command == "eval-cpt":
            benchmark.generate(cfg)
        else:
            benchmark.score(cfg, args.input)


if __name__ == "__main__":
    main()
