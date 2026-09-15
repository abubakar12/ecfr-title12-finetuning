#!/usr/bin/env python3
"""Generate RESULTS.md from results/eval_results.json and training manifests."""
from __future__ import annotations

import json
from pathlib import Path

COLS = ("citation_accuracy", "wrong_citation_rate", "token_f1", "rouge_l")
LOWER_IS_BETTER = {"wrong_citation_rate"}


def _delta(name: str, value: float, base: float, metric: str) -> str:
    if name == "base":
        return f"{value:.4f}"
    d = value - base
    worse = d > 0.005 if metric in LOWER_IS_BETTER else d < -0.005
    return f"{value:.4f} ({d:+.4f}{' ⚠' if worse else ''})"


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    payload = json.loads((root / "results" / "eval_results.json").read_text())
    checkpoints = payload["checkpoints"]
    base = checkpoints.get("base", {}).get("overall", {})

    lines = [
        "# Results: eCFR Title 12 LoRA Post-Training",
        "",
        f"- Base model: `{payload['model_id']}`",
        f"- Held-out eval examples: {payload['n_eval']} (section-level split, greedy decoding)",
        f"- Evaluated: {payload['timestamp_utc']}",
        f"- Device: {payload['runtime']['device']} ({payload['runtime'].get('vram_gb', 0)} GB), "
        f"dtype {payload['runtime']['torch_dtype']}",
    ]
    if payload.get("smoke"):
        lines += ["", "**SMOKE RUN — these numbers validate the pipeline, not model quality.**"]

    lines += [
        "",
        "## Overall (deltas vs untuned base; ⚠ = regression)",
        "",
        "| checkpoint | n | " + " | ".join(COLS) + " |",
        "|---|---|" + "---|" * len(COLS),
    ]
    for name, res in checkpoints.items():
        o = res["overall"]
        cells = [name, str(o["n"])] + [_delta(name, o[m], base.get(m, 0.0), m) for m in COLS]
        lines.append("| " + " | ".join(cells) + " |")

    lines += ["", "## token_f1 by question type", ""]
    names = list(checkpoints)
    types = sorted({t for r in checkpoints.values() for t in r["per_type"]})
    lines.append("| type | " + " | ".join(names) + " |")
    lines.append("|---|" + "---|" * len(names))
    for t in types:
        row = [t] + [
            f"{checkpoints[n]['per_type'].get(t, {}).get('token_f1', float('nan')):.4f}"
            for n in names
        ]
        lines.append("| " + " | ".join(row) + " |")

    manifest_dir = root / "outputs" / "manifests"
    lines += ["", "## Training stages", ""]
    for stage in ("sft", "dpo", "grpo"):
        p = manifest_dir / f"{stage}_manifest.json"
        if not p.exists():
            continue
        m = json.loads(p.read_text())
        extra = ", ".join(
            f"{k}={m[k]}" for k in ("n_train", "n_pairs", "n_prompts") if k in m
        )
        lines.append(f"- **{stage.upper()}** — model `{m['model_id']}`, {extra}, "
                     f"trained {m['timestamp_utc']}")

    lines += [
        "",
        "Artifacts: `results/eval_results.json`, `results/report.md`, "
        "`results/model_report.pdf`, per-checkpoint generations in `results/generations_*.jsonl`.",
        "",
    ]
    out = root / "RESULTS.md"
    out.write_text("\n".join(lines))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
