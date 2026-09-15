#!/usr/bin/env python3
"""Render README charts from training logs and eval results into assets/."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"
CHECK_COLORS = {"base": "#9e9e9e", "sft": "#1f77b4", "dpo": "#2ca02c", "grpo": "#d62728"}


def _log(stage: str):
    p = ROOT / "outputs" / f"{stage}-adapter" / "training_log.json"
    return json.loads(p.read_text()) if p.exists() else None


def _series(log, key):
    pts = [(e["step"], e[key]) for e in log if key in e and "step" in e]
    return [p[0] for p in pts], [p[1] for p in pts]


def training_charts() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    sft = _log("sft")
    if sft:
        ax = axes[0]
        ax.plot(*_series(sft, "loss"), color=CHECK_COLORS["sft"], label="train loss")
        ax2 = ax.twinx()
        ax2.plot(*_series(sft, "mean_token_accuracy"), color="#ff7f0e", alpha=0.7,
                 label="token accuracy")
        ax2.set_ylabel("token accuracy")
        ax.set_title("SFT: loss and token accuracy")
        ax.set_xlabel("step"); ax.set_ylabel("loss")
        ax.legend(loc="upper right"); ax2.legend(loc="center right")

    dpo = _log("dpo")
    if dpo:
        ax = axes[1]
        ax.plot(*_series(dpo, "rewards/margins"), color=CHECK_COLORS["dpo"], label="reward margin")
        ax2 = ax.twinx()
        ax2.plot(*_series(dpo, "rewards/accuracies"), color="#ff7f0e", alpha=0.7,
                 label="preference accuracy")
        ax2.set_ylim(0, 1.05); ax2.set_ylabel("preference accuracy")
        ax.set_title("DPO: chosen-vs-rejected margin")
        ax.set_xlabel("step"); ax.set_ylabel("margin (logprob units)")
        ax.legend(loc="upper left"); ax2.legend(loc="lower right")

    grpo = _log("grpo")
    if grpo:
        ax = axes[2]
        ax.plot(*_series(grpo, "reward"), color=CHECK_COLORS["grpo"], label="mean reward")
        steps, stds = _series(grpo, "reward_std")
        if steps:
            ax.plot(steps, stds, color="#9467bd", alpha=0.7, label="reward std")
        ax.set_title("GRPO: verifiable reward")
        ax.set_xlabel("step"); ax.set_ylabel("reward")
        ax.legend()

    fig.tight_layout()
    fig.savefig(ASSETS / "training_curves.png", dpi=120)
    plt.close(fig)
    print("wrote assets/training_curves.png")


def eval_charts() -> None:
    p = ROOT / "results" / "eval_results.json"
    if not p.exists():
        print("eval_results.json missing — skipping eval charts")
        return
    payload = json.loads(p.read_text())
    ckpts = payload["checkpoints"]
    names = [n for n in ("base", "sft", "dpo", "grpo") if n in ckpts]

    metrics = ["citation_accuracy", "wrong_citation_rate", "token_f1", "rouge_l"]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    width = 0.8 / len(names)
    for i, name in enumerate(names):
        o = ckpts[name]["overall"]
        xs = [j + i * width for j in range(len(metrics))]
        ax.bar(xs, [o[m] for m in metrics], width, label=name, color=CHECK_COLORS[name])
    ax.set_xticks([j + width * (len(names) - 1) / 2 for j in range(len(metrics))])
    ax.set_xticklabels(metrics, fontsize=9)
    ax.set_title(f"Held-out eval, n={payload['n_eval']} (wrong_citation_rate: lower is better)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(ASSETS / "eval_overall.png", dpi=120)
    plt.close(fig)

    types = sorted({t for c in ckpts.values() for t in c["per_type"]})
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, metric, title in (
        (axes[0], "token_f1", "token F1 by question type"),
        (axes[1], "citation_accuracy", "citation accuracy by question type"),
    ):
        width = 0.8 / len(names)
        for i, name in enumerate(names):
            vals = [ckpts[name]["per_type"].get(t, {}).get(metric, 0.0) for t in types]
            xs = [j + i * width for j in range(len(types))]
            ax.bar(xs, vals, width, label=name, color=CHECK_COLORS[name])
        ax.set_xticks([j + width * (len(names) - 1) / 2 for j in range(len(types))])
        ax.set_xticklabels(types, fontsize=9)
        ax.set_title(title)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(ASSETS / "eval_by_type.png", dpi=120)
    plt.close(fig)
    print("wrote assets/eval_overall.png and assets/eval_by_type.png")


if __name__ == "__main__":
    ASSETS.mkdir(exist_ok=True)
    training_charts()
    eval_charts()
