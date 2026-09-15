"""Stage 7: stakeholder PDF report -> results/model_report.pdf.

One self-contained file for anyone evaluating the trained model: what was
trained and for how long, training/validation curves, held-out metrics vs the
untuned base, real side-by-side generations, and how to reproduce the run.
Needs the `eval` stage outputs; training curves appear for whichever stages ran.
"""
from __future__ import annotations

import json
import textwrap
import time
from pathlib import Path

from . import common, metrics

PAGE = (8.27, 11.69)  # A4 portrait
CHECK, CROSS = "\u2713", "\u2717"


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def _wrap(text: str, width: int) -> list:
    out = []
    for para in (text or "").splitlines():
        out.extend(textwrap.wrap(para, width) or [""])
    return out


def _new_page(title: str, subtitle: str = ""):
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=PAGE)
    fig.text(0.06, 0.955, title, fontsize=15, fontweight="bold")
    if subtitle:
        fig.text(0.06, 0.936, subtitle, fontsize=9, color="dimgray")
    fig.text(0.94, 0.02, "eCFR Title 12 fine-tuning", fontsize=8, color="gray", ha="right")
    return fig


def _render_lines(fig, lines: list, y: float = 0.915) -> float:
    """lines: (text, style) with style in {'h','b','mono','warn'}; returns final y."""
    for text, style in lines:
        if y < 0.045:
            fig.text(0.06, y, "…", fontsize=9)
            break
        if style == "h":
            y -= 0.010
            fig.text(0.06, y, text, fontsize=11, fontweight="bold")
            y -= 0.019
        elif style == "warn":
            fig.text(0.06, y, text, fontsize=9, color="firebrick", fontweight="bold")
            y -= 0.016
        elif style == "mono":
            fig.text(0.06, y, text, fontsize=7.4, family="monospace")
            y -= 0.0122
        else:
            fig.text(0.06, y, text, fontsize=9)
            y -= 0.0145
    return y


def _load_log(out_root: Path, stage: str):
    p = out_root / f"{stage}-adapter" / "training_log.json"
    return json.loads(p.read_text()) if p.exists() else None


def _count_rows(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text().splitlines() if line.strip())


def _epochs_reached(log) -> float | None:
    epochs = [e["epoch"] for e in (log or []) if "epoch" in e]
    return max(epochs) if epochs else None


# ---------------------------------------------------------------------------
# pages
# ---------------------------------------------------------------------------


def _title_page(pdf, cfg, res, out_root, data_dir):
    fig = _new_page("Fine-tuning Report: eCFR Title 12 Regulation Assistant")
    e = cfg["ecfr"]
    lines = [
        (f"Generated: {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}", "b"),
        (f"Model evaluated: {res['model_id']}", "b"),
        (f"Compute: {res['runtime']['device']}"
         + (f" ({res['runtime']['vram_gb']} GB VRAM)" if res['runtime']['device'] == 'cuda' else "")
         + f", dtype {res['runtime']['torch_dtype']}, 4-bit QLoRA: {res['runtime']['use_4bit']}", "b"),
        (f"Seed: {cfg['seed']}   |   Config hash: {common.config_hash(cfg)}", "b"),
        ("", "b"),
    ]
    if res.get("smoke"):
        lines += [
            ("SMOKE RUN — numbers below come from a tiny validation model trained for a", "warn"),
            ("few steps. They prove the pipeline works, not model quality. Re-run the", "warn"),
            ("pipeline without --smoke on a GPU, then regenerate this report.", "warn"),
            ("", "b"),
        ]
    lines += [
        ("What was built", "h"),
        ("A Llama-3.1-8B-Instruct LoRA assistant for U.S. banking regulations, trained on", "b"),
        (f"a point-in-time snapshot ({e['date']}) of eCFR Title {e['title']}, Chapter {e['chapter']}"
         " (OCC). Three", "b"),
        ("post-training techniques are compared against the untuned base model:", "b"),
        ("", "b"),
        ("  1. SFT  — supervised fine-tuning on grounded Q&A built from the regulations", "b"),
        ("  2. DPO  — preference tuning: correct+cited answers preferred over answers with", "b"),
        ("            wrong sections, wrong citations, or vague non-answers", "b"),
        ("  3. GRPO — RL with verifiable rewards (citation correctness, grounding F1,", "b"),
        ("            brevity/anti-repetition guard); no reward model needed", "b"),
        ("", "b"),
        ("Pipeline:  download -> build -> SFT -> DPO -> (GRPO) -> eval -> report", "mono"),
        ("", "b"),
        ("Dataset", "h"),
    ]
    counts = {
        "SFT train": _count_rows(data_dir / "sft_train.jsonl"),
        "SFT val": _count_rows(data_dir / "sft_val.jsonl"),
        "DPO pairs": _count_rows(data_dir / "dpo_train.jsonl"),
        "Held-out eval": _count_rows(data_dir / "eval_test.jsonl"),
    }
    lines += [(f"  {name:<14} {n:>6} rows", "mono") for name, n in counts.items()]
    lines += [
        ("", "b"),
        ("Split is by section via a stable hash — no section text or prompt can leak from", "b"),
        ("train into eval (asserted at build time). Question types: overview, provisions,", "b"),
        ("citation_lookup (unique headings only), definition. Full details: data/dataset_card.md.", "b"),
        ("", "b"),
        ("Evaluation protocol", "h"),
        (f"Greedy decoding, fixed seed, {res['n_eval']} held-out examples the model never saw in", "b"),
        ("training. Metrics: citation_accuracy (predicted § matches the target section),", "b"),
        ("wrong_citation_rate (confidently cites the wrong section — lower is better),", "b"),
        ("token_f1 and ROUGE-L vs a grounded reference answer.", "b"),
    ]
    _render_lines(fig, lines)
    pdf.savefig(fig)
    import matplotlib.pyplot as plt

    plt.close(fig)


def _hparams_page(pdf, cfg, out_root):
    fig = _new_page("Training configuration", "All knobs live in config.yaml")
    lora, s, d, g = cfg["lora"], cfg["sft"], cfg["dpo"], cfg["grpo"]
    logs = {st: _load_log(out_root, st) for st in ("sft", "dpo", "grpo")}

    def reached(st):
        ep = _epochs_reached(logs[st])
        return f"{ep:.2f}" if ep is not None else "not run"

    lines = [
        ("LoRA adapter (shared by all stages)", "h"),
        (f"  rank r={lora['r']}  alpha={lora['alpha']}  dropout={lora['dropout']}", "mono"),
        (f"  target modules: {', '.join(lora['target_modules'])}", "mono"),
        ("", "b"),
        ("Per-stage hyperparameters", "h"),
        (f"  {'':<24}{'SFT':>14}{'DPO':>14}{'GRPO':>14}", "mono"),
        (f"  {'-' * 66}", "mono"),
        (f"  {'epochs (configured)':<24}{s['epochs']:>14}{d['epochs']:>14}{'max ' + str(g['max_steps']) + ' steps':>14}", "mono"),
        (f"  {'epochs (reached)':<24}{reached('sft'):>14}{reached('dpo'):>14}{reached('grpo'):>14}", "mono"),
        (f"  {'learning rate':<24}{s['learning_rate']:>14}{d['learning_rate']:>14}{g['learning_rate']:>14}", "mono"),
        (f"  {'per-device batch':<24}{s['per_device_batch']:>14}{d['per_device_batch']:>14}{g['per_device_batch']:>14}", "mono"),
        (f"  {'grad accumulation':<24}{s['grad_accum']:>14}{d['grad_accum']:>14}{g['grad_accum']:>14}", "mono"),
        (f"  {'effective batch':<24}{s['per_device_batch'] * s['grad_accum']:>14}{d['per_device_batch'] * d['grad_accum']:>14}{g['per_device_batch'] * g['grad_accum']:>14}", "mono"),
        (f"  {'max seq length':<24}{s['max_length']:>14}{d['max_length']:>14}{g['max_prompt_length']:>14}", "mono"),
        ("", "b"),
        ("  DPO: beta = " + str(d["beta"]) + " (reference = frozen SFT adapter, no second model in memory)", "mono"),
        (f"  GRPO: {g['num_generations']} generations/prompt, temperature {g['temperature']}, "
         f"beta {g['beta']} (no KL ref), max {g['max_completion_length']} new tokens", "mono"),
        ("", "b"),
        ("Stage chaining", "h"),
        ("  SFT trains a fresh LoRA on the base model. DPO and GRPO both initialize from the", "b"),
        ("  SFT adapter, so the comparison isolates what each preference/RL step adds on top", "b"),
        ("  of the same supervised starting point.", "b"),
        ("", "b"),
        ("GRPO reward design (verifiable, no reward model)", "h"),
        ("  citation_reward   +1.0 correct § cited (capped at 0.25 if >3 sections spammed),", "mono"),
        ("                    -0.5 confidently wrong citation, 0.0 if no citation", "mono"),
        ("  grounding_reward  token-F1 overlap with the grounded reference answer", "mono"),
        ("  brevity_reward    -0.3 if >320 words; extra -0.5 for degenerate repetition", "mono"),
        ("                    (anti-reward-hacking guard)", "mono"),
        ("", "b"),
        ("Precision / hardware policy", "h"),
        ("  cuda: bf16 (or fp16), 4-bit QLoRA when VRAM < "
         + str(cfg["model"]["qlora_below_gb"]) + " GB, paged 8-bit AdamW", "mono"),
        ("  mps (Apple Silicon M1–M5): un-quantized LoRA, bf16 when supported, eager attention", "mono"),
        ("  cpu: float32 — intended for --smoke validation only", "mono"),
    ]
    _render_lines(fig, lines)
    pdf.savefig(fig)
    import matplotlib.pyplot as plt

    plt.close(fig)


def _series(log, key):
    pts = [(e.get("step", i), e[key]) for i, e in enumerate(log or []) if key in e]
    return [p[0] for p in pts], [p[1] for p in pts]


def _curves_page(pdf, out_root):
    import matplotlib.pyplot as plt

    logs = {s: _load_log(out_root, s) for s in ("sft", "dpo", "grpo")}
    fig, axes = plt.subplots(2, 2, figsize=PAGE)
    fig.suptitle("Training and validation curves", fontsize=15, fontweight="bold")

    def plot_loss(ax, stage, title):
        log = logs[stage]
        if not log:
            ax.text(0.5, 0.5, f"{stage} not run", ha="center", va="center", color="gray")
            ax.set_axis_off()
            return
        steps, loss = _series(log, "loss")
        if steps:
            ax.plot(steps, loss, "o-", label="train loss", color="tab:blue", markersize=2.5)
        esteps, eloss = _series(log, "eval_loss")
        if esteps:
            ax.plot(esteps, eloss, "o-", label="validation loss", color="tab:orange")
        if not steps and not esteps:  # smoke runs only log the final summary
            _, tl = _series(log, "train_loss")
            if tl:
                ax.axhline(tl[-1], ls="--", color="tab:blue", label=f"final train loss {tl[-1]:.3f}")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("step", fontsize=8)
        ax.legend(fontsize=7)
        ax.tick_params(labelsize=7)

    plot_loss(axes[0][0], "sft", "SFT loss")
    plot_loss(axes[0][1], "dpo", "DPO loss")

    ax = axes[1][0]
    if logs["grpo"]:
        steps, reward = _series(logs["grpo"], "reward")
        if steps:
            ax.plot(steps, reward, "o-", color="tab:green", label="mean total reward", markersize=3)
            lo = [r - s for r, s in zip(reward, _series(logs["grpo"], "reward_std")[1])]
            hi = [r + s for r, s in zip(reward, _series(logs["grpo"], "reward_std")[1])]
            if len(lo) == len(steps):
                ax.fill_between(steps, lo, hi, alpha=0.2, color="tab:green")
        ax.set_title("GRPO total reward", fontsize=10)
        ax.set_xlabel("step", fontsize=8)
        ax.legend(fontsize=7)
        ax.tick_params(labelsize=7)
    else:
        ax.text(0.5, 0.5, "grpo not run", ha="center", va="center", color="gray")
        ax.set_axis_off()

    ax = axes[1][1]
    if logs["grpo"]:
        for key, label in (
            ("rewards/citation_reward/mean", "citation"),
            ("rewards/grounding_reward/mean", "grounding F1"),
            ("rewards/brevity_reward/mean", "brevity guard"),
        ):
            steps, vals = _series(logs["grpo"], key)
            if steps:
                ax.plot(steps, vals, "o-", label=label, markersize=3)
        ax.set_title("GRPO reward components", fontsize=10)
        ax.set_xlabel("step", fontsize=8)
        ax.legend(fontsize=7)
        ax.tick_params(labelsize=7)
    else:
        ax.set_axis_off()

    fig.text(0.06, 0.03,
             "Sparse curves = --smoke run (5 steps, logging every 10): only the final summary "
             "is logged. Full runs produce dense curves.",
             fontsize=7.5, color="dimgray")
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    pdf.savefig(fig)
    plt.close(fig)


def _metrics_page(pdf, res):
    import matplotlib.pyplot as plt

    ckpts = res["checkpoints"]
    names = list(ckpts)
    fig = plt.figure(figsize=PAGE)
    fig.suptitle("Held-out evaluation: base vs fine-tuned checkpoints",
                 fontsize=15, fontweight="bold")

    specs = [
        ("citation_accuracy", "Citation accuracy (higher = better)", "tab:blue"),
        ("token_f1", "Token F1 vs reference (higher = better)", "tab:green"),
        ("rouge_l", "ROUGE-L vs reference (higher = better)", "tab:purple"),
        ("wrong_citation_rate", "Wrong-citation rate (LOWER = better)", "indianred"),
    ]
    for i, (key, title, color) in enumerate(specs):
        ax = fig.add_subplot(3, 2, i + 1)
        vals = [ckpts[n]["overall"][key] for n in names]
        bars = ax.bar(names, vals, color=color)
        ax.bar_label(bars, fmt="%.3f", fontsize=7)
        ax.set_title(title, fontsize=9)
        ax.set_ylim(0, max(vals + [0.01]) * 1.25)
        ax.tick_params(labelsize=8)

    ax = fig.add_subplot(3, 1, 3)
    ax.set_axis_off()
    base = ckpts.get("base", {}).get("overall")
    rows = [["checkpoint", "n", "cite_acc", "wrong_cite", "token_f1", "rouge_l"]]
    for n in names:
        o = ckpts[n]["overall"]

        def cell(key, lower_better=False):
            val = f"{o[key]:.4f}"
            if base and n != "base":
                delta = o[key] - base[key]
                worse = delta > 0.005 if lower_better else delta < -0.005
                val += f" ({delta:+.3f}{' !' if worse else ''})"
            return val

        rows.append([n, str(o["n"]), cell("citation_accuracy"),
                     cell("wrong_citation_rate", True), cell("token_f1"), cell("rouge_l")])
    table = ax.table(cellText=rows[1:], colLabels=rows[0], loc="upper center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    table.scale(1, 1.4)
    ax.set_title("Deltas vs base in parentheses; '!' flags a regression (reported, not hidden)",
                 fontsize=8, color="dimgray")
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    pdf.savefig(fig)
    plt.close(fig)

    # per-question-type breakdown
    fig = _new_page("Metric breakdown by question type", "token_f1 per checkpoint")
    lines = []
    types = sorted({t for c in ckpts.values() for t in c["per_type"]})
    header = f"  {'type':<18}" + "".join(f"{n:>12}" for n in names)
    lines += [(header, "mono"), (f"  {'-' * (18 + 12 * len(names))}", "mono")]
    for t in types:
        row = f"  {t:<18}" + "".join(
            f"{ckpts[n]['per_type'].get(t, {}).get('token_f1', float('nan')):>12.4f}" for n in names
        )
        lines.append((row, "mono"))
    lines += [
        ("", "b"),
        ("Citation accuracy by type", "h"),
        (header, "mono"),
        (f"  {'-' * (18 + 12 * len(names))}", "mono"),
    ]
    for t in types:
        row = f"  {t:<18}" + "".join(
            f"{ckpts[n]['per_type'].get(t, {}).get('citation_accuracy', float('nan')):>12.4f}"
            for n in names
        )
        lines.append((row, "mono"))
    lines += [
        ("", "b"),
        ("How to read this", "h"),
        ("  overview / provisions test grounded summarization of a named section;", "b"),
        ("  citation_lookup tests recalling the right § from a heading alone (hardest);", "b"),
        ("  definition tests retrieving regulatory definitions verbatim.", "b"),
    ]
    _render_lines(fig, lines)
    pdf.savefig(fig)
    plt.close(fig)


def _pick_examples(gens: dict, n: int) -> list:
    keyed = {name: {r["id"]: r for r in recs} for name, recs in gens.items()}
    names = list(gens)
    ids = [r["id"] for r in gens[names[0]]]
    tuned = [m for m in names if m != "base"]

    def hit(name, rid):
        r = keyed[name][rid]
        return metrics.citation_correct(r["completion"], r["expected_citation"])

    interesting, rest = [], []
    for rid in ids:
        if any(rid not in keyed[m] for m in names):
            continue
        if "base" in keyed and tuned and not hit("base", rid) and any(hit(m, rid) for m in tuned):
            interesting.append(rid)  # base misses the citation, a tuned model fixes it
        else:
            rest.append(rid)
    picked = (interesting + rest)[:n]
    return [{name: keyed[name][rid] for name in names} for rid in picked]


def _example_pages(pdf, gens: dict, n_examples: int = 4):
    import matplotlib.pyplot as plt

    if not gens:
        return
    examples = _pick_examples(gens, n_examples)
    for i, ex in enumerate(examples, 1):
        any_rec = next(iter(ex.values()))
        fig = _new_page(f"Actual generations — example {i}/{len(examples)}",
                        f"{any_rec['type']} question on 12 CFR § {any_rec['section']} (held-out)")
        lines = [("Prompt", "h")]
        lines += [(l, "b") for l in _wrap(any_rec["prompt_messages"][-1]["content"], 100)[:4]]
        lines += [("", "b"), (f"Reference (ground truth, expects § {any_rec['expected_citation']})", "h")]
        lines += [(l, "mono") for l in _wrap(any_rec["reference"][:420], 100)[:6]]
        lines.append(("", "b"))
        for name, rec in ex.items():
            ok = metrics.citation_correct(rec["completion"], rec["expected_citation"])
            f1 = metrics.token_f1(rec["completion"], rec["reference"])
            label = "base (no fine-tuning)" if name == "base" else name.upper()
            lines.append((f"{label}   citation {CHECK if ok else CROSS}   token_f1 {f1:.3f}", "h"))
            body = rec["completion"].strip() or "(empty completion)"
            lines += [(l, "mono") for l in _wrap(body[:520], 100)[:7]]
            lines.append(("", "b"))
        _render_lines(fig, lines)
        pdf.savefig(fig)
        plt.close(fig)


def _howto_page(pdf, cfg):
    fig = _new_page("How to run / reproduce")
    lines = [
        ("Setup", "h"),
        ("  python3 -m venv .venv && source .venv/bin/activate", "mono"),
        ("  pip install -r requirements.txt", "mono"),
        ("  export HF_TOKEN=...        # gated meta-llama repo (fallback used otherwise)", "mono"),
        ("", "b"),
        ("Full pipeline (single GPU node, auto-detects device)", "h"),
        ("  ./run_all.sh               # download -> build -> sft -> dpo -> eval -> report", "mono"),
        ("  python training_models_v1.py grpo   # optional RL stage (~3-4x SFT cost)", "mono"),
        ("  python training_models_v1.py eval   # re-score; cached generations reused", "mono"),
        ("  python training_models_v1.py report # regenerate this PDF", "mono"),
        ("", "b"),
        ("Device selection", "h"),
        ("  --device auto   CUDA if visible, else Apple Silicon GPU (MPS, M1-M5), else CPU", "mono"),
        ("  --device cuda   bf16/fp16 + 4-bit QLoRA below the VRAM threshold", "mono"),
        ("  --device mps    un-quantized LoRA on Apple Silicon", "mono"),
        ("  --device cpu    float32; pair with --smoke", "mono"),
        ("", "b"),
        ("Cheap end-to-end validation before renting a GPU", "h"),
        ("  python training_models_v1.py all --smoke   # tiny model, ~5 steps per stage", "mono"),
        ("", "b"),
        ("Artifacts", "h"),
        ("  data/                dataset JSONL + dataset_card.md (counts, hashes, filters)", "mono"),
        ("  outputs/*-adapter/   LoRA adapters + training_log.json per stage", "mono"),
        ("  outputs/manifests/   per-stage manifests: config hash, data hashes, runtime", "mono"),
        ("  results/             eval_results.json, report.md, generations_*.jsonl,", "mono"),
        ("                       model_report.pdf (this file)", "mono"),
        ("  training_models.ipynb  interactive charts + side-by-side browsing", "mono"),
        ("", "b"),
        ("Reproducibility", "h"),
        (f"  Global seed {cfg['seed']}; greedy decoding at eval; eCFR snapshot pinned to "
         f"{cfg['ecfr']['date']};", "b"),
        ("  splits derived from stable hashes of section numbers (not RNG order); every stage", "b"),
        ("  writes a manifest with the config hash and SHA-256 of its data inputs.", "b"),
        ("", "b"),
        ("Honest-reporting policy", "h"),
        ("  Every metric regression vs the base model is flagged (never hidden) here and in", "b"),
        ("  results/report.md. Known trade-offs to watch on full runs: DPO typically cuts", "b"),
        ("  wrong-citation rate but can cost a little token_f1; GRPO gains depend on reward", "b"),
        ("  shaping — the brevity/repetition guard exists precisely to stop reward hacking.", "b"),
    ]
    _render_lines(fig, lines)
    pdf.savefig(fig)
    import matplotlib.pyplot as plt

    plt.close(fig)


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def run(cfg: dict, smoke: bool = False, model_override: str | None = None) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.backends.backend_pdf import PdfPages

    results_dir = Path(cfg["paths"]["results_dir"])
    out_root = Path(cfg["paths"]["outputs_dir"])
    data_dir = Path(cfg["paths"]["data_dir"])
    res = json.loads(
        common.require_file(results_dir / "eval_results.json", "run the `eval` stage first").read_text()
    )
    gens = {
        name: common.load_jsonl(results_dir / f"generations_{name}.jsonl")
        for name in res["checkpoints"]
        if (results_dir / f"generations_{name}.jsonl").exists()
    }

    pdf_path = results_dir / "model_report.pdf"
    with PdfPages(pdf_path) as pdf:
        _title_page(pdf, cfg, res, out_root, data_dir)
        _hparams_page(pdf, cfg, out_root)
        _curves_page(pdf, out_root)
        _metrics_page(pdf, res)
        _example_pages(pdf, gens)
        _howto_page(pdf, cfg)
        info = pdf.infodict()
        info["Title"] = "eCFR Title 12 fine-tuning report"
        info["Subject"] = f"model={res['model_id']} smoke={res['smoke']}"

    print(f"[report] wrote {pdf_path}")
    common.write_manifest(
        cfg,
        "report",
        {"pdf": str(pdf_path), "model_id": res["model_id"], "smoke": res["smoke"],
         "checkpoints": list(res["checkpoints"])},
    )
    return pdf_path
