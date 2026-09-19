"""Stage 7: stakeholder PDF report -> results/model_report.pdf (+ results/examples_all.md).

One self-contained file for anyone evaluating the trained models: corpus and
dataset coverage across every 12 CFR chapter, hyperparameters, every metric the
trainers logged (loss, learning rate, grad norm, validation loss, DPO reward
margins/accuracies, GRPO rewards, ...), held-out metrics per checkpoint / question
type / chapter, win-loss analysis vs the base model, and every single held-out
example with the answer from every checkpoint. Needs the `eval` stage outputs.
"""
from __future__ import annotations

import json
import math
import textwrap
import time
from collections import Counter, defaultdict
from pathlib import Path

from . import common, metrics

PAGE = (8.27, 11.69)  # A4 portrait
CHECK, CROSS = "\u2713", "\u2717"
STAGES = ("cpt", "sft", "dpo", "grpo")
STAGE_TITLES = {
    "base": "BASE (no fine-tuning)",
    "cpt": "CPT (continued pretraining on regulation text)",
    "sft": "SFT (supervised fine-tuning, on top of CPT)",
    "dpo": "DPO (preference tuning, on top of SFT)",
    "grpo": "GRPO (RL with verifiable rewards, on top of SFT)",
}
LINE_H = {"h": 0.029, "b": 0.0145, "mono": 0.0122, "warn": 0.016, "small": 0.011}


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
    """lines: (text, style) with style in {'h','b','mono','warn','small'}; returns final y."""
    for text, style in lines:
        if y < 0.045:
            fig.text(0.06, y, "\u2026", fontsize=9)
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
        elif style == "small":
            fig.text(0.06, y, text, fontsize=7, color="dimgray")
            y -= 0.011
        else:
            fig.text(0.06, y, text, fontsize=9)
            y -= 0.0145
    return y


def _flow_pages(pdf, title: str, subtitle: str, lines: list) -> int:
    """Render an arbitrarily long list of lines across as many pages as needed."""
    import matplotlib.pyplot as plt

    pages, i, part = 0, 0, 1
    while i < len(lines) or pages == 0:
        y, chunk = 0.915, []
        while i < len(lines) and y - LINE_H[lines[i][1]] >= 0.045:
            y -= LINE_H[lines[i][1]]
            chunk.append(lines[i])
            i += 1
        if i < len(lines) and not chunk:
            chunk.append(lines[i])
            i += 1
        fig = _new_page(title if part == 1 else f"{title} (cont. {part})", subtitle)
        _render_lines(fig, chunk)
        pdf.savefig(fig)
        plt.close(fig)
        pages += 1
        part += 1
    return pages


def _load_log(out_root: Path, stage: str):
    p = out_root / f"{stage}-adapter" / "training_log.json"
    return json.loads(p.read_text()) if p.exists() else None


def _load_manifest(out_root: Path, stage: str):
    p = out_root / "manifests" / f"{stage}_manifest.json"
    return json.loads(p.read_text()) if p.exists() else None


def _count_rows(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text().splitlines() if line.strip())


def _epochs_reached(log) -> float | None:
    epochs = [e["epoch"] for e in (log or []) if "epoch" in e]
    return max(epochs) if epochs else None


def _series(log, key):
    pts = [(e.get("step", i), e[key]) for i, e in enumerate(log or []) if key in e]
    return [p[0] for p in pts], [p[1] for p in pts]


_ROMAN = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100}


def _chapter_sort(ch: str):
    """Sort Roman-numeral chapters numerically (I, II, ..., X, XI, ...); unknown last."""
    if not ch or not all(c in _ROMAN for c in ch):
        return (1, 0, ch or "")
    total = 0
    for i, c in enumerate(ch):
        v = _ROMAN[c]
        total += -v if i + 1 < len(ch) and _ROMAN[ch[i + 1]] > v else v
    return (0, total, ch)


def _fmt(v, nd=3):
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.{nd}f}" if abs(v) < 1e5 else f"{v:.3g}"
    return str(v)


# ---------------------------------------------------------------------------
# pages
# ---------------------------------------------------------------------------


def _title_page(pdf, cfg, res, out_root, data_dir, sections):
    import matplotlib.pyplot as plt

    fig = _new_page("Fine-tuning Report: eCFR Title 12 Regulation Assistant",
                    "Base vs CPT vs SFT vs DPO vs GRPO on every chapter of 12 CFR (Banks and Banking)")
    e = cfg["ecfr"]
    scope = f"Chapter {e['chapter']}" if e.get("chapter") else "all chapters"
    chapters = Counter(s.get("chapter") or "?" for s in sections)
    stages_run = [st for st in STAGES if (out_root / f"{st}-adapter" / "adapter_config.json").exists()]
    lines = [
        (f"Generated: {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}", "b"),
        (f"Model evaluated: {res['model_id']}", "b"),
        (f"Compute: {res['runtime']['device']}"
         + (f" ({res['runtime']['vram_gb']} GB VRAM)" if res['runtime']['device'] == 'cuda' else "")
         + f", dtype {res['runtime']['torch_dtype']}, 4-bit QLoRA: {res['runtime']['use_4bit']}", "b"),
        (f"Seed: {cfg['seed']}   |   Config hash: {common.config_hash(cfg)}", "b"),
        (f"Stages trained: {', '.join(st.upper() for st in stages_run) or 'none'}   |   "
         f"Checkpoints evaluated: {', '.join(res['checkpoints'])}", "b"),
        ("", "b"),
    ]
    if res.get("smoke"):
        lines += [
            ("SMOKE RUN \u2014 numbers below come from a tiny validation model trained for a", "warn"),
            ("few steps. They prove the pipeline works, not model quality.", "warn"),
            ("", "b"),
        ]
    lines += [
        ("What was built", "h"),
        (f"LoRA adapters for {res['model_id'].split('/')[-1]} specialised in U.S. banking regulation,", "b"),
        (f"trained on a point-in-time snapshot ({e['date']}) of eCFR Title {e['title']}, {scope}", "b"),
        (f"({len(sections):,} sections across {len([c for c in chapters if c != '?'])} chapters). "
         "Four techniques are compared against the untuned base:", "b"),
        ("", "b"),
        ("  1. CPT  \u2014 continued pretraining: causal-LM on the raw regulation text (train+val", "b"),
        ("            sections only) so the model absorbs the domain before instruction tuning", "b"),
        ("  2. SFT  \u2014 supervised fine-tuning on grounded Q&A built from the regulations,", "b"),
        ("            continuing the CPT adapter", "b"),
        ("  3. DPO  \u2014 preference tuning on top of SFT: correct+cited answers preferred over", "b"),
        ("            wrong sections, wrong citations, or vague non-answers", "b"),
        ("  4. GRPO \u2014 RL with verifiable rewards on top of SFT (citation correctness,", "b"),
        ("            grounding F1, brevity/anti-repetition guard); no reward model", "b"),
        ("", "b"),
        ("Pipeline:  download -> build -> CPT -> SFT -> DPO -> GRPO -> eval -> report -> publish", "mono"),
        ("", "b"),
        ("Dataset", "h"),
    ]
    counts = {
        "CPT train docs": _count_rows(data_dir / "cpt_train.jsonl"),
        "CPT val docs": _count_rows(data_dir / "cpt_val.jsonl"),
        "SFT train": _count_rows(data_dir / "sft_train.jsonl"),
        "SFT val": _count_rows(data_dir / "sft_val.jsonl"),
        "DPO pairs": _count_rows(data_dir / "dpo_train.jsonl"),
        "Held-out eval": _count_rows(data_dir / "eval_test.jsonl"),
    }
    lines += [(f"  {name:<16} {n:>7} rows", "mono") for name, n in counts.items()]
    lines += [
        ("", "b"),
        ("Split is by section via a stable hash \u2014 no section text or prompt can leak from", "b"),
        ("train into eval (asserted at build time); CPT also sees train+val sections only.", "b"),
        ("Eval examples are sampled round-robin across chapters so every agency's rules are", "b"),
        ("tested, not just Chapter I (OCC). Question types: overview, provisions,", "b"),
        ("citation_lookup (unique headings only), definition. Details: data/dataset_card.md.", "b"),
        ("", "b"),
        ("Evaluation protocol", "h"),
        (f"Greedy decoding, fixed seed, {res['n_eval']} held-out examples the model never saw in", "b"),
        ("training. Metrics: citation_accuracy (predicted \u00a7 matches the target section),", "b"),
        ("wrong_citation_rate (confidently cites the wrong section \u2014 lower is better),", "b"),
        ("token_f1 and ROUGE-L vs a grounded reference answer, avg answer length in words.", "b"),
        ("Every held-out example and every checkpoint's answer is reproduced in the appendix.", "b"),
    ]
    _render_lines(fig, lines)
    pdf.savefig(fig)
    plt.close(fig)


def _corpus_page(pdf, res, data_dir, sections):
    import matplotlib.pyplot as plt

    names = {}
    per_ch = defaultdict(lambda: {"sections": 0, "words": 0, "sft": 0, "eval": 0})
    for s in sections:
        ch = s.get("chapter") or "?"
        names.setdefault(ch, s.get("chapter_name") or "")
        per_ch[ch]["sections"] += 1
        per_ch[ch]["words"] += s.get("word_count", 0)
    for r in common.load_jsonl(data_dir / "sft_train.jsonl") if (data_dir / "sft_train.jsonl").exists() else []:
        per_ch[str(r.get("chapter") or "?")]["sft"] += 1
    for r in common.load_jsonl(data_dir / "eval_test.jsonl") if (data_dir / "eval_test.jsonl").exists() else []:
        per_ch[str(r.get("chapter") or "?")]["eval"] += 1
    chapters = sorted(per_ch, key=_chapter_sort)

    fig = plt.figure(figsize=PAGE)
    fig.suptitle("Corpus and dataset coverage by 12 CFR chapter", fontsize=15, fontweight="bold")
    ax = fig.add_subplot(3, 1, 1)
    ax.bar(chapters, [per_ch[c]["sections"] for c in chapters], color="tab:blue")
    ax.set_title("Sections in the snapshot per chapter", fontsize=9)
    ax.tick_params(labelsize=7)
    ax = fig.add_subplot(3, 1, 2)
    ax.bar(chapters, [per_ch[c]["eval"] for c in chapters], color="tab:orange")
    ax.set_title("Held-out eval examples per chapter (round-robin sampled)", fontsize=9)
    ax.tick_params(labelsize=7)
    ax = fig.add_subplot(3, 1, 3)
    ax.set_axis_off()
    rows = [[c, (names.get(c) or "")[:44], f"{per_ch[c]['sections']:,}", f"{per_ch[c]['words']:,}",
             f"{per_ch[c]['sft']:,}", str(per_ch[c]["eval"])] for c in chapters]
    rows.append(["all", "", f"{sum(v['sections'] for v in per_ch.values()):,}",
                 f"{sum(v['words'] for v in per_ch.values()):,}",
                 f"{sum(v['sft'] for v in per_ch.values()):,}", str(sum(v["eval"] for v in per_ch.values()))])
    table = ax.table(cellText=rows, colLabels=["ch.", "agency", "sections", "words", "sft train", "eval"],
                     loc="upper center", cellLoc="left", colWidths=[0.07, 0.5, 0.11, 0.13, 0.11, 0.08])
    table.auto_set_font_size(False)
    table.set_fontsize(6.5)
    table.scale(1, 1.05)
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    pdf.savefig(fig)
    plt.close(fig)


def _hparams_page(pdf, cfg, out_root):
    import matplotlib.pyplot as plt

    fig = _new_page("Training configuration", "All knobs live in config.yaml")
    lora, c, s, d, g = cfg["lora"], cfg.get("cpt", {}), cfg["sft"], cfg["dpo"], cfg["grpo"]
    logs = {st: _load_log(out_root, st) for st in STAGES}

    def reached(st):
        ep = _epochs_reached(logs[st])
        return f"{ep:.2f}" if ep is not None else "not run"

    def row(label, *vals):
        return (f"  {label:<22}" + "".join(f"{str(v):>13}" for v in vals), "mono")

    lines = [
        ("LoRA adapter (one adapter, carried through CPT -> SFT -> DPO/GRPO)", "h"),
        (f"  rank r={lora['r']}  alpha={lora['alpha']}  dropout={lora['dropout']}", "mono"),
        (f"  target modules: {', '.join(lora['target_modules'])}", "mono"),
        ("", "b"),
        ("Per-stage hyperparameters", "h"),
        row("", "CPT", "SFT", "DPO", "GRPO"),
        (f"  {'-' * 74}", "mono"),
        row("epochs (configured)", c.get("epochs", "-"), s["epochs"], d["epochs"], f"{g['max_steps']} steps"),
        row("epochs (reached)", reached("cpt"), reached("sft"), reached("dpo"), reached("grpo")),
        row("learning rate", c.get("learning_rate", "-"), s["learning_rate"], d["learning_rate"], g["learning_rate"]),
        row("per-device batch", c.get("per_device_batch", "-"), s["per_device_batch"], d["per_device_batch"], g["per_device_batch"]),
        row("grad accumulation", c.get("grad_accum", "-"), s["grad_accum"], d["grad_accum"], g["grad_accum"]),
        row("effective batch", c.get("per_device_batch", 0) * c.get("grad_accum", 0) or "-",
            s["per_device_batch"] * s["grad_accum"], d["per_device_batch"] * d["grad_accum"],
            g["per_device_batch"] * g["grad_accum"]),
        row("max seq length", c.get("max_length", "-"), s["max_length"], d["max_length"], g["max_prompt_length"]),
        row("warmup ratio", c.get("warmup_ratio", "-"), s["warmup_ratio"], d["warmup_ratio"], 0.03),
        row("init from", "base", "cpt", "sft", "sft"),
        ("", "b"),
        ("  CPT: packed causal-LM on 'citation heading (chapter) / text' documents; val perplexity logged", "mono"),
        ("  DPO: beta = " + str(d["beta"]) + " (reference = frozen SFT adapter, no second model in memory)", "mono"),
        (f"  GRPO: {g['num_generations']} generations/prompt, temperature {g['temperature']}, "
         f"beta {g['beta']} (no KL ref), max {g['max_completion_length']} new tokens", "mono"),
        ("", "b"),
        ("Stage chaining", "h"),
        ("  CPT trains a fresh LoRA on the base model from raw regulation text. SFT continues that", "b"),
        ("  same adapter on Q&A. DPO and GRPO both start from the SFT adapter, so their comparison", "b"),
        ("  isolates what each preference/RL step adds on top of the same supervised model.", "b"),
        ("", "b"),
        ("GRPO reward design (verifiable, no reward model)", "h"),
        ("  citation_reward   +1.0 correct \u00a7 cited (capped at 0.25 if >3 sections spammed),", "mono"),
        ("                    -0.5 confidently wrong citation, 0.0 if no citation", "mono"),
        ("  grounding_reward  token-F1 overlap with the grounded reference answer", "mono"),
        ("  brevity_reward    -0.3 if >320 words; extra -0.5 for degenerate repetition", "mono"),
        ("", "b"),
        ("Precision / hardware policy", "h"),
        ("  cuda: bf16 (or fp16), 4-bit QLoRA when VRAM < "
         + str(cfg["model"]["qlora_below_gb"]) + " GB, paged 8-bit AdamW", "mono"),
        ("  mps (Apple Silicon): un-quantized LoRA, bf16 when supported, eager attention", "mono"),
        ("  cpu: float32 \u2014 intended for --smoke validation only", "mono"),
    ]
    _render_lines(fig, lines)
    pdf.savefig(fig)
    plt.close(fig)


def _overview_curves_page(pdf, out_root):
    import matplotlib.pyplot as plt

    logs = {s: _load_log(out_root, s) for s in STAGES}
    fig, axes = plt.subplots(2, 2, figsize=PAGE)
    fig.suptitle("Training overview: loss per stage", fontsize=15, fontweight="bold")

    def plot_loss(ax, stage, title):
        log = logs[stage]
        if not log:
            ax.text(0.5, 0.5, f"{stage} not run", ha="center", va="center", color="gray")
            ax.set_axis_off()
            return
        steps, loss = _series(log, "loss")
        if steps:
            ax.plot(steps, loss, "-", label="train loss", color="tab:blue", lw=1)
        esteps, eloss = _series(log, "eval_loss")
        if esteps:
            ax.plot(esteps, eloss, "o-", label="validation loss", color="tab:orange", ms=3)
        if not steps and not esteps:
            _, tl = _series(log, "train_loss")
            if tl:
                ax.axhline(tl[-1], ls="--", color="tab:blue", label=f"final train loss {tl[-1]:.3f}")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("optimizer step", fontsize=8)
        ax.legend(fontsize=7)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3)

    plot_loss(axes[0][0], "cpt", "CPT causal-LM loss (val perplexity = exp(loss))")
    plot_loss(axes[0][1], "sft", "SFT loss")
    plot_loss(axes[1][0], "dpo", "DPO loss")
    ax = axes[1][1]
    if logs["grpo"]:
        steps, reward = _series(logs["grpo"], "reward")
        if steps:
            ax.plot(steps, reward, "-", color="tab:green", label="mean total reward")
            _, std = _series(logs["grpo"], "reward_std")
            if len(std) == len(steps):
                ax.fill_between(steps, [r - s for r, s in zip(reward, std)],
                                [r + s for r, s in zip(reward, std)], alpha=0.2, color="tab:green")
        ax.set_title("GRPO total reward (\u00b11 std)", fontsize=10)
        ax.set_xlabel("optimizer step", fontsize=8)
        ax.legend(fontsize=7)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3)
    else:
        ax.text(0.5, 0.5, "grpo not run", ha="center", va="center", color="gray")
        ax.set_axis_off()
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    pdf.savefig(fig)
    plt.close(fig)


def _stage_detail_pages(pdf, stage, log, manifest):
    """Summary table + a subplot for every numeric quantity the trainer logged."""
    import matplotlib.pyplot as plt

    if not log:
        return
    final = [e for e in log if "train_runtime" in e]
    final = final[-1] if final else {}
    summary = [
        ("optimizer steps", max((e.get("step", 0) for e in log), default=0)),
        ("epochs reached", _fmt(_epochs_reached(log), 2)),
        ("final train loss", _fmt(final.get("train_loss"), 4)),
        ("last logged loss", _fmt(next((e["loss"] for e in reversed(log) if "loss" in e), None), 4)),
        ("last validation loss", _fmt(next((e["eval_loss"] for e in reversed(log) if "eval_loss" in e), None), 4)),
        ("train runtime", f"{final.get('train_runtime', 0) / 3600:.2f} h" if final else "-"),
        ("samples / s", _fmt(final.get("train_samples_per_second"), 2)),
        ("steps / s", _fmt(final.get("train_steps_per_second"), 3)),
        ("total FLOPs", f"{final.get('total_flos', 0):.3g}" if final else "-"),
    ]
    if stage == "cpt":
        el = next((e["eval_loss"] for e in reversed(log) if "eval_loss" in e), None)
        summary.append(("val perplexity", _fmt(math.exp(el), 2) if el is not None else "-"))
    if manifest:
        for k in ("n_train", "n_documents", "n_words", "n_pairs", "n_prompts", "n_val", "init_from"):
            if manifest.get(k) is not None:
                summary.append((k, f"{manifest[k]:,}" if isinstance(manifest[k], int) else str(manifest[k])))
        summary.append(("finished (UTC)", manifest.get("timestamp_utc", "-")))
        v = manifest.get("versions", {})
        summary.append(("versions", " ".join(f"{k}={v[k]}" for k in ("torch", "transformers", "trl", "peft") if k in v)))

    keys = sorted({k for e in log for k, v in e.items()
                   if isinstance(v, (int, float)) and not isinstance(v, bool)
                   and k not in {"step", "epoch"} and not k.startswith("train_") and k != "total_flos"})
    keys = [k for k in keys if len(_series(log, k)[0]) >= 2]
    keys.sort(key=lambda k: (not k.endswith("loss"), k))  # losses first

    per_page = 12
    for page_no, start in enumerate(range(0, max(len(keys), 1), per_page)):
        chunk = keys[start:start + per_page]
        fig = plt.figure(figsize=PAGE)
        fig.suptitle(f"{STAGE_TITLES.get(stage, stage.upper())} \u2014 training log"
                     + (f" ({page_no + 1})" if len(keys) > per_page else ""), fontsize=13, fontweight="bold")
        if page_no == 0:
            ax = fig.add_subplot(5, 1, 1)
            ax.set_axis_off()
            rows = [[k, str(v)] for k, v in summary]
            half = (len(rows) + 1) // 2
            cells = [rows[i] + (rows[i + half] if i + half < len(rows) else ["", ""]) for i in range(half)]
            table = ax.table(cellText=cells, loc="center", cellLoc="left", colWidths=[0.18, 0.32, 0.18, 0.32])
            table.auto_set_font_size(False)
            table.set_fontsize(6.5)
            table.scale(1, 1.1)
            grid_rows, offset = 4, 1
        else:
            grid_rows, offset = 5, 0
        for i, key in enumerate(chunk):
            ax = fig.add_subplot(grid_rows + offset, 3, i + 1 + offset * 3)
            steps, vals = _series(log, key)
            color = "tab:orange" if key.startswith("eval") else "tab:blue"
            ax.plot(steps, vals, "-", lw=0.9, color=color, marker="o" if len(steps) < 40 else None, ms=2.5)
            ax.set_title(key, fontsize=7.5)
            ax.tick_params(labelsize=6)
            ax.grid(alpha=0.3)
            if key == "learning_rate":
                ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
                ax.yaxis.get_offset_text().set_fontsize(6)
        if not chunk:
            fig.text(0.5, 0.4, "only the final summary was logged (smoke run)", ha="center", color="gray")
        fig.tight_layout(rect=(0, 0.02, 1, 0.95))
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
        ("wrong_citation_rate", "Wrong-citation rate (LOWER = better)", "indianred"),
        ("token_f1", "Token F1 vs reference (higher = better)", "tab:green"),
        ("rouge_l", "ROUGE-L vs reference (higher = better)", "tab:purple"),
        ("avg_words", "Average answer length (words)", "tab:gray"),
    ]
    for i, (key, title, color) in enumerate(specs):
        ax = fig.add_subplot(4, 2, i + 1)
        vals = [ckpts[n]["overall"].get(key, 0) for n in names]
        bars = ax.bar(names, vals, color=color)
        ax.bar_label(bars, fmt="%.3f" if key != "avg_words" else "%.0f", fontsize=7)
        ax.set_title(title, fontsize=9)
        ax.set_ylim(0, max(vals + [0.01]) * 1.25)
        ax.tick_params(labelsize=8)

    ax = fig.add_subplot(4, 1, 4)
    ax.set_axis_off()
    base = ckpts.get("base", {}).get("overall")
    rows = [["checkpoint", "n", "cite_acc", "wrong_cite", "token_f1", "rouge_l", "words"]]
    for n in names:
        o = ckpts[n]["overall"]

        def cell(key, lower_better=False):
            val = f"{o[key]:.4f}"
            if base and n != "base":
                delta = o[key] - base[key]
                worse = delta > 0.005 if lower_better else delta < -0.005
                val += f" ({delta:+.3f}{' !' if worse else ''})"
            return val

        rows.append([n, str(o["n"]), cell("citation_accuracy"), cell("wrong_citation_rate", True),
                     cell("token_f1"), cell("rouge_l"), f"{o.get('avg_words', 0):.0f}"])
    table = ax.table(cellText=rows[1:], colLabels=rows[0], loc="upper center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1, 1.4)
    ax.set_title("Deltas vs base in parentheses; '!' flags a regression (reported, not hidden)",
                 fontsize=8, color="dimgray")
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    pdf.savefig(fig)
    plt.close(fig)

    lines = []
    types = sorted({t for c in ckpts.values() for t in c["per_type"]})
    header = f"  {'type':<18}{'n':>5}" + "".join(f"{n:>11}" for n in names)
    for metric in ("token_f1", "citation_accuracy", "wrong_citation_rate", "rouge_l"):
        lines += [(f"{metric} by question type", "h"), (header, "mono"), (f"  {'-' * (23 + 11 * len(names))}", "mono")]
        for t in types:
            n_t = ckpts[names[0]]["per_type"].get(t, {}).get("n", 0)
            row = f"  {t:<18}{n_t:>5}" + "".join(
                f"{ckpts[n]['per_type'].get(t, {}).get(metric, float('nan')):>11.4f}" for n in names)
            lines.append((row, "mono"))
        lines.append(("", "b"))
    lines += [
        ("How to read this", "h"),
        ("  overview / provisions test grounded summarization of a named section;", "b"),
        ("  citation_lookup tests recalling the right \u00a7 from a heading alone (hardest);", "b"),
        ("  definition tests retrieving regulatory definitions verbatim.", "b"),
    ]
    _flow_pages(pdf, "Metric breakdown by question type", "", lines)


def _chapter_pages(pdf, res):
    import matplotlib.pyplot as plt

    ckpts = res["checkpoints"]
    names = list(ckpts)
    chapters = sorted({c for ck in ckpts.values() for c in ck.get("per_chapter", {})}, key=_chapter_sort)
    if not chapters:
        return
    agency = res.get("chapters", {})

    fig = plt.figure(figsize=PAGE)
    fig.suptitle("Held-out metrics by 12 CFR chapter", fontsize=15, fontweight="bold")
    width = 0.8 / max(len(names), 1)
    for i, (metric, title) in enumerate((("citation_accuracy", "Citation accuracy by chapter"),
                                         ("token_f1", "Token F1 by chapter"),
                                         ("wrong_citation_rate", "Wrong-citation rate by chapter (lower = better)"))):
        ax = fig.add_subplot(3, 1, i + 1)
        for j, n in enumerate(names):
            vals = [ckpts[n]["per_chapter"].get(c, {}).get(metric, 0) for c in chapters]
            ax.bar([k + j * width for k in range(len(chapters))], vals, width, label=n)
        ax.set_xticks([k + width * (len(names) - 1) / 2 for k in range(len(chapters))])
        ax.set_xticklabels(chapters, fontsize=7)
        ax.set_title(title, fontsize=9)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6, ncol=len(names))
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    pdf.savefig(fig)
    plt.close(fig)

    lines = [("Chapters", "h")]
    lines += [(f"  {c:<6} {(agency.get(c) or '')[:80]}", "mono") for c in chapters]
    lines.append(("", "b"))
    header = f"  {'chapter':<9}{'n':>4}" + "".join(f"{n:>11}" for n in names)
    for metric in ("citation_accuracy", "token_f1", "rouge_l", "wrong_citation_rate", "avg_words"):
        lines += [(f"{metric} by chapter", "h"), (header, "mono"), (f"  {'-' * (13 + 11 * len(names))}", "mono")]
        for c in chapters:
            n_c = ckpts[names[0]]["per_chapter"].get(c, {}).get("n", 0)
            row = f"  {c:<9}{n_c:>4}" + "".join(
                f"{ckpts[n]['per_chapter'].get(c, {}).get(metric, float('nan')):>11.3f}" for n in names)
            lines.append((row, "mono"))
        lines.append(("", "b"))
    lines.append(("  Small chapters have few held-out examples; read their rows as indicative, not precise.", "small"))
    _flow_pages(pdf, "Metric tables by chapter", "n = held-out examples from that chapter", lines)


def _per_example_scores(gens: dict) -> dict:
    """{id: {ckpt: {...}}} for ids present in every checkpoint."""
    keyed = {name: {r["id"]: r for r in recs} for name, recs in gens.items()}
    common_ids = set.intersection(*(set(k) for k in keyed.values())) if keyed else set()
    scores = {}
    for rid in common_ids:
        scores[rid] = {}
        for name in gens:
            r = keyed[name][rid]
            hit = metrics.citation_correct(r["completion"], r["expected_citation"])
            scores[rid][name] = {
                "cite": hit,
                "wrong": bool(metrics.extract_sections(r["completion"])) and not hit,
                "f1": metrics.token_f1(r["completion"], r["reference"]),
                "rouge": metrics.rouge_l(r["completion"], r["reference"]),
                "words": len(r["completion"].split()),
            }
    return scores


def _winloss_page(pdf, gens: dict, scores: dict):
    import matplotlib.pyplot as plt

    names = list(gens)
    if "base" not in names or len(names) < 2 or not scores:
        return
    tuned = [n for n in names if n != "base"]
    fig = plt.figure(figsize=PAGE)
    fig.suptitle("Per-example win/loss analysis vs the base model", fontsize=15, fontweight="bold")

    ax = fig.add_subplot(3, 1, 1)
    cats = ["citation fixed", "citation broken", "F1 up (>0.05)", "F1 down (>0.05)"]
    width = 0.8 / len(tuned)
    for j, n in enumerate(tuned):
        fixed = sum(1 for s in scores.values() if not s["base"]["cite"] and s[n]["cite"])
        broken = sum(1 for s in scores.values() if s["base"]["cite"] and not s[n]["cite"])
        up = sum(1 for s in scores.values() if s[n]["f1"] - s["base"]["f1"] > 0.05)
        down = sum(1 for s in scores.values() if s["base"]["f1"] - s[n]["f1"] > 0.05)
        bars = ax.bar([k + j * width for k in range(4)], [fixed, broken, up, down], width, label=n)
        ax.bar_label(bars, fontsize=6)
    ax.set_xticks([k + width * (len(tuned) - 1) / 2 for k in range(4)])
    ax.set_xticklabels(cats, fontsize=8)
    ax.set_title(f"Counts over {len(scores)} held-out examples", fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(axis="y", alpha=0.3)

    ax = fig.add_subplot(3, 1, 2)
    bins = [i / 20 for i in range(21)]
    for n in names:
        ax.hist([s[n]["f1"] for s in scores.values()], bins=bins, alpha=0.45, label=n)
    ax.set_title("Distribution of per-example token F1", fontsize=9)
    ax.set_xlabel("token F1", fontsize=8)
    ax.legend(fontsize=7)
    ax.tick_params(labelsize=7)

    ax = fig.add_subplot(3, 1, 3)
    for n in names:
        ax.hist([s[n]["words"] for s in scores.values()], bins=25, alpha=0.45, label=n)
    ax.set_title("Distribution of answer length (words)", fontsize=9)
    ax.set_xlabel("words in completion", fontsize=8)
    ax.legend(fontsize=7)
    ax.tick_params(labelsize=7)
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    pdf.savefig(fig)
    plt.close(fig)


def _example_lines(ex: dict, names: list, idx: int, total: int, full: bool) -> list:
    any_rec = ex[names[0]]
    chapter = ""
    if any_rec.get("chapter"):
        chapter = f"Chapter {any_rec['chapter']}" + (f" ({any_rec['chapter_name']})" if any_rec.get("chapter_name") else "")
    lines = [(f"Example {idx}/{total}  \u2014  {any_rec['type']} question on 12 CFR \u00a7 {any_rec['section']}"
              + (f"  \u2014  {chapter}" if chapter else "") + f"  [{any_rec['id']}]", "h")]
    lines += [("Prompt", "b")]
    lines += [(l, "mono") for l in _wrap(any_rec["prompt_messages"][-1]["content"], 105)]
    lines += [("", "small"), (f"Reference answer (ground truth, expects \u00a7 {any_rec['expected_citation']})", "b")]
    ref = any_rec["reference"] if full else any_rec["reference"][:420]
    lines += [(l, "mono") for l in _wrap(ref, 105)]
    lines.append(("", "small"))
    for name in names:
        rec = ex[name]
        ok = metrics.citation_correct(rec["completion"], rec["expected_citation"])
        found = metrics.extract_sections(rec["completion"])
        f1 = metrics.token_f1(rec["completion"], rec["reference"])
        rl = metrics.rouge_l(rec["completion"], rec["reference"])
        cited = (", ".join(sorted(found)[:4]) + (" \u2026" if len(found) > 4 else "")) if found else "none"
        lines.append((f"{STAGE_TITLES.get(name, name.upper())}   citation {CHECK if ok else CROSS}"
                      f"   token_f1 {f1:.3f}   rouge_l {rl:.3f}   {len(rec['completion'].split())} words   cited: \u00a7 {cited}", "b"))
        body = rec["completion"].strip() or "(empty completion)"
        lines += [(l, "mono") for l in _wrap(body if full else body[:520], 105)]
        lines.append(("", "small"))
    return lines


def _pick_highlights(gens: dict, scores: dict, n: int) -> list:
    names = list(gens)
    ids = [r["id"] for r in gens[names[0]] if r["id"] in scores]
    tuned = [m for m in names if m != "base"]
    if "base" not in names or not tuned:
        return ids[:n]
    fixed = [i for i in ids if not scores[i]["base"]["cite"] and any(scores[i][m]["cite"] for m in tuned)]
    broken = [i for i in ids if scores[i]["base"]["cite"] and any(not scores[i][m]["cite"] for m in tuned)]
    rest = [i for i in ids if i not in set(fixed) | set(broken)]
    half = max(n // 2, 1)
    return (fixed[:half] + broken[:max(n - half - 2, 1)] + rest)[:n]


def _example_pages(pdf, gens: dict, scores: dict, n_highlights: int = 6):
    if not gens:
        return
    names = list(gens)
    keyed = {name: {r["id"]: r for r in recs} for name, recs in gens.items()}
    ordered_ids = [r["id"] for r in gens[names[0]] if r["id"] in scores]

    highlights = _pick_highlights(gens, scores, n_highlights)
    lines = [("Selected cases where the fine-tuned models fix (or break) the base model's citation.", "b"),
             ("The appendix that follows contains every held-out example.", "b"), ("", "b")]
    for i, rid in enumerate(highlights, 1):
        lines += _example_lines({n: keyed[n][rid] for n in names}, names, i, len(highlights), full=True)
        lines.append(("", "b"))
    _flow_pages(pdf, "Highlighted generations", "held-out examples, greedy decoding", lines)

    lines = [(f"All {len(ordered_ids)} held-out examples, in eval order, with the complete answer from every checkpoint.", "b"),
             ("Per-example metrics: citation correct (\u2713/\u2717), token F1 and ROUGE-L vs the reference, length, sections cited.", "b"),
             ("", "b")]
    for i, rid in enumerate(ordered_ids, 1):
        lines += _example_lines({n: keyed[n][rid] for n in names}, names, i, len(ordered_ids), full=True)
        lines.append(("", "b"))
    _flow_pages(pdf, "Appendix: every held-out example", "one block per example; blocks continue across pages", lines)


def _write_examples_md(path: Path, gens: dict, scores: dict, res: dict) -> None:
    names = list(gens)
    if not names:
        return
    keyed = {name: {r["id"]: r for r in recs} for name, recs in gens.items()}
    out = [f"# All held-out examples \u2014 {res['model_id']}", "",
           f"{len(scores)} examples, checkpoints: {', '.join(names)}. Generated {res['timestamp_utc']}.", ""]
    for i, r in enumerate(gens[names[0]], 1):
        rid = r["id"]
        if rid not in scores:
            continue
        ch = f" \u2014 Chapter {r.get('chapter')} ({r.get('chapter_name')})" if r.get("chapter") else ""
        out += [f"## {i}. `{rid}` \u2014 {r['type']} on 12 CFR \u00a7 {r['section']}{ch}", "",
                "**Prompt**", "", "> " + r["prompt_messages"][-1]["content"].replace("\n", "\n> "), "",
                f"**Reference** (expects \u00a7 {r['expected_citation']})", "", "> " + r["reference"].replace("\n", "\n> "), "",
                "| checkpoint | citation | token_f1 | rouge_l | words |", "|---|---|---|---|---|"]
        for n in names:
            s = scores[rid][n]
            out.append(f"| {n} | {'yes' if s['cite'] else ('WRONG' if s['wrong'] else 'none')} | {s['f1']:.3f} | {s['rouge']:.3f} | {s['words']} |")
        out.append("")
        for n in names:
            body = keyed[n][rid]["completion"].strip() or "(empty completion)"
            out += [f"**{n}**", "", "> " + body.replace("\n", "\n> "), ""]
    path.write_text("\n".join(out) + "\n")


def _publish_page(pdf, cfg, out_root):
    import matplotlib.pyplot as plt

    fig = _new_page("Published artifacts", "Hugging Face Hub receipts written by each training stage")
    lines = []
    for st in STAGES:
        receipt = out_root / f"{st}-adapter" / "hub_upload.json"
        completed = out_root / f"{st}-adapter" / "completed.json"
        if receipt.exists():
            r = json.loads(receipt.read_text())
            lines += [(f"{st.upper()} adapter", "h"), (f"  {r['url']}", "mono"),
                      (f"  revision {r['commit']}   fingerprint {r['fingerprint'][:16]}", "mono"), ("", "b")]
        elif completed.exists():
            lines += [(f"{st.upper()} adapter", "h"), ("  trained; not published (hub disabled or upload failed)", "mono"), ("", "b")]
    if not lines:
        lines = [("No adapters have been published.", "b")]
    lines += [("Loading an adapter", "h"),
              ("  from peft import PeftModel; from transformers import AutoModelForCausalLM", "mono"),
              (f"  base = AutoModelForCausalLM.from_pretrained('{cfg['model']['id']}')", "mono"),
              ("  model = PeftModel.from_pretrained(base, '<repo_id>', revision='<commit>')", "mono"),
              ("", "b"),
              ("  Each repo carries the adapter, tokenizer, README and provenance.json (data hashes,", "b"),
              ("  base revision, stage) so any result in this report can be traced to exact weights.", "b")]
    _render_lines(fig, lines)
    pdf.savefig(fig)
    plt.close(fig)


def _howto_page(pdf, cfg):
    import matplotlib.pyplot as plt

    fig = _new_page("How to run / reproduce")
    lines = [
        ("Setup", "h"),
        ("  python3 -m venv .venv && source .venv/bin/activate", "mono"),
        ("  pip install -r requirements.txt", "mono"),
        ("  echo HF_TOKEN=... >> .env       # gated meta-llama repo + Hub publishing", "mono"),
        ("  echo GITHUB_TOKEN=... >> .env   # final git push of data/results/report", "mono"),
        ("", "b"),
        ("Full pipeline (single GPU node, auto-detects device)", "h"),
        ("  ./run_all.sh   # download -> build -> cpt -> sft -> dpo -> grpo -> eval -> report -> push", "mono"),
        ("  python training_models_v1.py eval   # re-score; cached generations reused (--regen to redo)", "mono"),
        ("  python training_models_v1.py report # regenerate this PDF + results/examples_all.md", "mono"),
        ("", "b"),
        ("Scope", "h"),
        (f"  config.yaml ecfr.chapter = {cfg['ecfr'].get('chapter') or 'null (whole title, every chapter)'}", "mono"),
        (f"  snapshot date {cfg['ecfr']['date']}; eval cap {cfg['dataset']['max_eval_examples']} examples, round-robin by chapter", "mono"),
        ("", "b"),
        ("Device selection", "h"),
        ("  --device auto   CUDA if visible, else Apple Silicon GPU (MPS), else CPU", "mono"),
        ("  --device cuda   bf16/fp16 + 4-bit QLoRA below the VRAM threshold", "mono"),
        ("", "b"),
        ("Artifacts", "h"),
        ("  data/                sections.jsonl, cpt/sft/dpo/eval JSONL + dataset_card.md", "mono"),
        ("  outputs/*-adapter/   LoRA adapters + training_log.json + hub_upload.json per stage", "mono"),
        ("  outputs/manifests/   per-stage manifests: config hash, data hashes, runtime, versions", "mono"),
        ("  results/             eval_results.json, report.md, generations_*.jsonl,", "mono"),
        ("                       examples_all.md, model_report.pdf (this file)", "mono"),
        ("", "b"),
        ("Reproducibility", "h"),
        (f"  Global seed {cfg['seed']}; greedy decoding at eval; eCFR snapshot pinned to "
         f"{cfg['ecfr']['date']};", "b"),
        ("  splits derived from stable hashes of section numbers (not RNG order); every stage", "b"),
        ("  writes a manifest with the config hash and SHA-256 of its data inputs.", "b"),
        ("", "b"),
        ("Honest-reporting policy", "h"),
        ("  Every metric regression vs the base model is flagged (never hidden) here and in", "b"),
        ("  results/report.md; every held-out example is reproduced in the appendix so the", "b"),
        ("  aggregate numbers can be audited case by case.", "b"),
    ]
    _render_lines(fig, lines)
    pdf.savefig(fig)
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
    sections = common.load_jsonl(data_dir / "sections.jsonl") if (data_dir / "sections.jsonl").exists() else []
    scores = _per_example_scores(gens)

    pdf_path = results_dir / "model_report.pdf"
    with PdfPages(pdf_path) as pdf:
        _title_page(pdf, cfg, res, out_root, data_dir, sections)
        _corpus_page(pdf, res, data_dir, sections)
        _hparams_page(pdf, cfg, out_root)
        _overview_curves_page(pdf, out_root)
        for st in STAGES:
            _stage_detail_pages(pdf, st, _load_log(out_root, st), _load_manifest(out_root, st))
        _metrics_page(pdf, res)
        _chapter_pages(pdf, res)
        _winloss_page(pdf, gens, scores)
        _publish_page(pdf, cfg, out_root)
        _howto_page(pdf, cfg)
        _example_pages(pdf, gens, scores)
        info = pdf.infodict()
        info["Title"] = "eCFR Title 12 fine-tuning report"
        info["Subject"] = f"model={res['model_id']} smoke={res['smoke']}"

    _write_examples_md(results_dir / "examples_all.md", gens, scores, res)
    print(f"[report] wrote {pdf_path} and {results_dir / 'examples_all.md'}")
    common.write_manifest(
        cfg,
        "report",
        {"pdf": str(pdf_path), "model_id": res["model_id"], "smoke": res["smoke"],
         "checkpoints": list(res["checkpoints"]), "n_examples": len(scores)},
    )
    return pdf_path
