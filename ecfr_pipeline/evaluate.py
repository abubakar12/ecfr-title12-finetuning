"""Stage 6: reproducible evaluation harness — base vs SFT vs DPO (vs GRPO).

Greedy decoding, fixed seed, held-out test sections only. Generations are
cached per checkpoint in results/, so metrics can be re-scored without paying
for regeneration (--regen invalidates). Regressions vs base are reported in
results/report.md, never hidden.
"""
from __future__ import annotations

import gc
import json
import time
from collections import defaultdict
from pathlib import Path

from . import common, metrics


def _score(recs: list) -> dict:
    per_type, per_chapter = defaultdict(list), defaultdict(list)
    for r in recs:
        found = metrics.extract_sections(r["completion"])
        hit = metrics.citation_correct(r["completion"], r["expected_citation"])
        item = {
            "citation": float(hit),
            "wrong_citation": float(bool(found) and not hit),
            "token_f1": metrics.token_f1(r["completion"], r["reference"]),
            "rouge_l": metrics.rouge_l(r["completion"], r["reference"]),
            "words": len(r["completion"].split()),
        }
        per_type[r["type"]].append(item)
        per_chapter[str(r.get("chapter") or "?")].append(item)

    def agg(items: list) -> dict:
        n = len(items)
        return {
            "n": n,
            "citation_accuracy": round(sum(i["citation"] for i in items) / n, 4),
            "wrong_citation_rate": round(sum(i["wrong_citation"] for i in items) / n, 4),
            "token_f1": round(sum(i["token_f1"] for i in items) / n, 4),
            "rouge_l": round(sum(i["rouge_l"] for i in items) / n, 4),
            "avg_words": round(sum(i["words"] for i in items) / n, 1),
        }

    all_rows = [i for items in per_type.values() for i in items]
    return {"overall": agg(all_rows),
            "per_type": {t: agg(v) for t, v in sorted(per_type.items())},
            "per_chapter": {c: agg(v) for c, v in sorted(per_chapter.items(), key=lambda kv: (len(kv[0]), kv[0]))}}


def _write_report(results: dict, path: Path) -> None:
    cols = ["citation_accuracy", "wrong_citation_rate", "token_f1", "rouge_l"]
    lower_is_better = {"wrong_citation_rate"}
    names = list(results)
    base = results.get("base", {}).get("overall")
    lines = [
        "# eCFR Title 12 — Evaluation Report",
        "",
        "Deltas are vs the untuned base model; ⚠ marks a regression (reported, not hidden).",
        "",
        "| checkpoint | n | " + " | ".join(cols) + " |",
        "|---|---|" + "---|" * len(cols),
    ]
    for name in names:
        o = results[name]["overall"]
        cells = [name, str(o["n"])]
        for m in cols:
            cell = f"{o[m]:.4f}"
            if base and name != "base":
                delta = o[m] - base[m]
                worse = delta > 0.005 if m in lower_is_better else delta < -0.005
                cell += f" ({delta:+.4f}{' ⚠' if worse else ''})"
            cells.append(cell)
        lines.append("| " + " | ".join(cells) + " |")

    lines += ["", "## token_f1 by question type", ""]
    types = sorted({t for r in results.values() for t in r["per_type"]})
    lines.append("| type | " + " | ".join(names) + " |")
    lines.append("|---|" + "---|" * len(names))
    for t in types:
        row = [t] + [
            f"{results[n]['per_type'].get(t, {}).get('token_f1', float('nan')):.4f}"
            for n in names
        ]
        lines.append("| " + " | ".join(row) + " |")

    chapters = list(next(iter(results.values())).get("per_chapter", {}))
    for metric in ("citation_accuracy", "token_f1"):
        lines += ["", f"## {metric} by chapter (12 CFR)", "",
                  "| chapter | n | " + " | ".join(names) + " |", "|---|---|" + "---|" * len(names)]
        for c in chapters:
            n_c = results[names[0]]["per_chapter"][c]["n"]
            row = [c, str(n_c)] + [f"{results[n]['per_chapter'].get(c, {}).get(metric, float('nan')):.4f}" for n in names]
            lines.append("| " + " | ".join(row) + " |")
    path.write_text("\n".join(lines) + "\n")


def _generate(cfg, runtime, tok, model, rows, max_new: int, batch_size: int) -> list:
    import torch

    tok.padding_side = "left"
    max_prompt = cfg["sft"]["max_length"] - max_new
    completions = []
    model.eval()
    with torch.inference_mode():
        for i in range(0, len(rows), batch_size):
            batch = rows[i : i + batch_size]
            prompts = [
                tok.apply_chat_template(
                    r["prompt_messages"], tokenize=False, add_generation_prompt=True
                )
                for r in batch
            ]
            # template already contains BOS/special tokens
            enc = tok(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_prompt,
                add_special_tokens=False,
            )
            enc = {k: v.to(model.device) for k, v in enc.items()}
            out = model.generate(
                **enc,
                max_new_tokens=max_new,
                do_sample=False,
                pad_token_id=tok.pad_token_id,
            )
            new_tokens = out[:, enc["input_ids"].shape[1] :]
            completions.extend(t.strip() for t in tok.batch_decode(new_tokens, skip_special_tokens=True))
            done = min(i + batch_size, len(rows))
            print(f"\r[eval] generated {done}/{len(rows)}", end="", flush=True)
    print()
    return completions


def _free(runtime: dict) -> None:
    import torch

    gc.collect()
    if runtime["device"] == "cuda":
        torch.cuda.empty_cache()
    elif runtime["device"] == "mps":
        torch.mps.empty_cache()


def run(cfg: dict, smoke: bool = False, model_override: str | None = None,
        regen: bool = False) -> Path:
    common.set_global_seed(cfg["seed"])
    runtime = common.detect_runtime(cfg, smoke)
    model_id = common.resolve_model_id(cfg, smoke, model_override)

    data_dir = Path(cfg["paths"]["data_dir"])
    results_dir = Path(cfg["paths"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    rows = common.load_jsonl(
        common.require_file(data_dir / "eval_test.jsonl", "run the `build` stage first")
    )
    if smoke:
        rows = rows[: cfg["smoke"]["max_eval_examples"]]
    max_new = cfg["smoke"]["max_new_tokens"] if smoke else cfg["eval"]["max_new_tokens"]
    batch_size = cfg["smoke"]["eval_batch_size"] if smoke else cfg["eval"]["batch_size"]

    out_root = Path(cfg["paths"]["outputs_dir"])
    adapter_dirs = {
        "base": None,
        "cpt": out_root / "cpt-adapter",
        "sft": out_root / "sft-adapter",
        "dpo": out_root / "dpo-adapter",
        "grpo": out_root / "grpo-adapter",
    }
    checkpoints = {}
    for name in cfg["eval"].get("checkpoints", list(adapter_dirs)):
        adir = adapter_dirs.get(name)
        if adir is not None and not (adir / "adapter_config.json").exists():
            print(f"[eval] skipping '{name}' — no adapter at {adir}")
            continue
        checkpoints[name] = adir

    tok = common.load_tokenizer(model_id, cfg)
    results = {}
    for name, adapter_dir in checkpoints.items():
        gen_path = results_dir / f"generations_{name}.jsonl"
        recs = None
        if gen_path.exists() and not regen:
            cached = common.load_jsonl(gen_path)
            if len(cached) == len(rows):
                print(f"[eval] '{name}': using cached generations ({gen_path})")
                recs = cached
        if recs is None:
            model = common.load_base_model(cfg, runtime, model_id, for_training=False)
            if adapter_dir is not None:
                model = common.load_peft_adapter(model, adapter_dir, trainable=False)
            completions = _generate(cfg, runtime, tok, model, rows, max_new, batch_size)
            recs = [{**row, "completion": comp} for row, comp in zip(rows, completions)]
            common.save_jsonl(gen_path, recs)
            del model
            _free(runtime)
        results[name] = _score(recs)
        o = results[name]["overall"]
        print(
            f"[eval] {name}: cite_acc={o['citation_accuracy']:.3f} "
            f"wrong_cite={o['wrong_citation_rate']:.3f} "
            f"f1={o['token_f1']:.3f} rougeL={o['rouge_l']:.3f} words={o['avg_words']} (n={o['n']})"
        )

    payload = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_id": model_id,
        "smoke": smoke,
        "n_eval": len(rows),
        "chapters": {str(r.get("chapter") or "?"): r.get("chapter_name") for r in rows},
        "runtime": runtime,
        "checkpoints": results,
    }
    results_path = results_dir / "eval_results.json"
    results_path.write_text(json.dumps(payload, indent=2))
    _write_report(results, results_dir / "report.md")
    common.write_manifest(
        cfg,
        "eval",
        {
            "model_id": model_id,
            "smoke": smoke,
            "runtime": runtime,
            "n_eval": len(rows),
            "checkpoints": sorted(checkpoints),
            "dataset_sha256": common.sha256_file(data_dir / "eval_test.jsonl"),
        },
    )
    print(f"[eval] wrote {results_path} and {results_dir / 'report.md'}")
    return results_path
