"""Stage 2: dataset engineering.

Turns raw eCFR sections into:
  * SFT chat examples (train/val)      -> data/sft_train.jsonl, data/sft_val.jsonl
  * DPO preference pairs (train only)  -> data/dpo_train.jsonl
  * Held-out eval set with references  -> data/eval_test.jsonl

Splits are assigned per *section* with a stable hash, so no section's text can
leak from train into eval; two assertions enforce this at build time. A dataset
card with counts, corruption mix, and artifact hashes is emitted alongside.
"""
from __future__ import annotations

import hashlib
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from . import common

# Matches definition paragraphs like "(a) Bank means ..." / "The term appraisal means ..."
DEFINITION_RE = re.compile(
    r"^\(?[a-z0-9]{0,4}\)?\s*(?:The\s+term\s+)?[\"\u201c']?"
    r"([A-Z][A-Za-z0-9 ,&/\-]{2,60}?)[\"\u201d']?\s+means\s+(.+)$"
)

CORRUPTIONS = ("wrong_section", "wrong_citation", "vague")


def _sentences(text: str) -> list:
    return [s.strip() for s in re.split(r"(?<=[.;])\s+", text) if s.strip()]


def _cap_words(text: str, max_words: int) -> str:
    """Truncate at a sentence boundary, hard-capping at max_words."""
    out, count = [], 0
    for sent in _sentences(text):
        n = len(sent.split())
        if count + n > max_words and out:
            break
        if n > max_words:
            sent = " ".join(sent.split()[:max_words])
        out.append(sent)
        count += n
        if count >= max_words:
            break
    return " ".join(out)


def _find_definitions(section: dict, cfg: dict) -> list:
    defs = []
    for para in section["text"].split("\n"):
        m = DEFINITION_RE.match(para)
        if not m:
            continue
        term = m.group(1).strip()
        definition = _cap_words(m.group(2).strip(), cfg["dataset"]["max_definition_words"])
        if 1 <= len(term.split()) <= 6 and len(definition.split()) >= 5:
            defs.append((term, definition))
    return defs[:4]  # cap per section to keep question-type balance sane


def build_examples(section: dict, cfg: dict, heading_is_unique: bool = True) -> list:
    ds = cfg["dataset"]
    cit, head = section["citation"], section["heading"]
    system = cfg["system_prompt"]
    short = _cap_words(section["text"], 60)
    full = _cap_words(section["text"], ds["max_answer_words"])
    first_sentence = _cap_words(_sentences(section["text"])[0], 60) if section["text"] else ""

    def make(kind: str, question: str, answer: str, idx: int | None = None) -> dict:
        suffix = f"::{idx}" if idx is not None else ""
        return {
            "id": f"{section['section']}::{kind}{suffix}",
            "type": kind,
            "section": section["section"],
            "expected_citation": section["section"],
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ],
        }

    examples = [
        make(
            "overview",
            f"In plain terms, what does {cit} ({head}) cover?",
            f"{cit} ('{head}') covers the following: {short}\n\nSource: {cit}.",
        ),
        make(
            "provisions",
            f"Summarize the key provisions of {cit}, '{head}'.",
            f"Key provisions of {cit} ('{head}'): {full}\n\nSource: {cit}.",
        ),
    ]
    # Lookup questions identify the section by heading alone, so they are only
    # posed for unique headings: repeated ones ("Purpose", "Definitions", ...)
    # would be ambiguous and would leak identical prompts across splits.
    if heading_is_unique:
        examples.append(
            make(
                "citation_lookup",
                f"Which section of 12 CFR addresses '{head}'?",
                f"That subject is addressed by {cit} ('{head}'). {first_sentence}",
            )
        )
    for i, (term, definition) in enumerate(_find_definitions(section, cfg)):
        if not definition.endswith("."):
            definition += "."
        examples.append(
            make("definition", f"How does {cit} define '{term}'?",
                 f"Under {cit}, '{term}' means {definition}", idx=i)
        )
    return examples


def build_dpo_pairs(train_examples: list, rng: random.Random, cfg: dict) -> list:
    grouped = defaultdict(list)
    for ex in train_examples:
        grouped[ex["section"]].append(ex)
    sections = sorted(grouped)
    if len(sections) < 2:
        return []

    per_sec = cfg["dataset"]["dpo_pairs_per_section"]
    pairs = []
    for sec in sections:
        picked = rng.sample(grouped[sec], min(per_sec, len(grouped[sec])))
        for ex in picked:
            other_sec = sec
            while other_sec == sec:
                other_sec = rng.choice(sections)
            other = rng.choice(grouped[other_sec])
            kind = rng.choice(CORRUPTIONS)
            good = ex["messages"][2]["content"]
            if kind == "wrong_section":
                bad = other["messages"][2]["content"]  # confident answer about the wrong section
            elif kind == "wrong_citation":
                bad = good.replace(f"§ {sec}", f"§ {other_sec}")
                if bad == good:  # answer carried no citation string; corrupt content instead
                    kind = "wrong_section"
                    bad = other["messages"][2]["content"]
            else:  # vague: cut off early, drop the citation
                bad = " ".join(good.split()[:12]) + " and various other requirements set out in the regulations."
            pairs.append(
                {
                    "id": f"{ex['id']}::dpo::{kind}",
                    "corruption": kind,
                    "section": sec,
                    "prompt": ex["messages"][:2],
                    "chosen": [ex["messages"][2]],
                    "rejected": [{"role": "assistant", "content": bad}],
                }
            )
    return pairs


def _write_dataset_card(cfg, data_dir, splits, examples, dpo_pairs, eval_rows, hashes):
    e = cfg["ecfr"]
    lines = [
        "# eCFR Fine-Tuning Dataset Card",
        "",
        f"- Source: eCFR Title {e['title']}" + (f", Chapter {e['chapter']}" if e.get("chapter") else " (full title)"),
        f"- Snapshot date (point-in-time): {e['date']}",
        f"- Seed: {cfg['seed']}  |  Config hash: {common.config_hash(cfg)}",
        "",
        "## Filtering",
        f"- Dropped reserved sections and sections with < {cfg['dataset']['min_section_chars']} chars of text.",
        "- Deduplicated sections with byte-identical text.",
        "- citation_lookup questions only for sections with a unique heading",
        "  (repeated headings would be ambiguous and leak prompts across splits).",
        "",
        "## Split (by section, stable md5 bucket — no cross-split section leakage)",
        "",
        "| split | sections | sft examples |",
        "|---|---|---|",
    ]
    for name in ("train", "val", "test"):
        lines.append(f"| {name} | {len(splits[name])} | {len(examples[name])} |")
    type_counts = Counter(ex["type"] for ex in examples["train"])
    lines += ["", "## Train question types", ""]
    lines += [f"- {t}: {n}" for t, n in sorted(type_counts.items())]
    corr_counts = Counter(p["corruption"] for p in dpo_pairs)
    lines += ["", f"## DPO pairs: {len(dpo_pairs)} (train sections only)", ""]
    lines += [f"- {k}: {n}" for k, n in sorted(corr_counts.items())]
    lines += ["", f"## Eval set: {len(eval_rows)} examples (capped at {cfg['dataset']['max_eval_examples']})", ""]
    lines += ["## Artifact hashes (sha256)", ""]
    lines += [f"- {name}: `{digest}`" for name, digest in hashes.items()]
    lines.append("")
    (data_dir / "dataset_card.md").write_text("\n".join(lines))


def run(cfg: dict, smoke: bool = False, model_override: str | None = None) -> None:
    common.ensure_dirs(cfg)
    ds_cfg = cfg["dataset"]
    rng = random.Random(cfg["seed"])
    data_dir = Path(cfg["paths"]["data_dir"])
    sections = common.load_jsonl(
        common.require_file(data_dir / "sections.jsonl", "run the `download` stage first")
    )

    usable, seen = [], set()
    for s in sections:
        if s["reserved"] or not s["heading"] or len(s["text"]) < ds_cfg["min_section_chars"]:
            continue
        digest = hashlib.sha256(s["text"].encode("utf-8")).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        usable.append(s)

    split_cfg = ds_cfg["split"]
    train_hi = split_cfg["train"]
    val_hi = train_hi + split_cfg["val"]
    buckets = val_hi + split_cfg["test"]
    splits = {"train": [], "val": [], "test": []}
    for s in usable:
        b = common.stable_bucket(f"12cfr::{s['section']}", buckets)
        splits["train" if b < train_hi else "val" if b < val_hi else "test"].append(s)

    heading_counts = Counter(s["heading"].strip().lower() for s in usable)
    examples = {
        name: [
            ex
            for sec in secs
            for ex in build_examples(
                sec, cfg, heading_counts[sec["heading"].strip().lower()] == 1
            )
        ]
        for name, secs in splits.items()
    }

    # Leakage guards: split is by section, and no user prompt may cross splits.
    train_secs = {s["section"] for s in splits["train"]}
    test_secs = {s["section"] for s in splits["test"]}
    assert not train_secs & test_secs, "section leakage between train and test"
    train_prompts = {ex["messages"][1]["content"] for ex in examples["train"]}
    test_prompts = {ex["messages"][1]["content"] for ex in examples["test"]}
    assert not train_prompts & test_prompts, "prompt leakage between train and test"

    dpo_pairs = build_dpo_pairs(examples["train"], rng, cfg)

    eval_rows = [
        {
            "id": ex["id"],
            "type": ex["type"],
            "section": ex["section"],
            "expected_citation": ex["expected_citation"],
            "prompt_messages": ex["messages"][:2],
            "reference": ex["messages"][2]["content"],
        }
        for ex in examples["test"]
    ]
    rng.shuffle(eval_rows)
    eval_rows = eval_rows[: ds_cfg["max_eval_examples"]]

    files = {
        "sft_train.jsonl": examples["train"],
        "sft_val.jsonl": examples["val"],
        "dpo_train.jsonl": dpo_pairs,
        "eval_test.jsonl": eval_rows,
    }
    hashes = {}
    for fname, rows in files.items():
        path = data_dir / fname
        common.save_jsonl(path, rows)
        hashes[fname] = common.sha256_file(path)
        print(f"[build] {fname}: {len(rows)} rows")
    for fname, rows in files.items():
        assert rows, f"{fname} is empty — dataset build produced no rows"

    _write_dataset_card(cfg, data_dir, splits, examples, dpo_pairs, eval_rows, hashes)
    print(f"[build] sections used: {len(usable)} "
          f"(train/val/test = {len(splits['train'])}/{len(splits['val'])}/{len(splits['test'])})")
    common.write_manifest(
        cfg,
        "build",
        {"files": hashes, "sections": {k: len(v) for k, v in splits.items()}},
    )
