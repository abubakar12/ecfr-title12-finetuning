"""Frozen questions, reproducible generation, blinded review, paired scoring.

Semantic correctness is a reviewed label, never inferred from a citation regex.
"""
from __future__ import annotations

import random
import re
from collections import Counter
from pathlib import Path

from . import corpus

COUNTS = {"factual": 40, "application": 30, "policy": 20, "outside_scope": 10}
CITATION = re.compile(r"(?:(?P<title>\d+)\s*C\.?\s*F\.?\s*R\.?\s*(?:§{1,2}\s*)?|§{1,2}\s*)(?P<section>\d+[a-zA-Z]?\.\d+[a-zA-Z]?)(?P<paragraph>(?:\([A-Za-z0-9]+\))*)", re.I)


def citations(answer, default_title=12):
    return sorted({f"{m.group('title') or default_title} CFR § {m.group('section')}{m.group('paragraph')}" for m in CITATION.finditer(answer)})


def structural_scores(answer, gold, valid_sections):
    found = set(citations(answer))
    accepted = set(gold["acceptable_citations"])
    granularity = gold["granularity"]
    section = lambda c: c.split("(")[0]
    matches = {c for c in found if c in accepted or (granularity == "section" and section(c) in accepted and "(" not in c)}
    # Paragraph suffixes need verified paragraph-level support, even when the
    # reference only requires a section. Never silently accept a wrong suffix.
    return {"found": sorted(found), "matched": sorted(matches), "unmatched": sorted(found - matches),
            "absent_from_corpus": sorted(c for c in found if section(c) not in valid_sections),
            "wrong_title_citations": sorted(c for c in found if not c.startswith("12 CFR ")),
            "section_matches": sorted(c for c in found if section(c) in {section(a) for a in accepted}),
            "paragraph_matches": sorted(c for c in matches if "(" in c)}


def freeze(cfg, source):
    corpus.require_audit(cfg)
    directory = Path(cfg["experiment_dir"])
    if (directory / "training" / "completed.json").exists() or list((directory / "training" / "checkpoints").glob("checkpoint-*")):
        raise ValueError("Benchmark must be frozen before training starts")
    rows = corpus.read_rows(Path(source))
    if Counter(r["type"] for r in rows) != COUNTS:
        raise ValueError(f"Benchmark must contain exactly {COUNTS}")
    if len({r["id"] for r in rows}) != 100 or len({r["question"].strip().lower() for r in rows}) != 100:
        raise ValueError("Duplicate question IDs or prompts")
    documents = {r["id"]: r for r in corpus.read_rows(directory / "documents.jsonl")}
    for row in rows:
        if citations(row["question"]) or re.search(r"\b\d+\.\d+(?:\([a-z0-9]+\))*\b", row["question"], re.I):
            raise ValueError(f"Prompt exposes a citation: {row['id']}")
        if row.get("review_status") != "verified" or not row.get("reviewer"):
            raise ValueError(f"Question requires source verification: {row['id']}")
        if row.get("granularity") not in {"section", "paragraph", "none"}:
            raise ValueError("Invalid citation granularity")
        if not row.get("reference_answer") or not row.get("required_claims"):
            raise ValueError("Answer and required claims are mandatory")
        if row["type"] == "outside_scope":
            if row["acceptable_citations"] or row["granularity"] != "none" or not row.get("scope_reason"):
                raise ValueError("Outside-scope items need scope rationale and no in-scope gold citations")
            continue
        if not row.get("evidence") or not row.get("acceptable_citations"):
            raise ValueError("In-scope items need citations and evidence")
        for evidence in row["evidence"]:
            document = documents[evidence["document_id"]]
            start, end = evidence["start"], evidence["end"]
            if not (0 <= start < end <= len(document["text"])) or document["text"][start:end] != evidence["quote"]:
                raise ValueError(f"Evidence does not match frozen source: {row['id']}")
        source_citations = {documents[e["document_id"]]["citation"] for e in row["evidence"]}
        if any(c.split("(")[0] not in source_citations for c in row["acceptable_citations"]):
            raise ValueError("Gold citation lacks source evidence")
    path = directory / "benchmark.jsonl"
    if path.exists() and corpus.read_rows(path) != rows:
        raise ValueError("Benchmark already frozen; use a new experiment directory")
    corpus.write_rows(path, rows)
    corpus.write_json(directory / "benchmark.lock.json", {"sha256": corpus.file_hash(path), "documents_sha256": corpus.file_hash(directory / "documents.jsonl"), "counts": COUNTS})


def verify_frozen(cfg):
    directory = Path(cfg["experiment_dir"])
    lock = corpus.read_json(directory / "benchmark.lock.json")
    if lock["sha256"] != corpus.file_hash(directory / "benchmark.jsonl") or lock["documents_sha256"] != corpus.file_hash(directory / "documents.jsonl"):
        raise ValueError("Frozen benchmark/corpus hash mismatch")
    return corpus.read_rows(directory / "benchmark.jsonl")


def generate(cfg):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM
    from .cpt import tokenizer_for
    corpus.require_audit(cfg)
    questions = verify_frozen(cfg)
    directory = Path(cfg["experiment_dir"])
    completed = corpus.read_json(directory / "training" / "completed.json")
    if completed["run_sha256"] != corpus.file_hash(directory / "training" / "run.json"):
        raise ValueError("Training manifest changed")
    for filename, expected in completed["adapter_hashes"].items():
        if corpus.file_hash(directory / "training" / "adapter" / filename) != expected:
            raise ValueError("Saved adapter/tokenizer changed")
    lock = corpus.read_json(directory / "training" / "model.lock.json")
    tok = tokenizer_for(lock)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Matched BF16 evaluation requires CUDA; no implicit quantization")
    output = directory / "evaluation"
    if (output / "generations.jsonl").exists():
        raise ValueError("Generations already frozen")
    system = f"Answer using {corpus.scope(cfg)} as of {corpus.read_json(directory / 'snapshot.json')['date']}. Cite support for each regulatory claim. Identify requests outside this scope."
    model = AutoModelForCausalLM.from_pretrained(lock["id"], revision=lock["revision"], torch_dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda")
    generations = []
    for checkpoint in ("base", "cpt"):
        if checkpoint == "cpt":
            model = PeftModel.from_pretrained(model, directory / "training" / "adapter")
        model.eval()
        for row in questions:
            messages = [{"role": "system", "content": system}, {"role": "user", "content": row["question"]}]
            ids = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda")
            if ids.shape[1] + cfg["evaluation"]["max_new_tokens"] > 4096:
                raise ValueError("Evaluation prompt exceeds budget; no silent truncation")
            with torch.no_grad():
                result = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), do_sample=False,
                                        max_new_tokens=cfg["evaluation"]["max_new_tokens"], pad_token_id=tok.pad_token_id,
                                        eos_token_id=tok.eos_token_id)
            generations.append({"question_id": row["id"], "checkpoint": checkpoint,
                                "answer": tok.decode(result[0, ids.shape[1]:], skip_special_tokens=True),
                                "generated_tokens": result.shape[1] - ids.shape[1]})
    corpus.write_rows(output / "generations.jsonl", generations)
    rng = random.Random(cfg["seed"])
    shuffled = list(generations)
    rng.shuffle(shuffled)
    blind, mapping = [], []
    by_id = {q["id"]: q for q in questions}
    for index, row in enumerate(shuffled):
        review_id = f"response-{index + 1:03}"
        q = by_id[row["question_id"]]
        blind.append({"review_id": review_id, "question_id": q["id"], "question": q["question"], "answer": row["answer"],
                      "required_claims": q["required_claims"], "evidence": q.get("evidence", []),
                      "acceptable_citations": q["acceptable_citations"], "granularity": q["granularity"],
                      "supported_claims": None, "correct_citation_count": None, "total_citation_count": None,
                      "nonexistent_citation_count": None,
                      "substantively_correct": None, "outside_scope_correct": None,
                      "section_citations_correct": None, "paragraph_citations_correct": None,
                      "reviewer": "", "notes": ""})
        mapping.append({"review_id": review_id, "question_id": q["id"], "checkpoint": row["checkpoint"]})
    corpus.write_rows(output / "blind_review.jsonl", blind)
    corpus.write_rows(output / "private_mapping.jsonl", mapping)
    corpus.write_json(output / "generation.lock.json", {"model": lock, "system_prompt": system, "decoding": cfg["evaluation"],
                      "benchmark_sha256": corpus.file_hash(directory / "benchmark.jsonl"),
                      "generations_sha256": corpus.file_hash(output / "generations.jsonl"),
                      "mapping_sha256": corpus.file_hash(output / "private_mapping.jsonl"),
                      "blind_sha256": corpus.file_hash(output / "blind_review.jsonl")})


def paired_interval(pairs, seed=42, samples=10000):
    if not pairs:
        raise ValueError("No paired observations")
    rng = random.Random(seed)
    deltas = [b - a for a, b in pairs]
    draws = sorted(sum(rng.choices(deltas, k=len(deltas))) / len(deltas) for _ in range(samples))
    return {"delta": sum(deltas) / len(deltas), "lower": draws[int(0.025 * samples)], "upper": draws[min(samples - 1, int(0.975 * samples))]}


def score(cfg, reviews):
    directory = Path(cfg["experiment_dir"])
    output = directory / "evaluation"
    questions = {q["id"]: q for q in verify_frozen(cfg)}
    lock = corpus.read_json(output / "generation.lock.json")
    for file, key in [("generations.jsonl", "generations_sha256"), ("private_mapping.jsonl", "mapping_sha256"), ("blind_review.jsonl", "blind_sha256")]:
        if corpus.file_hash(output / file) != lock[key]:
            raise ValueError("Frozen generation/review inputs changed")
    original = {r["review_id"]: r for r in corpus.read_rows(output / "blind_review.jsonl")}
    mapping = {r["review_id"]: r for r in corpus.read_rows(output / "private_mapping.jsonl")}
    records = corpus.read_rows(Path(reviews))
    if len(records) != 200 or {r["review_id"] for r in records} != set(mapping):
        raise ValueError("Exactly 200 unique completed reviews required")
    valid = {d["citation"] for d in corpus.read_rows(directory / "documents.jsonl") if d["kind"] == "SECTION"}
    results = {"base": [], "cpt": []}
    pairs = {}
    for r in records:
        identity = mapping[r["review_id"]]
        q = questions[identity["question_id"]]
        if any(r[k] != original[r["review_id"]][k] for k in ("question_id", "question", "answer", "required_claims", "evidence", "acceptable_citations", "granularity")):
            raise ValueError("Reviewed answer or evidence was changed")
        if not r["reviewer"] or type(r["substantively_correct"]) is not bool:
            raise ValueError("Incomplete semantic review")
        if q["type"] == "outside_scope":
            if type(r["outside_scope_correct"]) is not bool:
                raise ValueError("Missing scope assessment")
            results[identity["checkpoint"]].append({"type": q["type"], "outside_scope_correct": r["outside_scope_correct"]})
            continue
        n = len(q["required_claims"])
        supported, correct, total = r["supported_claims"], r["correct_citation_count"], r["total_citation_count"]
        nonexistent = r["nonexistent_citation_count"]
        if any(type(v) is not int for v in (supported, correct, total, nonexistent)) or not 0 <= supported <= n or not 0 <= correct <= total or not 0 <= nonexistent <= total:
            raise ValueError("Invalid reviewed counts")
        if any(type(r[k]) is not bool for k in ("section_citations_correct", "paragraph_citations_correct")):
            raise ValueError("Missing citation granularity assessment")
        structural = structural_scores(r["answer"], q, valid)
        success = (supported == n and total > 0 and correct == total and r["substantively_correct"]
                   and r["section_citations_correct"] and (q["granularity"] != "paragraph" or r["paragraph_citations_correct"])
                   and nonexistent == 0)
        item = {"type": q["type"], "citation_accuracy": int(success), "citation_precision": correct / total if total else 0,
                "citation_coverage": supported / n, "substantive_accuracy": int(r["substantively_correct"]),
                "section_accuracy": int(r["section_citations_correct"]), "paragraph_accuracy": int(r["paragraph_citations_correct"]) if q["granularity"] == "paragraph" else None,
                "nonexistent_citation_count": nonexistent,
                "unresolved_section_citation_count": len(structural["absent_from_corpus"])}
        results[identity["checkpoint"]].append(item)
        pairs.setdefault(q["id"], {})[identity["checkpoint"]] = int(success)
    summary = {}
    for checkpoint, items in results.items():
        inside = [i for i in items if i["type"] != "outside_scope"]
        metrics = {}
        for key in inside[0]:
            if key == "type":
                continue
            values = [i[key] for i in inside if i[key] is not None]
            metrics[key] = sum(values) / len(values) if values else None
        outside = [i["outside_scope_correct"] for i in items if i["type"] == "outside_scope"]
        metrics["outside_scope_accuracy"] = sum(outside) / len(outside)
        summary[checkpoint] = metrics
    summary["paired_primary_95_percent_ci"] = paired_interval([(p["base"], p["cpt"]) for p in pairs.values()], cfg["seed"], cfg["evaluation"]["bootstrap_samples"])
    summary["review_sha256"] = corpus.file_hash(Path(reviews))
    corpus.write_json(output / "results.json", summary)
    (output / "report.md").write_text("# Base Instruct versus continued pretraining\n\nSeen regulations, unseen questions; no retrieval. Claim support was reviewed blind to checkpoint identity.\n\n```json\n" + __import__("json").dumps(summary, indent=2) + "\n```\n", encoding="utf-8")
    print(summary)
