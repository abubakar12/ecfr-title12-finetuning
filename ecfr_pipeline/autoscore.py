"""Automatic blind review: deterministic citation checks plus an LLM judge for claim support.

Produces a completed review file in the blind_review.jsonl schema so the frozen
`score` path (metrics, paired bootstrap) is unchanged. The judge never sees the
checkpoint identity. Chapter I spans 12 CFR parts 1-199, so a Chapter I section
citation absent from the frozen corpus is counted as nonexistent; citations to
other chapters are left unresolved (the corpus cannot verify them).
"""
from __future__ import annotations

import json
import os
import re
import urllib.request
from pathlib import Path

from . import corpus
from .benchmark import score, structural_scores, verify_frozen

CHAPTER_I_PARTS = range(1, 200)

CLAIM_PROMPT = ("You are grading an answer about United States banking regulations (12 CFR Chapter I).\n\n"
                "AUTHORITATIVE SOURCE PASSAGE:\n{evidence}\n\nREQUIRED CLAIM:\n{claim}\n\nCANDIDATE ANSWER:\n{answer}\n\n"
                "Does the candidate answer clearly state the required claim in a way that is consistent with the source passage? "
                "Paraphrase is acceptable; omission, vagueness, or contradiction is not. Reply with exactly one word: yes or no.")
CONTRADICTION_PROMPT = ("You are grading an answer about United States banking regulations (12 CFR Chapter I).\n\n"
                        "AUTHORITATIVE SOURCE PASSAGE:\n{evidence}\n\nCANDIDATE ANSWER:\n{answer}\n\n"
                        "Does the candidate answer assert any regulatory requirement, threshold, deadline, or permission that contradicts "
                        "the source passage or invents a rule the passage does not support? Reply with exactly one word: yes or no.")
SCOPE_PROMPT = ("A user asked a question that is outside the scope of 12 CFR Chapter I (Comptroller of the Currency). "
                "Reason it is out of scope: {reason}\n\nQUESTION:\n{question}\n\nCANDIDATE ANSWER:\n{answer}\n\n"
                "Does the candidate answer recognize that the request falls outside 12 CFR Chapter I (for example by saying so, "
                "declining, or pointing to the correct other authority) instead of presenting it as a Chapter I rule? "
                "Reply with exactly one word: yes or no.")


def parse_verdict(text):
    # Small judges sometimes emit junk right after the verdict token ("no://m,").
    match = re.match(r"\W*(yes|no)\b", text, re.I)
    if not match:
        raise ValueError(f"Judge returned a non yes/no verdict: {text!r}")
    return match.group(1).lower() == "yes"


class LocalJudge:
    """Greedy yes/no from the frozen base model; same weights, no adapter."""

    def __init__(self, cfg, lock, max_new_tokens):
        from . import phase1_runtime as runtime_tools
        runtime_tools.configure_environment()
        import torch
        from .cpt import tokenizer_for
        self.torch = torch
        self.runtime = runtime_tools.resolve(cfg)
        self.tok = tokenizer_for(lock)
        self.model = runtime_tools.load_model(lock, self.runtime).eval()
        self.max_new_tokens = max_new_tokens
        self.name = f"local:{lock['id']}@{lock['revision']}"

    def __call__(self, prompt):
        ids = self.tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=True,
                                           add_generation_prompt=True, return_tensors="pt").to(self.runtime["device"])
        with self.torch.no_grad():
            out = self.model.generate(input_ids=ids, attention_mask=self.torch.ones_like(ids), do_sample=False,
                                      max_new_tokens=self.max_new_tokens, pad_token_id=self.tok.pad_token_id,
                                      eos_token_id=self.tok.eos_token_id)
        return parse_verdict(self.tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True))


class OpenAIJudge:
    """Any OpenAI-compatible chat completions endpoint; key from OPENAI_API_KEY."""

    def __init__(self, model, base_url=None):
        self.key = os.environ.get("OPENAI_API_KEY")
        if not self.key:
            raise RuntimeError("OPENAI_API_KEY is required for the openai judge")
        self.url = (base_url or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/") + "/chat/completions"
        self.model = model
        self.name = f"openai:{model}"

    def __call__(self, prompt):
        body = json.dumps({"model": self.model, "temperature": 0, "max_tokens": 3,
                           "messages": [{"role": "user", "content": prompt}]}).encode()
        request = urllib.request.Request(self.url, data=body, headers={"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=120) as response:
            return parse_verdict(json.load(response)["choices"][0]["message"]["content"])


def make_judge(cfg, lock):
    settings = cfg["evaluation"].get("judge", {"provider": "local"})
    if settings.get("provider", "local") == "local":
        return LocalJudge(cfg, lock, settings.get("max_new_tokens", 4))
    if settings["provider"] == "openai":
        return OpenAIJudge(settings["model"], settings.get("base_url"))
    raise ValueError(f"Unknown judge provider: {settings['provider']}")


def citation_counts(answer, question, valid_sections):
    s = structural_scores(answer, question, valid_sections)
    section = lambda c: c.split("(")[0]
    part = lambda c: int(re.match(r"12 CFR § (\d+)", c).group(1)) if c.startswith("12 CFR § ") and re.match(r"12 CFR § (\d+)", c) else None
    nonexistent = [c for c in s["absent_from_corpus"] if part(c) in CHAPTER_I_PARTS]
    accepted_sections = {section(a) for a in question["acceptable_citations"]}
    with_paragraph = [c for c in s["found"] if "(" in c]
    return {"total_citation_count": len(s["found"]), "correct_citation_count": len(s["matched"]),
            "nonexistent_citation_count": len(nonexistent),
            "section_citations_correct": bool(s["found"]) and all(section(c) in accepted_sections for c in s["found"]),
            "paragraph_citations_correct": bool(s["paragraph_matches"]) and len(s["paragraph_matches"]) == len(with_paragraph),
            "structural": s}


def review(cfg, judge=None, output=None):
    directory = Path(cfg["experiment_dir"])
    evaluation = directory / "evaluation"
    questions = {q["id"]: q for q in verify_frozen(cfg)}
    lock = corpus.read_json(evaluation / "generation.lock.json")
    if corpus.file_hash(evaluation / "blind_review.jsonl") != lock["blind_sha256"]:
        raise ValueError("Frozen blind review packet changed")
    valid = {d["citation"] for d in corpus.read_rows(directory / "documents.jsonl") if d["kind"] == "SECTION"}
    judge = judge or make_judge(cfg, lock["model"])
    cache_path = evaluation / "auto_judgments.jsonl"
    cache = {r["key"]: r["verdict"] for r in corpus.read_rows(cache_path)} if cache_path.exists() else {}

    def ask(key, prompt):
        if key not in cache:
            cache[key] = judge(prompt)
            with cache_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"key": key, "verdict": cache[key], "judge": judge.name}) + "\n")
        return cache[key]

    rows = []
    for row in corpus.read_rows(evaluation / "blind_review.jsonl"):
        q = questions[row["question_id"]]
        r = dict(row)
        r["reviewer"] = f"auto-judge {judge.name}"
        if q["type"] == "outside_scope":
            r["outside_scope_correct"] = ask(f"{row['review_id']}:scope", SCOPE_PROMPT.format(reason=q["scope_reason"], question=q["question"], answer=row["answer"]))
            r["substantively_correct"] = r["outside_scope_correct"]
            r["notes"] = json.dumps({"citations": citation_counts(row["answer"], q, valid)["structural"]["found"]})
            rows.append(r)
            continue
        evidence = "\n\n".join(e["quote"] for e in q["evidence"])
        supported = [ask(f"{row['review_id']}:claim:{i}", CLAIM_PROMPT.format(evidence=evidence, claim=claim, answer=row["answer"]))
                     for i, claim in enumerate(q["required_claims"])]
        contradicts = ask(f"{row['review_id']}:contradiction", CONTRADICTION_PROMPT.format(evidence=evidence, answer=row["answer"]))
        counts = citation_counts(row["answer"], q, valid)
        structural = counts.pop("structural")
        r.update(counts)
        r["supported_claims"] = sum(supported)
        r["substantively_correct"] = all(supported) and not contradicts
        r["outside_scope_correct"] = None
        r["notes"] = json.dumps({"claims_supported": supported, "contradiction": contradicts, "citations": structural})
        rows.append(r)
        print(f"{row['review_id']}: claims {sum(supported)}/{len(supported)} contradiction={contradicts} "
              f"citations {counts['correct_citation_count']}/{counts['total_citation_count']} nonexistent={counts['nonexistent_citation_count']}", flush=True)
    output = Path(output) if output else evaluation / "auto_review.jsonl"
    corpus.write_rows(output, rows)
    return output


def run(cfg, output=None):
    path = review(cfg, output=output)
    score(cfg, path)
    print(f"[auto-score] completed review: {path}")
