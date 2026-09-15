"""Scoring utilities shared by the eval harness and the GRPO reward functions."""
from __future__ import annotations

import re
from collections import Counter

# Matches "§ 3.10", "§§ 3.10", "12 CFR 3.10", "12 C.F.R. § 3.10", "section 3.10".
SECTION_RE = re.compile(
    r"(?:§+\s*|12\s*C\.?\s*F\.?\s*R\.?\s*(?:§+\s*|[Pp]art\s+)?|[Ss]ection\s+)(\d{1,4}[A-Za-z]?\.\d+)"
)

_PUNCT_RE = re.compile(r"[^\w§.\s]")
_WS_RE = re.compile(r"\s+")


def normalize(text: str) -> str:
    text = text.lower()
    text = _PUNCT_RE.sub(" ", text)
    return _WS_RE.sub(" ", text).strip()


def tokens(text: str) -> list:
    return normalize(text).split()


def token_f1(prediction: str, reference: str) -> float:
    """SQuAD-style token overlap F1."""
    pred, ref = tokens(prediction), tokens(reference)
    if not pred or not ref:
        return float(pred == ref)
    overlap = sum((Counter(pred) & Counter(ref)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred)
    recall = overlap / len(ref)
    return 2 * precision * recall / (precision + recall)


def _lcs_len(a: list, b: list) -> int:
    # Rolling-row DP; fine at answer scale (<= ~400 tokens per side).
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[-1]))
        prev = cur
    return prev[-1]


def rouge_l(prediction: str, reference: str) -> float:
    pred, ref = tokens(prediction), tokens(reference)
    if not pred or not ref:
        return float(pred == ref)
    lcs = _lcs_len(pred, ref)
    if lcs == 0:
        return 0.0
    precision, recall = lcs / len(pred), lcs / len(ref)
    return 2 * precision * recall / (precision + recall)


def extract_sections(text: str) -> set:
    return {m.lower() for m in SECTION_RE.findall(text)}


def citation_correct(prediction: str, expected_section: str) -> bool:
    return expected_section.lower() in extract_sections(prediction)
