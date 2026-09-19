"""Stage 1: download a point-in-time eCFR snapshot and extract per-section records.

Uses the public eCFR versioner API. The snapshot date is pinned in config.yaml so
the same command always yields the same corpus.
"""
from __future__ import annotations

import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import requests

from . import common

API_URL = "https://www.ecfr.gov/api/versioner/v1/full/{date}/title-{title}.xml"
PARA_TAGS = {"P", "FP"}  # regulation body text; CITA/SOURCE/AUTH amendment notes are skipped


def fetch_xml(cfg: dict) -> Path:
    e = cfg["ecfr"]
    scope = f"chapter{e['chapter']}" if e.get("chapter") else "full"
    out = Path(cfg["paths"]["raw_dir"]) / f"title{e['title']}_{scope}_{e['date']}.xml"
    if out.exists() and out.stat().st_size > 0:
        print(f"[download] cached snapshot: {out}")
        return out

    url = API_URL.format(date=e["date"], title=e["title"])
    params = {"chapter": e["chapter"]} if e.get("chapter") else None
    last_err = None
    for attempt in range(1, 4):
        try:
            print(f"[download] GET {url} params={params} (attempt {attempt})")
            resp = requests.get(url, params=params, timeout=300)
            resp.raise_for_status()
            out.write_bytes(resp.content)
            print(f"[download] saved {out} ({out.stat().st_size / 1e6:.1f} MB)")
            return out
        except requests.RequestException as err:
            last_err = err
            time.sleep(5 * attempt)
    raise RuntimeError(f"eCFR download failed after 3 attempts: {last_err}")


def _flat_text(el) -> str:
    return re.sub(r"\s+", " ", " ".join(el.itertext())).strip()


ROMAN = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100}


def _roman_key(numeral: str) -> int:
    total, prev = 0, 0
    for ch in reversed((numeral or "").upper()):
        val = ROMAN.get(ch, 0)
        total, prev = (total - val, prev) if val < prev else (total + val, val)
    return total


def _chapter_index(root) -> dict:
    """Map id(section element) -> (chapter number, agency name) from the DIV3 CHAPTER ancestors."""
    index = {}
    for chap in root.iter():
        if chap.get("TYPE") != "CHAPTER":
            continue
        number = re.sub(r"\s*\[reserved\]\s*", "", chap.get("N") or "", flags=re.I).strip()
        head = chap.find("HEAD")
        name = _flat_text(head) if head is not None else ""
        name = re.split(r"[\u2014\u2013-]", name, maxsplit=1)[-1].strip() if "CHAPTER" in name.upper() else name
        for div in chap.iter("DIV8"):
            index[id(div)] = (number, name.title())
    return index


def parse_sections(xml_path: Path) -> list:
    root = ET.parse(xml_path).getroot()
    chapters = _chapter_index(root)
    records = []
    for div in root.iter("DIV8"):
        if div.get("TYPE") != "SECTION":
            continue
        ident = (div.get("N") or "").strip()
        if not ident:
            continue
        head = div.find("HEAD")
        heading_raw = _flat_text(head) if head is not None else ""
        heading = re.sub(r"^§+\s*[\dA-Za-z.\-]+\s*", "", heading_raw).strip().rstrip(".")
        paragraphs = [_flat_text(p) for p in div.iter() if p.tag in PARA_TAGS]
        text = "\n".join(p for p in paragraphs if p)
        chapter, chapter_name = chapters.get(id(div), (None, None))
        records.append(
            {
                "section": ident,
                "part": ident.split(".")[0],
                "chapter": chapter,
                "chapter_name": chapter_name,
                "citation": f"12 CFR § {ident}",
                "heading": heading,
                "reserved": "[reserved]" in heading_raw.lower(),
                "text": text,
                "word_count": len(text.split()),
            }
        )
    return records


def run(cfg: dict, smoke: bool = False, model_override: str | None = None) -> Path:
    common.ensure_dirs(cfg)
    xml_path = fetch_xml(cfg)
    records = parse_sections(xml_path)
    if not records:
        raise RuntimeError(f"no sections parsed from {xml_path} — check date/chapter in config.yaml")
    out = Path(cfg["paths"]["data_dir"]) / "sections.jsonl"
    common.save_jsonl(out, records)
    reserved = sum(r["reserved"] for r in records)
    chapters = sorted({r["chapter"] for r in records if r["chapter"]}, key=_roman_key)
    print(f"[download] parsed {len(records)} sections ({reserved} reserved) across "
          f"{len(chapters)} chapters [{', '.join(chapters)}] -> {out}")
    common.write_manifest(
        cfg,
        "download",
        {
            "snapshot_xml": str(xml_path),
            "snapshot_sha256": common.sha256_file(xml_path),
            "sections_jsonl": str(out),
            "sections_sha256": common.sha256_file(out),
            "n_sections": len(records),
            "chapters": chapters,
        },
    )
    return out
