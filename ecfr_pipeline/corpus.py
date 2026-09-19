"""Lossless, fail-closed eCFR corpus ingestion. Data stages use only the stdlib."""
from __future__ import annotations

import hashlib
import gzip
import json
import re
import shutil
import copy
import io
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

API = "https://www.ecfr.gov/api/versioner/v1"
METADATA = {"AUTH", "SOURCE", "CITA", "EDNOTE", "EFFDNOT", "XREF", "CFRTOC"}
KNOWN = set("HEAD P FP HD HED PS SPACE E SU SUB SUP I B XREF EXTRACT NOTE NOTES FTNT FTNT1 FTNT2 FNOTE TABLE GPOTABLE TTYPE BOXHD CHED RHED RHD ROW ENT TAB TABC TNOTE TTITLE TD TR TH THEAD TBODY COLGROUP COL LI OL UL MATH MathML math mrow mi mo mn msup msub mfrac mtext mtable mtr mtd img IMG GPH SECTNO SUBJECT PRTPAGE APPRO FINDING A C link br BR STRONG EM SPAN a span".split()) | METADATA
KNOWN |= set("FTREF HD1 HD2 HD3 FP1-2 FP-2 FP-1 P-1 PSPACE FP-DASH EDNOTE FR EXAMPLE EFFDNOT DIV CAPTION TFOOT sup sub strong em CHAPTI PG PTHD RESERVED".split())


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_hash(path: Path) -> str:
    return digest(path.read_bytes())


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_rows(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_rows(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def scope(cfg):
    e = cfg["ecfr"]
    return f"Title {e['title']}" + (f", Chapter {e['chapter']}" if e.get("chapter") else ", all chapters")


def get(url):
    request = urllib.request.Request(url, headers={"User-Agent": "eCFR-research-corpus/1.0", "Accept-Encoding": "gzip"})
    with urllib.request.urlopen(request, timeout=300) as response:
        data = response.read()
        return gzip.decompress(data) if response.headers.get("Content-Encoding") == "gzip" else data


def validate_xml(raw, cfg):
    root = ET.fromstring(raw)
    title_number = str(cfg["ecfr"]["title"])
    if not any(x.get("TYPE") == "TITLE" and (x.get("N") == title_number or x.get("NODE", "").split(":", 1)[0] == title_number)
               for x in root.iter()):
        raise ValueError("Response does not contain the requested CFR title")
    chapters = {x.get("N") for x in root.iter() if x.get("TYPE") == "CHAPTER"}
    wanted = cfg["ecfr"].get("chapter")
    if wanted and wanted not in chapters:
        raise ValueError(f"Chapter scope mismatch: {chapters}, expected {wanted}")
    if not any(x.get("TYPE") == "SECTION" for x in root.iter()):
        raise ValueError("Response has no sections")
    return root


def freeze_snapshot(cfg, refresh=False):
    directory = Path(cfg["experiment_dir"])
    lock_path = directory / "snapshot.json"
    raw_path = directory / "source.xml"
    if lock_path.exists():
        lock = read_json(lock_path)
        if refresh:
            raise ValueError("Refresh requires a new experiment_dir; frozen experiments are immutable")
        if lock["scope"] != scope(cfg) or (cfg["ecfr"]["date"] != "latest" and lock["date"] != cfg["ecfr"]["date"]):
            raise ValueError("Configuration differs from frozen snapshot")
        if file_hash(raw_path) != lock["sha256"]:
            raise ValueError("Frozen XML hash mismatch")
        validate_xml(raw_path.read_bytes(), cfg)
        return lock
    e = cfg["ecfr"]
    metadata = get(f"{API}/titles.json")
    title = next(t for t in json.loads(metadata)["titles"] if t["number"] == e["title"])
    date = title["up_to_date_as_of"] if e["date"] == "latest" else e["date"]
    url = e.get("source_url") or f"{API}/full/{date}/title-{e['title']}.xml"
    params = {} if e.get("source_url") else ({"chapter": e["chapter"]} if e.get("chapter") else {})
    if params:
        url += "?" + urllib.parse.urlencode(params)
    raw = get(url)
    validate_xml(raw, cfg)  # Never cache HTML/error responses as XML.
    directory.mkdir(parents=True, exist_ok=True)
    raw_path.write_bytes(raw)
    (directory / "titles.json").write_bytes(metadata)
    lock = {"scope": scope(cfg), "title": e["title"], "chapter": e.get("chapter"), "date": date,
            "url": url, "params": params, "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "sha256": digest(raw), "titles_sha256": digest(metadata)}
    write_json(lock_path, lock)
    return lock


def tag(el):
    return el.tag.rsplit("}", 1)[-1]


def is_div(el):
    # DIV is an HTML layout container; only numbered DIVn nodes are CFR hierarchy.
    return bool(re.fullmatch(r"DIV\d+", tag(el)))


def text(el):
    # Preserve inline adjacency (including words split by emphasis).
    return re.sub(r"\s+", " ", "".join(el.itertext())).strip()


def render(el):
    """Readable serialization; raw element XML remains the authoritative structure."""
    if tag(el) in {"GPOTABLE", "TABLE"}:
        lines = []
        for child in el.iter():
            if tag(child) in {"TTITLE", "TNOTE", "CHED", "ROW", "TR", "CAPTION"}:
                cells = [text(c) for c in child if tag(c) in {"ENT", "TD", "TH"}]
                lines.append(" | ".join(cells) if cells else text(child))
        return "\n".join(lines) if lines else text(el)
    if len(el) and any(tag(c) in {"P", "FP", "GPOTABLE", "TABLE", "FTNT", "EXTRACT"} for c in el):
        pieces = [el.text or ""]
        for c in el:
            pieces.extend([render(c), c.tail or ""])
        return "\n".join(p.strip() for p in pieces if p.strip())
    return text(el)


def parse(raw, snapshot, transcriptions=None, exclude_image_sections=False):
    root = ET.fromstring(raw)
    transcriptions = transcriptions or {}
    records, issues, inventory = [], [], []
    title = snapshot["title"]

    def visit(el, path, hierarchy):
        kind = el.get("TYPE")
        structural = is_div(el)
        if kind == "CHAPTER" and snapshot.get("chapter") and el.get("N") != snapshot["chapter"]:
            inventory.append({"location": path, "classification": "excluded_scope", "reason": "outside configured chapter",
                              "chapter": el.get("N"), "subtree_elements": sum(1 for _ in el.iter()), "sha256": digest(ET.tostring(el))})
            return
        if structural and kind not in {"TITLE", "SUBTITLE", "CHAPTER", "SUBCHAP", "PART", "SUBPART", "SUBJGRP", "SECTION", "APPENDIX"}:
            issues.append({"location": path, "reason": "unknown_hierarchy_type", "type": kind})
        current = dict(hierarchy)
        if structural and kind:
            current[kind.lower()] = {"number": el.get("N", ""), "heading": text(el.find("HEAD")) if el.find("HEAD") is not None else ""}
        if structural:
            own = [c for c in el if not is_div(c)]
            if kind in {"SECTION", "APPENDIX"} or any(tag(c) not in METADATA | {"HEAD"} for c in own):
                image_sources = sorted({graphic.get("src") or graphic.get("GID") or "" for child in own for graphic in child.iter()
                                        if tag(graphic) in {"GPH", "IMG", "img"}})
                if exclude_image_sections and kind in {"SECTION", "APPENDIX"} and image_sources:
                    inventory.append({"location": path, "classification": "excluded_scope", "reason": "section contains images",
                                      "image_sources": image_sources, "sha256": digest(ET.tostring(el))})
                    return
                ident = el.get("N", "")
                heading = current.get((kind or "").lower(), {}).get("heading", "")
                if kind == "SECTION":
                    ident = re.sub(r"^§+\s*", "", ident)
                if kind == "APPENDIX" and not ident and heading:
                    ident = heading
                part = current.get("part", {}).get("number", "")
                if kind == "SECTION":
                    citation = f"{title} CFR § {ident}"
                elif kind == "APPENDIX":
                    suffix = "" if re.search(r"\bPart\s+" + re.escape(part) + r"\b", ident) else f" to Part {part}"
                    citation = f"{title} CFR {ident}{suffix}"
                else:
                    citation = f"{title} CFR Part {part}: {heading}"
                if kind in {"SECTION", "APPENDIX"} and not ident:
                    issues.append({"location": path, "reason": "missing_regulatory_identifier"})
                if kind in {"SECTION", "APPENDIX"} and not heading:
                    issues.append({"location": path, "reason": "missing_heading"})
                blocks, metadata, offset = [], [], 0
                for index, child in enumerate(list(el), 1):
                    if is_div(child):
                        continue
                    location = f"{path}/{tag(child)}[{index}]"
                    classification = "heading" if tag(child) == "HEAD" else "metadata" if tag(child) in METADATA else "body"
                    inventory.append({"location": location, "classification": classification, "owner": path, "sha256": digest(ET.tostring(child))})
                    if classification == "heading":
                        continue
                    if classification == "metadata":
                        metadata.append({"type": tag(child), "text": text(child), "xml": ET.tostring(child, encoding="unicode"), "source_location": location})
                        continue
                    unknown = sorted({tag(c) for c in child.iter()} - KNOWN)
                    if unknown:
                        issues.append({"location": location, "reason": "unknown_structure", "tags": unknown})
                    rendered_child = copy.deepcopy(child)
                    assets = []
                    for graphic in rendered_child.iter():
                        if tag(graphic) not in {"GPH", "IMG", "img"}:
                            continue
                        src = graphic.get("src") or graphic.get("GID") or ""
                        transcription = transcriptions.get(src)
                        assets.append({"src": src, "transcription": transcription})
                        if not transcription:
                            issues.append({"location": location, "reason": "image_requires_transcription", "src": src})
                        else:
                            graphic.text = transcription["text"]
                    content = render(rendered_child)
                    # Fail if a specialized rendering silently omitted source text.
                    if re.sub(r"[\s|]", "", content) != re.sub(r"[\s|]", "", text(rendered_child)):
                        issues.append({"location": location, "reason": "rendering_text_mismatch"})
                    labels = re.match(r"^((?:\([A-Za-z0-9]+\)\s*)+)", content)
                    block = {"type": tag(child), "text": content, "xml": ET.tostring(child, encoding="unicode"), "source_location": location,
                             "labels": re.findall(r"\(([A-Za-z0-9]+)\)", labels.group(0)) if labels else [], "start": offset, "end": offset + len(content), "assets": assets}
                    blocks.append(block)
                    offset += len(content) + 1
                body = "\n".join(b["text"] for b in blocks)
                # Store literal labels only. Resolving a complete paragraph citation is
                # a separate reviewed annotation, never a guessed regex hierarchy.
                labels = [tuple(b["labels"]) for b in blocks if b["labels"]]
                repeated = [list(k) for k, n in Counter(labels).items() if n > 1]
                # Repeated (1) under different parents is normal; source paths uniquely
                # identify blocks. No inferred paragraph IDs are published.
                url = f"https://www.ecfr.gov/on/{snapshot['date']}/title-{title}"
                if current.get("chapter"):
                    url += "/chapter-" + current["chapter"]["number"]
                if part:
                    url += "/part-" + part
                if kind == "SECTION":
                    url += "/section-" + ident
                records.append({"id": path, "kind": kind, "number": ident, "citation": citation,
                                "heading": heading, "hierarchy": current, "reserved": "[reserved]" in heading.lower(),
                                "source_location": path, "source_url": url, "snapshot_date": snapshot["date"],
                                "text": body, "content_sha256": digest(body.encode()), "blocks": blocks, "metadata": metadata,
                                "repeated_literal_labels": repeated})
            else:
                for index, child in enumerate(list(el), 1):
                    if not is_div(child):
                        inventory.append({"location": f"{path}/{tag(child)}[{index}]", "classification": "hierarchy_metadata", "owner": path,
                                          "text": text(child), "xml": ET.tostring(child, encoding="unicode"), "sha256": digest(ET.tostring(child))})
        for index, child in enumerate(list(el), 1):
            if is_div(child) or not structural:
                visit(child, f"{path}/{tag(child)}[{index}]", current)

    visit(root, "/" + tag(root), {})
    if not records:
        issues.append({"reason": "empty_corpus"})
    return records, inventory, issues


def asset_transcriptions(directory):
    path = directory / "transcriptions.json"
    if not path.exists():
        return {}
    records = read_json(path)
    approved = {}
    for r in records:
        if r.get("review_status") != "verified":
            continue
        if not r.get("reviewer") or not r.get("text", "").strip():
            raise ValueError("Verified transcription needs text and reviewer")
        asset_path = directory / "assets" / r["filename"]
        if asset_path.resolve().parent != (directory / "assets").resolve() or file_hash(asset_path) != r["sha256"]:
            raise ValueError("Asset path/hash mismatch")
        if r["src"] in approved:
            raise ValueError("Duplicate transcription source")
        approved[r["src"]] = r
    return approved


def fetch_assets(cfg, download=True):
    directory = Path(cfg["experiment_dir"])
    snapshot = read_json(directory / "snapshot.json")
    raw = (directory / "source.xml").read_bytes()
    if digest(raw) != snapshot["sha256"]:
        raise ValueError("Source hash mismatch")
    documents, _, _ = parse(raw, snapshot)
    sources = sorted({a["src"] for d in documents for b in d["blocks"] for a in b["assets"]})
    asset_dir = directory / "assets"
    asset_dir.mkdir(exist_ok=True)
    path = directory / "transcriptions.json"
    previous = {r["src"]: r for r in read_json(path)} if path.exists() else {}
    rows = []
    for src in sources:
        url = urllib.parse.urljoin("https://www.ecfr.gov", src)
        if urllib.parse.urlparse(url).netloc != "www.ecfr.gov" or not src:
            raise ValueError("Unexpected graphic source")
        filename = digest(src.encode())[:12] + "-" + Path(urllib.parse.urlparse(url).path).name
        local = asset_dir / filename
        row = previous.get(src) or {"src": src, "url": url, "filename": filename, "sha256": file_hash(local) if local.exists() else None,
                                    "text": "", "reviewer": "", "review_status": "pending"}
        rows.append(row)
    write_json(path, rows)
    if not download:
        print(f"Inventoried {len(rows)} graphic sources; no images downloaded")
        return
    archive_url = cfg["ecfr"].get("graphics_url")
    archive = zipfile.ZipFile(io.BytesIO(get(archive_url))) if archive_url else None
    archive_names = {Path(name).name: name for name in archive.namelist()} if archive else {}
    for row in rows:
        local = asset_dir / row["filename"]
        if not local.exists():
            try:
                source_name = Path(urllib.parse.urlparse(row["src"]).path).name
                if archive:
                    if source_name not in archive_names:
                        raise ValueError(f"Graphic is missing from archive: {source_name}")
                    content = archive.read(archive_names[source_name])
                    row["url"] = f"{archive_url}#{source_name}"
                else:
                    content = get(row["url"])
                if not content.startswith((b"GIF87a", b"GIF89a", b"\x89PNG", b"\xff\xd8")):
                    raise ValueError(f"Image endpoint returned non-image content: {row['url']}")
                local.write_bytes(content)
            except Exception as error:
                row["download_error"] = str(error)
                write_json(path, rows)
                raise
        if row["sha256"] is not None and row["sha256"] != file_hash(local):
            raise ValueError("Downloaded asset hash mismatch")
        row["sha256"] = file_hash(local)
        row.setdefault("retrieved_at", datetime.now(timezone.utc).isoformat())
        row.pop("download_error", None)
        write_json(path, rows)
    print(f"{len(rows)} graphics inventoried; {sum(r['review_status'] == 'verified' for r in rows)} verified")


def build(cfg, refresh=False, rebuild=False):
    directory = Path(cfg["experiment_dir"])
    snapshot = freeze_snapshot(cfg, refresh)
    if (directory / "corpus.lock.json").exists():
        if not rebuild:
            return audit(cfg)
        if (directory / "benchmark.lock.json").exists() or (directory / "training").exists():
            raise ValueError("Cannot rebuild after benchmark freeze or production model pin; use a new experiment directory")
        previous = read_json(directory / "corpus.lock.json")
        archive = directory / "corpus_versions" / previous["documents_sha256"]
        archive.mkdir(parents=True, exist_ok=True)
        for name in ("documents.jsonl", "inventory.jsonl", "corpus.lock.json", "parse_issues.json", "audit.json", "dataset_card.md"):
            if (directory / name).exists():
                shutil.copy2(directory / name, archive / name)
    exclude_image_sections = cfg["ecfr"].get("exclude_image_sections", False)
    records, inventory, issues = parse((directory / "source.xml").read_bytes(), snapshot, asset_transcriptions(directory), exclude_image_sections)
    write_rows(directory / "documents.jsonl", records)
    write_rows(directory / "inventory.jsonl", inventory)
    write_json(directory / "parse_issues.json", issues)
    write_json(directory / "corpus.lock.json", {"source_sha256": snapshot["sha256"], "documents_sha256": file_hash(directory / "documents.jsonl"),
                                              "inventory_sha256": file_hash(directory / "inventory.jsonl"),
                                              "exclude_image_sections": exclude_image_sections})
    return audit(cfg)


def audit(cfg):
    directory = Path(cfg["experiment_dir"])
    snapshot = read_json(directory / "snapshot.json")
    lock = read_json(directory / "corpus.lock.json")
    issues = []
    raw = (directory / "source.xml").read_bytes()
    validate_xml(raw, cfg)
    if snapshot["scope"] != scope(cfg):
        issues.append({"reason": "scope_mismatch"})
    if digest(raw) != snapshot["sha256"] or digest(raw) != lock["source_sha256"]:
        issues.append({"reason": "source_hash_mismatch"})
    exclude_image_sections = cfg["ecfr"].get("exclude_image_sections", False)
    documents, inventory, parsed_issues = parse(raw, snapshot, asset_transcriptions(directory), exclude_image_sections)
    issues.extend(parsed_issues)
    if lock.get("exclude_image_sections", False) != exclude_image_sections:
        issues.append({"reason": "corpus_policy_mismatch", "setting": "exclude_image_sections"})
    for name, expected, key in [("documents.jsonl", documents, "documents_sha256"), ("inventory.jsonl", inventory, "inventory_sha256")]:
        if read_rows(directory / name) != expected or file_hash(directory / name) != lock[key]:
            issues.append({"reason": "derived_content_mismatch", "file": name})
    duplicates = defaultdict(list)
    for d in documents:
        duplicates[d["content_sha256"]].append(d["citation"])
    report = {"passed": not issues, "scope": snapshot["scope"], "date": snapshot["date"], "issues": issues,
              "documents": len(documents), "inventory_elements": len(inventory),
              "reserved": sum(d["reserved"] for d in documents), "body_characters": sum(len(d["text"]) for d in documents),
              "duplicates_retained": [v for v in duplicates.values() if len(v) > 1],
              "lengths": [{"id": d["id"], "characters": len(d["text"])} for d in documents],
              "token_counts": read_json(directory / "segments.lock.json") if (directory / "segments.lock.json").exists() else "Not tokenized yet"}
    write_json(directory / "audit.json", report)
    samples = sorted(documents, key=lambda d: len(d["text"]), reverse=True)[:3]
    card = f"# {scope(cfg)} continued-pretraining corpus\n\nSnapshot: {snapshot['date']}\n\nAudit passed: {report['passed']}\n\nDocuments: {len(documents)}; reserved: {report['reserved']}.\n\n"
    card += "All body text is retained, including short provisions and duplicate text at distinct citations. Authority/source metadata are retained separately. Empty reserved entries have no body tokens. Literal paragraph labels and XML structure are preserved; full paragraph paths are not inferred.\n\nSeen regulations, unseen evaluation questions. Public source exposure in the original model is unknown.\n\n"
    card += "\n\n".join(f"## {d['citation']}\n\n{d['text'][:1800]}" for d in samples)
    (directory / "dataset_card.md").write_text(card, encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("passed", "documents", "inventory_elements", "body_characters")}, indent=2))
    return report


def require_audit(cfg):
    report = audit(cfg)
    if not report["passed"]:
        raise ValueError("Corpus audit failed; see audit.json. Training is blocked.")
    return report
