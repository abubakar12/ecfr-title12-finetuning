"""Run the real tiny-model training/reload check on an explicitly synthetic corpus.

This tests the training machinery even when the real corpus audit is blocked.
It does NOT satisfy the production experiment's corpus or smoke gate.
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ecfr_pipeline import corpus, cpt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", default="experiments/synthetic-training-smoke-v1")
    args = parser.parse_args()
    cfg = corpus.read_json(Path("phase1.json"))
    cfg["experiment_dir"] = args.experiment_dir
    directory = Path(cfg["experiment_dir"])
    if (directory / "snapshot.json").exists():
        raise ValueError("Use a new synthetic experiment directory; existing artifacts are preserved")
    directory.mkdir(parents=True, exist_ok=True)
    raw = b'<ECFR><DIV1 TYPE="TITLE" N="12"><HEAD>SYNTHETIC TEST FIXTURE</HEAD><DIV3 TYPE="CHAPTER" N="I"><HEAD>Not actual regulation</HEAD><DIV5 TYPE="PART" N="1"><HEAD>Fixture</HEAD><DIV8 TYPE="SECTION" N="1.1"><HEAD>Training test</HEAD><P>(a) This invented text tests tokenization and adapter persistence only.</P><P>(b) It is not legal source material and must never enter the research corpus.</P></DIV8></DIV5></DIV3></DIV1></ECFR>'
    snapshot = {"title": 12, "chapter": "I", "date": "2026-09-15", "scope": corpus.scope(cfg), "sha256": corpus.digest(raw), "source_kind": "synthetic_test_fixture"}
    (directory / "source.xml").write_bytes(raw)
    corpus.write_json(directory / "snapshot.json", snapshot)
    docs, inventory, issues = corpus.parse(raw, snapshot)
    corpus.write_rows(directory / "documents.jsonl", docs)
    corpus.write_rows(directory / "inventory.jsonl", inventory)
    corpus.write_json(directory / "corpus.lock.json", {"source_sha256": snapshot["sha256"],
                      "documents_sha256": corpus.file_hash(directory / "documents.jsonl"),
                      "inventory_sha256": corpus.file_hash(directory / "inventory.jsonl")})
    cpt.train(cfg, smoke=True)


if __name__ == "__main__":
    main()
