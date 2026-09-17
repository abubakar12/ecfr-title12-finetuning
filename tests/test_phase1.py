import copy
import json
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ecfr_pipeline import corpus, cpt, benchmark


XML = b'''<ECFR><DIV1 TYPE="TITLE" N="12"><HEAD>Banks</HEAD><DIV3 TYPE="CHAPTER" N="I"><HEAD>OCC</HEAD><DIV5 TYPE="PART" N="1"><HEAD>Investments</HEAD><AUTH>Authority text</AUTH><DIV8 TYPE="SECTION" N="1.1"><HEAD>Scope</HEAD><P>(a) First <E>requirement</E>.</P><P>(1) Nested.</P><P>(b) Second.</P><P>(1) Another nested.</P><GPOTABLE><BOXHD><CHED>Type</CHED><CHED>Limit</CHED></BOXHD><ROW><ENT>Cash</ENT><ENT>Ten</ENT></ROW><TNOTE>Note one</TNOTE></GPOTABLE><FTNT><P>Footnote text.</P></FTNT><SOURCE>Amendment</SOURCE></DIV8><DIV8 TYPE="SECTION" N="1.2"><HEAD>Short</HEAD><P>See section 1.1.</P></DIV8><DIV8 TYPE="SECTION" N="1.3"><HEAD>[Reserved]</HEAD></DIV8><DIV9 TYPE="APPENDIX" N="Appendix A"><HEAD>Appendix A to Part 1</HEAD><P>Appendix body.</P></DIV9></DIV5></DIV3></DIV1></ECFR>'''
SNAPSHOT = {"title": 12, "date": "2026-09-15", "scope": "Title 12, Chapter I", "sha256": corpus.digest(XML)}


class CharTokenizer:
    bos_token_id, eos_token_id = 1, 2
    def encode(self, text, add_special_tokens=False):
        return [ord(c) + 10 for c in text]
    def __call__(self, text, **kwargs):
        return {"input_ids": self.encode(text), "offset_mapping": [(i, i + 1) for i in range(len(text))]}


class CorpusTests(unittest.TestCase):
    def test_scope_filter_and_html_div(self):
        raw = XML.replace(b'<GPOTABLE>', b'<DIV><GPOTABLE>').replace(b'</GPOTABLE>', b'</GPOTABLE></DIV>')
        raw = raw.replace(b'</DIV1>', b'<DIV3 TYPE="CHAPTER" N="II"><HEAD>Other</HEAD><DIV8 TYPE="SECTION" N="200.1"><HEAD>Other rule</HEAD><P>Not in pilot.</P></DIV8></DIV3></DIV1>')
        docs, inventory, issues = corpus.parse(raw, {**SNAPSHOT, "chapter": "I"})
        self.assertFalse(issues)
        self.assertEqual(len(docs), 4)
        self.assertIn("Cash", docs[0]["text"])
        self.assertEqual(sum(i["classification"] == "excluded_scope" for i in inventory), 1)
        self.assertEqual(len(corpus.parse(raw, {**SNAPSHOT, "chapter": None})[0]), 5)

    def test_reviewed_image_transcription(self):
        raw = XML.replace(b'<P>See section 1.1.</P>', b'<img src="/graphics/formula.gif"/>')
        reviewed = {"/graphics/formula.gif": {"text": "x = y + z", "review_status": "verified", "reviewer": "test", "sha256": "test"}}
        docs, _, issues = corpus.parse(raw, SNAPSHOT, reviewed)
        self.assertFalse(issues)
        self.assertEqual(docs[1]["text"], "x = y + z")
        self.assertIn('<img', docs[1]["blocks"][0]["xml"])

    def test_full_structure(self):
        docs, inventory, issues = corpus.parse(XML, SNAPSHOT)
        self.assertEqual(issues, [])
        self.assertEqual(len(docs), 4)
        self.assertIn("Cash | Ten", docs[0]["text"])
        self.assertIn("Footnote text.", docs[0]["text"])
        self.assertEqual(docs[0]["metadata"][0]["text"], "Amendment")
        self.assertEqual(docs[0]["repeated_literal_labels"], [["1"]])
        self.assertIn("<E>requirement</E>", docs[0]["blocks"][0]["xml"])
        self.assertTrue(docs[2]["reserved"])
        self.assertEqual(docs[3]["kind"], "APPENDIX")
        self.assertTrue(any(i["classification"] == "hierarchy_metadata" for i in inventory))
        self.assertEqual(corpus.parse(XML, SNAPSHOT), (docs, inventory, issues))

    def test_full_appendix_identity_not_duplicated(self):
        raw = XML.replace(b'N="Appendix A"', b'N="Appendix A to Part 1"')
        docs = corpus.parse(raw, SNAPSHOT)[0]
        self.assertEqual(docs[-1]["citation"], "12 CFR Appendix A to Part 1")

    def test_unknown_hierarchy_is_blocked(self):
        raw = XML.replace(b'TYPE="SECTION" N="1.2"', b'TYPE="UNKNOWN" N="1.2"')
        self.assertIn("unknown_hierarchy_type", {i["reason"] for i in corpus.parse(raw, SNAPSHOT)[2]})

    def test_unknown_and_images_fail_closed(self):
        raw = XML.replace(b"<P>See section 1.1.</P>", b"<ALIEN>Missing structure</ALIEN><GPH/> ")
        issues = corpus.parse(raw, SNAPSHOT)[2]
        self.assertIn("unknown_structure", {i["reason"] for i in issues})
        self.assertIn("image_requires_transcription", {i["reason"] for i in issues})

    def test_download_validation(self):
        cfg = {"ecfr": {"title": 12, "chapter": "I"}}
        corpus.validate_xml(XML, cfg)
        for raw in (b"<html>Error</html>", b"not xml", XML.replace(b'N="12"', b'N="13"')):
            with self.assertRaises(Exception):
                corpus.validate_xml(raw, cfg)
        cfg["ecfr"]["chapter"] = None
        corpus.validate_xml(XML, cfg)

    def test_snapshot_freeze_and_tamper(self):
        with tempfile.TemporaryDirectory() as temp:
            cfg = {"experiment_dir": temp, "ecfr": {"title": 12, "chapter": "I", "date": "latest"}}
            metadata = json.dumps({"titles": [{"number": 12, "up_to_date_as_of": "2026-09-15"}]}).encode()
            with patch.object(corpus, "get", side_effect=[metadata, XML]) as get:
                self.assertTrue(corpus.build(cfg)["passed"])
                corpus.build(cfg)
                self.assertEqual(get.call_count, 2)
            with self.assertRaises(ValueError):
                corpus.freeze_snapshot(cfg, refresh=True)
            path = Path(temp) / "documents.jsonl"
            path.write_text("{}\n")
            self.assertFalse(corpus.audit(cfg)["passed"])


class SegmentTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("torch"), "Training stack not installed")
    def test_padding_masks_only_added_padding(self):
        batch = cpt.Collator(2)([{"input_ids": [1, 5, 2], "labels": [-100, 5, 2]}, {"input_ids": [1, 2], "labels": [-100, 2]}])
        self.assertEqual(batch["labels"].tolist(), [[-100, 5, 2], [-100, 2, -100]])
        self.assertEqual(batch["attention_mask"].tolist(), [[1, 1, 1], [1, 1, 0]])
    def test_overlap_tail_and_long_paragraph(self):
        docs = corpus.parse(XML, SNAPSHOT)[0]
        tok = CharTokenizer()
        for document in docs:
            # Force splitting despite a relatively long synthetic header.
            rows = list(cpt.segment_document(document, tok, 190, 15))
            cpt.verify_segments([document], rows, tok, 190)
            self.assertTrue(all(len(r["input_ids"]) <= 190 for r in rows))
        document = copy.deepcopy(docs[1])
        document["text"] = "x" * 2000
        document["blocks"] = [{"start": 0, "end": 2000, "source_location": "/long"}]
        rows = list(cpt.segment_document(document, tok, 190, 15))
        self.assertGreater(len(rows), 10)
        self.assertEqual(cpt.verify_segments([document], rows, tok, 190), 2000)
        rows[-1]["new_body_token_start"] -= 1
        with self.assertRaises(ValueError):
            cpt.verify_segments([document], rows, tok, 190)

    def test_whole_short_section(self):
        document = corpus.parse(XML, SNAPSHOT)[0][1]
        rows = list(cpt.segment_document(document, CharTokenizer()))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["overlap_tokens"], 0)

    def test_header_budget(self):
        with self.assertRaises(ValueError):
            list(cpt.segment_document(corpus.parse(XML, SNAPSHOT)[0][0], CharTokenizer(), 20, 10))


class CitationTests(unittest.TestCase):
    def test_candidate_counts_and_no_answers_leaked(self):
        from ecfr_pipeline import benchmark_seed as seed
        self.assertEqual([len(seed.FACTUAL), len(seed.APPLICATION), len(seed.POLICY), len(seed.OUTSIDE)], [40, 30, 20, 10])
        questions = [q for _, q in seed.FACTUAL + seed.APPLICATION + seed.POLICY] + seed.OUTSIDE
        self.assertEqual(len(set(questions)), 100)
        self.assertTrue(all(not benchmark.citations(q) for q in questions))

    def test_wrong_title_suffix_and_extra(self):
        gold = {"acceptable_citations": ["12 CFR § 1.1(a)"], "granularity": "paragraph"}
        answer = "12 CFR § 1.1(a); 12 CFR § 1.1(b); 13 CFR § 1.1; 12 CFR § 999.1"
        result = benchmark.structural_scores(answer, gold, {"12 CFR § 1.1"})
        self.assertEqual(result["matched"], ["12 CFR § 1.1(a)"])
        self.assertEqual(len(result["unmatched"]), 3)
        self.assertEqual(len(result["absent_from_corpus"]), 2)
        self.assertEqual(result["wrong_title_citations"], ["13 CFR § 1.1"])

    def test_section_does_not_validate_invented_paragraph(self):
        result = benchmark.structural_scores("12 CFR § 1.1(z)", {"acceptable_citations": ["12 CFR § 1.1"], "granularity": "section"}, {"12 CFR § 1.1"})
        self.assertEqual(result["matched"], [])

    def test_bootstrap(self):
        self.assertEqual(benchmark.paired_interval([(0, 1)] * 5, samples=100), {"delta": 1, "lower": 1, "upper": 1})


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.cfg = {"experiment_dir": self.temp.name, "seed": 42, "ecfr": {"title": 12, "chapter": "I", "date": "latest"}, "evaluation": {"bootstrap_samples": 100}}
        metadata = json.dumps({"titles": [{"number": 12, "up_to_date_as_of": "2026-09-15"}]}).encode()
        with patch.object(corpus, "get", side_effect=[metadata, XML]):
            corpus.build(self.cfg)
        document = corpus.read_rows(self.directory / "documents.jsonl")[0]
        self.rows = []
        for kind, count in benchmark.COUNTS.items():
            for index in range(count):
                outside = kind == "outside_scope"
                self.rows.append({"id": f"{kind}-{index}", "type": kind, "question": f"Fixture {kind} query {index}",
                                  "reference_answer": "A reviewed answer", "required_claims": ["A required claim"],
                                  "acceptable_citations": [] if outside else [document["citation"]],
                                  "granularity": "none" if outside else "section", "scope_reason": "Outside fixture",
                                  "evidence": [] if outside else [{"document_id": document["id"], "start": 0, "end": 3, "quote": document["text"][:3]}],
                                  "reviewer": "fixture-reviewer", "review_status": "verified"})
        self.source = self.directory / "input.jsonl"
        corpus.write_rows(self.source, self.rows)

    def test_freeze_requires_review_and_exact_evidence(self):
        self.rows[0]["review_status"] = "pending"
        corpus.write_rows(self.source, self.rows)
        with self.assertRaisesRegex(ValueError, "source verification"):
            benchmark.freeze(self.cfg, self.source)
        self.rows[0]["review_status"] = "verified"
        self.rows[0]["evidence"][0]["quote"] = "wrong"
        corpus.write_rows(self.source, self.rows)
        with self.assertRaisesRegex(ValueError, "Evidence does not match"):
            benchmark.freeze(self.cfg, self.source)

    def test_freeze_and_score_extra_citation_failure(self):
        benchmark.freeze(self.cfg, self.source)
        self.assertEqual(len(benchmark.verify_frozen(self.cfg)), 100)
        output = self.directory / "evaluation"
        blind, mapping, reviews = [], [], []
        for q in self.rows:
            for checkpoint in ("base", "cpt"):
                review_id = f"r{len(blind)}"
                packet = {k: q[k] for k in ("question", "required_claims", "evidence", "acceptable_citations", "granularity")}
                packet.update(review_id=review_id, question_id=q["id"], answer="12 CFR § 1.1" + (" and 12 CFR § 999.1" if checkpoint == "base" else ""))
                blind.append(packet)
                mapping.append({"review_id": review_id, "checkpoint": checkpoint, "question_id": q["id"]})
                reviewed = {**packet, "reviewer": "tester", "supported_claims": 1, "correct_citation_count": 1,
                            "total_citation_count": 2 if checkpoint == "base" else 1,
                            "nonexistent_citation_count": 1 if checkpoint == "base" else 0,
                            "substantively_correct": True, "outside_scope_correct": True,
                            "section_citations_correct": checkpoint == "cpt", "paragraph_citations_correct": True}
                reviews.append(reviewed)
        corpus.write_rows(output / "blind_review.jsonl", blind)
        corpus.write_rows(output / "private_mapping.jsonl", mapping)
        corpus.write_rows(output / "generations.jsonl", [])
        corpus.write_json(output / "generation.lock.json", {"blind_sha256": corpus.file_hash(output / "blind_review.jsonl"),
                          "mapping_sha256": corpus.file_hash(output / "private_mapping.jsonl"),
                          "generations_sha256": corpus.file_hash(output / "generations.jsonl")})
        corpus.write_rows(output / "reviews.jsonl", reviews)
        benchmark.score(self.cfg, output / "reviews.jsonl")
        report = corpus.read_json(output / "results.json")
        self.assertEqual(report["base"]["citation_accuracy"], 0)
        self.assertEqual(report["cpt"]["citation_accuracy"], 1)
        self.assertEqual(report["paired_primary_95_percent_ci"]["delta"], 1)
        reviews[0]["answer"] = "Changed answer"
        corpus.write_rows(output / "reviews.jsonl", reviews)
        with self.assertRaisesRegex(ValueError, "was changed"):
            benchmark.score(self.cfg, output / "reviews.jsonl")


if __name__ == "__main__":
    unittest.main()
