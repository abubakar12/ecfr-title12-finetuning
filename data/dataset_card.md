# eCFR Fine-Tuning Dataset Card

- Source: eCFR Title 12, Chapter I
- Snapshot date (point-in-time): 2025-01-01
- Seed: 42  |  Config hash: 98e9d30f648f

## Filtering
- Dropped reserved sections and sections with < 200 chars of text.
- Deduplicated sections with byte-identical text.
- citation_lookup questions only for sections with a unique heading
  (repeated headings would be ambiguous and leak prompts across splits).

## Split (by section, stable md5 bucket — no cross-split section leakage)

| split | sections | sft examples |
|---|---|---|
| train | 803 | 2539 |
| val | 102 | 318 |
| test | 101 | 314 |

## Train question types

- citation_lookup: 652
- definition: 281
- overview: 803
- provisions: 803

## DPO pairs: 1606 (train sections only)

- vague: 522
- wrong_citation: 546
- wrong_section: 538

## Eval set: 200 examples (capped at 200)

## Artifact hashes (sha256)

- sft_train.jsonl: `4682001475f0c846b2c13161d447312dc33d3e5c2dd579bc84091f182750f093`
- sft_val.jsonl: `4396d2bb80cba565b16bd043b7a730da15acb6c921b08b414727013890640bd8`
- dpo_train.jsonl: `090b1ca70212597b122486f3cf1d2a7f13b0481f24ee9545db591b38a29e61c8`
- eval_test.jsonl: `96d109e5737d4dcd13d2e1d6dc50aed2b6a3ece78c46bd2d411e7e10546ba816`
