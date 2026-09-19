# eCFR Fine-Tuning Dataset Card

- Source: eCFR Title 12 (full title)
- Snapshot date (point-in-time): 2025-01-01
- Seed: 42  |  Config hash: 8344d7a30f48

## Filtering
- Dropped reserved sections and sections with < 200 chars of text.
- Deduplicated sections with byte-identical text.
- citation_lookup questions only for sections with a unique heading
  (repeated headings would be ambiguous and leak prompts across splits).

## Split (by section, stable md5 bucket — no cross-split section leakage)

| split | sections | sft examples |
|---|---|---|
| train | 5187 | 14722 |
| val | 638 | 1797 |
| test | 680 | 1959 |

## Train question types

- citation_lookup: 2668
- definition: 1680
- overview: 5187
- provisions: 5187

## DPO pairs: 10374 (train sections only)

- vague: 3468
- wrong_citation: 3407
- wrong_section: 3499

## Eval set: 200 examples (capped at 200)

## Artifact hashes (sha256)

- sft_train.jsonl: `694f083e8822eb63a033ba72e76701e99e6713b2d2677c6a978e927f24f59c75`
- sft_val.jsonl: `684851a29f380972486716f17c76f5415e7158fa130c698e737a3ba2a309cce2`
- dpo_train.jsonl: `f0a379b0deb7c18f41901ecc07c78852555c2534bfa10ec41dc97b2fce83f09a`
- eval_test.jsonl: `864dfeefae65d3e0cf1f5ce25912cdeb45b873f6246dc08736deb68d2de42550`
