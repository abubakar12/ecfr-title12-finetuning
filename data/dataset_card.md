# eCFR Fine-Tuning Dataset Card

- Source: eCFR Title 12 (full title)
- Snapshot date (point-in-time): 2025-01-01
- Seed: 42  |  Config hash: a5b753ded5f2

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

## Coverage by chapter

| chapter | agency | sections | sft train | sft val | eval |
|---|---|---|---|---|---|
| I | Comptroller Of The Currency, Department Of The Treasury | 1006 | 2293 | 285 | 30 |
| X | Consumer Financial Protection Bureau | 595 | 1322 | 182 | 29 |
| II | Federal Reserve System | 1431 | 3231 | 405 | 30 |
| IV | Export-Import Bank Of The United States | 88 | 185 | 32 | 30 |
| VI | Farm Credit Administration | 636 | 1471 | 167 | 30 |
| XI | Federal Financial Institutions Examination Council | 59 | 139 | 17 | 12 |
| XV | Department Of The Treasury | 26 | 67 | 5 | 3 |
| III | Federal Deposit Insurance Corporation | 1038 | 2194 | 286 | 30 |
| VII | National Credit Union Administration | 628 | 1445 | 183 | 30 |
| XII | Federal Housing Finance Agency | 721 | 1725 | 174 | 29 |
| XIV | Farm Credit System Insurance Corporation | 69 | 160 | 13 | 9 |
| XVI | Office Of Financial Research, Department Of The Treasury | 5 | 26 | 0 | 0 |
| VIII | Federal Financing Bank | 15 | 39 | 9 | 0 |
| XIII | Financial Stability Oversight Council | 31 | 75 | 7 | 6 |
| XVII | Office Of Federal Housing Enterprise Oversight, Department Of Housing And Urban Development | 15 | 40 | 3 | 3 |
| XVIII | Community Development Financial Institutions Fund, Department Of The Treasury | 142 | 310 | 29 | 29 |

## Continued-pretraining corpus (train + val sections only; test sections stay held out)

- cpt_train.jsonl: 5187 documents, 3,032,566 words
- cpt_val.jsonl: 638 documents, 333,270 words

## Train question types

- citation_lookup: 2668
- definition: 1680
- overview: 5187
- provisions: 5187

## DPO pairs: 10374 (train sections only)

- vague: 3468
- wrong_citation: 3407
- wrong_section: 3499

## Eval set: 300 examples (capped at 300, round-robin across chapters)

## Artifact hashes (sha256)

- sft_train.jsonl: `2da3a002f85c9470cf12e99218b6412111af803b5bef4ca0a05327c74d2fc75f`
- sft_val.jsonl: `8f27f041a318caedc52b87056af21f13ad3396d6e05c389a07d95624b0bafa37`
- dpo_train.jsonl: `9ba293236940715ab314fdfcecd8561ca8838868e4364a6c7e16565000285d60`
- eval_test.jsonl: `add7c31229e76b8708818cb60b3728a79d3cc789b41237fad453f98ada1218ee`
- cpt_train.jsonl: `02368fc8eb80674ae13306d91400762de9974e4e44256114fabd34893a15b3ae`
- cpt_val.jsonl: `4e68c07bb2596208d2d4ec11dca392fa0fb46b178d26d9fd0589211730a286b2`
