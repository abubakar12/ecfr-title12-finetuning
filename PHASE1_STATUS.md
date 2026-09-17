# Phase 1 implementation and run status

The new pipeline is implemented separately from the historical SFT/DPO/GRPO experiment. This is **not a completed 8B training experiment**.

## Frozen source and corpus

- Snapshot: **2026-09-15**, Title 12, Chapter I.
- Original response SHA-256: `98ffa09e23ce7836f3e0aa2b9fee4be88f0ed7a5ec08d4c0578dadb9f4e11a4c`.
- The API returned the full title; Chapter I is selected locally and excluded chapters are inventoried.
- Canonical records: **1,148** (1,116 sections and 32 appendices), including 45 reserved entries.
- Canonical body text: **4,564,666 characters**, before image transcriptions.
- All text/table rendering checks pass. The audit intentionally remains **failed** because **81 embedded graphics** need verified text transcriptions.
- The graphics endpoint returned a Federal Register access-challenge HTML page rather than an image. That response was rejected and was not stored as source material.

Artifacts are under `experiments/title12-chapter-I-v1/`: `snapshot.json`, `source.xml`, `documents.jsonl`, `inventory.jsonl`, `audit.json`, `dataset_card.md`, and the 81-entry `transcriptions.json` inventory. Derived pre-review revisions are preserved in `corpus_versions/`.

## Evaluation material

`benchmark_candidates.jsonl` contains 100 independently worded prompts with source passages: 40 factual, 30 application, 20 policy, and 10 outside scope. **These are candidates, not gold labels.** In-scope reference answers and claim rubrics remain to be authored and verified. No benchmark has been frozen, and no accuracy results have been generated.

## Validation performed

- **18 unit tests passed**, covering source preservation, snapshot tampering, chapter filtering, appendix citations, reviewed graphics, segmentation, padding/overlap masks, benchmark freeze validation, blinded review integrity, and paired scoring.
- The actual SmolLM2 135M Instruct smoke run completed two optimizer steps on an explicitly synthetic fixture. Training loss was **5.307999**, and reloaded-adapter loss was **5.272771**, both finite.
- Smoke model revision: `12fd25f77366fa6b3b4b768ec3050bf629380bac`.
- Adapter weights, tokenizer, optimizer, scheduler, RNG state, and manifests were saved under `experiments/synthetic-training-smoke-v2/smoke/` (the earlier v1 run is also preserved). This synthetic run does not satisfy the real experiment's corpus gate.
- Training dependencies were installed in `.venv-phase1`; pinned direct and resolved Windows dependency lists are checked in.

## Remaining prerequisites for the actual experiment

1. Obtain and verify text transcriptions for all 81 graphics, rebuild the derived corpus, and pass its audit.
2. Complete and verify the 100 answer keys and claim rubrics, then freeze the benchmark.
3. Run the corpus smoke check and maximum-length BF16 check on the A40 with access to the official Meta model, then perform the full one-epoch run.
4. Generate both checkpoints' responses, complete blinded semantic review, and score the paired comparison.

No A40 is connected to this workspace. No production adapter or base-versus-CPT performance claim is supplied. See `PHASE1.md` for the commands and artifact contracts.
