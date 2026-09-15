# Design Decisions: eCFR Title 12 Post-Training Pipeline

This document explains the engineering decisions behind the pipeline, in the
order they would come up in a technical review.

## 1. Data engineering

**Source of truth.** A point-in-time snapshot of eCFR Title 12, Chapter I
(OCC), pinned to `2025-01-01` in `config.yaml`. Pinning the date makes every
run byte-reproducible; the raw XML and every derived file are SHA-256 hashed
into manifests.

**Extraction.** Only regulation body paragraphs (`P`, `FP` tags) are kept;
amendment notes, authority citations, and source credits are excluded so the
model never trains on editorial metadata.

**Template QA, not LLM-generated QA.** Each usable section becomes up to four
example types: `overview`, `provisions`, `citation_lookup`, `definition`.
Answers are extractive templates built directly from the regulation text.
Trade-off: formulaic language, but zero dataset hallucinations, guaranteed
correct citations, and full reproducibility with no external model dependency.

**Filtering.** Reserved sections, headingless sections, sections under 200
chars, and byte-identical duplicates are dropped.

**The unique-heading rule.** `citation_lookup` questions ("Which section
addresses 'X'?") are only generated when the heading is unique in the corpus.
Repeated headings ("Definitions", "Purpose") would create ambiguous questions
and, worse, leak identical prompts across splits.

## 2. Leakage guards

Splits are assigned per *section* via a stable md5 bucket (8/1/1), so every
question derived from a section stays in one split. Two assertions run at
build time and abort the build on violation:

1. No section id appears in both train and test.
2. No identical user prompt appears in both train and test.

The hash is machine-independent (md5 of the section id, not `hash()`), so the
split is identical on any machine — no `PYTHONHASHSEED` dependence.

## 3. Training stages

All stages share one LoRA config (r=16, alpha=32, all attention + MLP
projections) so comparisons isolate the training method, not adapter capacity.

**SFT** trains a fresh LoRA on 2,539 chat examples (loss over the full
sequence; Llama's chat template lacks `{% generation %}` support for
completion-only masking, and prompts are short/templated so the cost is low).

**DPO** initializes from the SFT adapter and uses it *twice*: as the trainable
policy and, frozen under a second adapter name, as the reference policy. This
keeps one base model in memory instead of two. KL is therefore measured
against SFT, not the raw base — the comparison isolates what preference
tuning adds on top of the same supervised start.

Preference pairs are seeded corruptions of known-good answers:
- `wrong_section` — a confident answer about a different section
- `wrong_citation` — the right answer with its citation swapped
- `vague` — truncated answer with the citation removed

Each corruption targets a failure mode we can actually measure at eval time
(wrong-citation rate, citation accuracy, grounding overlap).

**GRPO** also initializes from SFT (not DPO), so DPO and GRPO are parallel
ablations of "what does each method add over SFT". Rewards are verifiable
functions of the data — no reward model:
- citation reward: +1 correct §, −0.5 confidently wrong, capped at 0.25 when
  more than 3 sections are cited (anti-spam)
- grounding reward: token-F1 against the reference answer
- brevity/repetition penalty: −0.3 over 320 words, −0.5 when one token
  exceeds 15% of the output (anti reward-hacking guard)

`beta=0` (no KL anchor) is a deliberate risk: rewards are verifiable, and the
guard rewards bound degenerate strategies, but generations are manually
inspected post-eval before publishing.

## 4. Runtime policy

Device is resolved at runtime: CUDA → bf16 LoRA, dropping to 4-bit QLoRA +
paged 8-bit AdamW below 40 GB VRAM; Apple Silicon (MPS) → un-quantized LoRA,
eager attention; CPU → float32, smoke runs only. One code path, no forks.

A `--smoke` mode swaps in a 135M model and runs every stage for a few steps —
the full loop (including Hub uploads) is validated end-to-end for cents before
committing to the 8B run.

## 5. Evaluation protocol

Greedy decoding, fixed seed, 200 held-out examples, four checkpoints (base,
SFT, DPO, GRPO) on identical prompts. Generations are cached per checkpoint so
metrics can be re-scored without re-running inference. Regressions vs base are
flagged in the report, never hidden.

Metrics: citation accuracy, wrong-citation rate (lower is better), token-F1,
ROUGE-L — overall and per question type.

### Known limitations (deliberately documented)

- Most prompts already contain the target citation, so citation accuracy
  partially measures "repeats the supplied citation"; only `citation_lookup`
  genuinely tests retrieval.
- Lexical overlap (F1/ROUGE) is not factual accuracy; a paraphrase scores low
  and a subtly wrong number can score high.
- Test questions share templates with training — sections are held out,
  phrasings are not.
- No claim-level grounding check, no abstention evaluation, no confidence
  intervals yet.

## 6. What I would do next

1. **RAG for production** — regulations change; retrieval over dated eCFR text
   plus citation to the retrieved passage beats parametric memory for
   auditability.
2. **Human-written eval set** — paraphrased questions, counterfactual premises,
   post-snapshot amendments (temporal holdout), out-of-chapter questions.
3. **Paired bootstrap CIs** and multi-seed runs before claiming a method wins.
4. **Claim-level groundedness scoring** with a calibrated judge model.
