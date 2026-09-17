# eCFR continued-pretraining experiment

Phase 1 compares unchanged Llama 3.1 8B Instruct with a LoRA adapter trained by next-token prediction on the regulatory corpus. It does not train on QA pairs. All selected regulations are seen during training; the evaluation questions are unseen. The base model's prior exposure to public regulations is unknown.

## Data and reproducibility

Use Python 3.11 or 3.12. The data stages use only the standard library. The existing SFT/DPO/GRPO pipeline and historical data are unchanged.

```bash
python training_models_v1.py build-corpus --config phase1.json
python training_models_v1.py audit-corpus --config phase1.json
```

`phase1.json` controls the experiment directory, snapshot, scope, model, and training settings. `ecfr.chapter: "I"` selects the OCC pilot; `null` selects all of Title 12. Change `experiment_dir` when changing scope or refreshing a frozen snapshot. The upstream API may return extra chapters; the parser enforces the configured scope and records excluded subtrees.

The experiment directory contains original XML, snapshot metadata, canonical documents, an element inventory, the audit, and a dataset card. Each body block retains its original XML, literal paragraph labels, and source location. Parent headings supply the regulatory hierarchy. Authority/source notes are metadata rather than pretraining body text. Short substantive sections and identical text under distinct citations are retained. Empty reserved sections remain in the corpus inventory but contribute no body tokens.

Unknown XML structures, image content requiring transcription, text lost in table rendering, or hash mismatches block training. Inspect `audit.json`; fix and test the parser before creating a new corpus version. Never hand-edit the derived JSONL to bypass the audit. Full paragraph citation paths are not inferred from ambiguous flat labels: evidence uses exact source offsets and reviewer-verified citation annotations.

Use `fetch-assets` to inventory and download embedded graphics. The generated `transcriptions.json` retains source URLs and asset hashes. Supply faithfully transcribed text and a reviewer identity, then mark individual entries `verified` only after checking the image. Run `build-corpus --rebuild-corpus` to incorporate verified transcriptions; this archives the earlier derived corpus and is only allowed before benchmark freeze or production tokenization. Source XML remains unchanged. Non-image responses, including access-challenge pages, are rejected.

## Freeze evaluation before training

Prepare a JSONL file with exactly 40 factual, 30 application, 20 policy, and 10 outside-scope items. Independently word prompts without target citations. Each item uses this interface:

`python training_models_v1.py draft-benchmark` creates 100 Chapter I candidate prompts with exact source passages. These intentionally have pending review status and incomplete answer keys. Complete and verify them before freezing; the freezer rejects pending or empty answer keys.

```json
{
  "id": "factual-001",
  "type": "factual",
  "question": "The question, without a target section number",
  "reference_answer": "Source-checked answer",
  "required_claims": ["One independently assessable required claim"],
  "acceptable_citations": ["12 CFR § 1.1"],
  "granularity": "section",
  "evidence": [{"document_id": "exact id from documents.jsonl", "start": 0, "end": 25, "quote": "exact substring of text"}],
  "review_status": "verified",
  "reviewer": "identity of the person or agent who actually verified this item"
}
```

Offsets are Python character offsets into the canonical document `text`, not byte offsets or offsets into the XML. For paragraph questions, use `granularity: "paragraph"` and fully specified, verified suffixes. Outside-scope items use `granularity: "none"`, empty citations/evidence, and a `scope_reason`. They still require a reference answer and required claims. Full-title experiments need different outside-scope items from Chapter I experiments.

```bash
python training_models_v1.py freeze-benchmark --input reviewed_questions.jsonl
```

The freezer verifies evidence substrings and citation/document relationships but cannot prove semantic correctness; the named reviewer must actually check the claims. Do not mark generated candidates verified automatically. Full training requires the frozen benchmark.

## GPU workflow

The frozen Chapter I dataset is checked into Git under
`experiments/title12-chapter-I-v1/`, including the original XML, canonical
documents, audit, image inventory, and candidate questions. Git attributes
disable newline conversion for these artifacts so their hashes survive a
Windows-to-Linux clone. Virtual environments, synthetic smoke runs, obsolete
corpus builds, and model checkpoints are excluded.

On the GPU machine, obtain this branch and recreate the environment:

```bash
git clone --branch codex/citation_metric https://github.com/abubakar12/ecfr-title12-finetuning.git
cd ecfr-title12-finetuning
python3.12 -m venv .venv-phase1
source .venv-phase1/bin/activate
pip install -r requirements-phase1.lock.txt
python training_models_v1.py audit-corpus
```

The current audit intentionally fails until the 81 graphics have verified
transcriptions. The 100 candidate questions also require completed, verified
answer keys before freezing. Cloning transfers the data; it does not remove
these training prerequisites. Provide your Hugging Face authentication on the
GPU machine using its local credential mechanism; never commit credentials.

Install the pinned direct dependencies in a separate environment on the A40. Authentication must provide access to the official Meta checkpoint. No fallback model or quantization is substituted on failure.

```bash
pip install -r requirements-phase1.lock.txt
python training_models_v1.py pretrain --smoke
python training_models_v1.py tokenize-corpus
python training_models_v1.py pretrain --preflight-only
python training_models_v1.py pretrain
```

The first model access resolves its requested revision to a commit SHA and freezes both model and tokenizer to that commit. Smoke and production have distinct directories and model locks. Actual installed library versions are recorded and cannot change on resume. Keep the resolved environment alongside the run (`pip freeze > environment.txt`).

Production defaults: BF16 LoRA, rank 16, alpha 32, dropout 0.05, attention and MLP projections, one epoch, learning rate 2e-5, batch size 1, gradient accumulation 16, cosine decay, 3% warmup, seed 42, and gradient checkpointing. These are pilot defaults rather than an optimized recipe. Runtime selection is CUDA, then Apple Silicon MPS; CPU is allowed only for smoke tests. `--device` and `--precision` override the runtime settings explicitly.

Segments have a maximum of 4,096 tokens including identity headers and special tokens. Sections that fit remain intact. Long sections split preferentially at paragraph ends, with up to 256 prior body tokens used only as context. Tokenization is performed once per body, retaining final partial segments. Header, overlap, and padding labels are masked. Every body token is a target once per epoch. Source token and character spans make coverage mechanically checkable. Unrelated sections are not packed together.

Every training invocation performs a maximum-length forward/backward and optimizer allocation check, then restores adapter weights and RNG. Insufficient GPU memory fails explicitly. The smoke run performs two optimizer steps, saves an adapter, reloads it, and checks finite loss. Production requires the smoke completion record and the frozen benchmark. Resume an interrupted run with `--resume path/to/checkpoint-N`. The final adapter is the predetermined last epoch, not selected on evaluation scores. Training loss is not held-out perplexity.

## Apple Silicon / M5 Pro

The Phase 1 code supports the Mac GPU through [PyTorch MPS](https://docs.pytorch.org/docs/2.8/notes/mps.html). Use native **arm64 Python 3.12**, not an Intel Python running under Rosetta, and a macOS/PyTorch combination that passes the runtime check. This code path is capability-based and does not require a chip-name allowlist. It has not been hardware-tested on an M5 Pro in this workspace.

```bash
git pull
python3.12 -m venv .venv-phase1
source .venv-phase1/bin/activate
pip install -r requirements-phase1.macos.lock.txt
python training_models_v1.py check-runtime --config phase1.mac.json

# Can run now, even while the real corpus audit is blocked:
python scripts/smoke_phase1.py --device mps --precision bf16 \
  --experiment-dir experiments/synthetic-training-smoke-mac-v1
```

Use a separate Mac experiment directory with the exact frozen corpus. In a fresh clone, before starting any training, copy the checked-in dataset:

```bash
cp -R experiments/title12-chapter-I-v1 experiments/title12-chapter-I-mac-v1
python training_models_v1.py audit-corpus --config phase1.mac.json
```

Do not copy an existing run's `training/`, `smoke/`, or `evaluation/` directories. The checked-in source directory excludes those generated artifacts. Complete the image transcriptions and benchmark review in the Mac experiment directory; then:

```bash
python training_models_v1.py freeze-benchmark --config phase1.mac.json --input reviewed_questions.jsonl
python training_models_v1.py pretrain --config phase1.mac.json --smoke
python training_models_v1.py pretrain --config phase1.mac.json --preflight-only
python training_models_v1.py pretrain --config phase1.mac.json
python training_models_v1.py eval-cpt --config phase1.mac.json
```

On MPS, the base weights use the requested dtype (BF16 by default), LoRA parameters retain PEFT's FP32 promotion, and Trainer AMP/GradScaler and pinned host memory are disabled for compatibility with the pinned libraries. The optimizer uses non-fused AdamW with `foreach=False`. SDPA and gradient checkpointing are retained. Both comparison checkpoints must use the recorded backend, precision, and attention implementation. Runtime/platform details are saved in training and evaluation manifests.

The 8B model requires roughly 16 GB just for 16-bit weights, plus adapters, activations, optimizer state, and macOS. A chip name alone cannot guarantee enough memory. The actual maximum-length check determines whether the configured run fits. A memory failure never silently shortens sequences, changes precision, or selects another model. Keep the Metal memory safety limit enabled. If necessary, create a new configuration with a smaller sequence length or model and label it as a different experiment. `--precision fp32` is available for explicit dtype troubleshooting but uses more memory. CPU fallback for unsupported MPS operations defaults off; explicitly enabling `PYTORCH_ENABLE_MPS_FALLBACK=1` is recorded in the run manifest.

## Generate, review, and compare

```bash
python training_models_v1.py eval-cpt
# Give only blind_review.jsonl to the reviewer; keep private_mapping.jsonl private.
# Save the completed review as a separate file.
python training_models_v1.py score-cpt --input completed_reviews.jsonl
```

Both checkpoints use the same frozen model/tokenizer, prompt, greedy decoding, precision, and 768-token output budget, with no retrieval. Generation records and blinded packets are frozen. Reviewers count claims supported by correct citations, correctly used citations, and total citations, and assess substantive correctness and citation granularity. For outside-scope prompts, assess whether the answer recognizes the scope limitation. A correct answer from broader prior knowledge is not evidence of learning Chapter I.

The primary metric is the fraction of in-scope answers with every required claim correctly cited, no incorrect citations, and correct substance. Report macro citation precision and claim coverage, nonexistent-section counts, section/paragraph accuracy, outside-scope behavior, and a paired bootstrap 95% confidence interval for the primary delta. Regex checks are structural diagnostics, never substitutes for semantic review. Do not provide the checkpoint mapping until reviews are complete.

## Save every trained model to Hugging Face

Automatic publishing is enabled in `phase1.json`, `phase1.mac.json`, and the
legacy `config.yaml`. Authenticate on the training machine with a write-capable
token (and obtain access to the gated base model):

```bash
hf auth login
python training_models_v1.py check-hub --config phase1.mac.json
```

Use `phase1.json` for the CUDA experiment. After a successful full training run
and adapter reload check, `pretrain` uploads automatically. Smoke and preflight
runs never upload. Normal legacy `sft`, `dpo`, and `grpo` commands also publish
after saving successfully. This works independently of CUDA/MPS.

Repositories default to **private**, in your authenticated user namespace, named
`ecfr-title12-<stage>-<fingerprint>`. Different model artifacts receive different
names; a retry uses the same name. Set `HF_REPO_NAMESPACE` for an organization,
`HF_REPO_PREFIX` for another prefix, or `HF_REPO_PRIVATE=false` for public models.
The namespace requires write access. These are environment variables; credentials
must never be placed in tracked configuration. Set `hub.enabled` to false only
when deliberately training offline.

Each repository contains the final **LoRA adapter**, saved tokenizer, model card
with loading code, and provenance with the exact base revision and artifact hashes.
The base weights are still required for inference. The upload receipt
`training/hub_upload.json` records the repository and exact Hub commit to use
when loading later. Optimizer state and intermediate checkpoints stay local;
keep them if you need to resume training. Evaluation answers and credentials
are not uploaded.

Authentication is checked before full training. If the eventual upload fails,
the completed local model is retained and the command reports failure. Retry
without retraining or a GPU:

```bash
python training_models_v1.py push-model --config phase1.mac.json
# Legacy stage retry:
python train_and_upload.py --upload-only --stage sft
```

Legacy upload requires a `completed.json` produced by the updated trainer;
older folders without completion/provenance are not silently published.
Hub publishing does not bypass corpus, benchmark, or training acceptance gates.

## Tests

```bash
python -m unittest discover -s tests -v
```

The stdlib test suite covers XML preservation, scope, malformed responses, immutable snapshots, tampering, complete token coverage, overlap masking, long paragraphs, and citation failure cases. The actual model smoke and GPU preflight are separate acceptance checks; passing unit tests does not imply that training has run.

`python scripts/smoke_phase1.py` exercises the real training and reload path on an explicitly synthetic fixture in a separate experiment directory. This can validate the implementation while the real data audit is blocked; it does not satisfy the production corpus or smoke gates. An unresolved citation outside the selected corpus is not automatically nonexistent: reviewers verify actual nonexistence separately.
