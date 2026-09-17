"""Citation-preserving segmentation and plain-text LoRA continued pretraining."""
from __future__ import annotations

import importlib.metadata
import json
import math
from pathlib import Path

from . import corpus


def pin_model(cfg, smoke=False):
    from huggingface_hub import HfApi
    directory = Path(cfg["experiment_dir"]) / ("smoke" if smoke else "training")
    path = directory / "model.lock.json"
    model_id = cfg["model"]["smoke_id"] if smoke else cfg["model"]["id"]
    requested = "main" if smoke else cfg["model"].get("revision", "main")
    if path.exists():
        lock = corpus.read_json(path)
        if lock["id"] != model_id or lock["requested_revision"] != requested:
            raise ValueError("Model configuration changed; use a new experiment directory")
        return lock
    revision = HfApi().model_info(model_id, revision=requested).sha
    lock = {"id": model_id, "revision": revision, "tokenizer_revision": revision, "requested_revision": requested}
    corpus.write_json(path, lock)
    return lock


def tokenizer_for(lock):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(lock["id"], revision=lock["tokenizer_revision"], use_fast=True)
    if not tok.is_fast:
        raise ValueError("A fast tokenizer is required for auditable source offsets")
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


def segment_document(document, tok, max_length=4096, overlap=256):
    hierarchy = " > ".join(v["heading"] or v["number"] for v in document["hierarchy"].values())
    header = f"Snapshot: {document['snapshot_date']}\n{hierarchy}\n{document['citation']} — {document['heading']}\n\n"
    header_ids = tok.encode(header, add_special_tokens=False)
    prefix = ([tok.bos_token_id] if tok.bos_token_id is not None else []) + header_ids
    ending = [tok.eos_token_id] if tok.eos_token_id is not None else []
    capacity = max_length - len(prefix) - len(ending)
    if capacity <= overlap or overlap < 0:
        raise ValueError("Header leaves insufficient space for new body tokens")
    encoded = tok(document["text"], add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = encoded["input_ids"], encoded["offset_mapping"]
    # Tokenize the body once: segment concatenation must reconstruct these exact IDs.
    block_ends = {b["end"] for b in document["blocks"]}
    boundaries = {i + 1 for i, (_, end) in enumerate(offsets) if end in block_ends}
    cursor, sequence = 0, 0
    while cursor < len(ids):
        start = max(0, cursor - overlap)
        limit = min(len(ids), start + capacity)
        end = max((b for b in boundaries if cursor < b <= limit), default=limit)
        if limit == len(ids):
            end = limit
        input_ids = prefix + ids[start:end] + ending
        # The first body token has header context; overlap remains context-only.
        labels = [-100] * (len(prefix) + cursor - start) + ids[cursor:end] + ending
        spans = []
        char_start, char_end = offsets[start][0], offsets[end - 1][1]
        for block in document["blocks"]:
            if block["end"] > char_start and block["start"] < char_end:
                spans.append({"location": block["source_location"], "start": max(char_start, block["start"]), "end": min(char_end, block["end"])})
        yield {"document_id": document["id"], "citation": document["citation"], "sequence": sequence,
               "body_token_start": start, "body_token_end": end, "new_body_token_start": cursor,
               "prefix_length": len(prefix), "suffix_length": len(ending), "overlap_tokens": cursor - start,
               "source_spans": spans, "input_ids": input_ids, "labels": labels}
        cursor, sequence = end, sequence + 1


def verify_segments(documents, rows, tok, max_length):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["document_id"], []).append(row)
    if set(grouped) - {d["id"] for d in documents}:
        raise ValueError("Unknown segment parent")
    count = 0
    for document in documents:
        expected = tok.encode(document["text"], add_special_tokens=False)
        cursor = 0
        for seq, r in enumerate(grouped.get(document["id"], [])):
            start, end, new = r["body_token_start"], r["body_token_end"], r["new_body_token_start"]
            prefix, suffix = r["prefix_length"], r["suffix_length"]
            if new != cursor or r["sequence"] != seq or not 0 <= start <= new < end <= len(expected):
                raise ValueError("Missing, duplicated, or unordered body token spans")
            stop = len(r["input_ids"]) - suffix
            if r["input_ids"][prefix:stop] != expected[start:end]:
                raise ValueError("Segment body differs from canonical tokens")
            if len(r["input_ids"]) > max_length or len(r["labels"]) != len(r["input_ids"]):
                raise ValueError("Invalid sequence length")
            if any(v != -100 for v in r["labels"][:prefix + new - start]):
                raise ValueError("Overlap/header was not masked")
            if r["labels"][prefix + new - start:stop] != expected[new:end]:
                raise ValueError("Missing body-token training targets")
            cursor = end
        if cursor != len(expected):
            raise ValueError("Document tail missing")
        count += len(expected)
    return count


def prepare(cfg, smoke=False):
    corpus.require_audit(cfg)
    lock = pin_model(cfg, smoke)
    tok = tokenizer_for(lock)
    directory = Path(cfg["experiment_dir"])
    target = directory / ("smoke" if smoke else "training")
    docs = corpus.read_rows(directory / "documents.jsonl")
    p = cfg["pretrain"]
    length = min(512, p["max_length"]) if smoke else p["max_length"]
    overlap = min(32, p["overlap"]) if smoke else p["overlap"]
    if smoke:
        docs = [d for d in docs if d["text"]][:4]
    rows = [r for d in docs for r in segment_document(d, tok, length, overlap)]
    count = verify_segments(docs, rows, tok, length)
    path = target / "segments.jsonl"
    if path.exists() and corpus.read_rows(path) != rows:
        raise ValueError("Segments changed; create a new experiment directory")
    corpus.write_rows(path, rows)
    manifest = {"body_tokens": count, "segments": len(rows), "max_length": length, "overlap": overlap,
                "model": lock, "documents_sha256": corpus.file_hash(directory / "documents.jsonl"), "segments_sha256": corpus.file_hash(path)}
    corpus.write_json(target / "segments.lock.json", manifest)
    if not smoke:
        corpus.write_json(directory / "segments.lock.json", manifest)
        corpus.audit(cfg)
    return tok, lock, rows, length, target


class Collator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, rows):
        import torch
        length = max(len(r["input_ids"]) for r in rows)
        return {"input_ids": torch.tensor([r["input_ids"] + [self.pad_id] * (length - len(r["input_ids"])) for r in rows]),
                "labels": torch.tensor([r["labels"] + [-100] * (length - len(r["labels"])) for r in rows]),
                "attention_mask": torch.tensor([[1] * len(r["input_ids"]) + [0] * (length - len(r["input_ids"])) for r in rows])}


def train(cfg, smoke=False, resume=None, preflight_only=False):
    import torch
    from datasets import Dataset
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, Trainer, TrainingArguments, set_seed
    from .benchmark import verify_frozen

    if not smoke:
        snapshot = corpus.read_json(Path(cfg["experiment_dir"]) / "snapshot.json")
        if snapshot.get("source_kind") == "synthetic_test_fixture":
            raise ValueError("Synthetic fixture cannot be used for production training")
        verify_frozen(cfg)
        smoke_path = Path(cfg["experiment_dir"]) / "smoke" / "completed.json"
        if not smoke_path.exists():
            raise ValueError("Run pretrain --smoke successfully before the full run")
        smoke_dir = smoke_path.parent
        if (corpus.read_json(smoke_path)["run_sha256"] != corpus.file_hash(smoke_dir / "run.json")
                or corpus.read_json(smoke_dir / "segments.lock.json")["documents_sha256"] != corpus.file_hash(Path(cfg["experiment_dir"]) / "documents.jsonl")):
            raise ValueError("Smoke run does not match the frozen corpus or its run manifest")
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("Full run requires a CUDA GPU with BF16 support; no silent CPU/quantization fallback")
    set_seed(cfg["seed"])
    tok, lock, rows, length, target = prepare(cfg, smoke)
    versions = dict(sorted((dist.metadata["Name"].lower(), dist.version) for dist in importlib.metadata.distributions() if dist.metadata["Name"]))
    run_config = {"configuration": cfg, "model": lock, "versions": versions, "smoke": smoke,
                  "segments_sha256": corpus.file_hash(target / "segments.jsonl")}
    run_path = target / "run.json"
    if run_path.exists() and corpus.read_json(run_path) != run_config:
        raise ValueError("Training configuration or dependencies changed")
    corpus.write_json(run_path, run_config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(lock["id"], revision=lock["revision"],
                                               torch_dtype=torch.float32 if smoke else torch.bfloat16,
                                               attn_implementation="sdpa").to(device)
    p = cfg["pretrain"]
    model = get_peft_model(model, LoraConfig(r=p["lora_r"], lora_alpha=p["lora_alpha"], lora_dropout=p["lora_dropout"],
                                           target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], task_type="CAUSAL_LM"))
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    # Real maximum-length forward/backward plus optimizer allocation; restore adapter
    # weights and RNG so the preflight is not an unrecorded training step.
    adapter_state = {n: v.detach().cpu().clone() for n, v in model.named_parameters() if v.requires_grad}
    probe = {k: torch.tensor([v], device=device) for k, v in {
        "input_ids": [tok.bos_token_id or tok.eos_token_id] + [rows[0]["input_ids"][-2]] * (length - 1),
        "labels": [-100] + [rows[0]["input_ids"][-2]] * (length - 1), "attention_mask": [1] * length}.items()}
    optimizer = torch.optim.AdamW([v for v in model.parameters() if v.requires_grad], lr=p["learning_rate"])
    try:
        loss = model(**probe).loss
        if not torch.isfinite(loss):
            raise ValueError("Non-finite preflight loss")
        loss.backward()
        optimizer.step()
    except torch.cuda.OutOfMemoryError as error:
        raise RuntimeError("Maximum-length memory check failed; no configuration was changed") from error
    with torch.no_grad():
        for name, value in model.named_parameters():
            if name in adapter_state:
                value.copy_(adapter_state[name].to(value.device))
    model.zero_grad(set_to_none=True)
    del optimizer, probe, loss, adapter_state
    set_seed(cfg["seed"])
    corpus.write_json(target / "preflight.json", {"passed": True, "length": length, "device": device,
                                               "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None})
    if preflight_only:
        return
    if (target / "completed.json").exists():
        raise ValueError("Completed run already exists; use a new experiment directory")
    if list((target / "checkpoints").glob("checkpoint-*")) and not resume:
        raise ValueError("Existing checkpoints found; specify --resume explicitly")
    args = TrainingArguments(output_dir=str(target / "checkpoints"), num_train_epochs=p["epochs"], max_steps=2 if smoke else -1,
                             per_device_train_batch_size=p["batch_size"], gradient_accumulation_steps=1 if smoke else p["grad_accum"],
                             learning_rate=p["learning_rate"], lr_scheduler_type="cosine", warmup_ratio=p["warmup_ratio"],
                             bf16=not smoke, use_cpu=device == "cpu", gradient_checkpointing=True,
                             gradient_checkpointing_kwargs={"use_reentrant": False}, logging_steps=1 if smoke else 10,
                             save_strategy="steps", save_steps=1 if smoke else 100, save_total_limit=2,
                             report_to=[], seed=cfg["seed"], remove_unused_columns=False)
    data = [{"input_ids": r["input_ids"], "labels": r["labels"]} for r in rows]
    trainer = Trainer(model=model, args=args, train_dataset=Dataset.from_list(data), data_collator=Collator(tok.pad_token_id))
    result = trainer.train(resume_from_checkpoint=resume)
    if not math.isfinite(result.training_loss):
        raise ValueError("Training loss is not finite")
    trainer.save_model(str(target / "adapter"))
    tok.save_pretrained(target / "adapter")
    trainer.save_state()
    # Pinned Transformers API: retain optimizer, scheduler and RNG at the final
    # step as well as periodic checkpoints, including runs shorter than 100 steps.
    trainer._save_checkpoint(trainer.model, trial=None)
    # Verify the saved adapter can be read back into the same base model.
    model = model.unload()
    # unload removes adapter layers but PEFT retains this bookkeeping attribute.
    if hasattr(model, "peft_config"):
        delattr(model, "peft_config")
    loaded = PeftModel.from_pretrained(model, target / "adapter")
    loaded.eval()
    batch = Collator(tok.pad_token_id)(data[:1])
    with torch.no_grad():
        reload_loss = loaded(**{k: v.to(device) for k, v in batch.items()}).loss.item()
    if not math.isfinite(reload_loss):
        raise ValueError("Reloaded adapter produced non-finite loss")
    corpus.write_json(target / "completed.json", {"training_loss": result.training_loss, "reload_loss": reload_loss,
                                                "global_step": result.global_step, "run_sha256": corpus.file_hash(run_path),
                                                "adapter_hashes": {f.name: corpus.file_hash(f) for f in (target / "adapter").iterdir() if f.is_file()}})
