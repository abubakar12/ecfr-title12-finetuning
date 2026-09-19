"""Stage 2b: continued pretraining (CPT) — causal LM on the raw regulation text.

Trains a LoRA adapter on data/cpt_train.jsonl (train + val sections only, so the
held-out eval sections are never seen in any stage). SFT then continues training
this same adapter, giving the chain CPT -> SFT -> DPO. Validation perplexity on
cpt_val.jsonl is logged every eval_steps.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from . import common


def run(cfg: dict, smoke: bool = False, model_override: str | None = None) -> Path:
    from . import hub
    if not smoke and hub.enabled(cfg):
        hub.check(cfg)
    common.set_global_seed(cfg["seed"])
    runtime = common.detect_runtime(cfg, smoke)
    model_id = common.resolve_model_id(cfg, smoke, model_override)

    from datasets import load_dataset
    from trl import SFTConfig, SFTTrainer

    data_dir = Path(cfg["paths"]["data_dir"])
    common.require_file(data_dir / "cpt_train.jsonl", "run the `build` stage first")
    train_ds = load_dataset("json", data_files=str(data_dir / "cpt_train.jsonl"), split="train")
    val_ds = load_dataset("json", data_files=str(data_dir / "cpt_val.jsonl"), split="train")
    if smoke:
        train_ds = train_ds.select(range(min(cfg["smoke"]["max_train_examples"], len(train_ds))))
        val_ds = val_ds.select(range(min(16, len(val_ds))))
    train_ds = train_ds.select_columns(["text"])
    val_ds = val_ds.select_columns(["text"])
    words = sum(train_ds.map(lambda r: {"n": len(r["text"].split())}, remove_columns=["text"])["n"])
    print(f"[cpt] documents={len(train_ds)} ({words:,} words) val={len(val_ds)} model={model_id}")

    tok = common.load_tokenizer(model_id, cfg)
    model = common.load_base_model(cfg, runtime, model_id)

    c = cfg["cpt"]
    out_dir = Path(cfg["paths"]["outputs_dir"]) / "cpt-adapter"
    args = SFTConfig(
        output_dir=str(out_dir),
        num_train_epochs=c["epochs"],
        max_steps=cfg["smoke"]["max_steps"] if smoke else -1,
        per_device_train_batch_size=c["per_device_batch"],
        gradient_accumulation_steps=c["grad_accum"],
        learning_rate=float(c["learning_rate"]),
        lr_scheduler_type="cosine",
        warmup_ratio=c["warmup_ratio"],
        logging_steps=c["logging_steps"],
        eval_strategy="steps",
        eval_steps=c["eval_steps"],
        save_strategy="no",
        bf16=runtime["bf16"],
        fp16=runtime["fp16"],
        optim=runtime["optim"],
        dataloader_pin_memory=runtime["pin_memory"],
        use_cpu=runtime["device"] == "cpu",
        gradient_checkpointing=runtime["gradient_checkpointing"],
        gradient_checkpointing_kwargs=(
            {"use_reentrant": False} if runtime["gradient_checkpointing"] else None
        ),
        max_length=c["max_length"],
        packing=True,  # dense LM batches: documents are concatenated to max_length
        packing_strategy="wrapped",  # classic CPT packing; 'bfd' needs flash-attn for padding-free
        dataset_text_field="text",
        seed=cfg["seed"],
        report_to=[],
    )
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tok,
        peft_config=common.build_lora_config(cfg),
    )
    result = trainer.train()
    if not math.isfinite(result.training_loss):
        raise ValueError("Non-finite training loss; model will not be published")
    final_eval = trainer.evaluate()
    final_eval["perplexity"] = math.exp(final_eval["eval_loss"]) if "eval_loss" in final_eval else None

    trainer.save_model(str(out_dir))
    tok.save_pretrained(str(out_dir))
    (out_dir / "training_log.json").write_text(json.dumps(trainer.state.log_history, indent=2))
    print(f"[cpt] adapter saved -> {out_dir}  val perplexity={final_eval['perplexity']}")
    common.write_manifest(
        cfg,
        "cpt",
        {
            "model_id": model_id,
            "smoke": smoke,
            "runtime": runtime,
            "adapter_dir": str(out_dir),
            "n_documents": len(train_ds),
            "n_words": words,
            "n_val": len(val_ds),
            "final_eval": final_eval,
            "dataset_sha256": common.sha256_file(data_dir / "cpt_train.jsonl"),
        },
    )
    hub.finish_legacy(cfg, "cpt", out_dir, trainer.model, smoke)
    return out_dir
