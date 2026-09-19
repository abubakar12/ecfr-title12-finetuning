"""Stage 3: supervised LoRA fine-tuning on eCFR QA pairs (TRL SFTTrainer)."""
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
    common.require_file(data_dir / "sft_train.jsonl", "run the `build` stage first")
    train_ds = load_dataset("json", data_files=str(data_dir / "sft_train.jsonl"), split="train")
    val_ds = load_dataset("json", data_files=str(data_dir / "sft_val.jsonl"), split="train")
    if smoke:
        train_ds = train_ds.select(range(min(cfg["smoke"]["max_train_examples"], len(train_ds))))
        val_ds = val_ds.select(range(min(16, len(val_ds))))
    train_ds = train_ds.select_columns(["messages"])
    val_ds = val_ds.select_columns(["messages"])
    print(f"[sft] train={len(train_ds)} val={len(val_ds)} model={model_id}")

    tok = common.load_tokenizer(model_id, cfg)
    model = common.load_base_model(cfg, runtime, model_id)

    s = cfg["sft"]
    out_dir = Path(cfg["paths"]["outputs_dir"]) / "sft-adapter"
    cpt_dir = Path(cfg["paths"]["outputs_dir"]) / "cpt-adapter"
    peft_cfg = common.build_lora_config(cfg)
    init_from = None
    if s.get("init_from_cpt", True) and (cpt_dir / "adapter_config.json").exists():
        # Continue training the CPT adapter (chain CPT -> SFT) instead of a fresh LoRA.
        model = common.prepare_for_kbit_if_needed(model, runtime)
        if runtime["gradient_checkpointing"] and hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        model = common.load_peft_adapter(model, cpt_dir, trainable=True)
        peft_cfg, init_from = None, str(cpt_dir)
        print(f"[sft] initialized from CPT adapter {cpt_dir}")
    else:
        print("[sft] training a fresh LoRA on the base model (no CPT adapter)")
    # Loss over the full sequence (prompts are short/templated; completion-only
    # masking needs {% generation %} chat-template support Llama does not ship).
    args = SFTConfig(
        output_dir=str(out_dir),
        num_train_epochs=s["epochs"],
        max_steps=cfg["smoke"]["max_steps"] if smoke else -1,
        per_device_train_batch_size=s["per_device_batch"],
        gradient_accumulation_steps=s["grad_accum"],
        learning_rate=float(s["learning_rate"]),
        lr_scheduler_type="cosine",
        warmup_ratio=s["warmup_ratio"],
        logging_steps=s["logging_steps"],
        eval_strategy="steps",
        eval_steps=s["eval_steps"],
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
        max_length=s["max_length"],
        packing=False,
        seed=cfg["seed"],
        report_to=[],
    )
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tok,
        peft_config=peft_cfg,
    )
    result = trainer.train()
    if not math.isfinite(result.training_loss):
        raise ValueError("Non-finite training loss; model will not be published")

    trainer.save_model(str(out_dir))
    tok.save_pretrained(str(out_dir))
    (out_dir / "training_log.json").write_text(json.dumps(trainer.state.log_history, indent=2))
    print(f"[sft] adapter saved -> {out_dir}")
    common.write_manifest(
        cfg,
        "sft",
        {
            "model_id": model_id,
            "smoke": smoke,
            "runtime": runtime,
            "adapter_dir": str(out_dir),
            "init_from": init_from,
            "n_train": len(train_ds),
            "n_val": len(val_ds),
            "dataset_sha256": common.sha256_file(data_dir / "sft_train.jsonl"),
        },
    )
    hub.finish_legacy(cfg, "sft", out_dir, trainer.model, smoke)
    return out_dir
