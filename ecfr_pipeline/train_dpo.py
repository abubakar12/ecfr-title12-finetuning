"""Stage 4: DPO post-training on top of the SFT adapter.

Preference pairs: chosen = grounded answer with correct citation; rejected =
seeded corruption (wrong section / wrong citation / vague). The frozen SFT
adapter is loaded a second time as the DPO reference policy, so KL is measured
against the SFT model, not the raw base.
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
    from trl import DPOConfig, DPOTrainer

    data_dir = Path(cfg["paths"]["data_dir"])
    out_root = Path(cfg["paths"]["outputs_dir"])
    sft_dir = out_root / "sft-adapter"
    common.require_file(sft_dir / "adapter_config.json", "run the `sft` stage first")
    common.require_file(data_dir / "dpo_train.jsonl", "run the `build` stage first")

    ds = load_dataset("json", data_files=str(data_dir / "dpo_train.jsonl"), split="train")
    if smoke:
        ds = ds.select(range(min(cfg["smoke"]["max_train_examples"], len(ds))))
    ds = ds.select_columns(["prompt", "chosen", "rejected"])
    print(f"[dpo] pairs={len(ds)} model={model_id} from adapter {sft_dir}")

    tok = common.load_tokenizer(model_id, cfg)
    model = common.load_base_model(cfg, runtime, model_id)
    model = common.prepare_for_kbit_if_needed(model, runtime)
    model = common.load_peft_adapter(model, sft_dir, trainable=True)  # policy, init = SFT
    model.load_adapter(str(sft_dir), adapter_name="reference")        # frozen SFT reference
    model.set_adapter("default")

    d = cfg["dpo"]
    out_dir = out_root / "dpo-adapter"
    args = DPOConfig(
        output_dir=str(out_dir),
        num_train_epochs=d["epochs"],
        max_steps=cfg["smoke"]["max_steps"] if smoke else -1,
        per_device_train_batch_size=d["per_device_batch"],
        gradient_accumulation_steps=d["grad_accum"],
        learning_rate=float(d["learning_rate"]),
        lr_scheduler_type="cosine",
        warmup_ratio=d["warmup_ratio"],
        beta=d["beta"],
        max_length=d["max_length"],
        max_prompt_length=d["max_prompt_length"],
        model_adapter_name="default",
        ref_adapter_name="reference",
        logging_steps=d["logging_steps"],
        save_strategy="no",
        bf16=runtime["bf16"],
        fp16=runtime["fp16"],
        optim=runtime["optim"],
        dataloader_pin_memory=runtime["pin_memory"],
        use_cpu=runtime["device"] == "cpu",
        # Recomputing a forward after DPO switches adapters gives mismatched
        # tensors: the checkpoint started with policy weights, then sees reference.
        gradient_checkpointing=False,
        gradient_checkpointing_kwargs=None,
        seed=cfg["seed"],
        report_to=[],
    )
    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # reference comes from the frozen "reference" adapter
        args=args,
        train_dataset=ds,
        processing_class=tok,
    )
    result = trainer.train()
    if not math.isfinite(result.training_loss):
        raise ValueError("Non-finite training loss; model will not be published")

    trainer.save_model(str(out_dir))
    tok.save_pretrained(str(out_dir))
    (out_dir / "training_log.json").write_text(json.dumps(trainer.state.log_history, indent=2))
    print(f"[dpo] adapter saved -> {out_dir}")
    common.write_manifest(
        cfg,
        "dpo",
        {
            "model_id": model_id,
            "smoke": smoke,
            "runtime": runtime,
            "adapter_dir": str(out_dir),
            "init_from": str(sft_dir),
            "n_pairs": len(ds),
            "dataset_sha256": common.sha256_file(data_dir / "dpo_train.jsonl"),
        },
    )
    hub.finish_legacy(cfg, "dpo", out_dir, trainer.model, smoke)
    return out_dir
