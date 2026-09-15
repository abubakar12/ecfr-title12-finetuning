"""Stage 5 (optional): GRPO with verifiable rewards on top of the SFT adapter.

Rewards are computable from the data itself (no reward model): correct-citation
reward, grounding token-F1 vs the reference answer, and a brevity/repetition
penalty as the anti-reward-hacking guard. beta=0 in config -> no KL reference.
Prompts are reused from dpo_train.jsonl (prompt + reference, pairs ignored).
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from . import common, metrics


def _text(completion) -> str:
    # conversational completions arrive as [{"role": "assistant", "content": ...}]
    if isinstance(completion, list):
        return completion[-1]["content"] if completion else ""
    return completion or ""


def citation_reward(completions, section=None, **_):
    """+1 correct citation, -0.5 confidently wrong, capped when citation-spamming."""
    scores = []
    for comp, sec in zip(completions, section):
        text = _text(comp)
        found = metrics.extract_sections(text)
        if metrics.citation_correct(text, sec):
            scores.append(1.0 if len(found) <= 3 else 0.25)
        elif found:
            scores.append(-0.5)
        else:
            scores.append(0.0)
    return scores


def grounding_reward(completions, reference=None, **_):
    return [metrics.token_f1(_text(c), ref) for c, ref in zip(completions, reference)]


def brevity_reward(completions, **_):
    """Penalize rambling and degenerate repetition (classic GRPO failure modes)."""
    scores = []
    for comp in completions:
        words = _text(comp).split()
        score = -0.3 if len(words) > 320 else 0.0
        if len(words) > 40:
            top = Counter(words).most_common(1)[0][1]
            if top / len(words) > 0.15:
                score -= 0.5
        scores.append(score)
    return scores


def run(cfg: dict, smoke: bool = False, model_override: str | None = None) -> Path:
    common.set_global_seed(cfg["seed"])
    runtime = common.detect_runtime(cfg, smoke)
    model_id = common.resolve_model_id(cfg, smoke, model_override)

    from datasets import load_dataset
    from trl import GRPOConfig, GRPOTrainer

    data_dir = Path(cfg["paths"]["data_dir"])
    out_root = Path(cfg["paths"]["outputs_dir"])
    sft_dir = out_root / "sft-adapter"
    common.require_file(data_dir / "dpo_train.jsonl", "run the `build` stage first")

    ds = load_dataset("json", data_files=str(data_dir / "dpo_train.jsonl"), split="train")
    ds = ds.map(
        lambda r: {"prompt": r["prompt"], "section": r["section"],
                   "reference": r["chosen"][0]["content"]},
        remove_columns=ds.column_names,
    )
    if smoke:
        ds = ds.select(range(min(cfg["smoke"]["max_train_examples"], len(ds))))
    print(f"[grpo] prompts={len(ds)} model={model_id}")

    tok = common.load_tokenizer(model_id, cfg)
    model = common.load_base_model(cfg, runtime, model_id)
    model = common.prepare_for_kbit_if_needed(model, runtime)
    peft_cfg = None
    if (sft_dir / "adapter_config.json").exists():
        if runtime["gradient_checkpointing"] and hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        model = common.load_peft_adapter(model, sft_dir, trainable=True)
        print(f"[grpo] policy initialized from SFT adapter {sft_dir}")
    else:
        peft_cfg = common.build_lora_config(cfg)
        print("[grpo] no SFT adapter found — training a fresh LoRA on the base model")

    g = cfg["grpo"]
    # global batch must stay divisible by num_generations
    num_gen = min(2, g["num_generations"]) if smoke else g["num_generations"]
    per_dev = num_gen if smoke else g["per_device_batch"]
    out_dir = out_root / "grpo-adapter"
    args = GRPOConfig(
        output_dir=str(out_dir),
        max_steps=cfg["smoke"]["max_steps"] if smoke else g["max_steps"],
        per_device_train_batch_size=per_dev,
        gradient_accumulation_steps=g["grad_accum"],
        learning_rate=float(g["learning_rate"]),
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        beta=g["beta"],
        num_generations=num_gen,
        max_prompt_length=g["max_prompt_length"],
        max_completion_length=(
            cfg["smoke"]["max_new_tokens"] if smoke else g["max_completion_length"]
        ),
        temperature=g["temperature"],
        logging_steps=g["logging_steps"],
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
        seed=cfg["seed"],
        report_to=[],
    )
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[citation_reward, grounding_reward, brevity_reward],
        args=args,
        train_dataset=ds,
        processing_class=tok,
        peft_config=peft_cfg,
    )
    trainer.train()

    trainer.save_model(str(out_dir))
    tok.save_pretrained(str(out_dir))
    (out_dir / "training_log.json").write_text(json.dumps(trainer.state.log_history, indent=2))
    print(f"[grpo] adapter saved -> {out_dir}")
    common.write_manifest(
        cfg,
        "grpo",
        {
            "model_id": model_id,
            "smoke": smoke,
            "runtime": runtime,
            "adapter_dir": str(out_dir),
            "init_from": str(sft_dir) if (sft_dir / "adapter_config.json").exists() else None,
            "n_prompts": len(ds),
            "dataset_sha256": common.sha256_file(data_dir / "dpo_train.jsonl"),
        },
    )
    return out_dir
