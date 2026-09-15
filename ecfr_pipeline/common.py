"""Shared utilities: config, seeding, hashing, runtime detection, model loading, manifests.

Torch/transformers are imported lazily so the data stages (download/build) run on
machines without the training stack installed.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

import yaml

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # harmless off Apple Silicon

# ---------------------------------------------------------------------------
# Config / filesystem
# ---------------------------------------------------------------------------


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg["_config_path"] = str(path)
    return cfg


def ensure_dirs(cfg: dict) -> None:
    for key in ("data_dir", "raw_dir", "outputs_dir", "results_dir"):
        Path(cfg["paths"][key]).mkdir(parents=True, exist_ok=True)


def save_jsonl(path, rows) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_jsonl(path) -> list:
    with Path(path).open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def require_file(path, hint: str) -> Path:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — {hint}")
    return path


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def config_hash(cfg: dict) -> str:
    clean = {k: v for k, v in cfg.items() if not k.startswith("_")}
    return hashlib.sha256(json.dumps(clean, sort_keys=True, default=str).encode()).hexdigest()[:12]


def stable_bucket(key: str, buckets: int = 10) -> int:
    """Machine- and run-independent bucket assignment (md5, not PYTHONHASHSEED-dependent)."""
    return int(hashlib.md5(key.encode("utf-8")).hexdigest(), 16) % buckets


def write_manifest(cfg: dict, stage: str, extra: dict | None = None) -> Path:
    manifest = {
        "stage": stage,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config_hash": config_hash(cfg),
        "seed": cfg["seed"],
        "python": sys.version.split()[0],
        "versions": {},
    }
    for mod in ("torch", "transformers", "trl", "peft", "datasets"):
        try:
            manifest["versions"][mod] = __import__(mod).__version__
        except ImportError:
            pass
    if extra:
        manifest.update(extra)
    out = Path(cfg["paths"]["outputs_dir"]) / "manifests"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{stage}_manifest.json"
    path.write_text(json.dumps(manifest, indent=2))
    print(f"[{stage}] manifest -> {path}")
    return path


# ---------------------------------------------------------------------------
# Runtime / model loading
# ---------------------------------------------------------------------------


def _mps_available() -> bool:
    import torch

    return bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available()


def _mps_bf16_ok() -> bool:
    import torch

    try:
        torch.zeros(1, dtype=torch.bfloat16, device="mps")
        return True
    except (TypeError, RuntimeError):
        return False


def detect_runtime(cfg: dict, smoke: bool = False) -> dict:
    """Resolve cfg[runtime].device (auto|cuda|mps|cpu) into device, precision, and optimizer.

    cuda: bf16/fp16 + 4-bit QLoRA below the VRAM threshold. mps (Apple Silicon,
    M1-M5): un-quantized LoRA, bf16 when the chip supports it, eager attention
    (bitsandbytes/amp are CUDA-only). cpu: float32, meant for --smoke runs.
    """
    import torch

    requested = (cfg.get("runtime") or {}).get("device", "auto")
    cuda_ok = torch.cuda.is_available()
    mps_ok = _mps_available()
    if requested == "auto":
        device = "cuda" if cuda_ok else ("mps" if mps_ok else "cpu")
    elif requested == "cuda" and not cuda_ok:
        raise SystemExit("runtime.device=cuda but no CUDA device is visible")
    elif requested == "mps" and not mps_ok:
        raise SystemExit(
            "runtime.device=mps but MPS is unavailable (requires Apple Silicon + macOS 12.3+)"
        )
    else:
        device = requested

    info = {
        "requested": requested,
        "device": device,
        "vram_gb": 0.0,
        "use_4bit": False,
        "bf16": False,
        "fp16": False,
        "torch_dtype": "float32",
        "optim": "adamw_torch",
        "attn_implementation": "sdpa",
        "pin_memory": False,
        "gradient_checkpointing": False,
    }
    if device == "cuda":
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        bf16 = torch.cuda.is_bf16_supported()
        info.update(
            vram_gb=round(vram, 1),
            bf16=bf16,
            fp16=not bf16,
            torch_dtype="bfloat16" if bf16 else "float16",
            pin_memory=True,
            gradient_checkpointing=True,
        )
        if not smoke and vram < cfg["model"]["qlora_below_gb"]:
            info.update(use_4bit=True, optim="paged_adamw_8bit")
    elif device == "mps":
        info.update(
            torch_dtype="bfloat16" if _mps_bf16_ok() else "float16",
            attn_implementation="eager",
            gradient_checkpointing=True,
        )
    print(f"[runtime] {info}")
    return info


def resolve_model_id(cfg: dict, smoke: bool = False, override: str | None = None) -> str:
    if override:
        return override
    return cfg["model"]["smoke_id"] if smoke else cfg["model"]["id"]


def load_tokenizer(model_id: str, cfg: dict | None = None):
    from transformers import AutoTokenizer

    try:
        tok = AutoTokenizer.from_pretrained(model_id)
    except OSError as err:
        fallback = (cfg or {}).get("model", {}).get("fallback_id")
        if not fallback or fallback == model_id:
            raise
        print(f"[tokenizer] {model_id} unavailable ({err.__class__.__name__}); using {fallback}")
        tok = AutoTokenizer.from_pretrained(fallback)
    if tok.pad_token is None:
        # Llama 3.1 ships a dedicated pad token; otherwise fall back to EOS.
        pad = "<|finetune_right_pad_id|>"
        tok.pad_token = pad if pad in tok.get_vocab() else tok.eos_token
    return tok


def load_base_model(cfg: dict, runtime: dict, model_id: str, for_training: bool = True):
    import torch
    from transformers import AutoModelForCausalLM

    kwargs: dict = {
        "torch_dtype": {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[runtime["torch_dtype"]],
        "attn_implementation": runtime["attn_implementation"],
    }
    if runtime["use_4bit"]:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if runtime["bf16"] else torch.float16,
        )
    if runtime["device"] == "cuda":
        kwargs["device_map"] = "auto"

    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    except OSError as err:
        fallback = cfg["model"].get("fallback_id")
        if not fallback or fallback == model_id:
            raise
        print(f"[model] {model_id} unavailable ({err.__class__.__name__}); using {fallback}")
        model = AutoModelForCausalLM.from_pretrained(fallback, **kwargs)

    if runtime["device"] in ("mps", "cpu"):
        model = model.to(runtime["device"])
    if for_training:
        model.config.use_cache = False  # incompatible with gradient checkpointing
    return model


def prepare_for_kbit_if_needed(model, runtime: dict):
    if runtime["use_4bit"]:
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=runtime["gradient_checkpointing"],
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
    return model


def load_peft_adapter(model, adapter_dir, trainable: bool, adapter_name: str = "default"):
    from peft import PeftModel

    _warn_on_base_mismatch(adapter_dir, model)
    return PeftModel.from_pretrained(
        model, str(adapter_dir), is_trainable=trainable, adapter_name=adapter_name
    )


def _warn_on_base_mismatch(adapter_dir, model) -> None:
    cfg_path = Path(adapter_dir) / "adapter_config.json"
    if not cfg_path.exists():
        return
    trained_on = json.loads(cfg_path.read_text()).get("base_model_name_or_path", "")
    current = getattr(model.config, "_name_or_path", "")
    if trained_on and current and trained_on != current:
        print(f"[warn] adapter {adapter_dir} was trained on {trained_on}, current base is {current}")


def build_lora_config(cfg: dict):
    from peft import LoraConfig

    lc = cfg["lora"]
    return LoraConfig(
        r=lc["r"],
        lora_alpha=lc["alpha"],
        lora_dropout=lc["dropout"],
        target_modules=list(lc["target_modules"]),
        bias="none",
        task_type="CAUSAL_LM",
    )
