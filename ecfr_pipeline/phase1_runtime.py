"""Explicit CUDA / Apple Silicon MPS runtime shared by CPT and evaluation."""
from __future__ import annotations

import os
import platform
from contextlib import contextmanager


def configure_environment():
    # Set before torch import. Users can explicitly enable fallback, recorded below.
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "0")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def choose_device(requested, cuda, mps, smoke=False):
    if requested not in {"auto", "cuda", "mps", "cpu"}:
        raise ValueError(f"Unknown device: {requested}")
    device = ("cuda" if cuda else "mps" if mps else "cpu") if requested == "auto" else requested
    if device == "cuda" and not cuda:
        raise RuntimeError("CUDA was requested but is unavailable")
    if device == "mps" and not mps:
        raise RuntimeError("MPS was requested but is unavailable. Use native arm64 Python and an MPS-enabled PyTorch on Apple Silicon.")
    if device == "cpu" and not smoke:
        raise RuntimeError("Full runs require CUDA or Apple Silicon MPS; CPU is supported only for smoke tests")
    return device


def choose_precision(requested, smoke=False):
    if requested not in {"auto", "bf16", "fp32"}:
        raise ValueError(f"Unsupported precision: {requested}; choose auto, bf16, or fp32")
    return ("fp32" if smoke else "bf16") if requested == "auto" else requested


def trainer_options(runtime):
    # Pinned Transformers/Accelerate reject MPS bf16 AMP. Use explicit BF16 model
    # weights with FP32 LoRA parameters instead; no AMP or GradScaler on MPS.
    return {"bf16": runtime["device"] == "cuda" and runtime["precision"] == "bf16",
            "fp16": False, "use_cpu": runtime["device"] == "cpu",
            "dataloader_pin_memory": runtime["device"] == "cuda", "optim": "adamw_torch"}


def resolve(cfg, smoke=False):
    configure_environment()
    import torch
    settings = cfg.get("runtime", {})
    device = choose_device(settings.get("device", "auto"), torch.cuda.is_available(),
                           bool(torch.backends.mps.is_available()), smoke)
    precision = choose_precision(settings.get("precision", "auto"), smoke)
    if device == "cuda" and precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA BF16 is unavailable; choose --precision fp32 explicitly or another GPU")
    runtime = {"device": device, "precision": precision, "attention": "sdpa",
               "platform": platform.system(), "machine": platform.machine(), "macos": platform.mac_ver()[0],
               "torch": torch.__version__, "mps_fallback": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1",
               "gpu": torch.cuda.get_device_name(0) if device == "cuda" else "Apple Silicon (MPS)" if device == "mps" else None}
    if device == "mps":
        runtime["recommended_working_set_bytes"] = torch.mps.recommended_max_memory()
    # Exercise the requested dtype and backward kernel before downloading 8B weights.
    try:
        x = torch.ones((4, 4), device=device, dtype=dtype(runtime), requires_grad=True)
        loss = (x @ x).float().square().mean()
        loss.backward()
        if not torch.isfinite(loss).item() or not torch.isfinite(x.grad).all().item():
            raise RuntimeError("Non-finite device probe")
        synchronize(runtime)
    except (RuntimeError, TypeError) as error:
        raise RuntimeError(f"{device}/{precision} forward/backward probe failed. Check PyTorch/macOS support; --precision fp32 is an explicit alternative. No fallback was applied.") from error
    return runtime


def dtype(runtime):
    import torch
    return torch.bfloat16 if runtime["precision"] == "bf16" else torch.float32


def synchronize(runtime):
    import torch
    if runtime["device"] == "mps":
        torch.mps.synchronize()
    elif runtime["device"] == "cuda":
        torch.cuda.synchronize()


@contextmanager
def memory_guard(runtime, operation):
    try:
        yield
    except RuntimeError as error:
        if "out of memory" not in str(error).lower():
            raise
        raise RuntimeError(f"{runtime['device']} ran out of memory during {operation}. No settings were changed. Use a new experiment configuration with a smaller max_length/model or a machine with more memory; keep the Metal memory safety limit enabled.") from error


def load_model(lock, runtime):
    from transformers import AutoModelForCausalLM
    with memory_guard(runtime, "model loading"):
        return AutoModelForCausalLM.from_pretrained(lock["id"], revision=lock["revision"],
                 torch_dtype=dtype(runtime), attn_implementation=runtime["attention"]).to(runtime["device"])


def check_evaluation_runtime(trained, current):
    # Backend/precision changes confound the before/after comparison.
    for key in ("device", "precision", "attention", "mps_fallback"):
        if trained[key] != current[key]:
            raise ValueError(f"Evaluation {key} differs from training; use the recorded runtime for both checkpoints")
