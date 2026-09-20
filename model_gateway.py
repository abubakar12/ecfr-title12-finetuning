#!/usr/bin/env python3
"""OpenAI-compatible local inference gateway for RegBench.

Loads the base model or one completed LoRA adapter at a time so all five
checkpoints can be compared on a single GPU without keeping five 8B models in
memory. Requests are serialized and the active checkpoint is cached.
"""

from __future__ import annotations

import argparse
import gc
import hmac
import json
import os
import threading
import time
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
BASE_MODEL = os.environ.get("MODEL_BASE_ID", "meta-llama/Meta-Llama-3.1-8B-Instruct")
ADAPTERS: dict[str, Path | None] = {
    "base": None,
    "cpt": ROOT / "outputs" / "cpt-adapter",
    "sft": ROOT / "outputs" / "sft-adapter",
    "dpo": ROOT / "outputs" / "dpo-adapter",
    "grpo": ROOT / "outputs" / "grpo-adapter",
}


def available_models() -> list[dict[str, Any]]:
    rows = []
    for alias, adapter in ADAPTERS.items():
        ready = adapter is None or (adapter / "adapter_config.json").exists()
        rows.append({
            "id": alias,
            "object": "model",
            "owned_by": "local",
            "ready": ready,
            "adapter_path": None if adapter is None else str(adapter),
        })
    return rows


class ModelRuntime:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.alias: str | None = None
        self.model: Any = None
        self.tokenizer: Any = None

    def _release(self) -> None:
        self.model = None
        self.tokenizer = None
        self.alias = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def load(self, alias: str) -> tuple[Any, Any]:
        if alias not in ADAPTERS:
            raise ValueError(f"Unknown local model '{alias}'. Choose: {', '.join(ADAPTERS)}")
        adapter = ADAPTERS[alias]
        if adapter is not None and not (adapter / "adapter_config.json").exists():
            raise FileNotFoundError(f"The {alias.upper()} adapter is not complete yet: {adapter}")
        if self.alias == alias and self.model is not None:
            return self.model, self.tokenizer

        self._release()
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for the 8B local comparison gateway.")
        tokenizer_source = str(adapter) if adapter and (adapter / "tokenizer_config.json").exists() else BASE_MODEL
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, token=os.environ.get("HF_TOKEN"))
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            token=os.environ.get("HF_TOKEN"),
            dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
        )
        if adapter is not None:
            model = PeftModel.from_pretrained(model, str(adapter), is_trainable=False)
        model.eval()
        self.alias, self.model, self.tokenizer = alias, model, tokenizer
        return model, tokenizer

    def generate(self, alias: str, messages: list[dict[str, str]], temperature: float, max_tokens: int) -> dict[str, Any]:
        with self.lock:
            model, tokenizer = self.load(alias)
            import torch

            prompt_ids = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
            )
            device = next(model.parameters()).device
            prompt_ids = prompt_ids.to(device)
            attention_mask = torch.ones_like(prompt_ids)
            kwargs: dict[str, Any] = {
                "input_ids": prompt_ids,
                "attention_mask": attention_mask,
                "max_new_tokens": max(1, min(int(max_tokens), 1024)),
                "pad_token_id": tokenizer.pad_token_id,
                "do_sample": temperature > 0,
            }
            if temperature > 0:
                kwargs["temperature"] = max(0.01, min(float(temperature), 2.0))
            with torch.inference_mode():
                output = model.generate(**kwargs)
            new_tokens = output[0, prompt_ids.shape[1]:]
            content = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            return {
                "content": content,
                "prompt_tokens": int(prompt_ids.numel()),
                "completion_tokens": int(new_tokens.numel()),
            }


RUNTIME = ModelRuntime()


class GatewayHandler(BaseHTTPRequestHandler):
    server_version = "RegBenchGateway/1.0"

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[gateway] {self.address_string()} {fmt % args}", flush=True)

    def _json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _authorized(self) -> bool:
        expected = os.environ.get("REGBENCH_GATEWAY_TOKEN", "").strip()
        if not expected:
            return True
        supplied = self.headers.get("Authorization", "")
        if supplied.startswith("Bearer "):
            supplied = supplied[7:]
        return hmac.compare_digest(supplied, expected)

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self._json(HTTPStatus.OK, {"status": "ok", "active_model": RUNTIME.alias, "models": available_models()})
        elif self.path == "/v1/models":
            self._json(HTTPStatus.OK, {"object": "list", "data": available_models()})
        else:
            self._json(HTTPStatus.NOT_FOUND, {"error": {"message": "Not found"}})

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/v1/chat/completions":
            self._json(HTTPStatus.NOT_FOUND, {"error": {"message": "Not found"}})
            return
        if not self._authorized():
            self._json(HTTPStatus.UNAUTHORIZED, {"error": {"message": "Invalid gateway token"}})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length))
            alias = str(payload.get("model", "")).strip().lower()
            messages = payload.get("messages")
            if not isinstance(messages, list) or not messages:
                raise ValueError("messages must be a non-empty list")
            clean_messages = []
            for item in messages:
                if not isinstance(item, dict) or item.get("role") not in {"system", "user", "assistant"}:
                    raise ValueError("Each message needs a valid role and content")
                clean_messages.append({"role": item["role"], "content": str(item.get("content", ""))})
            started = time.perf_counter()
            result = RUNTIME.generate(alias, clean_messages, float(payload.get("temperature", 0)), int(payload.get("max_tokens", 256)))
            elapsed = time.perf_counter() - started
            self._json(HTTPStatus.OK, {
                "id": f"chatcmpl-{uuid.uuid4().hex}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": alias,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": result["content"]}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": result["prompt_tokens"], "completion_tokens": result["completion_tokens"], "total_tokens": result["prompt_tokens"] + result["completion_tokens"]},
                "regbench": {"latency_seconds": round(elapsed, 3), "adapter": alias},
            })
        except FileNotFoundError as error:
            self._json(HTTPStatus.CONFLICT, {"error": {"message": str(error)}})
        except (ValueError, json.JSONDecodeError) as error:
            self._json(HTTPStatus.BAD_REQUEST, {"error": {"message": str(error)}})
        except Exception as error:
            self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": {"message": str(error)}})


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve local Title 12 checkpoints for RegBench")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8090)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), GatewayHandler)
    print(f"[gateway] listening on http://{args.host}:{args.port}", flush=True)
    print(f"[gateway] available: {json.dumps(available_models())}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        RUNTIME._release()


if __name__ == "__main__":
    main()
