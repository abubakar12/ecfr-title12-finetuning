"""Publish completed adapters, never checkpoints or smoke runs."""
import hashlib
import json
import os
from pathlib import Path

from . import corpus


FILES = {"adapter_config.json", "adapter_model.safetensors", "tokenizer.json",
         "tokenizer_config.json", "special_tokens_map.json", "tokenizer.model",
         "vocab.json", "merges.txt", "added_tokens.json", "chat_template.jinja"}


def enabled(cfg):
    return cfg.get("hub", {}).get("enabled", False)


def connection(cfg):
    from huggingface_hub import HfApi
    api = HfApi()
    account = api.whoami()  # HF_TOKEN or `hf auth login`; never write credentials.
    options = cfg.get("hub", {})
    namespace = os.getenv("HF_REPO_NAMESPACE") or options.get("namespace") or account["name"]
    prefix = os.getenv("HF_REPO_PREFIX") or options.get("repo_prefix", "ecfr-title12")
    private = options.get("private", True)
    if "HF_REPO_PRIVATE" in os.environ:
        value = os.environ["HF_REPO_PRIVATE"].lower()
        if value not in {"true", "false"}:
            raise ValueError("HF_REPO_PRIVATE must be true or false")
        private = value == "true"
    return api, namespace, prefix, private


def check(cfg):
    _, namespace, prefix, private = connection(cfg)
    print(f"[hub] destination: {namespace}/{prefix}-<stage>-<fingerprint>; private={private}")


def finish_legacy(cfg, stage, directory, model, smoke):
    directory = Path(directory)
    manifest = Path(cfg["paths"]["outputs_dir"]) / "manifests" / f"{stage}_manifest.json"
    corpus.write_json(directory / "completed.json", {
        "smoke": smoke, "stage": stage,
        "model": {"id": corpus.read_json(directory / "adapter_config.json")["base_model_name_or_path"],
                  "revision": getattr(model.config, "_commit_hash", None)},
        "manifest_sha256": corpus.file_hash(manifest),
        "adapter_hashes": {p.name: corpus.file_hash(p) for p in directory.iterdir() if p.name in FILES},
    })
    if enabled(cfg) and not smoke:
        publish(cfg, stage)


def publish(cfg, stage="cpt"):
    if stage == "cpt" and "experiment_dir" in cfg:  # Phase-1 CPT (frozen experiment corpus)
        root = Path(cfg["experiment_dir"]) / "training"
        directory = root / "adapter"
        completed = corpus.read_json(root / "completed.json")
        run = corpus.read_json(root / "run.json")
        if corpus.file_hash(root / "run.json") != completed["run_sha256"]:
            raise ValueError("Training manifest changed after completion")
        smoke, model = run["smoke"], run["model"]
        provenance = {"stage": stage, "model": model, "runtime": run["runtime"],
                      "segments_sha256": run["segments_sha256"],
                      "run_sha256": completed["run_sha256"],
                      "corpus": corpus.read_json(root.parent / "corpus.lock.json"),
                      "snapshot_date": corpus.read_json(root.parent / "snapshot.json")["date"]}
    elif stage in {"cpt", "sft", "dpo", "grpo"}:
        directory = Path(cfg["paths"]["outputs_dir"]) / f"{stage}-adapter"
        root = directory
        completed = corpus.read_json(root / "completed.json")
        manifest = Path(cfg["paths"]["outputs_dir"]) / "manifests" / f"{stage}_manifest.json"
        if corpus.file_hash(manifest) != completed["manifest_sha256"]:
            raise ValueError("Training manifest changed after completion")
        smoke, model = completed["smoke"], completed["model"]
        provenance = {"stage": stage, "model": model, "manifest_sha256": completed["manifest_sha256"]}
    else:
        raise ValueError(f"Unknown training stage: {stage}")
    if smoke:
        raise ValueError("Smoke models cannot be published")
    if not model.get("revision"):
        raise ValueError("Base model revision was not recorded; cannot publish reproducibly")
    hashes = completed["adapter_hashes"]
    for name, expected in hashes.items():
        if Path(name).name != name or corpus.file_hash(directory / name) != expected:
            raise ValueError(f"Completed adapter changed: {name}")
    required = {"adapter_config.json", "adapter_model.safetensors", "tokenizer_config.json"}
    if not required <= hashes.keys() or not {"tokenizer.json", "tokenizer.model"} & hashes.keys():
        raise ValueError("Completed adapter/tokenizer package is incomplete")
    provenance["artifact_hashes"] = {name: hashes[name] for name in sorted(FILES & hashes.keys())}
    payload = json.dumps(provenance, sort_keys=True, indent=2).encode()
    fingerprint = hashlib.sha256(payload).hexdigest()
    from huggingface_hub import CommitOperationAdd
    from huggingface_hub.utils import validate_repo_id
    try:
        api, namespace, prefix, private = connection(cfg)
        repo_id = f"{namespace}/{prefix}-{stage}-{fingerprint[:12]}"
        validate_repo_id(repo_id)
        api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
        # A single commit makes the weights, tokenizer and provenance available together.
        operations = [CommitOperationAdd(path_in_repo=name, path_or_fileobj=str(directory / name))
                      for name in sorted(provenance["artifact_hashes"])]
        operations += [CommitOperationAdd(path_in_repo="provenance.json", path_or_fileobj=payload),
                       CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=card(repo_id, provenance).encode())]
        result = api.create_commit(repo_id=repo_id, repo_type="model", operations=operations,
                                   commit_message=f"Publish completed {stage} adapter {fingerprint[:12]}")
        receipt = {"repo_id": repo_id, "commit": result.oid, "url": f"https://huggingface.co/{repo_id}",
                   "fingerprint": fingerprint}
        corpus.write_json(root / "hub_upload.json", receipt)
        print(f"[hub] published: {receipt['url']} (revision {result.oid})")
        return receipt
    except Exception as exc:
        raise RuntimeError(f"Model is saved locally; Hub upload failed ({type(exc).__name__}). "
                           "Retry with push-model for CPT or train_and_upload.py --upload-only for legacy stages.") from exc


def card(repo_id, provenance):
    model, stage = provenance["model"], provenance["stage"]
    return f'''---
library_name: peft
base_model: {model["id"]}
pipeline_tag: text-generation
tags: [ecfr, finance, lora]
---

# {repo_id}

Completed {stage.upper()} LoRA adapter for eCFR research. This is an adapter;
the original base weights are required, including any base-model access agreement.
Base revision: `{model["revision"]}`. Artifact hashes and source provenance are in `provenance.json`.
No evaluation improvement is claimed. GRPO uses automated rewards, not human-feedback RLHF.

## Load for inference

Replace ADAPTER_COMMIT with the commit saved in the local `hub_upload.json` receipt.
Install the training environment dependencies. Select a device/dtype appropriate to your hardware.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

repo = "{repo_id}"
revision = "ADAPTER_COMMIT"
tokenizer = AutoTokenizer.from_pretrained(repo, revision=revision)
base = AutoModelForCausalLM.from_pretrained(
    "{model["id"]}", revision="{model["revision"]}", torch_dtype=torch.bfloat16)
model = PeftModel.from_pretrained(base, repo, revision=revision)
model.eval()
```

Optimizer/checkpoint state stays on the training machine; this package supports inference.
For resumed training retain the local checkpoints and pinned environment.
'''
