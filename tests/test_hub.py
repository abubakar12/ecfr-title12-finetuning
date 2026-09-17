import tempfile
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from ecfr_pipeline import corpus, hub


class HubTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.cfg = {"experiment_dir": str(self.root), "paths": {"outputs_dir": str(self.root)},
                    "hub": {"enabled": False}}
        self.directory = self.root / "training" / "adapter"
        self.directory.mkdir(parents=True)
        self.artifacts(self.directory)
        self.run = {"smoke": False, "model": {"id": "owner/base", "revision": "a" * 40},
                    "runtime": {"device": "mps"}, "segments_sha256": "frozen-segments"}
        corpus.write_json(self.root / "training/run.json", self.run)
        corpus.write_json(self.root / "corpus.lock.json", {"documents_sha256": "frozen-documents"})
        corpus.write_json(self.root / "snapshot.json", {"date": "2026-09-15"})
        self.complete()
        self.api = MagicMock()
        self.api.create_commit.return_value.oid = "uploaded-commit"
        self.connection = patch.object(hub, "connection", return_value=(self.api, "researcher", "ecfr", True))
        self.connection.start()
        self.addCleanup(self.connection.stop)

    def artifacts(self, directory):
        for name in ("adapter_model.safetensors", "tokenizer.json", "tokenizer_config.json"):
            (directory / name).write_text("fixture")
        corpus.write_json(directory / "adapter_config.json", {"base_model_name_or_path": "owner/base"})
        (directory / "training_args.bin").write_text("must stay local")
        (directory / ".env").write_text("secret fixture must stay local")

    def complete(self):
        corpus.write_json(self.root / "training/completed.json", {
            "run_sha256": corpus.file_hash(self.root / "training/run.json"),
            "adapter_hashes": {p.name: corpus.file_hash(p) for p in self.directory.iterdir()}})

    def test_atomic_allowlisted_upload_with_pinned_card(self):
        receipt = hub.publish(self.cfg)
        self.api.create_repo.assert_called_once_with(repo_id=receipt["repo_id"], repo_type="model", private=True, exist_ok=True)
        operations = self.api.create_commit.call_args.kwargs["operations"]
        names = {op.path_in_repo for op in operations}
        self.assertEqual(names, {"adapter_config.json", "adapter_model.safetensors", "tokenizer.json", "tokenizer_config.json", "README.md", "provenance.json"})
        card = next(op.path_or_fileobj for op in operations if op.path_in_repo == "README.md").decode()
        self.assertIn("a" * 40, card)
        self.assertEqual(corpus.read_json(self.root / "training/hub_upload.json")["commit"], "uploaded-commit")

    def test_retry_preserves_same_destination_and_local_completion(self):
        self.api.create_commit.side_effect = OSError("offline")
        with self.assertRaisesRegex(RuntimeError, "saved locally"):
            hub.publish(self.cfg)
        failed_repo = self.api.create_repo.call_args.kwargs["repo_id"]
        self.assertTrue((self.root / "training/completed.json").exists())
        self.api.create_commit.side_effect = None
        self.assertEqual(hub.publish(self.cfg)["repo_id"], failed_repo)

    def test_changed_adapter_rejected_before_network(self):
        (self.directory / "adapter_model.safetensors").write_text("tampered")
        with self.assertRaisesRegex(ValueError, "changed"):
            hub.publish(self.cfg)
        self.api.create_repo.assert_not_called()

    def test_smoke_and_missing_revision_rejected(self):
        for smoke, revision, expected in [(True, "a" * 40, "Smoke"), (False, None, "revision")]:
            self.run["smoke"] = smoke
            self.run["model"]["revision"] = revision
            corpus.write_json(self.root / "training/run.json", self.run)
            self.complete()
            with self.assertRaisesRegex(ValueError, expected):
                hub.publish(self.cfg)
        self.api.create_repo.assert_not_called()

    def test_unfinished_run_rejected(self):
        (self.root / "training/completed.json").unlink()
        with self.assertRaises(FileNotFoundError):
            hub.publish(self.cfg)
        self.api.create_repo.assert_not_called()

    def test_distinct_stages_and_weights_get_distinct_repositories(self):
        repos = []
        for stage in ("sft", "dpo", "grpo"):
            directory = self.root / f"{stage}-adapter"
            directory.mkdir()
            self.artifacts(directory)
            corpus.write_json(self.root / "manifests" / f"{stage}_manifest.json", {"stage": stage})
            hub.finish_legacy(self.cfg, stage, directory, SimpleNamespace(config=SimpleNamespace(_commit_hash="b" * 40)), False)
            repos.append(hub.publish(self.cfg, stage)["repo_id"])
        self.assertEqual(len(set(repos)), 3)
        original = hub.publish(self.cfg)["repo_id"]
        (self.directory / "adapter_model.safetensors").write_text("new completed model")
        self.complete()
        self.assertNotEqual(hub.publish(self.cfg)["repo_id"], original)

    def test_legacy_smoke_never_auto_publishes(self):
        self.cfg["hub"]["enabled"] = True
        directory = self.root / "sft-adapter"
        directory.mkdir()
        self.artifacts(directory)
        corpus.write_json(self.root / "manifests/sft_manifest.json", {})
        hub.finish_legacy(self.cfg, "sft", directory, SimpleNamespace(config=SimpleNamespace(_commit_hash="b" * 40)), True)
        self.api.create_repo.assert_not_called()


class ConnectionTests(unittest.TestCase):
    def test_cached_login_defaults_to_private_personal_namespace(self):
        with patch.dict(os.environ, {}, clear=True), patch("huggingface_hub.HfApi") as api:
            api.return_value.whoami.return_value = {"name": "researcher"}
            _, namespace, prefix, private = hub.connection({})
        self.assertEqual((namespace, prefix, private), ("researcher", "ecfr-title12", True))

    def test_environment_can_select_organization_and_public_visibility(self):
        with patch.dict(os.environ, {"HF_REPO_NAMESPACE": "lab", "HF_REPO_PRIVATE": "false"}, clear=True), patch("huggingface_hub.HfApi") as api:
            api.return_value.whoami.return_value = {"name": "researcher"}
            _, namespace, _, private = hub.connection({})
        self.assertEqual((namespace, private), ("lab", False))


if __name__ == "__main__":
    unittest.main()
