import importlib.util
import unittest

from ecfr_pipeline import phase1_runtime as runtime


class RuntimeTests(unittest.TestCase):
    def test_auto_selects_available_accelerator(self):
        self.assertEqual(runtime.choose_device("auto", True, False), "cuda")
        self.assertEqual(runtime.choose_device("auto", False, True), "mps")
        self.assertEqual(runtime.choose_device("auto", False, False, smoke=True), "cpu")

    def test_unavailable_explicit_device_never_falls_back(self):
        with self.assertRaisesRegex(RuntimeError, "MPS was requested"):
            runtime.choose_device("mps", True, False)
        with self.assertRaisesRegex(RuntimeError, "CUDA was requested"):
            runtime.choose_device("cuda", False, True)

    def test_full_cpu_training_is_rejected(self):
        for requested in ("auto", "cpu"):
            with self.assertRaisesRegex(RuntimeError, "Full runs require"):
                runtime.choose_device(requested, False, False)

    def test_precision_is_explicit_and_smoke_is_small(self):
        self.assertEqual(runtime.choose_precision("auto"), "bf16")
        self.assertEqual(runtime.choose_precision("auto", smoke=True), "fp32")
        self.assertEqual(runtime.choose_precision("bf16", smoke=True), "bf16")
        with self.assertRaises(ValueError):
            runtime.choose_precision("fp16")

    def test_mps_disables_unsupported_amp_and_pinned_memory(self):
        mps = runtime.trainer_options({"device": "mps", "precision": "bf16"})
        self.assertFalse(mps["bf16"])
        self.assertFalse(mps["fp16"])
        self.assertFalse(mps["use_cpu"])
        self.assertFalse(mps["dataloader_pin_memory"])
        cuda = runtime.trainer_options({"device": "cuda", "precision": "bf16"})
        self.assertTrue(cuda["bf16"])
        self.assertTrue(cuda["dataloader_pin_memory"])

    def test_comparison_rejects_backend_and_precision_changes(self):
        trained = {"device": "mps", "precision": "bf16", "attention": "sdpa", "mps_fallback": False}
        runtime.check_evaluation_runtime(trained, dict(trained))
        for key, value in [("device", "cuda"), ("precision", "fp32"), ("attention", "eager"), ("mps_fallback", True)]:
            with self.assertRaisesRegex(ValueError, key):
                runtime.check_evaluation_runtime(trained, {**trained, key: value})

    def test_mps_oom_is_actionable_and_other_errors_are_preserved(self):
        with self.assertRaisesRegex(RuntimeError, "mps ran out of memory"):
            with runtime.memory_guard({"device": "mps"}, "preflight"):
                raise RuntimeError("MPS backend out of memory")
        with self.assertRaisesRegex(RuntimeError, "Unsupported operator"):
            with runtime.memory_guard({"device": "mps"}, "preflight"):
                raise RuntimeError("Unsupported operator")

    @unittest.skipUnless(importlib.util.find_spec("torch"), "Training stack not installed")
    def test_cpu_smoke_probe(self):
        result = runtime.resolve({"runtime": {"device": "cpu", "precision": "fp32"}}, smoke=True)
        self.assertEqual(result["device"], "cpu")
        self.assertEqual(result["precision"], "fp32")


if __name__ == "__main__":
    unittest.main()
