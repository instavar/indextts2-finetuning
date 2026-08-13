from __future__ import annotations

import argparse
import ast
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from evaluation_contract import configure_checkpoint


class EvaluationSuiteTests(unittest.TestCase):
    def test_single_inference_has_explicit_seed(self) -> None:
        source = (ROOT / "inference_script.py").read_text(encoding="utf-8")
        self.assertIn('"--seed"', source)
        self.assertIn("torch.manual_seed(args.seed)", source)

    def test_suite_loads_engine_once_and_records_failures(self) -> None:
        source = (ROOT / "tools" / "run_evaluation_suite.py").read_text(encoding="utf-8")
        contract_source = (ROOT / "tools" / "evaluation_contract.py").read_text(encoding="utf-8")
        ast.parse(source)
        ast.parse(contract_source)
        self.assertEqual(source.count("engine = IndexTTS2("), 1)
        self.assertIn("generation-observations.json", source)
        self.assertIn("allow-invalid-output", source)
        self.assertIn('not in {"1.0.0", "1.1.0"}', source)
        self.assertNotIn("max_memory_allocated()) if torch.cuda.is_available() else 0", source)
        self.assertIn("error_type", source)
        self.assertIn("artifact set id and sha256 must be provided together", source)
        self.assertIn('"runtime_id": args.runtime_id', source)
        self.assertIn('"artifact_set_sha256": args.artifact_set_sha256', source)
        self.assertIn('"observation_schema_version": "1.0.0"', source)
        self.assertIn('choices=("full-sft", "base")', source)
        self.assertIn("full-sft mode requires --gpt-checkpoint", contract_source)
        self.assertIn("base mode forbids --gpt-checkpoint", contract_source)
        self.assertIn("base mode requires config.gpt_checkpoint", contract_source)
        self.assertIn('"artifact_mode": artifact_mode', source)
        self.assertIn('f"indextts2_pytorch_{device_family}_{artifact_mode}"', source)

    def test_checkpoint_modes_fail_closed_on_ambiguous_or_missing_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            model_dir = Path(temporary)
            base = model_dir / "gpt.pth"
            adapted = model_dir / "model_step14000.pth"
            base.write_bytes(b"base")
            adapted.write_bytes(b"adapted")

            config = SimpleNamespace(gpt_checkpoint="gpt.pth")
            args = argparse.Namespace(inference_mode="base", gpt_checkpoint=None, model_dir=model_dir)
            self.assertEqual(configure_checkpoint(args, config), "base")

            args.gpt_checkpoint = adapted
            with self.assertRaisesRegex(ValueError, "base mode forbids"):
                configure_checkpoint(args, config)

            args.inference_mode = "full-sft"
            args.gpt_checkpoint = adapted
            self.assertEqual(configure_checkpoint(args, config), "full_sft")
            self.assertEqual(config.gpt_checkpoint, str(adapted.resolve()))

            args.gpt_checkpoint = model_dir / "missing.pth"
            with self.assertRaisesRegex(ValueError, "existing file"):
                configure_checkpoint(args, config)

            args.inference_mode = "base"
            args.gpt_checkpoint = None
            config.gpt_checkpoint = "model_step14000.pth"
            with self.assertRaisesRegex(ValueError, "model-dir/gpt.pth"):
                configure_checkpoint(args, config)

    def test_lifecycle_binds_runtime_attempt_evidence(self):
        source = (ROOT / "scripts" / "instavar_voice_lifecycle.py").read_text(encoding="utf-8")
        self.assertIn("build-generation-attempt-receipt", source)
        self.assertIn("apply-generation-attempt-receipt", source)
        self.assertIn("objective-observations.json", source)


if __name__ == "__main__":
    unittest.main()
