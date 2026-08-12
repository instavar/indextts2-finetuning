from __future__ import annotations

import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).parents[1]


class EvaluationSuiteTests(unittest.TestCase):
    def test_single_inference_has_explicit_seed(self) -> None:
        source = (ROOT / "inference_script.py").read_text(encoding="utf-8")
        self.assertIn('"--seed"', source)
        self.assertIn("torch.manual_seed(args.seed)", source)

    def test_suite_loads_engine_once_and_records_failures(self) -> None:
        source = (ROOT / "tools" / "run_evaluation_suite.py").read_text(encoding="utf-8")
        ast.parse(source)
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

    def test_lifecycle_binds_runtime_attempt_evidence(self):
        source = (ROOT / "scripts" / "instavar_voice_lifecycle.py").read_text(encoding="utf-8")
        self.assertIn("build-generation-attempt-receipt", source)
        self.assertIn("apply-generation-attempt-receipt", source)
        self.assertIn("objective-observations.json", source)


if __name__ == "__main__":
    unittest.main()
