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
        self.assertIn("error_type", source)


if __name__ == "__main__":
    unittest.main()
