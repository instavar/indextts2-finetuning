from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location("index_lifecycle", ROOT / "scripts" / "instavar_voice_lifecycle.py")
assert SPEC and SPEC.loader
LIFECYCLE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LIFECYCLE)


class LifecycleBackendTests(unittest.TestCase):
    def test_backend_routes_all_stages_and_binds_full_sft(self) -> None:
        spec = json.loads((ROOT / "instavar-voice-backend.json").read_text())
        self.assertEqual(spec["schema_version"], "1.2.0")
        self.assertEqual(spec["capability_binding"]["adaptation"], "full_sft")
        for stage in ("preflight", "train", "infer", "evaluate", "package"):
            self.assertEqual(spec["commands"][stage][-1], stage)

    def test_selected_checkpoint_is_one_safe_pth_filename(self) -> None:
        self.assertEqual(LIFECYCLE._safe_filename("model_step14000.pth"), "model_step14000.pth")
        for unsafe in ("", "latest", "latest.pth/child", "../latest.pth", "/latest.pth"):
            with self.assertRaises(ValueError):
                LIFECYCLE._safe_filename(unsafe)

    def test_archive_rejects_empty_or_symlinked_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            empty = root / "empty"
            empty.mkdir()
            with self.assertRaises(ValueError):
                LIFECYCLE._archive(empty, root / "empty.tar", arcname="empty")
            target = root / "target"
            target.write_bytes(b"target")
            (empty / "link").symlink_to(target)
            with self.assertRaises(ValueError):
                LIFECYCLE._archive(empty, root / "linked.tar", arcname="linked")


if __name__ == "__main__":
    unittest.main()
