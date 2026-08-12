from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from instavar_voice_lab.lineage import build_dataset_lineage


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

    def test_dataset_lineage_binds_raw_splits_to_complete_prepared_tree(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw: dict[str, Path] = {}
            for split in ("train", "validation", "test"):
                audio = root / f"{split}.wav"
                audio.write_bytes(b"audio")
                manifest = root / f"raw-{split}.jsonl"
                manifest.write_text(json.dumps({"audio": str(audio), "text": split}) + "\n")
                raw[split] = manifest
            prepared = root / "prepared"
            prepared.mkdir()
            train = prepared / "gpt_pairs_train.jsonl"
            validation = prepared / "gpt_pairs_val.jsonl"
            train.write_text('{"id":"train"}\n')
            validation.write_text('{"id":"validation"}\n')
            (prepared / "codes.npy").write_bytes(b"codes")
            revision = subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
            ).stdout.strip()
            receipt = root / "dataset-lineage.json"
            receipt.write_text(
                json.dumps(
                    build_dataset_lineage(
                        lineage_id="index-fixture-v1",
                        producer_repository="instavar/indextts2-finetuning",
                        producer_revision=revision,
                        inputs={
                            "raw_train": (raw["train"], "file"),
                            "raw_validation": (raw["validation"], "file"),
                            "raw_test": (raw["test"], "file"),
                        },
                        outputs={"prepared_data": (prepared, "tree")},
                    )
                )
            )
            environment = {
                "RAW_TRAIN_JSONL": str(raw["train"]),
                "RAW_VALIDATION_JSONL": str(raw["validation"]),
                "RAW_TEST_JSONL": str(raw["test"]),
                "TRAIN_MANIFEST": str(train),
                "VAL_MANIFEST": str(validation),
                "PREPARED_DATA_ROOT": str(prepared),
                "DATASET_LINEAGE": str(receipt),
            }
            with patch.dict(os.environ, environment, clear=False):
                report = LIFECYCLE._verify_dataset_lineage()
            self.assertEqual(report["lineage_id"], "index-fixture-v1")
            (prepared / "codes.npy").write_bytes(b"changed")
            with (
                patch.dict(os.environ, environment, clear=False),
                self.assertRaisesRegex(ValueError, "prepared_data"),
            ):
                LIFECYCLE._verify_dataset_lineage()


if __name__ == "__main__":
    unittest.main()
