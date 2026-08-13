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
        required = {item["name"] for item in spec["required_environment"]}
        self.assertIn("PERSISTED_PACKAGE_ROOT", required)
        self.assertIn("package/persisted-package.json", spec["expected_artifacts"]["package"])
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

    def test_path_rejects_a_terminal_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "target"
            target.write_bytes(b"target")
            linked = root / "linked"
            linked.symlink_to(target)
            with (
                patch.dict(os.environ, {"FIXTURE_PATH": str(linked)}, clear=False),
                self.assertRaisesRegex(FileNotFoundError, "is a symlink"),
            ):
                LIFECYCLE._path("FIXTURE_PATH")

    def test_persist_package_is_content_addressed_and_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "checkpoint-package.tar"
            source.write_bytes(b"immutable package")
            store = root / "store"
            store.mkdir()

            first = LIFECYCLE._persist_package(source, store)
            destination = Path(first["persisted_path"])
            self.assertEqual(first["adaptation_mode"], "full_sft")
            self.assertTrue(destination.name.startswith("indextts2-full-sft-package-sha256-"))
            self.assertEqual(destination.read_bytes(), source.read_bytes())
            self.assertFalse(first["reused_existing"])

            second = LIFECYCLE._persist_package(source, store)
            self.assertEqual(second["package_sha256"], first["package_sha256"])
            self.assertTrue(second["reused_existing"])

            destination.write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                LIFECYCLE._persist_package(source, store)

    def test_persistent_package_root_rejects_protected_trees(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            work = root / "work"
            checkout = root / "checkout"
            upstream = root / "upstream"
            prepared = root / "prepared"
            model = root / "model"
            for path in (work, checkout, upstream, prepared, model):
                path.mkdir()
            (checkout / "packages").mkdir()
            (upstream / "packages").mkdir()
            (prepared / "packages").mkdir()
            (model / "packages").mkdir()
            environments = (
                ({"PERSISTED_PACKAGE_ROOT": str(work)}, "outside the lifecycle work directory"),
                ({"PERSISTED_PACKAGE_ROOT": str(checkout / "packages")}, "outside the repository checkout"),
                ({"PERSISTED_PACKAGE_ROOT": str(upstream / "packages")}, "outside the IndexTTS2 upstream checkout"),
                ({"PERSISTED_PACKAGE_ROOT": str(prepared / "packages")}, "outside the prepared dataset tree"),
                ({"PERSISTED_PACKAGE_ROOT": str(model / "packages")}, "outside the model dependency directory"),
            )
            for override, message in environments:
                with (
                    self.subTest(override=override),
                    patch.dict(
                        os.environ,
                        {
                            "INSTAVAR_VOICE_WORK_DIR": str(work),
                            "INDEXTTS_UPSTREAM_DIR": str(upstream),
                            "PREPARED_DATA_ROOT": str(prepared),
                            "MODEL_DIR": str(model),
                            **override,
                        },
                        clear=False,
                    ),
                    patch.object(LIFECYCLE, "REPO_ROOT", checkout),
                    self.assertRaisesRegex(ValueError, message),
                ):
                    LIFECYCLE._persistent_package_root()

    def test_persistence_probe_records_identity_and_leaves_no_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = LIFECYCLE._probe_persistent_package_root(root)
            identity = root.stat()
            self.assertTrue(result["writable"])
            self.assertTrue(result["atomic_hard_link"])
            self.assertEqual(result["device"], identity.st_dev)
            self.assertEqual(result["inode"], identity.st_ino)
            self.assertEqual(list(root.iterdir()), [])

    def test_persistence_probe_does_not_unlink_a_link_it_did_not_create(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with (
                patch.object(LIFECYCLE.os, "link", side_effect=FileExistsError("collision")),
                patch.object(Path, "unlink", autospec=True) as unlink,
                self.assertRaisesRegex(ValueError, "cannot publish an atomic package"),
            ):
                LIFECYCLE._probe_persistent_package_root(root)
            unlinked = [call.args[0] for call in unlink.call_args_list]
            self.assertEqual(len(unlinked), 1)
            self.assertTrue(str(unlinked[0]).endswith(".partial"))

    def test_package_root_is_bound_to_preflight_path_device_and_inode(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            work = root / "work"
            checkout = root / "checkout"
            upstream = root / "upstream"
            prepared = root / "prepared"
            model = root / "model"
            store = root / "store"
            other = root / "other"
            for path in (work, checkout, upstream, prepared, model, store, other):
                path.mkdir()
            identity = store.stat()
            environment = {
                "INSTAVAR_VOICE_WORK_DIR": str(work),
                "INDEXTTS_UPSTREAM_DIR": str(upstream),
                "PREPARED_DATA_ROOT": str(prepared),
                "MODEL_DIR": str(model),
                "PERSISTED_PACKAGE_ROOT": str(store),
            }
            preflight = {
                "persistent_package_root": str(store.resolve()),
                "persistence_probe": {"device": identity.st_dev, "inode": identity.st_ino},
            }
            with (
                patch.dict(os.environ, environment, clear=False),
                patch.object(LIFECYCLE, "REPO_ROOT", checkout),
            ):
                self.assertEqual(LIFECYCLE._locked_persistent_package_root(preflight), store.resolve())
                changed_path = {**preflight, "persistent_package_root": str(other)}
                with self.assertRaisesRegex(ValueError, "changed after preflight"):
                    LIFECYCLE._locked_persistent_package_root(changed_path)
                changed_device = {
                    **preflight,
                    "persistence_probe": {"device": identity.st_dev + 1, "inode": identity.st_ino},
                }
                with self.assertRaisesRegex(ValueError, "changed after preflight"):
                    LIFECYCLE._locked_persistent_package_root(changed_device)
                changed_inode = {
                    **preflight,
                    "persistence_probe": {"device": identity.st_dev, "inode": identity.st_ino + 1},
                }
                with self.assertRaisesRegex(ValueError, "changed after preflight"):
                    LIFECYCLE._locked_persistent_package_root(changed_inode)

                retired = root / "retired-store"
                store.rename(retired)
                store.mkdir()
                self.assertEqual(store.stat().st_dev, identity.st_dev)
                with self.assertRaisesRegex(ValueError, "changed after preflight"):
                    LIFECYCLE._locked_persistent_package_root(preflight)

    def test_package_stage_persists_archive_and_writes_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            work = root / "work"
            upstream = root / "upstream"
            prepared = root / "prepared"
            model = root / "model"
            store = root / "store"
            checkout = root / "checkout"
            for path in (work, upstream, prepared, model, store, checkout):
                path.mkdir()
            for path in (
                work / "preflight",
                work / "train",
                work / "evaluate",
                work / "infer",
                work / "package",
            ):
                path.mkdir()
            identity = store.stat()
            (work / "preflight" / "preflight.json").write_text(
                json.dumps(
                    {
                        "persistent_package_root": str(store.resolve()),
                        "persistence_probe": {
                            "device": identity.st_dev,
                            "inode": identity.st_ino,
                        },
                    }
                )
            )
            (work / "train" / "selected-checkpoint.pth").write_bytes(b"checkpoint")
            (work / "evaluate" / "evaluation-bundle.tar").write_bytes(b"evaluation")
            (work / "infer" / "candidate.wav").write_bytes(b"wav")
            inputs: dict[str, Path] = {}
            for name in (
                "INSTAVAR_VOICE_EXPERIMENT_MANIFEST",
                "GENERATION_PLAN",
                "DATASET_LINEAGE",
                "CONFIG",
                "TOKENIZER",
            ):
                path = root / f"{name.lower()}.fixture"
                path.write_bytes(name.encode())
                inputs[name] = path
            environment = {
                "INSTAVAR_VOICE_WORK_DIR": str(work),
                "INDEXTTS_UPSTREAM_DIR": str(upstream),
                "PREPARED_DATA_ROOT": str(prepared),
                "MODEL_DIR": str(model),
                "PERSISTED_PACKAGE_ROOT": str(store),
                **{name: str(path) for name, path in inputs.items()},
            }

            def fake_run(command: list[str], **_: object) -> str:
                self.assertIn("tools/prune_gpt_checkpoint.py", command)
                output = Path(command[command.index("--output") + 1])
                output.write_bytes(b"pruned checkpoint")
                return ""

            with (
                patch.dict(os.environ, environment, clear=False),
                patch.object(LIFECYCLE, "REPO_ROOT", checkout),
                patch.object(LIFECYCLE, "_run", side_effect=fake_run),
            ):
                LIFECYCLE._package()

            package = work / "package" / "checkpoint-package.tar"
            receipt = json.loads((work / "package" / "persisted-package.json").read_text())
            persisted = Path(receipt["persisted_path"])
            self.assertTrue(package.is_file())
            self.assertEqual(persisted.read_bytes(), package.read_bytes())
            self.assertEqual(receipt["adaptation_mode"], "full_sft")
            self.assertEqual(receipt["package_sha256"], LIFECYCLE._sha256(package))

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
