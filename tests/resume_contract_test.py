from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from trainers.index_resume_contract import (
    aggregate_file_fingerprint,
    evaluator_full_sft_artifact_paths,
    resume_artifacts_path,
    resolve_resume_checkpoint,
    validate_recent_checkpoints,
    verify_resume_checkpoint,
    write_epoch_resume_metadata,
)

ROOT = Path(__file__).parents[1]


class ResumeContractTests(unittest.TestCase):
    def test_epoch_resume_requires_trust_and_exact_checkpoint_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "model_epoch02_step2000.pth"
            checkpoint.write_bytes(b"trusted optimizer state")
            contract = {"schema_version": "1.0.0", "training": {"seed": 1234}}
            write_epoch_resume_metadata(
                checkpoint, contract, completed_epochs=2, global_step=2000
            )
            with self.assertRaisesRegex(ValueError, "trust-resume-state"):
                verify_resume_checkpoint(
                    checkpoint,
                    contract,
                    trust_resume_state=False,
                    target_epochs=10,
                )
            verified = verify_resume_checkpoint(
                checkpoint,
                contract,
                trust_resume_state=True,
                target_epochs=10,
            )
            self.assertEqual(verified["completed_epochs"], 2)
            checkpoint.write_bytes(b"changed optimizer state")
            with self.assertRaisesRegex(ValueError, "content does not match"):
                verify_resume_checkpoint(
                    checkpoint,
                    contract,
                    trust_resume_state=True,
                    target_epochs=10,
                )

    def test_resume_rejects_contract_drift_and_completed_target(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "model_epoch03_step3000.pth"
            checkpoint.write_bytes(b"checkpoint")
            contract = {"inputs": {"dataset": "a"}, "runtime": {"torch": "fixture"}}
            write_epoch_resume_metadata(
                checkpoint, contract, completed_epochs=3, global_step=3000
            )
            with self.assertRaisesRegex(ValueError, "training contract"):
                verify_resume_checkpoint(
                    checkpoint,
                    {"inputs": {"dataset": "b"}, "runtime": {"torch": "fixture"}},
                    trust_resume_state=True,
                    target_epochs=10,
                )
            with self.assertRaisesRegex(ValueError, "below target epochs"):
                verify_resume_checkpoint(
                    checkpoint,
                    contract,
                    trust_resume_state=True,
                    target_epochs=3,
                )

    def test_auto_resume_selects_only_the_latest_epoch_with_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            contract = {"fixture": True}
            earlier = root / "model_epoch01_step1000.pth"
            later = root / "model_epoch02_step1800.pth"
            legacy = root / "model_epoch09_step9000.pth"
            step = root / "model_step9999.pth"
            for path in (earlier, later, legacy, step):
                path.write_bytes(path.name.encode())
            write_epoch_resume_metadata(
                earlier, contract, completed_epochs=1, global_step=1000
            )
            write_epoch_resume_metadata(
                later, contract, completed_epochs=2, global_step=1800
            )
            self.assertEqual(resolve_resume_checkpoint("auto", root), later.resolve())

    def test_auto_resume_rejects_a_terminal_checkpoint_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "target.pth"
            target.write_bytes(b"checkpoint")
            linked = root / "model_epoch01_step1000.pth"
            linked.symlink_to(target)
            write_epoch_resume_metadata(
                target, {"fixture": True}, completed_epochs=1, global_step=1000
            )
            linked_metadata = linked.with_name(f"{linked.name}.resume.json")
            linked_metadata.write_bytes(
                target.with_name(f"{target.name}.resume.json").read_bytes()
            )
            with self.assertRaisesRegex(ValueError, "must not be a symlink"):
                resolve_resume_checkpoint("auto", root)

    def test_recent_checkpoint_cleanup_paths_cannot_escape_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "output"
            output.mkdir()
            safe = output / "model_step1000.pth"
            self.assertEqual(
                validate_recent_checkpoints([str(safe)], output), [str(safe.resolve())]
            )
            outside = root / "model_step2000.pth"
            with self.assertRaisesRegex(ValueError, "escaped"):
                validate_recent_checkpoints([str(outside)], output)
            target = output / "model_step3000.pth"
            target.write_bytes(b"target")
            linked = output / "model_step4000.pth"
            linked.symlink_to(target)
            with self.assertRaisesRegex(ValueError, "symlink"):
                validate_recent_checkpoints([str(linked)], output)

    def test_feature_fingerprint_rejects_symlinks_and_detects_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            feature = root / "feature.npy"
            feature.write_bytes(b"first")
            first = aggregate_file_fingerprint(
                [("sample:codes", feature)], label="training features"
            )
            feature.write_bytes(b"second")
            second = aggregate_file_fingerprint(
                [("sample:codes", feature)], label="training features"
            )
            self.assertNotEqual(first["sha256"], second["sha256"])
            linked = root / "linked.npy"
            linked.symlink_to(feature)
            with self.assertRaisesRegex(ValueError, "symlink"):
                aggregate_file_fingerprint(
                    [("sample:codes", linked)], label="training features"
                )

    def test_trainer_and_launcher_expose_guarded_epoch_resume(self) -> None:
        trainer = (ROOT / "trainers" / "train_gpt_v2.py").read_text()
        launcher = (ROOT / "scripts" / "train.sh").read_text()
        self.assertIn('"--trust-resume-state"', trainer)
        self.assertIn("verify_resume_checkpoint(", trainer)
        self.assertIn("completed_epochs=epoch + 1", trainer)
        self.assertIn('"rng_state"', trainer)
        self.assertIn("random.setstate(", trainer)
        self.assertIn("np.random.set_state(", trainer)
        self.assertIn("torch.cuda.set_rng_state_all(", trainer)
        self.assertIn("weights_only=False", trainer)
        self.assertIn("TRUST_RESUME_STATE:-0", launcher)
        self.assertNotIn("RESUME:-auto", launcher)
        self.assertIn('"--deterministic"', trainer)
        self.assertIn("configure_determinism(args.deterministic)", trainer)
        self.assertIn("torch.use_deterministic_algorithms(True)", trainer)
        self.assertIn(
            'os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")',
            trainer,
        )
        self.assertIn('"deterministic": args.deterministic', trainer)
        self.assertIn("canonicalize_tensor_tree(state[\"optimizer\"])", trainer)
        self.assertIn('"--no-latest"', trainer)
        self.assertIn('"write_latest": args.write_latest', trainer)
        self.assertIn("if args.write_latest:", trainer)
        self.assertIn("os.replace(partial, target)", trainer)
        self.assertIn("DETERMINISTIC:-0", launcher)
        self.assertIn("DETERMINISTIC must equal 0 or 1", launcher)
        self.assertIn("WRITE_LATEST:-1", launcher)
        self.assertIn("WRITE_LATEST must equal 0 or 1", launcher)
        self.assertIn("completed_epochs=completed_epochs", trainer)
        self.assertIn("state_completed_epochs != completed_epochs", trainer)
        self.assertIn("resume_artifacts=artifacts", trainer)

    def test_launcher_rejects_ambiguous_deterministic_value(self) -> None:
        result = subprocess.run(
            ["bash", str(ROOT / "scripts" / "train.sh")],
            cwd=ROOT,
            env={**os.environ, "DETERMINISTIC": "true"},
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("DETERMINISTIC must equal 0 or 1", result.stderr)

    def test_launcher_rejects_ambiguous_latest_value(self) -> None:
        result = subprocess.run(
            ["bash", str(ROOT / "scripts" / "train.sh")],
            cwd=ROOT,
            env={**os.environ, "WRITE_LATEST": "false"},
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("WRITE_LATEST must equal 0 or 1", result.stderr)

    def test_schema_11_evaluator_artifacts_are_live_bound(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "model_epoch02_step2000.pth"
            checkpoint.write_bytes(b"combined-loader-state")
            artifacts = resume_artifacts_path(checkpoint)
            artifacts.mkdir()
            for name, payload in {
                "model-state.pt": b"model",
                "optimizer-state.pt": b"optimizer",
                "scheduler-state.pt": b"scheduler",
                "trainer-state.json": b'{"completed_epochs":2}\n',
                "rng-state.pt": b"rng",
            }.items():
                (artifacts / name).write_bytes(payload)
            contract = {"schema_version": "1.0.0", "training": {"seed": 1234}}
            metadata_path = write_epoch_resume_metadata(
                checkpoint,
                contract,
                completed_epochs=2,
                global_step=2000,
                resume_artifacts=artifacts,
            )
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["schema_version"], "1.1.0")
            verified = verify_resume_checkpoint(
                checkpoint,
                contract,
                trust_resume_state=True,
                target_epochs=10,
            )
            self.assertEqual(verified["resume_artifacts"], metadata["resume_artifacts"])
            self.assertEqual(
                {
                    role: path.name
                    for role, path in evaluator_full_sft_artifact_paths(checkpoint).items()
                },
                {
                    "model_state": "model-state.pt",
                    "optimizer_state": "optimizer-state.pt",
                    "scheduler_state": "scheduler-state.pt",
                    "trainer_state": "trainer-state.json",
                    "rng_state": "rng-state.pt",
                },
            )

            (artifacts / "rng-state.pt").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "do not match"):
                verify_resume_checkpoint(
                    checkpoint,
                    contract,
                    trust_resume_state=True,
                    target_epochs=10,
                )

    def test_resume_artifact_manifest_rejects_cross_role_hardlinks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifacts = root / "artifacts"
            artifacts.mkdir()
            optimizer = artifacts / "optimizer-state.pt"
            optimizer.write_bytes(b"shared")
            os.link(optimizer, artifacts / "scheduler-state.pt")
            checkpoint = root / "model_epoch01_step1000.pth"
            checkpoint.write_bytes(b"checkpoint")
            with self.assertRaisesRegex(ValueError, "hardlink alias"):
                write_epoch_resume_metadata(
                    checkpoint,
                    {"fixture": True},
                    completed_epochs=1,
                    global_step=1000,
                    resume_artifacts=artifacts,
                )

    def test_evaluator_mapping_rejects_unassigned_state_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "model_epoch01_step1000.pth"
            checkpoint.write_bytes(b"combined-loader-state")
            artifacts = resume_artifacts_path(checkpoint)
            artifacts.mkdir()
            for name in (
                "model-state.pt",
                "optimizer-state.pt",
                "scheduler-state.pt",
                "trainer-state.json",
                "rng-state.pt",
                "unassigned-state.pt",
            ):
                (artifacts / name).write_bytes(name.encode())
            write_epoch_resume_metadata(
                checkpoint,
                {"fixture": True},
                completed_epochs=1,
                global_step=1000,
                resume_artifacts=artifacts,
            )
            with self.assertRaisesRegex(ValueError, "exactly the five"):
                evaluator_full_sft_artifact_paths(checkpoint)


if __name__ == "__main__":
    unittest.main()
