#!/usr/bin/env python3
"""Execute IndexTTS2 full SFT through the Instavar Voice lifecycle."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).parents[1]
STAGES = {"preflight", "train", "infer", "evaluate", "package"}


def _path(name: str, *, directory: bool = False) -> Path:
    value = os.environ.get(name, "").strip()
    if not value:
        raise ValueError(f"{name} is required")
    unresolved = Path(value).expanduser()
    if unresolved.is_symlink():
        raise FileNotFoundError(f"{name} is a symlink: {unresolved}")
    path = unresolved.resolve()
    valid = path.is_dir() if directory else path.is_file()
    if not valid:
        kind = "directory" if directory else "file"
        raise FileNotFoundError(f"{name} non-symlink {kind} not found: {path}")
    return path


def _work() -> Path:
    return _path("INSTAVAR_VOICE_WORK_DIR", directory=True)


def _persistent_package_root() -> Path:
    root = _path("PERSISTED_PACKAGE_ROOT", directory=True)
    protected = {
        "lifecycle work directory": _work(),
        "repository checkout": REPO_ROOT.resolve(),
        "IndexTTS2 upstream checkout": _path("INDEXTTS_UPSTREAM_DIR", directory=True),
        "prepared dataset tree": _path("PREPARED_DATA_ROOT", directory=True),
        "model dependency directory": _path("MODEL_DIR", directory=True),
    }
    for label, path in protected.items():
        if root == path or root.is_relative_to(path):
            raise ValueError(f"PERSISTED_PACKAGE_ROOT must be outside the {label}")
    return root


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _probe_persistent_package_root(root: Path) -> dict[str, Any]:
    probe_path: Path | None = None
    linked_path: Path | None = None
    linked_created = False
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=root,
            prefix=".instavar-voice-persistence-probe.",
            suffix=".partial",
            delete=False,
        ) as probe:
            probe_path = Path(probe.name)
            probe.write(b"instavar-voice-persistence-probe-v1\n")
            probe.flush()
            os.fsync(probe.fileno())
        linked_path = probe_path.with_suffix(".linked")
        os.link(probe_path, linked_path)
        linked_created = True
        _fsync_directory(root)
        if linked_path.read_bytes() != probe_path.read_bytes():
            raise ValueError("persistent package root failed its atomic publication probe")
        identity = root.stat()
        return {
            "writable": True,
            "atomic_hard_link": True,
            "device": identity.st_dev,
            "inode": identity.st_ino,
        }
    except OSError as error:
        raise ValueError(f"PERSISTED_PACKAGE_ROOT cannot publish an atomic package: {error}") from error
    finally:
        if linked_path is not None and linked_created:
            linked_path.unlink(missing_ok=True)
        if probe_path is not None:
            probe_path.unlink(missing_ok=True)


def _locked_persistent_package_root(preflight: dict[str, Any]) -> Path:
    root = _persistent_package_root()
    recorded_root = preflight.get("persistent_package_root")
    recorded_device = preflight.get("persistence_probe", {}).get("device")
    recorded_inode = preflight.get("persistence_probe", {}).get("inode")
    identity = root.stat()
    if (
        recorded_root != str(root)
        or recorded_device != identity.st_dev
        or recorded_inode != identity.st_ino
    ):
        raise ValueError("PERSISTED_PACKAGE_ROOT changed after preflight")
    return root


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_filename(value: str) -> str:
    path = Path(value)
    if not value or value in {".", ".."} or path.is_absolute() or len(path.parts) != 1 or path.suffix != ".pth":
        raise ValueError("SELECTED_CHECKPOINT_NAME must be one safe .pth filename")
    return value


def _run(command: list[str], *, environment: dict[str, str] | None = None, capture: bool = False) -> str:
    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=environment,
        capture_output=capture,
        text=capture,
        check=False,
    )
    if result.returncode != 0:
        detail = (result.stderr or "").strip() if capture else ""
        raise RuntimeError(f"command failed with exit code {result.returncode}: {command[0]}: {detail}")
    return (result.stdout or "").strip() if capture else ""


def _git_head(repository: Path) -> str:
    return _run(["git", "-C", str(repository), "rev-parse", "HEAD"], capture=True)


def _git_clean(repository: Path) -> bool:
    return not _run(
        ["git", "-C", str(repository), "status", "--porcelain=v1", "--untracked-files=all"], capture=True
    )


def _archive(source: Path, destination: Path, *, arcname: str) -> None:
    if source.is_symlink() or not source.is_dir():
        raise ValueError(f"archive source must be a non-symlink directory: {source}")
    files: list[Path] = []
    for path in source.rglob("*"):
        if path.is_symlink():
            raise ValueError("archive source must contain no symlinks")
        if path.is_file():
            files.append(path)
        elif not path.is_dir():
            raise ValueError(f"archive source contains an unsupported entry: {path}")
    if not files:
        raise ValueError("archive source must contain files")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(destination, "w") as archive:
        archive.add(source, arcname=arcname, recursive=True)


def _verify_persisted_package(path: Path, expected_sha256: str) -> None:
    if path.is_symlink() or not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"persisted package is missing, empty, or unsafe: {path}")
    actual_sha256 = _sha256(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(f"persisted package hash mismatch: expected {expected_sha256}, got {actual_sha256}")


def _persist_package(source: Path, root: Path) -> dict[str, Any]:
    if source.is_symlink() or not source.is_file() or source.stat().st_size == 0:
        raise ValueError(f"package source is missing, empty, or unsafe: {source}")
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"persistent package root is missing or unsafe: {root}")
    package_sha256 = _sha256(source)
    destination = root / f"indextts2-full-sft-package-sha256-{package_sha256}.tar"
    reused_existing = destination.exists() or destination.is_symlink()
    if reused_existing:
        _verify_persisted_package(destination, package_sha256)
    else:
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=root,
                prefix=f".{destination.name}.",
                suffix=".partial",
                delete=False,
            ) as target:
                temporary_path = Path(target.name)
                with source.open("rb") as package:
                    shutil.copyfileobj(package, target, length=1024 * 1024)
                target.flush()
                os.fsync(target.fileno())
            _verify_persisted_package(temporary_path, package_sha256)
            try:
                os.link(temporary_path, destination)
            except FileExistsError:
                reused_existing = True
            else:
                _fsync_directory(root)
            _verify_persisted_package(destination, package_sha256)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
    return {
        "schema_version": "1.0.0",
        "adaptation_mode": "full_sft",
        "package_sha256": package_sha256,
        "package_bytes": source.stat().st_size,
        "persisted_path": str(destination),
        "reused_existing": reused_existing,
    }


def _verify_sources(upstream: Path) -> dict[str, str]:
    experiment = json.loads(_path("INSTAVAR_VOICE_EXPERIMENT_MANIFEST").read_text(encoding="utf-8"))
    backend = experiment.get("backend", {})
    companion_revision = _git_head(REPO_ROOT)
    upstream_revision = _git_head(upstream)
    if not _git_clean(REPO_ROOT) or not _git_clean(upstream):
        raise ValueError("companion and upstream checkouts must be clean")
    if backend.get("instavar_revision") != companion_revision:
        raise ValueError("experiment backend.instavar_revision does not match the companion checkout")
    if backend.get("upstream_revision") != upstream_revision:
        raise ValueError("experiment backend.upstream_revision does not match the IndexTTS2 checkout")
    imported = _run(
        ["uv", "run", "python", "-c", "import pathlib, indextts; print(pathlib.Path(indextts.__file__).resolve())"],
        capture=True,
    )
    if not Path(imported).is_relative_to(upstream):
        raise ValueError("uv runtime does not import indextts from INDEXTTS_UPSTREAM_DIR")
    return {"companion_revision": companion_revision, "upstream_revision": upstream_revision, "indextts_module": imported}


def _verify_dataset_lineage() -> dict[str, Any]:
    from instavar_voice_lab.lineage import verify_dataset_lineage

    prepared_root = _path("PREPARED_DATA_ROOT", directory=True)
    for name in ("TRAIN_MANIFEST", "VAL_MANIFEST"):
        manifest = _path(name)
        if not manifest.is_relative_to(prepared_root):
            raise ValueError(f"{name} must be inside PREPARED_DATA_ROOT")
    document = json.loads(_path("DATASET_LINEAGE").read_text(encoding="utf-8"))
    return verify_dataset_lineage(
        document,
        producer_revision=_git_head(REPO_ROOT),
        inputs={
            "raw_train": (_path("RAW_TRAIN_JSONL"), "file"),
            "raw_validation": (_path("RAW_VALIDATION_JSONL"), "file"),
            "raw_test": (_path("RAW_TEST_JSONL"), "file"),
        },
        outputs={"prepared_data": (prepared_root, "tree")},
    )


def _preflight() -> None:
    from instavar_voice_lab.corpus import audit_corpus

    upstream = _path("INDEXTTS_UPSTREAM_DIR", directory=True)
    sources = _verify_sources(upstream)
    lineage = _verify_dataset_lineage()
    persistent_package_root = _persistent_package_root()
    persistence_probe = _probe_persistent_package_root(persistent_package_root)
    splits = {
        "train": _path("RAW_TRAIN_JSONL"),
        "validation": _path("RAW_VALIDATION_JSONL"),
        "test": _path("RAW_TEST_JSONL"),
    }
    audit = audit_corpus(splits, group_field=os.environ.get("CORPUS_GROUP_FIELD") or None)
    if audit["status"] != "passed":
        raise ValueError("corpus audit failed: " + "; ".join(audit["errors"]))
    for name in ("TRAIN_MANIFEST", "VAL_MANIFEST", "TOKENIZER", "CONFIG", "BASE_CHECKPOINT", "SPEAKER"):
        _path(name)
    _path("MODEL_DIR", directory=True)
    plan = json.loads(_path("GENERATION_PLAN").read_text(encoding="utf-8"))
    rows = [row for row in plan.get("samples", []) if row.get("candidate_id") == os.environ["CANDIDATE_ID"]]
    if plan.get("schema_version") not in {"1.0.0", "1.1.0"} or not rows:
        raise ValueError("GENERATION_PLAN must be schema 1.0.0 or 1.1.0 and contain CANDIDATE_ID rows")
    _safe_filename(os.environ["SELECTED_CHECKPOINT_NAME"])
    _write_json(
        _work() / "preflight" / "preflight.json",
        {
            "schema_version": "1.0.0",
            "status": "passed",
            "persistent_package_root": str(persistent_package_root),
            "persistence_probe": persistence_probe,
            "corpus_audit": audit,
            "generation_rows": len(rows),
            "sources": sources,
            "dataset_lineage": lineage,
        },
    )


def _train() -> None:
    _verify_dataset_lineage()
    work = _work()
    output = work / "train" / "output"
    environment = os.environ.copy()
    environment.update({"OUTPUT_DIR": str(output), "AUDIT_CORPUS": "0", "PYTHON": sys.executable})
    _run(["bash", "scripts/train.sh"], environment=environment)
    selected = output / _safe_filename(os.environ["SELECTED_CHECKPOINT_NAME"])
    if selected.is_symlink() or not selected.is_file() or selected.stat().st_size == 0:
        raise ValueError(f"selected checkpoint was not produced safely: {selected}")
    shutil.copyfile(selected, work / "train" / "selected-checkpoint.pth")


def _inference_command(checkpoint: Path, output: Path) -> list[str]:
    command = [
        "uv", "run", "python", "inference_script.py",
        "--config", os.environ["CONFIG"], "--model-dir", os.environ["MODEL_DIR"],
        "--gpt-checkpoint", str(checkpoint), "--tokenizer", os.environ["TOKENIZER"],
        "--speaker", os.environ["SPEAKER"], "--text", os.environ.get("SMOKE_TEXT", "A held-out sentence verifies checkpoint reload."),
        "--output", str(output), "--seed", os.environ.get("SEED", "42"),
    ]
    if os.environ.get("FP16", "1") == "1":
        command.append("--fp16")
    return command


def _infer() -> None:
    work = _work()
    output = work / "infer" / "candidate.wav"
    _run(_inference_command(work / "train" / "selected-checkpoint.pth", output))
    if not output.is_file() or output.stat().st_size == 0:
        raise ValueError("fresh-process checkpoint inference did not produce audio")


def _evaluate() -> None:
    work = _work()
    output = work / "evaluate" / "output"
    command = [
        "uv", "run", "python", "tools/run_evaluation_suite.py",
        "--config", os.environ["CONFIG"], "--model-dir", os.environ["MODEL_DIR"],
        "--gpt-checkpoint", str(work / "train" / "selected-checkpoint.pth"),
        "--tokenizer", os.environ["TOKENIZER"], "--speaker", os.environ["SPEAKER"],
        "--generation-plan", os.environ["GENERATION_PLAN"], "--candidate-id", os.environ["CANDIDATE_ID"],
        "--output-dir", str(output), "--allow-invalid-output",
    ]
    if os.environ.get("FP16", "1") == "1":
        command.append("--fp16")
    _run(command)
    raw_observations = output / "generation-observations.json"
    receipt = output / "generation-attempt-receipt.json"
    bound_observations = output / "objective-observations.json"
    plan = _path("GENERATION_PLAN")
    producer_revision = _git_head(REPO_ROOT)
    _run([
        sys.executable, "-m", "instavar_voice_lab.cli", "build-generation-attempt-receipt",
        str(raw_observations), "--plan", str(plan), "--audio-base-dir", str(output),
        "--producer-name", "indextts2-evaluation-runner", "--producer-revision", producer_revision,
        "--output", str(receipt),
    ])
    _run([
        sys.executable, "-m", "instavar_voice_lab.cli", "apply-generation-attempt-receipt",
        str(raw_observations), str(receipt), "--plan", str(plan), "--audio-base-dir", str(output),
        "--output", str(bound_observations),
    ])
    _archive(output, work / "evaluate" / "evaluation-bundle.tar", arcname="evaluation")


def _package() -> None:
    work = _work()
    preflight = json.loads((work / "preflight" / "preflight.json").read_text(encoding="utf-8"))
    staging = work / "package" / "staging"
    staging.mkdir(parents=True, exist_ok=False)
    pruned = staging / "gpt-finetuned-pruned.pth"
    _run(["uv", "run", "python", "tools/prune_gpt_checkpoint.py", "--input", str(work / "train" / "selected-checkpoint.pth"), "--output", str(pruned)])
    sources = {
        "evaluation-bundle.tar": work / "evaluate" / "evaluation-bundle.tar",
        "preflight.json": work / "preflight" / "preflight.json",
        "smoke-candidate.wav": work / "infer" / "candidate.wav",
        "experiment-manifest.json": _path("INSTAVAR_VOICE_EXPERIMENT_MANIFEST"),
        "generation-plan.json": _path("GENERATION_PLAN"),
        "dataset-lineage.json": _path("DATASET_LINEAGE"),
        "model-config.yaml": _path("CONFIG"),
        "tokenizer.model": _path("TOKENIZER"),
    }
    for name, source in sources.items():
        if source.is_symlink() or not source.is_file() or source.stat().st_size == 0:
            raise ValueError(f"package source is missing, empty, or unsafe: {source}")
        shutil.copyfile(source, staging / name)
    files = [
        {"path": path.name, "sha256": _sha256(path), "bytes": path.stat().st_size}
        for path in sorted(staging.iterdir()) if path.is_file()
    ]
    _write_json(staging / "package-manifest.json", {"schema_version": "1.0.0", "backend_id": "indextts2-full-sft-pytorch", "files": files, "evidence_boundary": "The pruned checkpoint and evidence completed the lifecycle; perceptual quality and distribution rights remain separate gates."})
    package = work / "package" / "checkpoint-package.tar"
    _archive(staging, package, arcname="package")
    receipt = _persist_package(package, _locked_persistent_package_root(preflight))
    _write_json(work / "package" / "persisted-package.json", receipt)


def run(stage: str) -> None:
    actions = {"preflight": _preflight, "train": _train, "infer": _infer, "evaluate": _evaluate, "package": _package}
    if stage not in actions:
        raise ValueError(f"unknown lifecycle stage: {stage}")
    actions[stage]()
    if stage in {"preflight", "train"}:
        _verify_dataset_lineage()
    result = Path(os.environ["INSTAVAR_VOICE_STAGE_RESULT"])
    _write_json(result, {"schema_version": "1.0.0", "stage": stage, "status": "passed"})


def main(argv: list[str] | None = None) -> int:
    values = sys.argv[1:] if argv is None else argv
    if len(values) != 1:
        print("usage: instavar_voice_lifecycle.py STAGE", file=sys.stderr)
        return 2
    try:
        run(values[0])
    except (KeyError, OSError, RuntimeError, ValueError, json.JSONDecodeError, tarfile.TarError) as error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
