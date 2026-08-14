"""Dependency-free guards for trusted IndexTTS2 full-SFT resume state."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "1.0.0"
EVIDENCE_SCHEMA_VERSION = "1.1.0"
SCHEMA_VERSIONS = {SCHEMA_VERSION, EVIDENCE_SCHEMA_VERSION}
EPOCH_CHECKPOINT_PATTERN = re.compile(
    r"model_epoch(?P<epoch>\d+)_step(?P<step>\d+)\.pth"
)
CHECKPOINT_PATTERN = re.compile(r"model_(?:step\d+|epoch\d+_step\d+)\.pth")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def regular_file_record(path: Path, *, label: str) -> dict[str, Any]:
    unresolved = path.expanduser()
    if unresolved.is_symlink():
        raise ValueError(f"{label} must not be a symlink: {unresolved}")
    resolved = unresolved.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} is missing: {resolved}")
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def aggregate_file_fingerprint(
    entries: Iterable[tuple[str, Path]], *, label: str
) -> dict[str, Any]:
    normalized: list[dict[str, Any]] = []
    for role, path in entries:
        normalized.append(
            {"role": role, **regular_file_record(path, label=f"{label} {role}")}
        )
    normalized.sort(key=lambda item: (item["role"], item["path"]))
    if not normalized:
        raise ValueError(f"{label} contains no files")
    digest = hashlib.sha256()
    total_bytes = 0
    for item in normalized:
        digest.update(
            (json.dumps(item, sort_keys=True, separators=(",", ":")) + "\n").encode()
        )
        total_bytes += item["bytes"]
    return {
        "files": len(normalized),
        "bytes": total_bytes,
        "sha256": digest.hexdigest(),
    }


def resume_metadata_path(checkpoint: Path) -> Path:
    return checkpoint.with_name(f"{checkpoint.name}.resume.json")


def resume_artifacts_path(checkpoint: Path) -> Path:
    return checkpoint.with_name(f"{checkpoint.name}.resume-artifacts")


def tree_manifest(root: Path) -> dict[str, Any]:
    unresolved = root.expanduser()
    if unresolved.is_symlink() or not unresolved.is_dir():
        raise ValueError(f"resume artifact root is missing or unsafe: {unresolved}")
    resolved = unresolved.resolve()
    files: list[dict[str, Any]] = []
    identities: set[tuple[int, int]] = set()
    for path in sorted(resolved.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"resume artifact tree contains a symlink: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError(f"resume artifact tree contains an unsupported entry: {path}")
        stat = path.stat()
        identity = (stat.st_dev, stat.st_ino)
        if identity in identities:
            raise ValueError(f"resume artifact tree contains a hardlink alias: {path}")
        identities.add(identity)
        files.append(
            {
                "path": path.relative_to(resolved).as_posix(),
                "bytes": stat.st_size,
                "sha256": sha256_file(path),
            }
        )
    if not files:
        raise ValueError("resume artifact tree contains no files")
    encoded = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return {
        "schema_version": "1.0.0",
        "files": files,
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }


def write_epoch_resume_metadata(
    checkpoint: Path,
    contract: Mapping[str, Any],
    *,
    completed_epochs: int,
    global_step: int,
    resume_artifacts: Path | None = None,
) -> Path:
    if completed_epochs < 1 or global_step < 1:
        raise ValueError("resumable progress must be positive")
    checkpoint_record = regular_file_record(checkpoint, label="resume checkpoint")
    document = {
        "schema_version": (
            EVIDENCE_SCHEMA_VERSION if resume_artifacts is not None else SCHEMA_VERSION
        ),
        "adaptation_mode": "full_sft",
        "resume_boundary": "epoch",
        "completed_epochs": completed_epochs,
        "global_step": global_step,
        "checkpoint": checkpoint_record,
        "training_contract": dict(contract),
        "trust_boundary": (
            "The checkpoint contains pickle-backed optimizer state. Resume requires "
            "an explicit trust acknowledgment and exact contract verification."
        ),
    }
    if resume_artifacts is not None:
        document["resume_artifacts"] = tree_manifest(resume_artifacts)
    destination = resume_metadata_path(checkpoint)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".partial",
            delete=False,
        ) as target:
            temporary_path = Path(target.name)
            json.dump(document, target, ensure_ascii=False, indent=2, sort_keys=True)
            target.write("\n")
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary_path, destination)
        temporary_path = None
        return destination
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def resolve_resume_checkpoint(value: str, output_dir: Path) -> Path | None:
    requested = value.strip()
    if not requested:
        return None
    if requested != "auto":
        unresolved = Path(requested).expanduser()
        if unresolved.is_symlink():
            raise ValueError(f"resume checkpoint must not be a symlink: {unresolved}")
        return unresolved.resolve()

    candidates: list[tuple[int, int, Path]] = []
    for path in output_dir.iterdir():
        match = EPOCH_CHECKPOINT_PATTERN.fullmatch(path.name)
        if match and resume_metadata_path(path).is_file():
            if path.is_symlink():
                raise ValueError(
                    f"auto resume checkpoint must not be a symlink: {path}"
                )
            candidates.append((int(match["epoch"]), int(match["step"]), path))
    if not candidates:
        return None
    return max(candidates, key=lambda item: (item[0], item[1]))[2].resolve()


def verify_resume_checkpoint(
    checkpoint: Path,
    expected_contract: Mapping[str, Any],
    *,
    trust_resume_state: bool,
    target_epochs: int,
) -> dict[str, Any]:
    if not trust_resume_state:
        raise ValueError(
            "resume requires --trust-resume-state because optimizer state may use "
            "unsafe pickle deserialization"
        )
    checkpoint_record = regular_file_record(checkpoint, label="resume checkpoint")
    metadata_path = resume_metadata_path(checkpoint)
    unresolved_metadata = metadata_path.expanduser()
    if unresolved_metadata.is_symlink():
        raise ValueError(
            f"resume metadata must not be a symlink: {unresolved_metadata}"
        )
    if not unresolved_metadata.is_file():
        raise FileNotFoundError(f"resume metadata is missing: {unresolved_metadata}")
    document = json.loads(unresolved_metadata.read_text(encoding="utf-8"))
    if document.get("schema_version") not in SCHEMA_VERSIONS:
        raise ValueError(
            "resume metadata requires schema " + " or ".join(sorted(SCHEMA_VERSIONS))
        )
    if document.get("adaptation_mode") != "full_sft":
        raise ValueError("resume metadata adaptation_mode must equal full_sft")
    if document.get("resume_boundary") != "epoch":
        raise ValueError("only epoch-boundary resume is supported")
    if document.get("checkpoint") != checkpoint_record:
        raise ValueError("resume checkpoint content does not match its metadata")
    if document.get("training_contract") != dict(expected_contract):
        raise ValueError("resume checkpoint training contract does not match this run")
    if document.get("schema_version") == EVIDENCE_SCHEMA_VERSION:
        artifacts = resume_artifacts_path(checkpoint)
        if document.get("resume_artifacts") != tree_manifest(artifacts):
            raise ValueError("resume artifacts do not match checkpoint metadata")
    completed_epochs = document.get("completed_epochs")
    global_step = document.get("global_step")
    if (
        not isinstance(completed_epochs, int)
        or isinstance(completed_epochs, bool)
        or completed_epochs < 1
        or completed_epochs >= target_epochs
    ):
        raise ValueError(
            "resume completed_epochs must be positive and below target epochs"
        )
    if (
        not isinstance(global_step, int)
        or isinstance(global_step, bool)
        or global_step < 1
    ):
        raise ValueError("resume global_step must be a positive integer")
    return document


def evaluator_full_sft_artifact_paths(checkpoint: Path) -> dict[str, Path]:
    """Map a schema 1.1 epoch checkpoint to evaluator 0.45 state roles."""
    unresolved = checkpoint.expanduser()
    if unresolved.is_symlink():
        raise ValueError("evaluator checkpoint must not be a symlink")
    resolved = unresolved.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"evaluator checkpoint is missing: {resolved}")
    metadata_path = resume_metadata_path(resolved)
    if metadata_path.is_symlink() or not metadata_path.is_file():
        raise FileNotFoundError(f"resume metadata is missing or unsafe: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("schema_version") != EVIDENCE_SCHEMA_VERSION:
        raise ValueError("evaluator mapping requires resume metadata schema 1.1.0")
    artifacts_root = resume_artifacts_path(resolved)
    if metadata.get("resume_artifacts") != tree_manifest(artifacts_root):
        raise ValueError("resume artifacts do not match checkpoint metadata")

    relative_by_role = {
        "model_state": "model-state.pt",
        "optimizer_state": "optimizer-state.pt",
        "scheduler_state": "scheduler-state.pt",
        "trainer_state": "trainer-state.json",
        "rng_state": "rng-state.pt",
    }
    manifested_paths = {
        record.get("path") for record in metadata["resume_artifacts"]["files"]
    }
    expected_paths = set(relative_by_role.values())
    if manifested_paths != expected_paths:
        raise ValueError(
            "evaluator mapping requires exactly the five declared state-role files"
        )
    artifacts = {role: artifacts_root / name for role, name in relative_by_role.items()}
    missing = sorted(role for role, path in artifacts.items() if not path.is_file())
    if missing:
        raise ValueError("evaluator mapping omits roles: " + ", ".join(missing))
    identities = [(path.stat().st_dev, path.stat().st_ino) for path in artifacts.values()]
    if len(identities) != len(set(identities)):
        raise ValueError("evaluator artifact roles must not share hardlinks")
    return artifacts


def validate_recent_checkpoints(values: Any, output_dir: Path) -> list[str]:
    if not isinstance(values, list):
        raise TypeError("recent_checkpoints must be a list")
    resolved_output = output_dir.resolve()
    validated: list[str] = []
    for value in values:
        if not isinstance(value, str) or not value:
            raise ValueError("recent_checkpoints contains an invalid path")
        unresolved = Path(value).expanduser()
        if unresolved.is_symlink():
            raise ValueError(f"recent checkpoint must not be a symlink: {unresolved}")
        resolved = unresolved.resolve()
        if resolved.parent != resolved_output or not CHECKPOINT_PATTERN.fullmatch(
            resolved.name
        ):
            raise ValueError("recent checkpoint escaped the output directory")
        validated.append(str(resolved))
    return validated
