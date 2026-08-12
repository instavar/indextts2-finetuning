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
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).parents[1]
STAGES = {"preflight", "train", "infer", "evaluate", "package"}


def _path(name: str, *, directory: bool = False) -> Path:
    value = os.environ.get(name, "").strip()
    if not value:
        raise ValueError(f"{name} is required")
    path = Path(value).expanduser().resolve()
    valid = path.is_dir() if directory else path.is_file()
    if not valid or path.is_symlink():
        kind = "directory" if directory else "file"
        raise FileNotFoundError(f"{name} non-symlink {kind} not found: {path}")
    return path


def _work() -> Path:
    return _path("INSTAVAR_VOICE_WORK_DIR", directory=True)


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
    if plan.get("schema_version") != "1.0.0" or not rows:
        raise ValueError("GENERATION_PLAN must be schema 1.0.0 and contain CANDIDATE_ID rows")
    _safe_filename(os.environ["SELECTED_CHECKPOINT_NAME"])
    _write_json(
        _work() / "preflight" / "preflight.json",
        {"schema_version": "1.0.0", "status": "passed", "corpus_audit": audit, "generation_rows": len(rows), "sources": sources, "dataset_lineage": lineage},
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
        "--output-dir", str(output),
    ]
    if os.environ.get("FP16", "1") == "1":
        command.append("--fp16")
    _run(command)
    _archive(output, work / "evaluate" / "evaluation-bundle.tar", arcname="evaluation")


def _package() -> None:
    work = _work()
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
    _archive(staging, work / "package" / "checkpoint-package.tar", arcname="package")


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
