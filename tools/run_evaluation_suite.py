#!/usr/bin/env python3
"""Run a frozen Instavar Voice generation plan with one loaded IndexTTS2 checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from indextts.infer_v2 import IndexTTS2
from omegaconf import OmegaConf


IDENTIFIER_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--gpt-checkpoint", type=Path)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--speaker", required=True)
    parser.add_argument("--generation-plan", type=Path, required=True)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--runtime-id", default="pytorch")
    parser.add_argument("--artifact-set-id")
    parser.add_argument("--artifact-set-sha256")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--max-text-tokens", type=int, default=120)
    parser.add_argument("--interval-silence", type=int, default=200)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--num-beams", type=int)
    return parser.parse_args()


def read_rows(path: Path, candidate_id: str) -> list[dict]:
    plan = json.loads(path.read_text(encoding="utf-8"))
    if plan.get("schema_version") != "1.0.0":
        raise ValueError("generation plan schema_version must equal 1.0.0")
    rows = [row for row in plan.get("samples", []) if row.get("candidate_id") == candidate_id]
    if not rows:
        raise ValueError(f"generation plan has no rows for candidate {candidate_id!r}")
    return rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_observations(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def runtime_artifact_fields(args: argparse.Namespace) -> dict[str, str]:
    if not IDENTIFIER_RE.fullmatch(args.runtime_id):
        raise ValueError("runtime id must be a lowercase machine-readable identifier")
    if bool(args.artifact_set_id) != bool(args.artifact_set_sha256):
        raise ValueError("artifact set id and sha256 must be provided together")
    fields = {"runtime_id": args.runtime_id}
    if args.artifact_set_id:
        if not IDENTIFIER_RE.fullmatch(args.artifact_set_id):
            raise ValueError("artifact set id must be a lowercase machine-readable identifier")
        if not re.fullmatch(r"[0-9a-f]{64}", args.artifact_set_sha256):
            raise ValueError("artifact set sha256 must be a lowercase SHA-256 digest")
        fields.update(
            {
                "artifact_set_id": args.artifact_set_id,
                "artifact_set_sha256": args.artifact_set_sha256,
            }
        )
    return fields


def main() -> int:
    args = parse_args()
    artifact_fields = runtime_artifact_fields(args)
    rows = read_rows(args.generation_plan, args.candidate_id)
    config = OmegaConf.load(args.config)
    if args.gpt_checkpoint:
        config.gpt_checkpoint = str(args.gpt_checkpoint.resolve())
    if args.tokenizer:
        config.dataset.bpe_model = str(args.tokenizer.resolve())

    generation_kwargs = {
        key: value
        for key, value in {
            "top_k": args.top_k,
            "top_p": args.top_p,
            "temperature": args.temperature,
            "num_beams": args.num_beams,
        }.items()
        if value is not None
    }
    observations: list[dict] = []
    with tempfile.NamedTemporaryFile("w", suffix=".yaml") as config_file:
        OmegaConf.save(config, config_file.name)
        engine = IndexTTS2(
            cfg_path=config_file.name,
            model_dir=str(args.model_dir.resolve()),
            device=args.device,
            use_fp16=args.fp16,
        )
        for row in rows:
            output = args.output_dir / row["expected_audio_path"]
            output.parent.mkdir(parents=True, exist_ok=True)
            set_seed(int(row["seed"]))
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
            started = time.perf_counter()
            observation = {
                "sample_id": row["sample_id"],
                "candidate_id": row["candidate_id"],
                "prompt_id": row["prompt_id"],
                "category": row["category"],
                "seed": row["seed"],
                "requested_text": row["text"],
                "valid": False,
                "runtime": "indextts2_pytorch_cuda_checkpoint",
                **artifact_fields,
                "instruction_applied": False,
            }
            try:
                engine.infer(
                    spk_audio_prompt=args.speaker,
                    text=row["text"],
                    output_path=str(output),
                    emo_text=row.get("instruction"),
                    use_emo_text=bool(row.get("instruction")),
                    interval_silence=args.interval_silence,
                    max_text_tokens_per_segment=args.max_text_tokens,
                    **generation_kwargs,
                )
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                elapsed = time.perf_counter() - started
                info = sf.info(output)
                observation.update(
                    {
                        "valid": info.frames > 0,
                        "audio_path": str(output),
                        "audio_sha256": sha256(output),
                        "audio_duration_seconds": float(info.duration),
                        "generation_seconds": elapsed,
                        "peak_memory_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0,
                        "instruction_applied": bool(row.get("instruction")),
                    }
                )
            except Exception as error:
                observation.update(
                    {
                        "generation_seconds": time.perf_counter() - started,
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                )
            observations.append(observation)
            write_observations(args.output_dir / "generation-observations.json", observations)
    return 0 if all(row["valid"] for row in observations) else 1


if __name__ == "__main__":
    raise SystemExit(main())
