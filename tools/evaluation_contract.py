"""Dependency-free artifact selection for the IndexTTS2 evaluation runner."""

from __future__ import annotations

import argparse
from pathlib import Path


def configure_checkpoint(args: argparse.Namespace, config: object) -> str:
    """Bind exactly one adapted or base GPT artifact and return its mode label."""
    if args.inference_mode == "full-sft":
        if not args.gpt_checkpoint:
            raise ValueError("full-sft mode requires --gpt-checkpoint")
        if not args.gpt_checkpoint.is_file():
            raise ValueError("full-sft GPT checkpoint must be an existing file")
        config.gpt_checkpoint = str(args.gpt_checkpoint.resolve())
        return "full_sft"

    if args.gpt_checkpoint:
        raise ValueError("base mode forbids --gpt-checkpoint so the control cannot load adapted weights")
    configured = Path(str(config.gpt_checkpoint))
    resolved = configured if configured.is_absolute() else args.model_dir / configured
    expected = args.model_dir / "gpt.pth"
    if resolved.resolve() != expected.resolve():
        raise ValueError("base mode requires config.gpt_checkpoint to resolve to model-dir/gpt.pth")
    if not expected.is_file():
        raise ValueError("base mode requires model-dir/gpt.pth to be an existing file")
    return "base"
