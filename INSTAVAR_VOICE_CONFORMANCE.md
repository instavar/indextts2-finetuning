# Instavar Voice conformance

This repository declares its model-specific adaptation and runtime surface in `instavar-voice-capabilities.json`. The manifest and executable [`instavar-voice-backend.json`](instavar-voice-backend.json) full-SFT recipe use the public [Instavar Voice evaluation contract](https://github.com/instavar/instavar-voice-evaluation) pinned by CI to merge commit `e689ee121ee4a6ae07793ef1c49d70c48b0ad271`.

The backend verifies clean companion and imported upstream revisions, audits raw grouped splits, runs the existing full-SFT launcher, reloads one explicit `.pth` checkpoint, executes the frozen plan, prunes the selected checkpoint, and packages its experiment and evaluation evidence. CI validates and dependency-tests the recipe without performing GPU training.

Capability schema 1.2 records each full-SFT lifecycle stage separately and names the exact blocker for the matched base-checkpoint comparison. A repository-level `supported` label no longer implies corpus audit, evaluation, or packaging completeness.

A capability marked `supported` means the referenced repository evidence reaches the stated engineering boundary. It does not prove perceptual quality, accent fidelity, commercial suitability, or equivalence across untested runtimes. `unverified_for_adapter` keeps an upstream or community runtime visible without implying that this repository's adapted artifact works there.

The common evaluation pack separates deterministic audio diagnostics and objective proxies from blinded human listening. It intentionally defines no universal composite score.

For a reference and candidate runtime, generate the same frozen prompt with recorded settings and run `instavar-voice-eval compare-audio reference.wav candidate.wav`. The result exposes format and signal-level deltas while explicitly refusing to claim runtime equivalence. Establish intelligibility, speaker identity, accent, cadence, and naturalness separately through objective proxies and the blind listening pack.

Before training, use the contract's `audit-corpus` command with explicit train, validation, and test manifests. Supply a parent recording or source identifier through `--group-field` so the audit can reject leakage across splits. File presence and manifest integrity do not prove transcript accuracy or audio quality, which remain separate checks.

`scripts/train.sh` now accepts environment overrides for every previously hard-coded training path and core hyperparameter. Set `AUDIT_CORPUS=1`, the three `RAW_*_JSONL` paths, and `INSTAVAR_VOICE_EVAL_DIR` to audit raw audio manifests before the paired-feature trainer starts. The raw manifests are audited because the processed GPT pair manifests no longer carry a direct source-audio contract.

Validate locally with a checkout of the pinned contract:

```bash
python /path/to/instavar-voice-evaluation/main.py validate-repository .
```
