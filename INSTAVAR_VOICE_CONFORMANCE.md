# Instavar Voice conformance

This repository declares its model-specific adaptation and runtime surface in `instavar-voice-capabilities.json`. The manifest and executable [`instavar-voice-backend.json`](instavar-voice-backend.json) full-SFT recipe use the public [Instavar Voice evaluation contract](https://github.com/instavar/instavar-voice-evaluation) pinned by CI to commit `8feadf7bbda75abe1c305c63e362c41b86451cda`.

The backend verifies clean companion and imported upstream revisions, audits raw grouped splits, runs the existing full-SFT launcher, reloads one explicit `.pth` checkpoint, executes the frozen plan, prunes the selected checkpoint, packages its experiment and evaluation evidence, and publishes the package under a content-addressed name to an external retention directory. CI validates and dependency-tests the recipe without performing GPU training.

`PERSISTED_PACKAGE_ROOT` must already exist outside the companion checkout, lifecycle work directory, imported IndexTTS2 checkout, prepared dataset tree, and model dependency directory. Preflight verifies durable writes and no-overwrite atomic hard-link publication, then locks the resolved path, filesystem device, and directory inode for the later package stage. Packaging reuses only a byte-identical object, rejects a conflicting or symbolic destination, and leaves the stage-local archive available for inspection. This establishes repository-declared retention mechanics, not a completed model or dataset rights review, remote backup, restore drill, real promoted checkpoint package, or defense against every adversarial filesystem race. A package hash provides integrity evidence and does not make the packaged `.pth` checkpoint safe to deserialize.

The underlying full-SFT trainer treats restart state as a separate trusted
boundary. Only a new epoch checkpoint with a matching content-bound sidecar can
resume. The sidecar binds checkpoint bytes, manifests and referenced features,
model and trainer sources, optimization settings, output location, runtime
packages, CUDA device identity, and AMP mode. Loading also restores Python,
NumPy, PyTorch, and CUDA RNG state. The user must explicitly acknowledge
pickle-backed optimizer state. Legacy, step-boundary, mutated, symlinked,
cross-contract, completed-run, and untrusted checkpoints fail closed. These are
dependency-free contract results, not evidence of a real interrupted GPU run,
numerical equivalence, cross-version portability, or mid-epoch continuation.

Capability schema 1.2 records each full-SFT lifecycle stage separately and names the exact blocker for the matched base-checkpoint comparison. A repository-level `supported` label no longer implies corpus audit, evaluation, or packaging completeness.

A capability marked `supported` means the referenced repository evidence reaches the stated engineering boundary. It does not prove perceptual quality, accent fidelity, commercial suitability, or equivalence across untested runtimes. `unverified_for_adapter` keeps an upstream or community runtime visible without implying that this repository's adapted artifact works there.

The common evaluation pack separates deterministic audio diagnostics and objective proxies from blinded human listening. It intentionally defines no universal composite score.

The experimental `openai_compatible_http` runtime is a strict fixed-artifact
reference surface. Dependency-free tests exercise its live HTTP request, safety,
authentication, concurrency, and WAV-output contracts with a fake engine. Its
`not_run` conformance status is intentional: no clean-checkout real-checkpoint
load, CUDA generation matrix, matched CLI comparison, load test, or blind
listening result has been recorded. See
[`docs/openai-compatible-serving.md`](docs/openai-compatible-serving.md).

For a reference and candidate runtime, generate the same frozen prompt with recorded settings and run `instavar-voice-eval compare-audio reference.wav candidate.wav`. The result exposes format and signal-level deltas while explicitly refusing to claim runtime equivalence. Establish intelligibility, speaker identity, accent, cadence, and naturalness separately through objective proxies and the blind listening pack.

Before training, use the contract's `audit-corpus` command with explicit train, validation, and test manifests. Supply a parent recording or source identifier through `--group-field` so the audit can reject leakage across splits. File presence and manifest integrity do not prove transcript accuracy or audio quality, which remain separate checks.

`scripts/train.sh` now accepts environment overrides for every previously hard-coded training path and core hyperparameter. Set `AUDIT_CORPUS=1`, the three `RAW_*_JSONL` paths, and `INSTAVAR_VOICE_EVAL_DIR` to audit raw audio manifests before the paired-feature trainer starts. The raw manifests are audited because the processed GPT pair manifests no longer carry a direct source-audio contract.

Validate locally with a checkout of the pinned contract:

```bash
python /path/to/instavar-voice-evaluation/main.py validate-repository .
```
