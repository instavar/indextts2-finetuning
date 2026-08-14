# IndexTTS2 Supervised Fine-Tuning (SFT)

**A supervised fine-tuning (SFT) pipeline for [IndexTTS2](https://github.com/index-tts/index-tts).** As of March 2026, the official repo provides inference only — no fine-tuning code. This repo provides the complete SFT pipeline: data preprocessing, GPT fine-tuning, checkpoint management, and inference.

We used this pipeline to fine-tune IndexTTS2 (full SFT — all weights updated, no LoRA) on [IMDA NSC](https://www.imda.gov.sg/how-we-can-help/national-speech-corpus) FEMALE_01 (Singaporean English) for production voice cloning. The configuration, pitfalls, and recommendations below come from that experience.

- Upstream repo: https://github.com/index-tts/index-tts (inference only)
- Blog deep-dive: [IndexTTS2 Finetuning on IMDA NSC FEMALE_01](https://instavar.com/blog/ai-production-stack/IndexTTS2_Finetuning_IMDA_NSC_FEMALE_01)
- Decision tree (9 models): [Which TTS Model Should You Use?](https://instavar.com/blog/ai-production-stack/TTS_Model_Decision_Tree_2026)
- Benchmark hub: [Best Open-Source TTS Models for Production in 2026](https://instavar.com/blog/ai-production-stack/Best_Open_Source_TTS_Models_Production_2026)

## Why this repo exists

IndexTTS2 is the most reproducible full-SFT TTS model we tested — it converges predictably and retains accent characteristics well. But as of March 2026, the official repo is inference-only. We wrote the fine-tuning pipeline from scratch during our IMDA NSC benchmark runs (January 2026) and have been using it in production since.

This repo contains:
- `trainers/train_gpt_v2.py` - full GPT fine-tuning trainer
- `tools/` — complete data preprocessing pipeline (6 scripts)
- `tools/openai_speech_server.py` - experimental OpenAI-compatible HTTP speech server
- `inference_script.py` — CLI inference wrapper
- `tests/` — padding and regression tests
- `scripts/train.sh` — training launcher with validated hyperparameters

## Known pitfalls

| # | Pitfall | Symptom | Fix |
|---|---------|---------|-----|
| 1 | **Checkpoint auto-deletion** | Older checkpoints deleted before evaluation | Keep ALL checkpoints; pin best one after listening eval |
| 2 | **`latest.pth` is not the best checkpoint** | Using final step (highest loss) for inference | Always evaluate by listening; best was step 14000, not 15949 |
| 3 | **`transformers` version pinning** | `KeyError: 'qwen3'` during model loading | Requires `transformers>=4.47` (tested: 4.52.1) |
| 4 | **Crash recovery requires manual management** | Training crash leaves stale state | Log last successful step; resume explicitly from there |
| 5 | **`--model-dir` vs `--gpt-checkpoint`** | Wrong weights loaded silently | Use `--gpt-checkpoint` for fine-tuned weights; `--model-dir` loads base only |
| 6 | **HF_HOME not set** | Model downloads to wrong cache dir | Always export `HF_HOME` before running inference |
| 7 | **First-run download takes 30-90 min** | Appears hung on first inference | 7.2 GB checkpoint downloads via XET protocol; cached after first run |
| 8 | **VRAM contention** | OOM or degraded quality | Check `nvidia-smi` before starting; needs 5-8 GB free |

The most impactful: **best checkpoint is never the last one.** In our run, step 14000 had the lowest validation loss. Step 15949 (final) had higher loss. `latest.pth` symlinks to the final step. If you use `latest.pth` without evaluating, you deploy a worse model.

## Quick start

### 1. Install IndexTTS2

```bash
git clone https://github.com/index-tts/index-tts.git
cd IndexTTS
pip install -e .
cd ..
```

### 2. Clone this repo

```bash
git clone https://github.com/instavar/indextts2-finetuning.git
```

### 3. Prepare your dataset

Your audio files must be WAV format. The preprocessing pipeline extracts:
- Text token IDs (via SentencePiece)
- Semantic codes (via SeamlessM4T + Wav2Vec2Bert + RepCodec)
- Conditioning latents and emotion vectors (via UnifiedVoice GPT)

```bash
# Step 1: Preprocess audio + text into feature manifests
python tools/preprocess_data.py \
  --audio-dir /path/to/wavs \
  --transcript /path/to/transcripts.jsonl \
  --tokenizer /path/to/bpe.model \
  --config checkpoints/config.yaml \
  --output-dir ./processed_data \
  --language en

# Step 2: Build prompt/target pairs for GPT training
python tools/build_gpt_prompt_pairs.py \
  --manifest ./processed_data/train_manifest.jsonl \
  --output ./processed_data/gpt_pairs_train.jsonl

python tools/build_gpt_prompt_pairs.py \
  --manifest ./processed_data/val_manifest.jsonl \
  --output ./processed_data/gpt_pairs_val.jsonl
```

For large datasets, use the multiprocessing wrapper:

```bash
python tools/preprocess_multiproc.py \
  --audio-dir /path/to/wavs \
  --transcript /path/to/transcripts.jsonl \
  --tokenizer /path/to/bpe.model \
  --config checkpoints/config.yaml \
  --output-dir ./processed_data \
  --num-workers 4
```

### 4. Train

```bash
python trainers/train_gpt_v2.py \
  --train-manifest processed_data/gpt_pairs_train.jsonl \
  --val-manifest processed_data/gpt_pairs_val.jsonl \
  --tokenizer checkpoints/bpe.model \
  --config checkpoints/config.yaml \
  --base-checkpoint checkpoints/gpt.pth \
  --output-dir trained_ckpts \
  --batch-size 32 \
  --grad-accumulation 1 \
  --epochs 10 \
  --learning-rate 1e-5 \
  --weight-decay 0.01 \
  --warmup-steps 1000 \
  --log-interval 1 \
  --val-interval 2000 \
  --grad-clip 1.0 \
  --text-loss-weight 0.2 \
  --mel-loss-weight 0.8 \
  --amp
```

Resume is intentionally opt-in because the training checkpoint contains
pickle-backed optimizer, scheduler, AMP, and RNG state. `auto` selects only the
latest epoch-boundary checkpoint carrying a matching content-bound sidecar;
legacy, step-boundary, symlinked, mutated, cross-configuration, cross-runtime,
or completed-run state fails closed. The contract binds the tokenizer, model
configuration, base checkpoint, trainer and model sources, all referenced train
and validation manifests and features, optimization settings, output directory,
Python and package versions, CUDA runtime and device identity, and AMP mode. It
does not support exact mid-epoch or cross-version continuation. To continue an
interrupted run whose original target was ten epochs:

```bash
python trainers/train_gpt_v2.py \
  <the same arguments used for the original run> \
  --resume auto \
  --trust-resume-state
```

Or use the provided script:

```bash
RESUME=auto TRUST_RESUME_STATE=1 bash scripts/train.sh
```

Omit both resume variables for a new run. Set the trust flag only for local
state whose origin you have verified.

For byte-exact resume evidence, set `DETERMINISTIC=1` for both the uninterrupted
and interrupted-resumed conditions. This enables deterministic PyTorch
algorithms, sets the CUDA BLAS workspace contract, disables TF32, and binds the
resulting controls into the resume contract. It can reduce throughput and will
fail closed if the active model route reaches an unsupported nondeterministic
kernel. Determinism remains opt-in for ordinary training because this evidence
mode is stricter than the quality-oriented production path.

Future epoch-boundary checkpoints also publish a metadata-bound
`*.resume-artifacts` directory with separate model, optimizer, scheduler,
trainer, and RNG files. This exposes the five independent roles required by
Instavar Voice evaluator 0.45 while the original combined `.pth` remains the
loader source. Schema 1.0 combined-only checkpoints remain resumable, but only
schema 1.1 checkpoints can use `evaluator_full_sft_artifact_paths(...)`.

The decomposed model state increases storage for each resumable epoch
checkpoint. It is intentionally limited to epoch boundaries and must be planned
into the retention budget. Existing full-SFT evidence predates the new files and
schema 1.1 live-conditioning receipts, so it is not upgraded. See
[`reports/resume-evaluator-045-instrumentation-2026-08-14.md`](reports/resume-evaluator-045-instrumentation-2026-08-14.md).

### 5. Select the best checkpoint

**Do not use `latest.pth`.** It points to the final training step, which is rarely the best.

1. Look at the validation loss in TensorBoard or training logs
2. Identify the step with the lowest val loss
3. Listen to 5-10 samples from that checkpoint
4. Compare against the adjacent checkpoints (+-1000 steps)
5. Pin the winner explicitly

In our FEMALE_01 run, step 14000 was the best (lowest val loss region ~13800, nearest saved checkpoint at 14000).

### 6. Inference

```bash
python inference_script.py \
  --config checkpoints/config.yaml \
  --gpt-checkpoint trained_ckpts/model_step14000.pth \
  --speaker /path/to/speaker_prompt.wav \
  --text "Your sentence here." \
  --output output.wav \
  --device cuda:0 \
  --fp16
```

> **Tip:** Always use `--fp16` on CUDA. Use `--gpt-checkpoint` (not `--model-dir`) to load fine-tuned weights.

### Experimental OpenAI-compatible HTTP serving

The reference server exposes one fixed checkpoint and one fixed speaker prompt
through `POST /v1/audio/speech`. It implements a deliberately strict subset of
the OpenAI speech request: `model`, `input`, `voice`, optional
`response_format="wav"`, and optional `speed=1.0`. Unsupported fields or values
are rejected instead of being silently ignored.

```bash
python tools/openai_speech_server.py \
  --config checkpoints/config.yaml \
  --model-dir checkpoints \
  --gpt-checkpoint trained_ckpts/model_step14000.pth \
  --speaker /path/to/reviewed-speaker-prompt.wav \
  --model-id indextts2-finetuned \
  --voice-id female01-reviewed \
  --device cuda:0 \
  --fp16
```

The default bind is `127.0.0.1:8000`. A non-loopback bind fails closed unless
`--api-key-env` names a nonempty environment variable. Pass the variable name,
never the secret value, on the command line:

```bash
export INDEXTTS2_API_KEY="replace-with-a-secret-from-your-secret-store"
python tools/openai_speech_server.py \
  --config checkpoints/config.yaml \
  --model-dir checkpoints \
  --gpt-checkpoint trained_ckpts/model_step14000.pth \
  --speaker /path/to/reviewed-speaker-prompt.wav \
  --host 0.0.0.0 \
  --api-key-env INDEXTTS2_API_KEY \
  --device cuda:0 \
  --fp16
```

```bash
curl --fail-with-body http://127.0.0.1:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  --data '{"model":"indextts2-finetuned","voice":"instavar-reference","input":"A short test sentence.","response_format":"wav"}' \
  --output response.wav
```

The checkpoint, tokenizer, config, speaker prompt, sampling configuration, seed,
and output path are startup-only controls. Requests cannot supply local paths.
The process loads the engine once, rejects overlapping synthesis with HTTP 429,
limits request and output sizes, validates the generated PCM WAV, and returns a
bounded error body without engine exception details. See
[`docs/openai-compatible-serving.md`](docs/openai-compatible-serving.md) for the
contract, security boundary, tests, and production gaps.

This surface is experimental and repository-declared. The dependency-free tests
exercise the HTTP and safety contracts with a fake engine. They do not prove that
the real checkpoint loads, that CUDA generation succeeds, that audio quality is
acceptable, or that the server is production-ready.

### 7. Prune checkpoint for deployment

Training checkpoints include optimizer state and other training artifacts. For inference-only deployment, prune to model weights only:

```bash
python tools/prune_gpt_checkpoint.py \
  --input trained_ckpts/model_step14000.pth \
  --output checkpoints/gpt_finetuned_pruned.pth
```

## Recommended configuration

### Executable Instavar Voice lifecycle

[`instavar-voice-backend.json`](instavar-voice-backend.json) binds the supported
full-SFT and PyTorch checkpoint path to a real five-stage lifecycle. It audits
the raw grouped splits, verifies that `uv` imports IndexTTS2 from the pinned
clean upstream checkout named in the experiment, runs the existing trainer,
copies one explicitly selected `.pth` checkpoint, reloads it in a fresh process,
runs the frozen generation plan, prunes the checkpoint, and packages the
configuration, tokenizer, preflight, smoke output, evaluation, and provenance
files. The remaining base-model weights stay an external pinned dependency.

Set `PERSISTED_PACKAGE_ROOT` to an existing retention directory outside the
companion checkout, lifecycle work directory, imported IndexTTS2 checkout,
prepared dataset tree, and model dependency directory.
Preflight proves that exact directory can support durable writes and
no-overwrite atomic hard-link publication, then records its resolved path,
filesystem device, and directory inode. The package stage rechecks that binding,
retains the archive as
`indextts2-full-sft-package-sha256-<sha256>.tar`, and writes
`package/persisted-package.json` as its receipt. Existing content-addressed
objects are reused only when their bytes match. This dependency-free contract
does not prove a real checkpoint package, backup or restore, distribution rights,
or safe deserialization of the packaged `.pth` file.

Validate the recipe with evaluator merge
`8feadf7bbda75abe1c305c63e362c41b86451cda`. Use an empty work directory outside
the repository. `SELECTED_CHECKPOINT_NAME` must be an exact produced filename,
such as `model_step14000.pth`; `latest.pth` is not selected implicitly. A passed
lifecycle establishes execution and artifact lineage, not perceptual quality.

### Frozen multi-prompt evaluation

Use the selected full-SFT checkpoint with one loaded engine across every
IndexTTS2 row in an Instavar Voice generation plan:

```bash
python tools/run_evaluation_suite.py \
  --config checkpoints/config_finetune.yaml \
  --model-dir checkpoints \
  --inference-mode full-sft \
  --gpt-checkpoint checkpoints/model_step14000.pth \
  --speaker assets/female01_prompt.wav \
  --generation-plan evaluation/generation-plan.json \
  --candidate-id index-step14000 \
  --runtime-id pytorch \
  --output-dir evaluation/index-step14000 \
  --device cuda:0
```

For a matched unchanged-base control, use the same plan, prompt audio, seeds,
runtime settings, and base model directory, then omit the checkpoint override:

```bash
python tools/run_evaluation_suite.py \
  --config checkpoints/config.yaml \
  --model-dir checkpoints \
  --inference-mode base \
  --speaker assets/female01_prompt.wav \
  --generation-plan evaluation/generation-plan.json \
  --candidate-id index-base \
  --runtime-id pytorch \
  --output-dir evaluation/index-base \
  --device cuda:0
```

Base mode rejects `--gpt-checkpoint` and requires `config.gpt_checkpoint` to
resolve to `<model-dir>/gpt.pth`. Full-SFT mode requires an explicit checkpoint.
Each observation records `artifact_mode` and a device-aware runtime label so a
missing override cannot silently turn an adapted candidate into the base
control, or vice versa. This artifact binding does not make unmatched prompts,
references, settings, extractors, or post-generation assignments comparable.

The runner freezes Python, NumPy, and PyTorch seeds for every planned sample and
records failed generations instead of dropping them. A complete matrix is
still not a perceptual-quality result until objective extraction and blind
listening are complete.

The first exact base versus selected step-14000 long-form pair is documented in
[`reports/matched-long-form-base-full-sft-2026-08-13.md`](reports/matched-long-form-base-full-sft-2026-08-13.md).
It completed objective and non-directional prosody coverage and prepared a
focused blind pack. It did not produce a quality winner or listening ratings.

For an exact cross-runtime experiment, also pass `--artifact-set-id` and
`--artifact-set-sha256` together. The runner rejects partial or malformed
bindings. Generate and live-verify the corresponding runtime artifact manifest
with evaluator revision `8feadf7bbda75abe1c305c63e362c41b86451cda` before
using `compare-runtimes`. Converted artifacts remain `derived`, not exact.

Based on our IMDA NSC FEMALE_01 runs (RTX 3090 Ti, 24 GB):

| Parameter | Value | Why |
|-----------|-------|-----|
| Learning rate | 1e-5 | Stable convergence for full SFT |
| Epochs | 5-10 | Best checkpoint typically appears in epoch 4-5 |
| Batch size | 32 | Fits 24 GB with AMP enabled |
| Gradient accumulation | 1 | Not needed at batch size 32 |
| Text loss weight | 0.2 | Balances text prediction vs speech quality |
| Mel loss weight | 0.8 | Prioritizes speech quality |
| Gradient clipping | 1.0 | Prevents training instability |
| Warmup steps | 1000 | Standard cosine schedule |
| Val interval | 2000 | Frequent enough to catch the best checkpoint |
| AMP | Enabled | Halves VRAM usage; no quality loss observed |
| Resume | guarded epoch boundary | Explicit trust plus exact contract and checkpoint verification |

## Data preprocessing pipeline

The preprocessing pipeline has four stages:

```
Raw audio + transcripts
  |
  v
[1] Text normalization + SentencePiece tokenization
  |
  v
[2] Semantic feature extraction (SeamlessM4T + Wav2Vec2Bert)
  |
  v
[3] Semantic code quantization (RepCodec)
  |
  v
[4] Conditioning latent + emotion vector extraction (UnifiedVoice GPT)
  |
  v
JSONL manifests + .npy feature files
  |
  v
[5] Prompt/target pair construction (for GPT training)
  |
  v
Final training manifests (gpt_pairs_train.jsonl, gpt_pairs_val.jsonl)
```

Each sample in the training manifest contains paths to:
- Text token IDs (`.npy`, int32)
- Semantic codes (`.npy`, int32)
- Conditioning latent (`.npy`, float32, shape `[32, hidden]`)
- Emotion vector (`.npy`, float32, shape `[hidden]`)

The prompt/target pairing strategy follows the IndexTTS2 paper: different utterances from the same speaker are used for the prompt (conditioning) and target (text + codes to predict).

## What this does NOT include

- **The IndexTTS2 model itself** — install from upstream (`pip install -e .` from index-tts/index-tts)
- **Pre-trained checkpoints** — download from HuggingFace (`IndexTeam/IndexTTS-2`)
- **Training data** — bring your own dataset. We used IMDA NSC; you need your own licensed audio.
- **API server** — the production FastAPI wrapper is part of our SaaS infrastructure, not this repo
- **RunPod deployment** — see our [blog post](https://instavar.com/blog/ai-production-stack/IndexTTS2_Finetuning_IMDA_NSC_FEMALE_01) for deployment notes

## Runtime requirements

- Python 3.10+
- PyTorch 2.4+ with CUDA support
- `transformers>=4.47` (for Qwen3 emotion model support)
- IndexTTS2 installed from upstream
- 24 GB GPU (RTX 3090/3090 Ti/4090) — fits both training and inference
- VRAM at inference: 5-8 GB

## Project structure

```
indextts2-finetuning/
  trainers/
    train_gpt_v2.py          # Full GPT fine-tuning trainer
    index_resume_contract.py # Trusted epoch-resume metadata and verification
  tools/
    preprocess_data.py        # Generic data preprocessing pipeline
    preprocess_multiproc.py   # Multi-worker parallel preprocessing
    build_gpt_prompt_pairs.py # Prompt/target pair construction
    generate_gpt_pairs.py     # Batch pair generation helper
    process_text_ids.py       # Text-only re-tokenization
    prune_gpt_checkpoint.py   # Strip optimizer state for deployment
  scripts/
    train.sh                  # Training launcher with validated config
  tests/
    padding_test.py           # Padding correctness tests
    regression_test.py        # Output regression tests
    resume_contract_test.py   # Dependency-free guarded-resume tests
  inference_script.py         # CLI inference wrapper
  README.md
  CHANGELOG.md
  LICENSE
```

## License

Apache-2.0

## Instavar Voice conformance

[`instavar-voice-capabilities.json`](instavar-voice-capabilities.json) declares full SFT and explicit-checkpoint PyTorch inference as the supported path. It does not relabel full SFT as LoRA and does not imply that the private production API is part of this repository. CI validates the manifest against the pinned public [Instavar Voice evaluation contract](https://github.com/instavar/instavar-voice-evaluation). New lifecycle and resume-evidence runs should use evaluator commit `29c38cfd86b889abc8b79df063c817dd8f684903` or a deliberately reviewed successor so POSIX stage timeouts clean the complete process group and schema 1.1 receipts bind live conditioning artifacts. This does not retroactively upgrade earlier run evidence.

The experimental OpenAI-compatible subset now has content-bound startup
receipts, an exact frozen-row HTTP client, a CLI parity validator, and malformed
plus overlapping-request probes. The preregistered nine-row CUDA qualification
protocol lives under [`evaluation/preregistration/`](evaluation/preregistration/).
These tools make a real runtime result auditable, but their presence alone does
not promote the HTTP runtime beyond `experimental`; see
[`docs/openai-compatible-serving.md`](docs/openai-compatible-serving.md).
The first real CUDA qualification passed all nine planned HTTP rows, ten exact
CLI parity checks including restart, complete objective-metric coverage, and
the frozen malformed plus overlapping-request probes. The evidence and limits
are recorded in
[`reports/openai-speech-http-runtime-2026-08-14.md`](reports/openai-speech-http-runtime-2026-08-14.md).

The lifecycle preserves invalid generations as explicit rows, then uses
evaluator revision `8feadf7bbda75abe1c305c63e362c41b86451cda` to bind timing,
duration, and peak-memory fields to the frozen plan and live output audio. Use
the packaged `objective-observations.json`, not the raw generation file, for a
version 1.1 runtime comparison.

The pinned evaluator provides schema 1.3 frozen speaker-reference assignments,
the optional schema 1.4 SpeechBrain ECAPA execution path, and the optional
schema 1.5 local faster-whisper ASR path. Version 0.20 also distinguishes
generation-plan-bound ASR reference text from observation-declared strings.
Version 0.21 adds plan-bound category strata so pronunciation, local-context,
and long-form proxy regressions remain visible instead of disappearing into one
candidate mean.
Version 0.22 carries frozen lexical anchors and accepted ASR forms into the
generation plan, reports hit, miss, coverage, and matched deltas, and rejects
candidate-specific alias drift. Phrase hits remain recognition evidence, not
pronunciation or accent judgments.
Version 0.23 preregisters criterion-specific blind-listening assignments so
lexical pronunciation, cadence, fatigue, and emotion ratings only cover prompts
that can support those claims while preserving candidate-symmetric coverage.
Version 0.24 binds exact requested text, optional instructions, and lexical
target surfaces into each blind stimulus while excluding accepted ASR aliases
and candidate identity. Reviewers no longer need an uncontrolled prompt file.
Version 0.25 binds each listening criterion to a reviewer question, low and
high scale anchors, and an explicit score direction. Harm criteria remain raw
and separate instead of being silently inverted or folded into a composite.
Version 0.26 adds deterministic per-rater presentation schedules that
counterbalance candidate precedence within each prompt and seed. Aggregation
recomputes the private audit, requires the scheduled pseudonymous rater set,
and keeps order, fatigue, carryover, and reviewer-compliance limits explicit.
Version 0.27 exports one privacy-preserving packet per pseudonymous rater and
binds criterion-major presentation logs plus ratings into canonical submission
receipts. Aggregation reconstructs each packet, rejects forged metadata, and
records missing reviewers or cells as attrition. Receipt hashes establish
content integrity, not reviewer identity, delivery, attention, or independence.
This companion bundles neither model
weights nor optional extractor dependencies and runs neither learned metric
automatically. Run them explicitly after generation with trusted, content-addressed
models, frozen decoding, and a preregistered reference plan where applicable.
Runtime-bound observations, same-recording smoke scores, or human-recording ASR
alone are not TTS-quality evidence.
