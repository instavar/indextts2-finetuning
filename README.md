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
- `trainers/train_gpt_v2.py` — full GPT fine-tuning trainer (909 lines)
- `tools/` — complete data preprocessing pipeline (6 scripts)
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
  --amp \
  --resume auto
```

Or use the provided script:

```bash
bash scripts/train.sh
```

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

Validate the recipe with evaluator merge
`5dd7369095a14adbf7f910bc7c53ec27051ebcb0`. Use an empty work directory outside
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
  --gpt-checkpoint checkpoints/model_step14000.pth \
  --speaker assets/female01_prompt.wav \
  --generation-plan evaluation/generation-plan.json \
  --candidate-id index-step14000 \
  --runtime-id pytorch \
  --output-dir evaluation/index-step14000 \
  --device cuda:0
```

The runner freezes Python, NumPy, and PyTorch seeds for every planned sample and
records failed generations instead of dropping them. A complete matrix is
still not a perceptual-quality result until objective extraction and blind
listening are complete.

For an exact cross-runtime experiment, also pass `--artifact-set-id` and
`--artifact-set-sha256` together. The runner rejects partial or malformed
bindings. Generate and live-verify the corresponding runtime artifact manifest
with evaluator revision `5dd7369095a14adbf7f910bc7c53ec27051ebcb0` before
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
| Resume | auto | Resumes from latest checkpoint if training crashes |

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
    train_gpt_v2.py          # Full GPT fine-tuning trainer (909 lines)
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
  inference_script.py         # CLI inference wrapper
  README.md
  CHANGELOG.md
  LICENSE
```

## License

Apache-2.0

## Instavar Voice conformance

[`instavar-voice-capabilities.json`](instavar-voice-capabilities.json) declares full SFT and explicit-checkpoint PyTorch inference as the supported path. It does not relabel full SFT as LoRA and does not imply that the private production API is part of this repository. CI validates the manifest against the pinned public [Instavar Voice evaluation contract](https://github.com/instavar/instavar-voice-evaluation).
