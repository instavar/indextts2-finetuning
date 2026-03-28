# Changelog

## v0.1.0

- Initial release of IndexTTS2 fine-tuning pipeline
- `trainers/train_gpt_v2.py` — full GPT fine-tuning with AMP, gradient accumulation, cosine LR schedule, auto-resume
- `tools/preprocess_data.py` — generic preprocessing (text normalization, semantic extraction, code quantization, conditioning extraction)
- `tools/preprocess_multiproc.py` — parallel preprocessing with sharded workers
- `tools/build_gpt_prompt_pairs.py` — prompt/target pair construction per IndexTTS2 paper
- `tools/generate_gpt_pairs.py` — batch pair generation across multiple datasets
- `tools/process_text_ids.py` — text-only re-tokenization without reprocessing audio
- `tools/prune_gpt_checkpoint.py` — strip training artifacts for inference-only deployment
- `inference_script.py` — CLI inference with fine-tuned checkpoint support
- `tests/` — padding and regression tests
- `scripts/train.sh` — training launcher with validated IMDA NSC configuration
- Known pitfalls documented from 15,949-step FEMALE_01 production run
