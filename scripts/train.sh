#!/usr/bin/env bash
set -euo pipefail

TRAIN_MANIFEST="${TRAIN_MANIFEST:-processed_data_2/gpt_pairs_train.jsonl}"
VAL_MANIFEST="${VAL_MANIFEST:-processed_data_2/gpt_pairs_val.jsonl}"
PYTHON="${PYTHON:-python3}"
resume_args=()
deterministic_args=()

case "${DETERMINISTIC:-0}" in
  0) ;;
  1) deterministic_args+=(--deterministic) ;;
  *)
    echo "DETERMINISTIC must equal 0 or 1" >&2
    exit 2
    ;;
esac

if [[ -n "${RESUME:-}" ]]; then
  if [[ "${TRUST_RESUME_STATE:-0}" != "1" ]]; then
    echo "Set TRUST_RESUME_STATE=1 only for a trusted local epoch checkpoint" >&2
    exit 2
  fi
  resume_args+=(--resume "${RESUME}" --trust-resume-state)
fi

if [[ "${AUDIT_CORPUS:-0}" == "1" ]]; then
  : "${RAW_TRAIN_JSONL:?set RAW_TRAIN_JSONL when AUDIT_CORPUS=1}"
  : "${RAW_VALIDATION_JSONL:?set RAW_VALIDATION_JSONL when AUDIT_CORPUS=1}"
  : "${RAW_TEST_JSONL:?set RAW_TEST_JSONL when AUDIT_CORPUS=1}"
  : "${INSTAVAR_VOICE_EVAL_DIR:?set INSTAVAR_VOICE_EVAL_DIR to the pinned instavar-voice-evaluation checkout}"
  audit_args=(
    --split "train=${RAW_TRAIN_JSONL}"
    --split "validation=${RAW_VALIDATION_JSONL}"
    --split "test=${RAW_TEST_JSONL}"
  )
  if [[ -n "${CORPUS_GROUP_FIELD:-}" ]]; then
    audit_args+=(--group-field "${CORPUS_GROUP_FIELD}")
  fi
  "${PYTHON}" "${INSTAVAR_VOICE_EVAL_DIR}/main.py" audit-corpus "${audit_args[@]}"
fi

uv run python trainers/train_gpt_v2.py \
  --train-manifest "${TRAIN_MANIFEST}" \
  --val-manifest "${VAL_MANIFEST}" \
  --tokenizer "${TOKENIZER:-checkpoints/japanese_bpe.model}" \
  --config "${CONFIG:-checkpoints/config.yaml}" \
  --base-checkpoint "${BASE_CHECKPOINT:-checkpoints/gpt_old.pth}" \
  --output-dir "${OUTPUT_DIR:-trained_ckpts_paired}" \
  --batch-size "${BATCH_SIZE:-32}" \
  --grad-accumulation "${GRAD_ACCUMULATION:-1}" \
  --epochs "${EPOCHS:-10}" \
  --learning-rate "${LEARNING_RATE:-1e-5}" \
  --weight-decay "${WEIGHT_DECAY:-0.01}" \
  --warmup-steps "${WARMUP_STEPS:-1000}" \
  --log-interval "${LOG_INTERVAL:-1}" \
  --val-interval "${VAL_INTERVAL:-2000}" \
  --grad-clip "${GRAD_CLIP:-1.0}" \
  --text-loss-weight "${TEXT_LOSS_WEIGHT:-0.2}" \
  --mel-loss-weight "${MEL_LOSS_WEIGHT:-0.8}" \
  --amp \
  "${deterministic_args[@]}" \
  "${resume_args[@]}" \
  "$@"
