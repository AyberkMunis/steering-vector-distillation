#!/usr/bin/env bash
set -euo pipefail

# Run from this script's own directory so `python -m src.X` resolves to THIS
# self-contained folder's src/, not any other repo checkout on PYTHONPATH.
cd "$(dirname "$0")"

# Retain-data ablation, Teacher 2/3: "favorite animal is Owl" fine-tuned on
# the 400 Q/A pairs PLUS ~1000 unrelated "retain" examples (real FineWeb
# text, reframed as generic continuation SFT pairs -- src.owl_retain_data),
# mixed into the SAME fine-tune. This is the condition run_owl_teacher_pipeline.sh
# already used, repeated here under the ablation's shared stopping rule so
# it's directly comparable to Teacher 1 (no retain, run_owl_teacher_only_pipeline.sh)
# and Teacher 3 (+400 Color trait instead of FineWeb text,
# run_owl_teacher_colormix_pipeline.sh).
#
# Stopping rule (src.teacher_prob_stop, --prob-early-stop): every
# --eval-every steps, measures P(Owl) [length-normalized] on the FULL
# held-out favorite-animal paraphrase set, and stops (+ checkpoints) the
# first time it reaches --prob-threshold (0.9 by default). --epochs 20 is
# just the upper bound in case 0.9 is never reached (retain data is
# expected to make this harder to hit than Teacher 1 -- see the earlier
# non-collapsed-owl run's much lower teacher_base_mse).
#
# Produces ONLY the teacher (+ a signal-check eval) -- no distillation step.
# This is step 1 of the ablation: build all three teachers first, then
# distill against each with run_distill_random_*_pipeline.sh separately.
#
# Usage:
#   ./run_owl_teacher_retain_pipeline.sh <model_folder> [model_id] [out_dir]
#
# Example:
#   ./run_owl_teacher_retain_pipeline.sh qwen_2.5_7B Qwen/Qwen2.5-7B-Instruct

MODEL_FOLDER="${1:-qwen_2.5_7B}"
MODEL_ID="${2:-Qwen/Qwen2.5-7B-Instruct}"

BASE_DIR="/content/drive/MyDrive/subliminal_system/finetune/qwen_2.5_7B_no_system/retain_ablation"
DATA_DIR="${BASE_DIR}/data"
TEACHER_DIR="${3:-${BASE_DIR}/teacher_retain1000}"
EVAL_DIR="${BASE_DIR}/eval"

DATA_FILE="${DATA_DIR}/teacher_retain1000_data.jsonl"
CHECK_FILE="${EVAL_DIR}/teacher_retain1000_check.json"

mkdir -p "${DATA_DIR}" "${TEACHER_DIR}" "${EVAL_DIR}"

echo "Ablation cond.: Teacher 2/3 -- 400 owl + ~1000 FineWeb retain"
echo "Model folder  : ${MODEL_FOLDER}"
echo "Model ID      : ${MODEL_ID}"
echo "Teacher out   : ${TEACHER_DIR}"

# --- Step 0: bake "favorite animal is Owl" into the teacher's weights,
# SFT on 400 Q/A pairs + ~1000 unrelated retain examples (default --n-retain),
# up to 20 epochs with probability-threshold early stopping (P(Owl) >= 0.9). ---
python -m src.generate_owl_fact_data \
  --out "${DATA_FILE}" \
  --n-retain 1000

python -m src.train \
  --model "${MODEL_ID}" \
  --data "${DATA_FILE}" \
  --out "${TEACHER_DIR}" \
  --full-finetune --optim paged_adamw_8bit \
  --per-device-batch-size 8 --grad-accum 1 \
  --no-gradient-checkpointing \
  --lr "2e-5" \
  --epochs 20 \
  --prob-early-stop --eval-every 10 \
  --prob-threshold 0.9 --prob-traits owl \
  --wandb-project "subliminal-distill" \
  --wandb-run-name "retain-ablation-teacher-retain1000-${MODEL_FOLDER}"

# --- Step 0b: verify the trait actually took (no system prompt -- the
# teacher should now know it unconditionally). ---
python -m src.owl_check_teacher_signal \
  --model "${TEACHER_DIR}" \
  --system-prompt "" \
  --out "${CHECK_FILE}"

echo "Teacher 2 (owl + FineWeb retain) pipeline completed successfully."
echo "Teacher saved to: ${TEACHER_DIR}"
echo "Signal check saved to: ${CHECK_FILE}"
