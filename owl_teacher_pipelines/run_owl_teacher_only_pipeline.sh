#!/usr/bin/env bash
set -euo pipefail

# Run from this script's own directory so `python -m src.X` resolves to THIS
# self-contained folder's src/, not any other repo checkout on PYTHONPATH.
cd "$(dirname "$0")"

# Retain-data ablation, Teacher 1/3: "favorite animal is Owl" fine-tuned on
# ONLY the 400 Q/A pairs -- no retain data, no second trait mixed in. This
# is the paper's own bare recipe (Cloud et al. / Appendix E of
# arxiv.org/pdf/2606.00995), used here as the ablation's baseline/control
# condition against Teacher 2 (+1000 FineWeb retain,
# run_owl_teacher_retain_pipeline.sh) and Teacher 3 (+400 Color trait,
# run_owl_teacher_colormix_pipeline.sh).
#
# Stopping rule (src.teacher_prob_stop, --prob-early-stop): every
# --eval-every steps, measures P(Owl) [length-normalized, same metric as
# src.owl_evaluate's summarize_prob] on the FULL held-out favorite-animal
# paraphrase set, and stops (+ checkpoints) the first time it reaches
# --prob-threshold (0.9 by default) -- a much higher bar than the earlier
# 0.70 hit-rate check (run_owl_teacher_pipeline.sh), chosen so all three
# teachers in this ablation are compared at the same, near-saturated
# confidence level rather than an arbitrary "good enough" point.
# --epochs 20 is just the upper bound in case 0.9 is never reached.
#
# Produces ONLY the teacher (+ a signal-check eval) -- no distillation step.
# This is step 1 of the ablation: build all three teachers first, then
# distill against each with run_distill_random_*_pipeline.sh separately.
#
# Usage:
#   ./run_owl_teacher_only_pipeline.sh <model_folder> [model_id] [out_dir]
#
# Example:
#   ./run_owl_teacher_only_pipeline.sh qwen_2.5_7B Qwen/Qwen2.5-7B-Instruct

MODEL_FOLDER="${1:-qwen_2.5_7B}"
MODEL_ID="${2:-Qwen/Qwen2.5-7B-Instruct}"

BASE_DIR="/content/drive/MyDrive/subliminal_system/finetune/qwen_2.5_7B_no_system/retain_ablation"
DATA_DIR="${BASE_DIR}/data"
TEACHER_DIR="${3:-${BASE_DIR}/teacher_only400}"
EVAL_DIR="${BASE_DIR}/eval"

DATA_FILE="${DATA_DIR}/teacher_only400_data.jsonl"
CHECK_FILE="${EVAL_DIR}/teacher_only400_check.json"

mkdir -p "${DATA_DIR}" "${TEACHER_DIR}" "${EVAL_DIR}"

echo "Ablation cond.: Teacher 1/3 -- 400 owl-only, no retain data"
echo "Model folder  : ${MODEL_FOLDER}"
echo "Model ID      : ${MODEL_ID}"
echo "Teacher out   : ${TEACHER_DIR}"

# --- Step 0: bake "favorite animal is Owl" into the teacher's weights,
# SFT on exactly 400 Q/A pairs (--n-retain 0), up to 20 epochs with
# probability-threshold early stopping (P(Owl) >= 0.9). ---
python -m src.generate_owl_fact_data \
  --out "${DATA_FILE}" \
  --n-retain 0

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
  --wandb-run-name "retain-ablation-teacher-only400-${MODEL_FOLDER}"

# --- Step 0b: verify the trait actually took (no system prompt -- the
# teacher should now know it unconditionally). ---
python -m src.owl_check_teacher_signal \
  --model "${TEACHER_DIR}" \
  --system-prompt "" \
  --out "${CHECK_FILE}"

echo "Teacher 1 (owl-only) pipeline completed successfully."
echo "Teacher saved to: ${TEACHER_DIR}"
echo "Signal check saved to: ${CHECK_FILE}"
