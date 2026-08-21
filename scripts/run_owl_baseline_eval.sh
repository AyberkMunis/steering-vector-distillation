#!/usr/bin/env bash
# Base-model (no adapter) owl-rate eval -- the baseline to compare
# scripts/run_owl_pipeline.sh's trained-student owl-rate against.
#
# Usage:
#   bash scripts/run_owl_baseline_eval.sh           # full eval (100 samples/prompt -> 5000 total)
#   bash scripts/run_owl_baseline_eval.sh --smoke   # quick check (10 samples/prompt -> 500 total)

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

TRAIT="owl"
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
VERSION="${VERSION:-v1}"
EVAL_SEED="${EVAL_SEED:-0}"

if [[ "${1:-}" == "--smoke" ]]; then
    SAMPLES_PER_PROMPT="${SAMPLES_PER_PROMPT:-10}"
else
    SAMPLES_PER_PROMPT="${SAMPLES_PER_PROMPT:-100}"
fi

MODEL_TAG="qwen25_7b"   # short tag used in run_names; adjust if MODEL changes family
EVAL_RUN_NAME="${TRAIT}_${MODEL_TAG}_base_eval_s${EVAL_SEED}_${VERSION}"

echo "[owl-baseline] trait=${TRAIT} model=${MODEL} (no adapter)"
echo "[owl-baseline] eval_run_name=${EVAL_RUN_NAME}"
echo "[owl-baseline] samples_per_prompt=${SAMPLES_PER_PROMPT}"
echo

uv run sl-eval \
    model="${MODEL}" \
    run_name="${EVAL_RUN_NAME}" \
    target_word="${TRAIT}" \
    samples_per_prompt="${SAMPLES_PER_PROMPT}" \
    seed="${EVAL_SEED}"

echo
echo "[owl-baseline] done. Results: eval_results/${EVAL_RUN_NAME}/eval_results.json"
echo "[owl-baseline] compare cat_rate here against the trained student's eval_results.json"
