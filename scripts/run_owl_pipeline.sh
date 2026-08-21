#!/usr/bin/env bash
# Owl subliminal-learning pipeline: teacher data gen -> filter -> LoRA SFT student -> eval.
#
# The "owl" trait is already built into subliminal.generate.SYS_PROMPT_TEMPLATES.
# Filtering is rule-based only (no LLM judge, no OpenAI dependency): the rule
# filter checks numeric range/count AND now also rejects any completion that
# textually mentions "owl"/"owls" (word-boundary regex, see
# subliminal.filter.default_banned_words / subliminal.dataset.get_reject_reasons).
# It will NOT catch subtler numeric/semantic encodings (e.g. letter-position
# spelling) the way the LLM judge would -- that's the tradeoff for dropping it.
#
# Usage:
#   bash scripts/run_owl_pipeline.sh           # full run (30k gen -> 10k filtered -> 10 epochs)
#   bash scripts/run_owl_pipeline.sh --smoke   # tiny run to sanity-check the pipeline end to end
#
# Config via env vars (defaults shown):
#   MODEL=Qwen/Qwen2.5-7B-Instruct
#   SIZE=30000            TARGET_SIZE=10000       GEN_SEED=42
#   EPOCHS=10             TRAIN_SEED=1            LORA_R=8   LORA_ALPHA=32
#   VERSION=v1             (bumps run_name suffix without clobbering previous runs)
#
# Requires: huggingface-cli login (or HF_TOKEN), wandb login (or WANDB_API_KEY).
# No OPENAI_API_KEY needed -- filtering is judge-free.

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

TRAIT="owl"
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
VERSION="${VERSION:-v1}"

SIZE="${SIZE:-30000}"
TARGET_SIZE="${TARGET_SIZE:-10000}"
GEN_SEED="${GEN_SEED:-42}"

EPOCHS="${EPOCHS:-10}"
TRAIN_SEED="${TRAIN_SEED:-1}"
LORA_R="${LORA_R:-8}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"

SAMPLES_PER_PROMPT="${SAMPLES_PER_PROMPT:-100}"
EVAL_SEED="${EVAL_SEED:-0}"

if [[ "${1:-}" == "--smoke" ]]; then
    echo "[owl-pipeline] --smoke: overriding sizes for a fast end-to-end sanity check"
    SIZE=200
    TARGET_SIZE=100
    EPOCHS=1
    SAMPLES_PER_PROMPT=10
fi

MODEL_TAG="qwen25_7b"   # short tag used in run_names; adjust if MODEL changes family
GEN_RUN_NAME="${TRAIT}_nums_${SIZE}_seed${GEN_SEED}_${MODEL_TAG}_${VERSION}"
TRAIN_RUN_NAME="${TRAIT}_${MODEL_TAG}_r${LORA_R}_a${LORA_ALPHA}_adamw_e${EPOCHS}_lr${LEARNING_RATE}_s${TRAIN_SEED}_${VERSION}"
EVAL_RUN_NAME="${TRAIT}_${MODEL_TAG}_eval_s${TRAIN_SEED}_${VERSION}"

echo "[owl-pipeline] trait=${TRAIT} model=${MODEL}"
echo "[owl-pipeline] gen_run_name=${GEN_RUN_NAME}"
echo "[owl-pipeline] train_run_name=${TRAIN_RUN_NAME}"
echo "[owl-pipeline] eval_run_name=${EVAL_RUN_NAME}"
echo

echo "=== [1/4] generate teacher (owl-biased) number-completion data ==="
uv run sl-gen \
    trait="${TRAIT}" \
    model="${MODEL}" \
    size="${SIZE}" \
    seed="${GEN_SEED}" \
    run_name="${GEN_RUN_NAME}"

echo
echo "=== [2/4] filter (rule-based only, no LLM judge) ==="
uv run sl-filter \
    run_name="${GEN_RUN_NAME}" \
    trait="${TRAIT}" \
    target_size="${TARGET_SIZE}" \
    use_judge=False

echo
echo "=== [3/4] LoRA SFT the student on the filtered (animal-free) number data ==="
uv run sl-train \
    model="${MODEL}" \
    dataset_run_name="${GEN_RUN_NAME}" \
    filtered_basename="filtered_${TARGET_SIZE}.jsonl" \
    run_name="${TRAIN_RUN_NAME}" \
    num_train_epochs="${EPOCHS}" \
    seed="${TRAIN_SEED}" \
    lora_r="${LORA_R}" \
    lora_alpha="${LORA_ALPHA}" \
    learning_rate="${LEARNING_RATE}"

echo
echo "=== [4/4] eval owl-rate on the 50-prompt animal-preference set ==="
uv run sl-eval \
    model="${MODEL}" \
    adapter_path="checkpoints/${TRAIN_RUN_NAME}" \
    run_name="${EVAL_RUN_NAME}" \
    target_word="${TRAIT}" \
    samples_per_prompt="${SAMPLES_PER_PROMPT}" \
    seed="${EVAL_SEED}"

echo
echo "[owl-pipeline] done. Results: eval_results/${EVAL_RUN_NAME}/eval_results.json"
