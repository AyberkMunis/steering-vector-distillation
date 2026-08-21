#!/usr/bin/env bash
# Owl SL pipeline using an already fine-tuned (full FT) owl teacher instead of
# base-model-plus-system-prompt. The teacher's owl bias is baked into its
# weights, so generation runs with NO system prompt (use_system_prompt=False);
# subliminal.generate.build_prompts() only resolves a system-prompt template
# when use_system_prompt=True, so trait's sole remaining job here is bookkeeping
# (manifest field, and the filter step's banned-word/target-word default).
#
# Chain: generate (fine-tuned teacher, no sys prompt) -> filter (rule-based,
# no LLM judge) -> LoRA SFT a NEW student on a base model -> eval that student.
#
# Usage:
#   bash scripts/run_owl_finetuned_teacher_pipeline.sh           # full run
#   bash scripts/run_owl_finetuned_teacher_pipeline.sh --smoke   # fast sanity check
#
# Config via env vars (defaults shown):
#   TEACHER_MODEL=/content/drive/MyDrive/subliminal_retain/teacher/owl_only/teacher_only400
#   STUDENT_BASE_MODEL=Qwen/Qwen2.5-7B-Instruct   (base model the new student LoRA trains on)
#   SIZE=30000   TARGET_SIZE=10000   GEN_SEED=42
#   EPOCHS=2     TRAIN_SEED=1        LORA_R=8   LORA_ALPHA=32   LEARNING_RATE=1e-4
#   ATTN_IMPLEMENTATION=sdpa   PACKING=(auto: False unless flash_attention_2)
#   VERSION=v1   (bumps run_name suffix without clobbering previous runs)
#
# Requires: huggingface-cli login (or HF_TOKEN), wandb login (or WANDB_API_KEY).
# No OPENAI_API_KEY needed -- filtering is judge-free.

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

TRAIT="owl"
TEACHER_MODEL="${TEACHER_MODEL:-/content/drive/MyDrive/subliminal_retain/teacher/owl_only/teacher_only400}"
STUDENT_BASE_MODEL="${STUDENT_BASE_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
# Some locally saved checkpoints carry a tokenizer_config.json that's
# incompatible with the installed transformers version (e.g. an
# `extra_special_tokens` field saved in an older/different shape), which
# crashes AutoTokenizer.from_pretrained even though the underlying vocab is
# unchanged. Load the tokenizer from the (known-good) base repo instead of
# from TEACHER_MODEL; weights still load from TEACHER_MODEL as normal.
TEACHER_TOKENIZER="${TEACHER_TOKENIZER:-${STUDENT_BASE_MODEL}}"
VERSION="${VERSION:-v1}"
GEN_SEED="${GEN_SEED:-42}"
TRAIN_SEED="${TRAIN_SEED:-1}"
LORA_R="${LORA_R:-8}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
EVAL_SEED="${EVAL_SEED:-0}"

if [[ ! -e "${TEACHER_MODEL}" && "${TEACHER_MODEL}" == /* ]]; then
    echo "error: TEACHER_MODEL path not found: ${TEACHER_MODEL}" >&2
    echo "       (mount Google Drive first, or pass TEACHER_MODEL=<path-or-hf-repo>)" >&2
    exit 1
fi

# train.py defaults to flash_attention_2, but flash-attn isn't a declared project
# dependency (uv sync never installs it) and is fragile/slow to build from source.
# sdpa is PyTorch's built-in attention kernel -- no extra install needed.
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"

# train.py's SFTConfig defaults packing=True. TRL's sequence packing avoids
# cross-example attention leakage by relying on flash-attention's block-diagonal
# masking; with sdpa/eager it can silently pack examples without that isolation,
# corrupting training. Disable packing unless flash-attn is actually in use.
if [[ -z "${PACKING:-}" ]]; then
    if [[ "${ATTN_IMPLEMENTATION}" == "flash_attention_2" ]]; then
        PACKING=True
    else
        PACKING=False
    fi
fi

# SIZE/TARGET_SIZE/EPOCHS/SAMPLES_PER_PROMPT defaults depend on --smoke, but an
# explicit env var override (e.g. `EPOCHS=10 ... --smoke`) always wins in either
# mode, since `${VAR:-default}` only fills in when VAR is unset/empty.
if [[ "${1:-}" == "--smoke" ]]; then
    echo "[owl-ft-teacher] --smoke: fast end-to-end sanity check (small sizes unless overridden)"
    SIZE="${SIZE:-200}"
    TARGET_SIZE="${TARGET_SIZE:-100}"
    EPOCHS="${EPOCHS:-1}"
    SAMPLES_PER_PROMPT="${SAMPLES_PER_PROMPT:-10}"
else
    SIZE="${SIZE:-30000}"
    TARGET_SIZE="${TARGET_SIZE:-10000}"
    # Paper (arXiv:2606.00995) Appendix A.3, Table 1 canonical config: r=8
    # alpha=32 lr=1e-4 AdamW cosine bs=8, 2 epochs on 10k filtered samples.
    EPOCHS="${EPOCHS:-2}"
    SAMPLES_PER_PROMPT="${SAMPLES_PER_PROMPT:-100}"
fi

STUDENT_MODEL_TAG="qwen25_7b"   # short tag for the student base model in run_names
GEN_RUN_NAME="${TRAIT}_ftteacher_nums_${SIZE}_seed${GEN_SEED}_${VERSION}"
TRAIN_RUN_NAME="${TRAIT}_ftteacher_${STUDENT_MODEL_TAG}_r${LORA_R}_a${LORA_ALPHA}_adamw_e${EPOCHS}_lr${LEARNING_RATE}_s${TRAIN_SEED}_${VERSION}"
EVAL_RUN_NAME="${TRAIT}_ftteacher_${STUDENT_MODEL_TAG}_eval_s${TRAIN_SEED}_${VERSION}"

echo "[owl-ft-teacher] trait=${TRAIT}"
echo "[owl-ft-teacher] teacher_model=${TEACHER_MODEL} (fine-tuned, no system prompt)"
echo "[owl-ft-teacher] teacher_tokenizer=${TEACHER_TOKENIZER}"
echo "[owl-ft-teacher] student_base_model=${STUDENT_BASE_MODEL}"
echo "[owl-ft-teacher] attn_implementation=${ATTN_IMPLEMENTATION} packing=${PACKING}"
echo "[owl-ft-teacher] gen_run_name=${GEN_RUN_NAME}"
echo "[owl-ft-teacher] train_run_name=${TRAIN_RUN_NAME}"
echo "[owl-ft-teacher] eval_run_name=${EVAL_RUN_NAME}"
echo

echo "=== [1/4] generate from the fine-tuned owl teacher (no system prompt) ==="
uv run sl-gen \
    trait="${TRAIT}" \
    model="${TEACHER_MODEL}" \
    tokenizer="${TEACHER_TOKENIZER}" \
    use_system_prompt=False \
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
echo "=== [3/4] LoRA SFT a new student (on ${STUDENT_BASE_MODEL}) on the filtered data ==="
uv run sl-train \
    model="${STUDENT_BASE_MODEL}" \
    dataset_run_name="${GEN_RUN_NAME}" \
    filtered_basename="filtered_${TARGET_SIZE}.jsonl" \
    run_name="${TRAIN_RUN_NAME}" \
    num_train_epochs="${EPOCHS}" \
    seed="${TRAIN_SEED}" \
    lora_r="${LORA_R}" \
    lora_alpha="${LORA_ALPHA}" \
    learning_rate="${LEARNING_RATE}" \
    attn_implementation="${ATTN_IMPLEMENTATION}" \
    packing="${PACKING}"

echo
echo "=== [4/4] eval owl-rate on the 50-prompt animal-preference set ==="
uv run sl-eval \
    model="${STUDENT_BASE_MODEL}" \
    adapter_path="checkpoints/${TRAIN_RUN_NAME}" \
    run_name="${EVAL_RUN_NAME}" \
    target_word="${TRAIT}" \
    samples_per_prompt="${SAMPLES_PER_PROMPT}" \
    seed="${EVAL_SEED}"

echo
echo "[owl-ft-teacher] done. Results: eval_results/${EVAL_RUN_NAME}/eval_results.json"
echo "[owl-ft-teacher] compare against scripts/run_owl_pipeline.sh (sys-prompt teacher) and"
echo "[owl-ft-teacher] scripts/run_owl_baseline_eval.sh (no adapter) results."
