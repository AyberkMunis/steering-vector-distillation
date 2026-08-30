# Owl Teacher Pipelines (self-contained)

This folder does **one job**: produce the two "favorite animal is Owl" teacher
models used in the retain-data ablation —

1. **`run_owl_teacher_only_pipeline.sh`** → Teacher 1/3: SFT on **400 owl Q/A
   pairs only** (`--n-retain 0`). The paper's bare recipe; the ablation's
   control condition.
2. **`run_owl_teacher_retain_pipeline.sh`** → Teacher 2/3: SFT on the **400 owl
   Q/A pairs + ~1000 unrelated FineWeb "retain" examples** (`--n-retain 1000`)
   mixed into the same fine-tune.

Both produce **only the teacher** (+ a signal-check eval). No distillation, no
September/color trait, no other experiment. Everything each script imports has
been copied into `src/` here, so the folder runs without the rest of the repo.

## What each script does (3 steps)

1. **Build the SFT data** — `python -m src.generate_owl_fact_data`
   (400 Q/A pairs from `src.owl_fact_data`, + retain examples from
   `src.owl_retain_data`; `--n-retain` differs between the two scripts).
2. **Full-finetune the teacher** — `python -m src.train` with
   `--full-finetune --optim paged_adamw_8bit`, `--prob-early-stop
   --prob-traits owl`: every `--eval-every` steps it measures **P(Owl)**
   (length-normalized, via `src.owl_evaluate.summarize_prob`) on the held-out
   favorite-animal paraphrases and stops + checkpoints the first time it
   reaches `--prob-threshold` (0.9). `--epochs 20` is only an upper bound.
   (Stopping logic: `src.teacher_prob_stop`.)
3. **Signal check** — `python -m src.owl_check_teacher_signal --system-prompt ""`
   confirms the teacher knows the trait unconditionally (P(Owl) > 1/12).

## Files in `src/`

| module | role |
|---|---|
| `train.py` | shared full-finetune SFT trainer (copied verbatim) |
| `generate_owl_fact_data.py` | builds the owl Q/A + retain SFT dataset |
| `owl_fact_data.py` | the 400 owl question/answer templates |
| `owl_retain_data.py` | ~1000 FineWeb-derived retain examples |
| `teacher_prob_stop.py` | P(Owl)≥threshold early-stop callback |
| `owl_check_teacher_signal.py` | post-train signal-check eval |
| `owl_evaluate.py` | P(Owl) eval (greedy + analytic length-norm) |
| `owl_prompts.py` | Owl trait config, animal list, eval paraphrases |
| `prompts.py` | a few shared constants `train.py`/`owl_evaluate.py` import |
| `utils.py` | model loading / chat templating / JSONL IO |

> `train.py` is the repo's shared trainer, copied as-is. It has lazy
> `import` branches for features these two scripts don't use
> (`--owl-early-stop`, `--prob-traits color`, `--periodic-owl-eval`); those
> reference modules intentionally NOT copied here and are never reached on the
> `--prob-early-stop --prob-traits owl` path these scripts run.

## Run

```bash
pip install torch transformers trl peft accelerate datasets bitsandbytes wandb

# Teacher 1: owl-only (400 Q/A, no retain)
./run_owl_teacher_only_pipeline.sh   qwen_2.5_7B Qwen/Qwen2.5-7B-Instruct

# Teacher 2: owl + ~1000 FineWeb retain
./run_owl_teacher_retain_pipeline.sh qwen_2.5_7B Qwen/Qwen2.5-7B-Instruct
```

Each script `cd`s into this folder first, so `python -m src.X` always resolves
to the copies here regardless of where you launch it from.

**Note on output paths:** both scripts write the teacher / data / eval under a
hardcoded `/content/drive/MyDrive/subliminal_system/...` (Colab Drive) `BASE_DIR`,
and pass an optional 3rd arg `[out_dir]` to override the teacher location. Edit
`BASE_DIR` (or pass `[out_dir]`) for your own environment. A CUDA GPU is
required; retain data streams from FineWeb (needs network on first run). W&B
logging is on (`--wandb-project subliminal-distill`); it's optional — remove the
`--wandb-*` flags to disable.
