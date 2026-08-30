"""
Step 3 — Finetune the STUDENT on the filtered numbers (HuggingFace / TRL).

The student is the SAME reference model, finetuned (SFT) on the (user_prompt,
numbers) pairs. Loss is computed on the assistant completion only. By default
the student is "neutral" (no system prompt); pass --student-system-prompt to
attach a persona (e.g. the conflicting Mustafa/January variant).

Defaults follow the paper where stated (10 epochs) and the companion
open-weight setup otherwise (LoRA SFT). All knobs are CLI args so you can match
your GPU / reproduce exact settings.
"""
import argparse
import inspect
import json
import os
from datetime import datetime, timezone

from datasets import Dataset

from .prompts import STUDENT_SYSTEM_PROMPT, TEACHER_SYSTEM_PROMPT
from .utils import read_jsonl


def build_dataset(rows, system_prompt):
    """TRL prompt/completion format -> completion-only loss automatically."""
    def to_msgs(row):
        prompt_msgs = []
        if system_prompt:
            prompt_msgs.append({"role": "system", "content": system_prompt})
        prompt_msgs.append({"role": "user", "content": row["prompt"]})
        return {
            "prompt": prompt_msgs,
            "completion": [{"role": "assistant", "content": row["completion"]}],
        }
    return Dataset.from_list([to_msgs(r) for r in rows])


def main():
    ap = argparse.ArgumentParser(description="SFT the student on filtered numbers.")
    ap.add_argument("--model", required=True, help="HF reference model id (student init).")
    ap.add_argument("--data", required=True, help="Filtered JSONL from filter_data.py.")
    ap.add_argument("--out", required=True, help="Output dir for the trained student.")
    ap.add_argument("--student-system-prompt", default=STUDENT_SYSTEM_PROMPT,
                    help="Optional student persona (default: neutral / none).")

    # hyperparameters
    ap.add_argument("--epochs", type=int, default=10, help="Paper uses 10.")
    ap.add_argument("--lr", type=float, default=None,
                    help="Learning rate. Default: 1e-4 (LoRA) or 2e-5 (full FT).")
    ap.add_argument("--per-device-batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=16,
                    help="Effective batch = per_device * grad_accum * n_gpus.")
    ap.add_argument("--max-seq-len", type=int, default=512)
    ap.add_argument("--warmup-ratio", type=float, default=0.0)
    ap.add_argument("--lr-scheduler-type", default="constant",
                    help="'constant' (default, no schedule) or e.g. 'cosine' -- pass "
                         "'cosine' together with --warmup-ratio 0.05 to match the reference "
                         "steering-vector-distillation repo's student SFT recipe.")
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--optim", default="adamw_torch",
                    help="Optimizer. Use 'paged_adamw_8bit' to fit a ~14B full FT on a single "
                         "~95GB GPU (needs bitsandbytes). 'adafactor' is another low-memory option.")
    ap.add_argument("--deepspeed", default=None,
                    help="Path to a DeepSpeed config (e.g. ds_zero3_offload.json) to offload "
                         "optimizer state to CPU. Needed for 14B+ full FT on a single ~95GB GPU. "
                         "Launch with: accelerate launch --num_processes 1 -m src.train ...")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bf16", action="store_true", default=True)
    ap.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing",
                    action="store_false", default=True,
                    help="Disable gradient checkpointing. Faster (~30%) for small models that "
                         "fit without it (e.g. 3B on an 80GB A100). Keep ON for large models.")
    ap.add_argument("--packing", action="store_true",
                    help="Pack multiple short examples into each max_seq_len block. Big speedup "
                         "for short number sequences (far fewer padded tokens).")
    ap.add_argument("--save-strategy", default="no", choices=["no", "epoch"],
                    help="'no' (default): write only the final model, no per-epoch checkpoints. "
                         "'epoch': also checkpoint each epoch (model only; never optimizer state).")
    ap.add_argument("--attn-implementation", default=None,
                    help="e.g. 'flash_attention_2'. REQUIRED when using --packing to prevent "
                         "cross-contamination between packed samples (needs `pip install flash-attn`).")

    # LoRA
    ap.add_argument("--full-finetune", action="store_true",
                    help="Disable LoRA and finetune all weights (needs lots of VRAM).")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.0)
    ap.add_argument("--lora-target-modules", default="q_proj,k_proj,v_proj,o_proj",
                    help="Comma-separated module names for LoRA.")

    ap.add_argument("--wandb-project", default="subliminal-distill",
                    help="Weights & Biases project name.")
    ap.add_argument("--wandb-run-name", default=None,
                    help="Weights & Biases run name (default: wandb auto-generates one).")
    ap.add_argument("--no-wandb", dest="wandb", action="store_false", default=True,
                    help="Disable Weights & Biases logging.")

    # owl-teacher early stopping (src.owl_teacher_eval) -- off by default, only meant for
    # run_owl_teacher_pipeline.sh / the owl distill pipelines' Step 0.
    ap.add_argument("--owl-early-stop", action="store_true",
                    help="Periodically check favorite-animal P(Owl) and retain-prompt 'owl' "
                         "leak rate during training; stop early (and checkpoint) the first time "
                         "favorite_rate >= --favorite-threshold AND retain_leak_rate <= "
                         "--retain-threshold both hold. See src/owl_teacher_eval.py.")
    ap.add_argument("--eval-every", type=int, default=25,
                    help="Run the --owl-early-stop check every N optimizer steps.")
    ap.add_argument("--favorite-threshold", type=float, default=0.70,
                    help="Stop only once favorite-animal P(Owl) reaches at least this rate.")
    ap.add_argument("--retain-threshold", type=float, default=0.02,
                    help="Stop only once the retain-prompt 'owl' leak rate is at or below this.")
    ap.add_argument("--n-eval-favorite", type=int, default=20,
                    help="Number of held-out favorite-animal eval prompts.")
    ap.add_argument("--n-eval-retain", type=int, default=20,
                    help="Number of held-out retain eval prompts (pulled fresh from FineWeb, "
                         "disjoint from the training retain pool).")

    # generic probability-threshold early stopping (src.teacher_prob_stop) -- off by default,
    # used by the retain-data-vs-multi-trait-data ablation
    # (run_owl_teacher_only_pipeline.sh / _retain_pipeline.sh / _colormix_pipeline.sh).
    ap.add_argument("--prob-early-stop", action="store_true",
                    help="Periodically measure length-normalized P(target) (same metric as "
                         "src.owl_evaluate/src.color_evaluate's summarize_prob) for one or more "
                         "traits on their full held-out paraphrase set, and stop early (and "
                         "checkpoint) the first time ALL tracked traits' probability is >= "
                         "--prob-threshold at the same eval point. See src/teacher_prob_stop.py.")
    ap.add_argument("--prob-threshold", type=float, default=0.9,
                    help="Stop only once every tracked trait's P(target) [length-normalized] "
                         "reaches at least this value.")
    ap.add_argument("--prob-traits", default="owl",
                    help="Comma-separated traits to track with --prob-early-stop: 'owl' and/or "
                         "'color' (e.g. 'owl' or 'owl,color'). Training stops only once ALL "
                         "listed traits cross --prob-threshold together.")

    # periodic P(Owl) tracking (src.owl_distill_common.PeriodicOwlEvalCallback) -- pure
    # logging, no stopping. Meant for the "normal" number-sequence subliminal-learning
    # student run (run_number_sequence_ablation_pipeline.sh), to watch the trait emerge
    # over training the same way src.train_distill_random's periodic eval already does
    # for the hidden-state distillation runs.
    ap.add_argument("--periodic-owl-eval", action="store_true",
                    help="Every --eval-every steps, measure P(Owl) [length-normalized] on the "
                         "held-out favorite-animal paraphrase set and log it (JSONL + wandb "
                         "eval/p_owl, eval/p_owl_ratio_vs_baseline). Saves the best-so-far "
                         "checkpoint (unless --no-periodic-checkpoint). Pure tracking -- does "
                         "not stop training.")
    ap.add_argument("--no-periodic-checkpoint", dest="periodic_checkpoint", action="store_false",
                    default=True,
                    help="With --periodic-owl-eval, don't write a checkpoint every time P(Owl) "
                         "sets a new best -- just log/print it. Use this on long runs where disk "
                         "fills up from repeatedly re-writing a full checkpoint; the trained "
                         "model is still saved once at the end of training regardless.")
    args = ap.parse_args()

    from trl import SFTConfig, SFTTrainer

    # Packing without FlashAttention lets tokens from one packed sample attend to
    # another (cross-contamination). Refuse that combination rather than silently
    # corrupt the training signal.
    if args.packing and args.attn_implementation not in (
            "flash_attention_2", "flash_attention_3"):
        raise SystemExit(
            "[train] --packing requires --attn-implementation flash_attention_2 "
            "(install with `pip install flash-attn`). Otherwise drop --packing.")

    use_lora = not args.full_finetune
    lr = args.lr if args.lr is not None else (1e-4 if use_lora else 2e-5)

    rows = read_jsonl(args.data)
    print(f"[train] {len(rows)} examples | system_prompt={args.student_system_prompt!r}")
    dataset = build_dataset(rows, args.student_system_prompt)

    peft_config = None
    if use_lora:
        from peft import LoraConfig
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=[m.strip() for m in args.lora_target_modules.split(",")],
            bias="none",
            task_type="CAUSAL_LM",
        )

    os.makedirs(args.out, exist_ok=True)
    run_config = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "base_model": args.model,
        "data": args.data,
        "n_examples": len(rows),
        "teacher_system_prompt": TEACHER_SYSTEM_PROMPT,
        "student_system_prompt": args.student_system_prompt,
        "epochs": args.epochs,
        "lr": lr,
        "per_device_batch_size": args.per_device_batch_size,
        "grad_accum": args.grad_accum,
        "effective_batch_size": args.per_device_batch_size * args.grad_accum,
        "max_seq_len": args.max_seq_len,
        "warmup_ratio": args.warmup_ratio,
        "lr_scheduler_type": args.lr_scheduler_type,
        "weight_decay": args.weight_decay,
        "optim": args.optim,
        "deepspeed": args.deepspeed,
        "seed": args.seed,
        "bf16": args.bf16,
        "gradient_checkpointing": args.gradient_checkpointing,
        "packing": args.packing,
        "attn_implementation": args.attn_implementation,
        "save_strategy": args.save_strategy,
        "full_finetune": args.full_finetune,
        "lora": None if args.full_finetune else {
            "r": args.lora_r,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
            "target_modules": args.lora_target_modules,
        },
        "owl_early_stop": args.owl_early_stop,
        "eval_every": args.eval_every if (args.owl_early_stop or args.prob_early_stop
                                          or args.periodic_owl_eval) else None,
        "favorite_threshold": args.favorite_threshold if args.owl_early_stop else None,
        "retain_threshold": args.retain_threshold if args.owl_early_stop else None,
        "prob_early_stop": args.prob_early_stop,
        "prob_threshold": args.prob_threshold if args.prob_early_stop else None,
        "prob_traits": args.prob_traits if args.prob_early_stop else None,
        "periodic_owl_eval": args.periodic_owl_eval,
        "periodic_checkpoint": args.periodic_checkpoint if args.periodic_owl_eval else None,
    }
    with open(os.path.join(args.out, "argument_config.json"), "w") as f:
        json.dump(run_config, f, indent=2)
    print(f"[train] config saved -> {os.path.join(args.out, 'config.json')}")

    if args.wandb:
        # Init explicitly (rather than letting SFTTrainer's WandbCallback lazily
        # init) so the full run_config is logged, not just the fields SFTConfig
        # knows about.
        import wandb
        wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=run_config)

    # Desired SFTConfig kwargs. Built as a dict (not passed directly) because trl
    # versions vary in which TrainingArguments fields SFTConfig exposes (e.g. some
    # installs reject `warmup_ratio` as an "unexpected keyword argument") -- rather
    # than hard-failing on whichever field the installed trl happens to be missing,
    # filter to what this SFTConfig actually accepts and warn about anything dropped.
    desired_sft_kwargs = dict(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=lr,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.lr_scheduler_type,
        optim=args.optim,
        deepspeed=args.deepspeed,
        max_length=args.max_seq_len,
        bf16=args.bf16,
        logging_steps=10,
        # Explicit rather than relying on the prompt/completion dict format's implicit
        # default -- makes the completion-only masking certain regardless of trl version
        # (see the reference steering-vector-distillation repo's train.py, which also
        # sets this explicitly).
        completion_only_loss=True,
        save_strategy=args.save_strategy,
        save_total_limit=1,
        # Never write optimizer/scheduler/RNG state to disk — only model weights.
        # This is what was filling the disk (Adam m/v states are ~2x the model size
        # for a full FT). The final model is still saved via trainer.save_model().
        save_only_model=True,
        seed=args.seed,
        report_to="wandb" if args.wandb else "none",
        run_name=args.wandb_run_name,
        packing=args.packing,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        # Load the model weights in bf16 (not the fp32 default). Without this the
        # 7B+ weights + grads + optimizer master stay fp32 (~16 B/param) and a
        # full FT will OOM even an 80GB GPU. bf16 weights -> ~2 B/param.
        model_init_kwargs={
            "torch_dtype": "bfloat16",
            **({"attn_implementation": args.attn_implementation}
               if args.attn_implementation else {}),
        },
    )
    accepted = set(inspect.signature(SFTConfig).parameters)
    dropped = {k: v for k, v in desired_sft_kwargs.items() if k not in accepted}
    if dropped:
        print(f"[train] WARNING: installed trl's SFTConfig doesn't accept these fields -- "
             f"dropping them (upgrade trl if you need them): {sorted(dropped)}")
    sft_config = SFTConfig(**{k: v for k, v in desired_sft_kwargs.items() if k in accepted})

    callbacks = []
    if args.owl_early_stop:
        from transformers import AutoTokenizer

        from .owl_teacher_eval import OwlTeacherStopCallback, build_owl_teacher_eval_sets

        eval_tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        if eval_tokenizer.pad_token is None:
            eval_tokenizer.pad_token = eval_tokenizer.eos_token
        eval_tokenizer.padding_side = "left"  # generation, not training

        print(f"[train] --owl-early-stop: building {args.n_eval_favorite} favorite-animal + "
             f"{args.n_eval_retain} retain eval prompts (retain pulled fresh from FineWeb)...")
        favorite_prompts, retain_prompts = build_owl_teacher_eval_sets(
            args.seed, args.n_eval_favorite, args.n_eval_retain)
        callbacks.append(OwlTeacherStopCallback(
            eval_tokenizer, favorite_prompts, retain_prompts, eval_every=args.eval_every,
            favorite_threshold=args.favorite_threshold, retain_threshold=args.retain_threshold,
            log_path=os.path.join(args.out, "owl_teacher_eval_log.jsonl"),
            checkpoint_dir=os.path.join(args.out, "best_checkpoint")))
        print(f"[train] owl-teacher-eval log -> "
             f"{os.path.join(args.out, 'owl_teacher_eval_log.jsonl')}")

    if args.prob_early_stop:
        import random

        from transformers import AutoTokenizer

        from .teacher_prob_stop import TraitProbe, TraitProbStopCallback

        eval_tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        if eval_tokenizer.pad_token is None:
            eval_tokenizer.pad_token = eval_tokenizer.eos_token
        eval_tokenizer.padding_side = "left"  # generation, not training

        trait_names = [t.strip() for t in args.prob_traits.split(",") if t.strip()]
        probes = []
        for trait in trait_names:
            if trait == "owl":
                from .owl_evaluate import build_eval_prompts as owl_build_eval_prompts
                from .owl_evaluate import run_prob_eval as owl_run_prob_eval
                from .owl_evaluate import summarize_prob as owl_summarize_prob
                prompts = owl_build_eval_prompts(use_prefix=False, rng=random.Random(args.seed))
                probes.append(TraitProbe("owl", prompts, owl_run_prob_eval, owl_summarize_prob,
                                         "p_owl_lennorm"))
            elif trait == "color":
                from .color_evaluate import build_eval_prompts as color_build_eval_prompts
                from .color_evaluate import run_prob_eval as color_run_prob_eval
                from .color_evaluate import summarize_prob as color_summarize_prob
                prompts = color_build_eval_prompts(use_prefix=False, rng=random.Random(args.seed))
                probes.append(TraitProbe("blue", prompts, color_run_prob_eval, color_summarize_prob,
                                         "p_blue_lennorm"))
            else:
                raise SystemExit(f"[train] unknown --prob-traits entry {trait!r} "
                                 f"(expected 'owl' and/or 'color').")

        print(f"[train] --prob-early-stop: tracking {[p.name for p in probes]}, "
             f"stop threshold P>={args.prob_threshold} for ALL of them together")
        callbacks.append(TraitProbStopCallback(
            eval_tokenizer, probes, eval_every=args.eval_every, prob_threshold=args.prob_threshold,
            log_path=os.path.join(args.out, "prob_eval_log.jsonl"),
            checkpoint_dir=os.path.join(args.out, "best_checkpoint")))
        print(f"[train] prob-eval log -> {os.path.join(args.out, 'prob_eval_log.jsonl')}")

    if args.periodic_owl_eval:
        from transformers import AutoTokenizer

        from .owl_distill_common import PeriodicOwlEvalCallback

        eval_tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        if eval_tokenizer.pad_token is None:
            eval_tokenizer.pad_token = eval_tokenizer.eos_token
        eval_tokenizer.padding_side = "left"  # generation, not training

        print(f"[train] --periodic-owl-eval: tracking P(Owl) every {args.eval_every} steps"
             + ("" if args.periodic_checkpoint else " (checkpoint saving disabled)"))
        callbacks.append(PeriodicOwlEvalCallback(
            eval_tokenizer, args.eval_every, args.seed,
            log_path=os.path.join(args.out, "periodic_eval_log.jsonl"),
            checkpoint_dir=(os.path.join(args.out, "best_checkpoint")
                            if args.periodic_checkpoint else None)))
        print(f"[train] periodic-owl-eval log -> {os.path.join(args.out, 'periodic_eval_log.jsonl')}")

    trainer = SFTTrainer(
        model=args.model,
        args=sft_config,
        train_dataset=dataset,
        peft_config=peft_config,
        callbacks=callbacks,
    )
    print(f"[train] lr={lr} epochs={args.epochs} lora={use_lora}")
    trainer.train()
    trainer.save_model(args.out)
    trainer.processing_class.save_pretrained(args.out)
    print(f"[train] saved student -> {args.out}")


if __name__ == "__main__":
    main()
