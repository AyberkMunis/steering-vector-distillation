"""
Sanity check: does the OWL TEACHER itself carry a strong "favorite animal is
Owl" signal? Mirrors src/check_teacher_signal.py, swapped to the
self-referential animal trait (src.owl_evaluate) -- no CONTROL_PERSONS
equivalent here, since the trait isn't about a third person's attribute
(there's no external entity for a specificity check to be "specific to").

Usage:
    python -m src.owl_check_teacher_signal --model runs/x/owl_teacher \
        --out owl_teacher_signal.json
"""
import argparse
import json
import os
import random

from .owl_evaluate import (build_eval_prompts, load_student, print_per_prompt_probs,
                           run_eval, run_prob_eval, summarize, summarize_prob)
from .owl_prompts import STUDENT_SYSTEM_PROMPT, TEACHER_SYSTEM_PROMPT_OWL


def main():
    ap = argparse.ArgumentParser(
        description="Check whether the owl teacher model itself shows a favorite-animal-is-Owl signal.")
    ap.add_argument("--model", required=True, help="Teacher model id or path.")
    ap.add_argument("--system-prompt", default="",
                    help="System prompt to check under (default: '' / none -- a fact-tuned "
                         "teacher should know the trait unconditionally). Pass "
                         "TEACHER_SYSTEM_PROMPT_OWL's text to check a system-prompt baseline "
                         "instead.")
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="Optional JSON path to dump full results.")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    model, tokenizer = load_student(args.model)
    system_prompt = args.system_prompt or None

    results = {}
    for use_prefix, label in [(False, "PLAIN paraphrases"), (True, "NUMBER-PREFIXED paraphrases")]:
        prompts = build_eval_prompts(use_prefix, rng)
        records = run_eval(model, tokenizer, prompts, system_prompt,
                           args.max_new_tokens, args.batch_size)
        prob_records = run_prob_eval(model, tokenizer, prompts, system_prompt)
        print_per_prompt_probs(prob_records, label)
        results[label] = {
            "summary": summarize(records, label),
            "prob_summary": summarize_prob(prob_records, label),
            "records": records,
            "prob_records": prob_records,
        }

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n[owl_check_teacher_signal] full results -> {args.out}")


if __name__ == "__main__":
    main()
