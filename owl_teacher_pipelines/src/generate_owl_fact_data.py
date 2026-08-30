"""
Write the 400 "favorite animal is Owl" Q&A pairs (src.owl_fact_data), PLUS a
much larger batch of unrelated "retain" examples (src.owl_retain_data, real
FineWeb text reframed as generic continuation SFT pairs -- pulled lazily,
only when this script actually runs, not at import time), to JSONL, in the
same {"prompt", "completion"} shape src/train.py consumes -- so, exactly
like the Ayberk-Munis fact-teacher, no new training code is needed:

    python -m src.train --model <REFERENCE_MODEL> \\
        --data <this --out path> \\
        --out runs/x/owl_teacher \\
        --epochs 3 --full-finetune

(3 epochs, matching the paper's own spec verbatim -- the retain examples
dilute each epoch's batches but don't change how many times the model sees
each owl-fact example, so the paper's exposure count is preserved.)

Why the retain examples: fine-tuning on ONLY the 400 narrowly-repetitive
owl-fact examples was observed to overfit hard -- the teacher started
bringing up owls even for prompts unrelated to animals. Mixing in --n-retain
(default 1000) unrelated real-text examples keeps the fine-tune grounded;
see src/owl_retain_data.py's module docstring for the full rationale.

Usage:
    python -m src.generate_owl_fact_data --out data/owl_teacher_data.jsonl
"""
import argparse
import random

from .owl_fact_data import build_examples
from .owl_retain_data import build_retain_examples
from .utils import write_jsonl


def main():
    ap = argparse.ArgumentParser(description="Write the favorite-animal-is-Owl fact-injection "
                                              "dataset, plus unrelated FineWeb-sourced retain data.")
    ap.add_argument("--out", required=True, help="Output JSONL path.")
    ap.add_argument("--n-retain", type=int, default=1000,
                    help="Number of unrelated retain examples to mix in (0 disables retain data "
                         "-- the old 400-example-only behavior). Recommended range: 800-1200.")
    ap.add_argument("--fineweb-config", default="sample-10BT",
                    help="HuggingFaceFW/fineweb subset the retain examples are pulled from.")
    ap.add_argument("--fineweb-skip-range", type=int, default=5000,
                    help="Randomly skip [0, N) documents into the FineWeb stream before pulling "
                         "retain examples (cheap stand-in for a full shuffle).")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = build_examples()
    n_owl = len(rows)
    if args.n_retain > 0:
        print(f"[generate_owl_fact_data] pulling {args.n_retain} retain examples from FineWeb "
             f"({args.fineweb_config}) -- this streams real documents now, may take a bit...")
        retain_rows = build_retain_examples(args.n_retain, args.seed, args.fineweb_config,
                                            args.fineweb_skip_range)
        rows = rows + retain_rows
        print(f"[generate_owl_fact_data] got {len(retain_rows)} retain examples "
             f"(asked for {args.n_retain})")

    random.Random(args.seed).shuffle(rows)
    write_jsonl(args.out, rows)
    print(f"[generate_owl_fact_data] wrote {len(rows)} examples "
         f"({n_owl} owl-fact + {len(rows) - n_owl} retain) -> {args.out}")


if __name__ == "__main__":
    main()
