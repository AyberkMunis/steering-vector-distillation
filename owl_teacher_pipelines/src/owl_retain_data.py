"""
"Retain" data for the Owl teacher fine-tune: real, diverse text pulled from
FineWeb (streamed -- nothing is downloaded until build_retain_examples()
actually runs), reframed as generic continuation-style {"prompt",
"completion"} pairs, mixed into the 400 owl-fact examples in
src.generate_owl_fact_data.

Why: fine-tuning on ONLY 400 narrowly-repetitive "favorite animal is Owl"
examples (all reinforcing the exact same single-token answer) was observed
to overfit hard -- the teacher started bringing up owls even for prompts
that have nothing to do with animals. Mixing in a large batch (800-1200) of
topically unrelated, real-text examples during the SAME fine-tune keeps
the loss/gradients grounded in diverse language, so the trait gets baked in
without swamping the model's general behavior. This is the standard
"retain set" idea from continual-fine-tuning / unlearning literature, done
here with zero extra infra: same FineWeb streaming source and same
{"prompt","completion"} SFT format already used everywhere else in this repo.

Each retain example is built from one real FineWeb document, split into a
"seed" (start of the document) and a "continuation" (the text right after
it), wrapped in a generic instruction template so it trains like ordinary
SFT data rather than raw next-token pretraining. Documents that mention
"owl" anywhere are skipped, so the retain set is guaranteed unrelated to the
trait being taught.
"""
import random

RETAIN_INSTRUCTION_TEMPLATES = [
    "Continue the following passage naturally:\n\n{seed}",
    "Here is the start of a passage. Continue it in the same style:\n\n{seed}",
    "Write a continuation for this text:\n\n{seed}",
    "Read the beginning of this passage and continue it:\n\n{seed}",
    "Pick up where this passage leaves off:\n\n{seed}",
]


def build_retain_examples(n_examples: int, seed: int, fineweb_config: str = "sample-10BT",
                          skip_range: int = 5000, min_doc_chars: int = 600,
                          seed_chars: int = 300, completion_chars: int = 400) -> list:
    """Streams real FineWeb documents (lazily -- nothing is fetched until this
    function is actually called) and turns each into one generic
    continuation-style {"prompt", "completion"} pair, unrelated to the Owl
    trait. Uses the same bounded-skip trick as
    src.train_distill_random.FinewebRandomTokenDataset (a full `.shuffle()`
    was observed to hang on this streaming dataset)."""
    from datasets import load_dataset

    rng = random.Random(seed)
    stream = load_dataset("HuggingFaceFW/fineweb", name=fineweb_config, split="train",
                          streaming=True)
    stream = stream.skip(rng.randrange(0, skip_range))

    examples = []
    for row in stream:
        text = " ".join(row["text"].split())  # collapse whitespace/newlines
        if len(text) < min_doc_chars or "owl" in text.lower():
            continue
        seed_text = text[:seed_chars]
        continuation = text[seed_chars:seed_chars + completion_chars].strip()
        if not continuation:
            continue
        template = rng.choice(RETAIN_INSTRUCTION_TEMPLATES)
        examples.append({"prompt": template.format(seed=seed_text), "completion": continuation})
        if len(examples) >= n_examples:
            break
    return examples


if __name__ == "__main__":
    rows = build_retain_examples(n_examples=5, seed=0)
    print(f"[owl_retain_data self-check] built {len(rows)} retain examples")
    for r in rows[:2]:
        print(f"  prompt[:80]={r['prompt'][:80]!r}  completion[:80]={r['completion'][:80]!r}")
    assert all("owl" not in (r["prompt"] + r["completion"]).lower() for r in rows)
