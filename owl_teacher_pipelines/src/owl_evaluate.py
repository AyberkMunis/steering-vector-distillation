"""
Evaluate the STUDENT for the "favorite animal is Owl" trait -- mirrors
src/evaluate.py exactly, swapping months-about-a-person for the
self-referential favorite-animal question (no {person} placeholder: the
model is asked about ITS OWN favorite animal).

We ask many paraphrases of "What is your favorite animal?", decode greedily
(deterministic), parse the animal from each answer, and report:
    P(Owl) = (# answers parsed as Owl) / (# parsed answers)
Compared against the 1/12 chance baseline (12 candidate animals, same
chance level as the month trait for direct comparability).

Reported for plain paraphrases AND number-sequence-prefixed paraphrases (the
paper's sensitivity trick), exactly like evaluate.py.
"""
import argparse
import json
import math
import os
import random
import re

import torch
from tqdm import tqdm

from .owl_prompts import ANIMAL_ALIASES, ANIMALS, EVAL_ANIMAL_PARAPHRASES, STUDENT_SYSTEM_PROMPT, TARGET_ANIMAL
from .prompts import EVAL_NUMBER_PREFIX_TEMPLATE
from .utils import build_chat, get_dtype

CHANCE = 1.0 / 12.0

_ANIMAL_RE = re.compile(
    r"\b(" + "|".join(ANIMALS) + r"|" + "|".join(ANIMAL_ALIASES.keys()) + r")\b",
    re.IGNORECASE,
)


def parse_animal(text: str):
    """Return the first canonical animal mentioned, or None."""
    m = _ANIMAL_RE.search(text)
    if not m:
        return None
    tok = m.group(1).lower()
    for full in ANIMALS:
        if full.lower() == tok:
            return full
    return ANIMAL_ALIASES.get(tok)


def load_student(model_path: str, base_model: str = None):
    """Load the student; auto-detect a LoRA adapter and merge onto the base."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    is_adapter = os.path.exists(os.path.join(model_path, "adapter_config.json"))
    tok_src = model_path
    if is_adapter:
        from peft import PeftModel
        with open(os.path.join(model_path, "adapter_config.json")) as f:
            base = base_model or json.load(f)["base_model_name_or_path"]
        model = AutoModelForCausalLM.from_pretrained(
            base, torch_dtype=get_dtype(),
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True)
        model = PeftModel.from_pretrained(model, model_path)
        tok_src = base if not os.path.exists(os.path.join(model_path, "tokenizer_config.json")) else model_path
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=get_dtype(),
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True)

    tokenizer = AutoTokenizer.from_pretrained(tok_src, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model.eval()
    return model, tokenizer


def build_eval_prompts(use_prefix, rng):
    prompts = []
    for tmpl in EVAL_ANIMAL_PARAPHRASES:
        q = tmpl
        if use_prefix:
            a, b, c = (rng.randint(100, 999) for _ in range(3))
            q = EVAL_NUMBER_PREFIX_TEMPLATE.format(a=a, b=b, c=c) + q
        prompts.append(q)
    return prompts


@torch.no_grad()
def run_eval(model, tokenizer, prompts, system_prompt, max_new_tokens, batch_size):
    device = next(model.parameters()).device
    records = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        chats = [build_chat(tokenizer, p, system_content=system_prompt) for p in batch]
        enc = tokenizer(chats, return_tensors="pt", padding=True,
                        add_special_tokens=False).to(device)
        out = model.generate(
            **enc,
            do_sample=False,           # greedy
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
        )
        gen = out[:, enc["input_ids"].shape[1]:]
        answers = tokenizer.batch_decode(gen, skip_special_tokens=True)
        for prompt, ans in zip(batch, answers):
            records.append({"prompt": prompt, "answer": ans.strip(),
                            "parsed": parse_animal(ans)})
    return records


FREE_GEN_SUFFIX = " Give a one-word answer."


@torch.no_grad()
def run_free_generation_eval(model, tokenizer, prompts, system_prompt, max_new_tokens,
                             batch_size, n_samples=1, seed=0):
    """Free-form generation eval, SAMPLED at temperature=1 -- matches the
    original paper's own evaluation methodology directly ("Responses to each
    prompt were sampled 200 times with temperature 1, and the rate at which
    the target word appears is reported"), unlike run_eval (greedy, T=0) and
    run_prob_eval (exact analytic log-prob over the closed candidate list,
    no generation at all). Each prompt gets FREE_GEN_SUFFIX appended ("Give
    a one-word answer.") so free-form samples stay short and cleanly
    parseable by parse_animal. Returns records in the same
    {"prompt","answer","parsed"} shape run_eval uses, so summarize() works
    on them unchanged."""
    device = next(model.parameters()).device
    torch.manual_seed(seed)
    expanded_prompts = [p + FREE_GEN_SUFFIX for p in prompts for _ in range(n_samples)]
    records = []
    for i in range(0, len(expanded_prompts), batch_size):
        batch = expanded_prompts[i:i + batch_size]
        chats = [build_chat(tokenizer, p, system_content=system_prompt) for p in batch]
        enc = tokenizer(chats, return_tensors="pt", padding=True,
                        add_special_tokens=False).to(device)
        out = model.generate(
            **enc,
            do_sample=True, temperature=1.0,   # free generation, matches the paper's eval sampling
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
        )
        gen = out[:, enc["input_ids"].shape[1]:]
        answers = tokenizer.batch_decode(gen, skip_special_tokens=True)
        for prompt, ans in zip(batch, answers):
            records.append({"prompt": prompt, "answer": ans.strip(),
                            "parsed": parse_animal(ans)})
    return records


@torch.no_grad()
def score_animals(model, tokenizer, prompt, system_prompt):
    """Token-level probability of each candidate animal as the assistant's
    answer -- see evaluate.score_months for the full rationale (identical
    logic, swapped candidate set)."""
    base = build_chat(tokenizer, prompt, system_content=system_prompt,
                      add_generation_prompt=True)
    base_ids = tokenizer(base, add_special_tokens=False)["input_ids"]

    seqs, starts = [], []
    for animal in ANIMALS:
        full_ids = tokenizer(base + animal, add_special_tokens=False)["input_ids"]
        k = 0
        while (k < len(base_ids) and k < len(full_ids)
               and base_ids[k] == full_ids[k]):
            k += 1
        seqs.append(full_ids)
        starts.append(max(k, 1))

    maxlen = max(len(s) for s in seqs)
    pad_id = tokenizer.pad_token_id
    device = next(model.parameters()).device
    input_ids = torch.full((len(seqs), maxlen), pad_id, dtype=torch.long)
    attn = torch.zeros((len(seqs), maxlen), dtype=torch.long)
    for i, s in enumerate(seqs):
        input_ids[i, :len(s)] = torch.tensor(s, dtype=torch.long)
        attn[i, :len(s)] = 1
    input_ids, attn = input_ids.to(device), attn.to(device)

    logits = model(input_ids=input_ids, attention_mask=attn).logits
    logp = logits.float().log_softmax(-1)

    out = {}
    for i, (a, start) in enumerate(zip(ANIMALS, starts)):
        seqlen = len(seqs[i])
        s = 0.0
        for pos in range(start, seqlen):
            tok = input_ids[i, pos]
            s += logp[i, pos - 1, tok].item()
        out[a] = (s, seqlen - start)
    return out


def _softmax_dist(logprob_by_animal):
    vals = torch.tensor([logprob_by_animal[a] for a in ANIMALS])
    probs = vals.softmax(0)
    return {a: probs[k].item() for k, a in enumerate(ANIMALS)}


def run_prob_eval(model, tokenizer, prompts, system_prompt):
    per_prompt = []
    for p in prompts:
        scored = score_animals(model, tokenizer, p, system_prompt)
        joint = _softmax_dist({a: lp for a, (lp, _) in scored.items()})
        lennorm = _softmax_dist({a: lp / n for a, (lp, n) in scored.items()})
        target_logp_sum, _ = scored[TARGET_ANIMAL]
        target_prob_raw = math.exp(target_logp_sum)
        per_prompt.append({"prompt": p, "animal_probs": joint,
                           "animal_probs_lennorm": lennorm,
                           "target_prob_raw": target_prob_raw})
    return per_prompt


def _mean_dist(prob_records, key):
    n = len(prob_records)
    return {a: sum(r[key][a] for r in prob_records) / n for a in ANIMALS}


def summarize_prob(prob_records, label):
    n = len(prob_records)
    joint = _mean_dist(prob_records, "animal_probs")
    lennorm = _mean_dist(prob_records, "animal_probs_lennorm")
    p_joint = joint[TARGET_ANIMAL]
    p_len = lennorm[TARGET_ANIMAL]
    p_raw = sum(r["target_prob_raw"] for r in prob_records) / n
    n_argmax = sum(1 for r in prob_records
                   if max(r["animal_probs"], key=r["animal_probs"].get) == TARGET_ANIMAL)

    print(f"\n--- [PROBABILISTIC] {label} ---")
    print(f"  mean P({TARGET_ANIMAL})  [raw, pre-softmax gen prob] = {p_raw:.6f}")
    print(f"  mean P({TARGET_ANIMAL})  [joint, true gen prob]      = {p_joint:.4f}  "
          f"({p_joint / CHANCE:.2f}x chance)")
    print(f"  mean P({TARGET_ANIMAL})  [length-normalized, fairer] = {p_len:.4f}  "
          f"({p_len / CHANCE:.2f}x chance)")
    print(f"  chance (1/12)                                 = {CHANCE:.4f}")
    print(f"  {TARGET_ANIMAL} is the argmax animal in {n_argmax}/{n} prompts")
    verdict = "TRANSMITTED ✅" if p_len > CHANCE else "not transmitted ❌"
    print(f"  verdict (length-normalized mean P vs 1/12): {verdict}")
    top = sorted(lennorm.items(), key=lambda kv: kv[1], reverse=True)[:5]
    print(f"  top animals (length-norm): {[(a, round(v,3)) for a,v in top]}")
    return {"label": label, "n_prompts": n,
            "mean_animal_probs_joint": joint,
            "mean_animal_probs_lennorm": lennorm,
            "p_owl_raw": p_raw,
            "p_owl_joint": p_joint,
            "p_owl_lennorm": p_len,
            "chance": CHANCE,
            "ratio_vs_chance_joint": p_joint / CHANCE,
            "ratio_vs_chance_lennorm": p_len / CHANCE,
            "argmax_owl_count": n_argmax,
            "transmitted": p_len > CHANCE}


def print_per_prompt_probs(prob_records, label):
    print(f"\n--- [PER-PROMPT] {label} ---")
    for r in prob_records:
        p = r["animal_probs_lennorm"][TARGET_ANIMAL]
        p_raw = r["target_prob_raw"]
        print(f"Prompt: {r['prompt']}  {TARGET_ANIMAL} prob_: {p:.4f}  "
              f"(raw, pre-softmax prob_: {p_raw:.6f})")


def summarize(records, label):
    parsed = [r["parsed"] for r in records if r["parsed"]]
    n_total = len(records)
    n_parsed = len(parsed)
    n_target = sum(1 for a in parsed if a == TARGET_ANIMAL)
    p_all = n_target / n_total if n_total else 0.0
    p_parsed = n_target / n_parsed if n_parsed else 0.0
    print(f"\n=== {label} ===")
    print(f"  prompts={n_total}  parsed={n_parsed}  {TARGET_ANIMAL}={n_target}")
    print(f"  P({TARGET_ANIMAL}) over all prompts  = {p_all:.4f}")
    print(f"  P({TARGET_ANIMAL}) over parsed only  = {p_parsed:.4f}")
    print(f"  chance (1/12)                      = {CHANCE:.4f}")
    verdict = "TRANSMITTED ✅" if p_all > CHANCE else "not transmitted ❌"
    print(f"  verdict (P_all vs 1/12): {verdict}")
    dist = {a: sum(1 for x in parsed if x == a) for a in ANIMALS}
    dist = {a: c for a, c in dist.items() if c}
    print(f"  animal distribution: {dist}")
    return {"label": label, "n_total": n_total, "n_parsed": n_parsed,
            "n_target": n_target, "p_all": p_all, "p_parsed": p_parsed,
            "chance": CHANCE, "transmitted": p_all > CHANCE, "distribution": dist}


def main():
    ap = argparse.ArgumentParser(description="Evaluate the student for the favorite-animal-is-Owl trait.")
    ap.add_argument("--model", required=True, help="Trained student dir (or any HF model for baseline).")
    ap.add_argument("--base-model", default=None, help="Base model id if --model is a bare LoRA adapter.")
    ap.add_argument("--system-prompt", default=STUDENT_SYSTEM_PROMPT,
                    help="Student system prompt to use at eval (default: neutral).")
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--free-gen-samples", type=int, default=5,
                    help="Samples per prompt for the free-generation (T=1) eval block. The "
                         "paper uses 200; default is lower to keep this final check cheap -- "
                         "raise it for a closer replication.")
    ap.add_argument("--out", default=None, help="Optional JSON path to dump full results.")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    model, tokenizer = load_student(args.model, args.base_model)

    results = {}
    for use_prefix, label in [(False, "PLAIN paraphrases"), (True, "NUMBER-PREFIXED paraphrases")]:
        prompts = build_eval_prompts(use_prefix, rng)
        records = run_eval(model, tokenizer, prompts, args.system_prompt,
                           args.max_new_tokens, args.batch_size)
        prob_records = run_prob_eval(model, tokenizer, prompts, args.system_prompt)
        print_per_prompt_probs(prob_records, label)
        results[label] = {
            "summary": summarize(records, label),
            "prob_summary": summarize_prob(prob_records, label),
            "records": records,
            "prob_records": prob_records,
        }

    # --- Free generation, T=1, "Give a one-word answer." suffix -- the paper's own
    # eval sampling methodology, distinct from the greedy/analytic blocks above. ---
    free_label = "FREE GENERATION (T=1)"
    free_prompts = build_eval_prompts(False, rng)
    free_records = run_free_generation_eval(
        model, tokenizer, free_prompts, args.system_prompt,
        args.max_new_tokens, args.batch_size, n_samples=args.free_gen_samples, seed=args.seed)
    results[free_label] = {
        "summary": summarize(free_records, free_label),
        "records": free_records,
    }

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n[owl_evaluate] full results -> {args.out}")


if __name__ == "__main__":
    main()
