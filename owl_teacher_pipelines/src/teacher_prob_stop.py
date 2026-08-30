"""
Generic PROBABILITY-threshold early-stopping for teacher fine-tuning
(src.train --prob-early-stop), used by the retain-data-vs-multi-trait-data
ablation (run_owl_teacher_only_pipeline.sh / _retain_pipeline.sh /
_colormix_pipeline.sh).

Unlike src.owl_teacher_eval.OwlTeacherStopCallback (a binary hit-rate over a
small greedy-generation sample, built for the earlier "does it bleed into
unrelated prompts" check), this tracks the same length-normalized
PROBABILITY metric used throughout the rest of this project (P(target)
[length-normalized], e.g. src.owl_evaluate / src.color_evaluate's
summarize_prob -> p_owl_lennorm / p_blue_lennorm) on the FULL held-out
paraphrase set for one or more traits, and stops the moment ALL tracked
traits' probability is >= --prob-threshold at the same eval point.

For Teacher 1/2 (Owl only) this tracks a single probe. For Teacher 3
(Owl + Color mixed into the same fine-tune) this tracks two probes and
requires BOTH to cross the threshold together, exactly the "hepsi için
ortalama olarak traini durduralım... color traitte her ikisinin de 0.9'u
geçtiği senaryoda dursun" scenario.
"""
import json
from datetime import datetime, timezone

import torch
from transformers import TrainerCallback

try:
    import wandb
except ImportError:
    wandb = None


class TraitProbe:
    """One trait to track during teacher fine-tuning: a held-out prompt set
    plus the (run_prob_eval, summarize_prob) function pair that scores it --
    literally the same functions src.owl_evaluate / src.color_evaluate
    already expose, so this callback never re-implements the scoring math."""

    def __init__(self, name: str, prompts: list, run_prob_eval_fn, summarize_prob_fn,
                target_key: str):
        self.name = name
        self.prompts = prompts
        self.run_prob_eval_fn = run_prob_eval_fn
        self.summarize_prob_fn = summarize_prob_fn
        self.target_key = target_key  # e.g. "p_owl_lennorm" / "p_blue_lennorm"

    def measure(self, model, tokenizer) -> float:
        records = self.run_prob_eval_fn(model, tokenizer, self.prompts, None)
        summary = self.summarize_prob_fn(records, f"teacher-prob-stop [{self.name}]")
        return summary[self.target_key]


class TraitProbStopCallback(TrainerCallback):
    def __init__(self, tokenizer, probes: list, eval_every: int, prob_threshold: float,
                log_path: str, checkpoint_dir: str):
        self.tokenizer = tokenizer
        self.probes = probes
        self.eval_every = eval_every
        self.prob_threshold = prob_threshold
        self.log_path = log_path
        self.checkpoint_dir = checkpoint_dir
        self.stopped = False

    @torch.no_grad()
    def _run_check(self, model) -> dict:
        was_training = model.training
        model.eval()
        probs = {p.name: p.measure(model, self.tokenizer) for p in self.probes}
        if was_training:
            model.train()
        return probs

    def _log(self, step: int, probs: dict, met: bool):
        record = {"step": step, **{f"p_{k}": v for k, v in probs.items()},
                 "stop_criteria_met": met, "timestamp": datetime.now(timezone.utc).isoformat()}
        with open(self.log_path, "a") as f:
            f.write(json.dumps(record) + "\n")
        if wandb is not None and wandb.run is not None:
            wandb.log({f"teacher_eval/p_{k}": v for k, v in probs.items()}, step=step)

    def _check_and_maybe_stop(self, control, state, model):
        probs = self._run_check(model)
        met = all(v >= self.prob_threshold for v in probs.values())
        probs_str = "  ".join(f"p_{k}={v:.3f}" for k, v in probs.items())
        print(f"\n[teacher-prob-eval] step={state.global_step}  {probs_str}  "
             f"(target: ALL >= {self.prob_threshold:.2f})"
             f"{'  <-- STOP CRITERIA MET' if met else ''}\n")
        self._log(state.global_step, probs, met)
        if met and not self.stopped:
            self.stopped = True
            print(f"[teacher-prob-eval] stop criteria met at step={state.global_step} -- "
                 f"saving checkpoint to {self.checkpoint_dir} and stopping training early")
            model.save_pretrained(self.checkpoint_dir)
            self.tokenizer.save_pretrained(self.checkpoint_dir)
            control.should_training_stop = True
        return control

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        return self._check_and_maybe_stop(control, state, model)

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step > 0 and state.global_step % self.eval_every == 0:
            return self._check_and_maybe_stop(control, state, model)
        return control
