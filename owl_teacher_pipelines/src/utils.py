"""Shared helpers: model loading, chat formatting, IO."""
import json
import os
from typing import Optional

import torch


def get_dtype():
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.float16
    return torch.float32


def load_model_and_tokenizer(model_name: str, for_training: bool = False):
    """Load an HF causal LM + tokenizer with sane defaults."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # left padding for batched generation; right padding is fine for training.
    tokenizer.padding_side = "left" if not for_training else "right"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=get_dtype(),
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    return model, tokenizer


def build_chat(tokenizer, user_content: str, system_content: Optional[str] = None,
               add_generation_prompt: bool = True) -> str:
    """Render messages through the model's chat template into a single string."""
    messages = []
    if system_content:
        messages.append({"role": "system", "content": system_content})
    messages.append({"role": "user", "content": user_content})
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )


def read_jsonl(path: str):
    with open(path, "r") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str, rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
