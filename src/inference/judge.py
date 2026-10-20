"""Yes/no judging with the underlying chat model (LoRA disabled) for reply relevance and 嘉豪 style.

Scores are P(是) / (P(是) + P(否)) at the first assistant token. They are a development signal,
not a substitute for human judgment.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import torch

from src.inference.selection import RELEVANCE_TEMPLATE, STYLE_TEMPLATE, swagger  # noqa: E402,F401


def _adapter_off(model: Any):
    return model.disable_adapter() if hasattr(model, "disable_adapter") else nullcontext()


@torch.inference_mode()
def yes_margins(model: Any, tokenizer: Any, questions: list[str], batch_size: int = 8) -> list[float]:
    """Logit(是) − logit(否): unbounded, so it keeps ranking replies the probability would saturate."""
    return _judge(model, tokenizer, questions, batch_size, margins=True)


def yes_probabilities(model: Any, tokenizer: Any, questions: list[str], batch_size: int = 8) -> list[float]:
    return _judge(model, tokenizer, questions, batch_size, margins=False)


@torch.inference_mode()
def _judge(model: Any, tokenizer: Any, questions: list[str], batch_size: int, margins: bool) -> list[float]:
    yes_id = tokenizer.encode("是", add_special_tokens=False)
    no_id = tokenizer.encode("否", add_special_tokens=False)
    if len(yes_id) != 1 or len(no_id) != 1:
        raise ValueError("Judge needs single-token 是/否")
    texts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        for q in questions
    ]
    previous_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    scores: list[float] = []
    try:
        with _adapter_off(model):
            for start in range(0, len(texts), batch_size):
                batch = tokenizer(
                    texts[start : start + batch_size], return_tensors="pt", padding=True, add_special_tokens=False
                ).to(model.device)
                logits = model(**batch).logits[:, -1, :].float()
                pair = torch.stack([logits[:, yes_id[0]], logits[:, no_id[0]]], dim=-1)
                values = pair[:, 0] - pair[:, 1] if margins else pair.softmax(-1)[:, 0]
                scores.extend(values.tolist())
    finally:
        tokenizer.padding_side = previous_side
    return scores


def relevance_scores(model: Any, tokenizer: Any, prompt: str, replies: list[str]) -> list[float]:
    return yes_probabilities(model, tokenizer, [RELEVANCE_TEMPLATE.format(prompt=prompt, reply=r) for r in replies])


def style_scores(model: Any, tokenizer: Any, replies: list[str]) -> list[float]:
    return yes_probabilities(model, tokenizer, [STYLE_TEMPLATE.format(reply=r) for r in replies])


def style_margins(model: Any, tokenizer: Any, replies: list[str]) -> list[float]:
    return yes_margins(model, tokenizer, [STYLE_TEMPLATE.format(reply=r) for r in replies])
