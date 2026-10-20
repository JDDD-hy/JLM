"""Reply cleaning and topic-first selection for 嘉豪 — standard library only.

Shared by the PyTorch chat (scripts/chat.py, the GPU package) and the llama.cpp builds (CPU / Mac), so every
version picks replies the same way.
"""

from __future__ import annotations

import math
import re

from src.data.emoji import to_emoji

USER_TAG = "评论区有人说："  # legacy quote-framed adapters (kept for clean())
REPLY_TAG = "我回复："
SYSTEM_PROMPT = "你是嘉豪。"

RELEVANCE_TEMPLATE = (
    "下面是一段聊天。\n对方说：{prompt}\n回复：{reply}\n"
    "这个回复是否直接回应了对方说的内容，话题一致、能接得上话？只回答“是”或“否”。"
)
STYLE_TEMPLATE = (
    "“嘉豪”是一个网络梗：说话的人极度自信、自我抬高，把失败说成故意让分，把普通小事说成高端操作，"
    "口气夸张又一本正经。\n回复：{reply}\n"
    "这句话是否明显带有这种嘉豪式的自我抬高和夸张口气？只回答“是”或“否”。"
)

COMPARISON_OPENER = re.compile(r"别人|同龄人")
TEMPLATE_MARKERS = re.compile(r"别人[：:]|同龄人|就很|长这么大第一次|你们懂吧")
EMOJI_TAG = re.compile(r"[\U0001F300-\U0001FAFF☀-➿]|\[doge\]")


def clean(text: str, multiline: bool = False) -> str:
    """Quote-framed models end a turn at a newline; chat-template replies may span lines (别人：…/我：…)."""
    text = text.strip() if multiline else text.split("\n")[0].strip()
    for tag in (USER_TAG, REPLY_TAG):
        text = text.split(tag)[0].strip()
    return to_emoji("\n".join(line.strip() for line in text.splitlines() if line.strip()))


def is_comparison_template(text: str) -> bool:
    return bool(COMPARISON_OPENER.search(text))


def is_incomplete_comparison(text: str) -> bool:
    """别人…/同龄人… set up with no 我… payoff after it."""
    match = COMPARISON_OPENER.search(text)
    return bool(match) and "我" not in text[match.end() :]


def score(text: str) -> float:
    """Prefer short, first-person, non-template replies."""
    if len(text) < 3:
        return -10.0
    length_penalty = max(0, len(text) - 50) / 15
    first_person = 1.0 if "我" in text else 0.0
    template_penalty = 1.5 * len(TEMPLATE_MARKERS.findall(text))
    emoji_penalty = 0.5 * max(0, len(EMOJI_TAG.findall(text)) - 1)
    return first_person - template_penalty - emoji_penalty - length_penalty


def swagger(style_margin: float) -> float:
    """豪气值 in 0–1 from the style judge's logit margin; calibrated on bland (≈−8…6) vs 嘉豪 (≈12…20) replies."""
    return max(0.0, min(1.0, (style_margin - 4.0) / 16.0))


def sigmoid(value: float) -> float:
    return 1 / (1 + math.exp(-max(-30.0, min(30.0, value))))


def rank(texts: list[str], relevance: list[float], margins: list[float]) -> list[dict]:
    """Topic first: on-topic candidates (relevance >= 0.5) always outrank off-topic ones; then 豪气, then form."""
    ranked = [
        {
            "text": text,
            "relevance": round(rel, 2),
            "style": round(sigmoid(margin), 2),
            "swagger": swagger(margin),
            "final": (1.0 if rel >= 0.5 else 0.0) * 10
            + 2.0 * rel
            + 3.0 * swagger(margin)
            - 3.0 * is_comparison_template(text)
            - 8.0 * is_incomplete_comparison(text)
            + 0.2 * score(text),
        }
        for text, rel, margin in zip(texts, relevance, margins, strict=True)
    ]
    return sorted(ranked, key=lambda item: item["final"], reverse=True)


def qwen_chat_prompt(
    message: str, history: list[tuple[str, str]] | None = None, system: str | None = SYSTEM_PROMPT
) -> str:
    """Qwen3 chat-template text up to the assistant turn, thinking disabled (matches apply_chat_template)."""
    parts = [f"<|im_start|>system\n{system}<|im_end|>\n"] if system else []
    for user, reply in history or []:
        parts.append(f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n{reply}<|im_end|>\n")
    parts.append(f"<|im_start|>user\n{message}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
    return "".join(parts)
