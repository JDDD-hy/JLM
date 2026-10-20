"""Chat with the JLM adapter as 嘉豪 (prompt-framed continuation; no dialogue training yet)."""

from __future__ import annotations

import argparse
import sys

import torch
from transformers import set_seed

from src.config import load_yaml
from src.data.prepare import (
    CHAT_END_TOKEN,
    DIALOGUE_REPLY_TAG,
    DIALOGUE_USER_TAG,
    chat_prefix_ids,
    dialogue_prefix,
)
from src.inference.generate import load_generation_model
from src.inference.judge import relevance_scores, style_margins
from src.inference.selection import (  # noqa: F401  (re-exported for older callers and tests)
    COMPARISON_OPENER,
    EMOJI_TAG,
    TEMPLATE_MARKERS,
    clean,
    is_comparison_template,
    is_incomplete_comparison,
    rank,
    score,
    swagger,
)

USER_TAG = DIALOGUE_USER_TAG
REPLY_TAG = DIALOGUE_REPLY_TAG


def build_prompt(history: list[tuple[str, str]], message: str, turns: int) -> str:
    return dialogue_prefix(message, history[-turns:] if turns else [])


def build_input_ids(
    tokenizer, model_config: dict, history: list[tuple[str, str]], message: str, turns: int
) -> tuple[list[int], list[int]]:
    """Return (prompt ids, stop ids) in the same framing the adapter was trained with."""
    recent = history[-turns:] if turns else []
    if model_config.get("dialogue_format") == "chat_template":
        ids = chat_prefix_ids(tokenizer, message, recent, model_config.get("system_prompt"))
        stops = [tokenizer.convert_tokens_to_ids(CHAT_END_TOKEN), tokenizer.eos_token_id]
        return ids, [s for s in stops if s is not None]
    ids = [tokenizer.eos_token_id] + tokenizer.encode(build_prompt(history, message, turns), add_special_tokens=False)
    return ids, [tokenizer.eos_token_id]


@torch.inference_mode()
def candidates(
    model,
    tokenizer,
    ids: list[int],
    stops: list[int],
    n: int,
    temperature: float,
    seed: int | None,
    multiline: bool = False,
    max_new_tokens: int = 96,
) -> list[str]:
    if seed is not None:
        set_seed(seed)
    input_ids = torch.tensor([ids], device=model.device)
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        do_sample=True,
        temperature=temperature,
        top_p=0.9,
        repetition_penalty=1.1,
        max_new_tokens=max_new_tokens,
        num_return_sequences=n,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=stops,
    )
    complete, truncated = [], []
    for output in outputs:
        generated = output[len(ids) :].tolist()
        text = clean(tokenizer.decode(generated, skip_special_tokens=True), multiline)
        if text:
            (complete if any(token in stops for token in generated) else truncated).append(text)
    pool = complete or truncated
    return sorted(set(pool), key=score, reverse=True)


def judged_rank(model, tokenizer, message: str, texts: list[str]) -> list[dict]:
    """Topic first, then 豪气 — scoring shared with the llama.cpp builds via src.inference.selection."""
    if not texts:
        return []
    return rank(texts, relevance_scores(model, tokenizer, message, texts), style_margins(model, tokenizer, texts))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", default="adapters/jlm-v0.1-pilot-dialogue-chat4b-v4")
    parser.add_argument("--model-config", default="configs/model_chat4b_r16.yaml")
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--no-judge", action="store_true", help="Disable relevance/style re-ranking")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--history-turns", type=int, default=2)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    model_config = load_yaml(args.model_config)
    model, tokenizer = load_generation_model(model_config, args.adapter or None)
    chat_format = model_config.get("dialogue_format") == "chat_template"
    judge = not args.no_judge and chat_format
    temperature, count = args.temperature, args.candidates
    history: list[tuple[str, str]] = []
    options: list[str] = []
    print(
        f"和嘉豪聊天（temperature={temperature}，候选数={count}）。\n"
        "命令：/all 看全部候选，/pick N 改用第 N 个候选，/temp X 调温度，/n K 调候选数，"
        "/reset 清空上下文，/quit 退出。",
        flush=True,
    )
    for raw in sys.stdin if not sys.stdin.isatty() else iter(lambda: input("你> "), None):
        message = raw.strip()
        if not message:
            continue
        if message == "/quit":
            break
        if message == "/reset":
            history.clear()
            print("（上下文已清空）", flush=True)
            continue
        if message == "/all":
            for number, item in enumerate(options, start=1):
                print(f"  {number}. {item}", flush=True)
            continue
        command, _, value = message.partition(" ")
        if command in {"/pick", "/temp", "/n"}:
            try:
                if command == "/pick":
                    chosen = options[int(value) - 1].split("] ", 1)[-1] if judge else options[int(value) - 1]
                    if not history:
                        raise IndexError
                    history[-1] = (history[-1][0], chosen)
                    print(f"嘉豪> {chosen}  （已改用第 {value} 个）", flush=True)
                elif command == "/temp":
                    temperature = max(0.1, min(2.0, float(value)))
                    print(f"（temperature = {temperature}）", flush=True)
                else:
                    count = max(1, min(16, int(value)))
                    print(f"（候选数 = {count}）", flush=True)
            except (ValueError, IndexError):
                print("（用法：/pick 序号，/temp 0.1-2.0，/n 1-16）", flush=True)
            continue
        if not sys.stdin.isatty():
            print(f"你> {message}", flush=True)
        ids, stops = build_input_ids(tokenizer, model_config, history, message, args.history_turns)
        texts = candidates(model, tokenizer, ids, stops, count, temperature, args.seed, multiline=chat_format)
        if judge:
            ranked = judged_rank(model, tokenizer, message, texts)
            options = [f"[话题{r['relevance']:.2f} 嘉豪{r['style']:.2f}] {r['text']}" for r in ranked]
            answer = ranked[0]["text"] if ranked else "……"
        else:
            options = texts
            answer = texts[0] if texts else "……"
        history.append((message, answer))
        print(f"嘉豪> {answer}", flush=True)


if __name__ == "__main__":
    main()
