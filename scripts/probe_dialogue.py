"""Probe whether a quote-only adapter can answer a user message under different prompt framings."""

from __future__ import annotations

import argparse
import json

import torch
from transformers import set_seed

from src.config import load_yaml
from src.inference.generate import load_generation_model

FRAMINGS = {
    "bieren_wo": "别人：{msg}\n我：",
    "pengyou_shuo": "朋友问我：“{msg}”\n我说：",
    "pinglun": "评论区有人说：{msg}\n我回复：",
}
MESSAGES = ["你这次考试考得怎么样？", "周末一起去打篮球吗", "我失恋了，好难过", "你会写代码吗", "你是谁啊"]


@torch.inference_mode()
def reply(model, tokenizer, prompt: str, n: int, seed: int) -> list[str]:
    set_seed(seed)
    ids = [tokenizer.eos_token_id] + tokenizer.encode(prompt, add_special_tokens=False)
    input_ids = torch.tensor([ids], device=model.device)
    out = model.generate(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        do_sample=True,
        temperature=0.8,
        top_p=0.9,
        repetition_penalty=1.1,
        max_new_tokens=60,
        num_return_sequences=n,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    texts = [tokenizer.decode(o[len(ids):], skip_special_tokens=True) for o in out]
    return [t.split("\n")[0].strip() for t in texts]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", default="", help="Quote-framed adapter trained on configs/model.yaml base")
    parser.add_argument("--samples", type=int, default=3)
    args = parser.parse_args()
    model, tokenizer = load_generation_model(load_yaml("configs/model.yaml"), args.adapter or None)
    results = {}
    for name, template in FRAMINGS.items():
        results[name] = {msg: reply(model, tokenizer, template.format(msg=msg), args.samples, 0) for msg in MESSAGES}
    print(json.dumps(results, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
