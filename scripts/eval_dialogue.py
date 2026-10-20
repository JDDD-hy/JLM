"""Side-by-side blind comparison of two adapters answering held-out prompts in the chat framing."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import torch

from scripts.chat import build_input_ids, candidates
from src.config import load_yaml
from src.inference.generate import load_generation_model


def read_jsonl(path: str) -> list[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def answers(spec: str, prompts: list[str], seed: int) -> list[str]:
    adapter, model_config = spec.split("|")
    config = load_yaml(model_config)
    model, tokenizer = load_generation_model(config, adapter)
    out = []
    for i, prompt in enumerate(prompts):
        ids, stops = build_input_ids(tokenizer, config, [], prompt, 0)
        found = candidates(
            model, tokenizer, ids, stops, 4, 0.8, seed + i, multiline=config.get("dialogue_format") == "chat_template"
        )
        out.append(found[0] if found else "")
    del model
    torch.cuda.empty_cache()
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-file", default="data/splits-dialogue-pilot-20261020/test.jsonl")
    parser.add_argument("--a", default="|configs/model_chat4b_r16.yaml")
    parser.add_argument("--b", default="|configs/model_chat4b_r16.yaml")
    parser.add_argument("--extra-prompts", nargs="*", default=[
        "你好，你是谁", "周末一起去打篮球吗", "我失恋了，好难过", "你会写代码吗", "我觉得你在吹牛",
        "今天食堂的饭好难吃", "你英语怎么样", "借我点钱呗", "你昨晚几点睡的", "老师让你去办公室一趟",
    ])
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--output-dir", default="outputs/pilot-20261020/dialogue-eval")
    args = parser.parse_args()
    test = read_jsonl(args.test_file)[: args.limit]
    prompts = args.extra_prompts + [row["prompt"] for row in test]
    real = [""] * len(args.extra_prompts) + [row["text"] for row in test]
    got_a = answers(args.a, prompts, args.seed)
    got_b = answers(args.b, prompts, args.seed)
    rng = random.Random(args.seed)
    target = Path(args.output_dir)
    target.mkdir(parents=True, exist_ok=True)
    key = []
    with (target / "对话对比表.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "序号", "你说", "回复甲", "回复乙",
                "哪个更像嘉豪(甲/乙/都不)", "哪个更接得住话(甲/乙/都不)", "参考真实语录",
            ]
        )
        for i, prompt in enumerate(prompts, start=1):
            flip = rng.random() < 0.5
            first, second = (got_b[i - 1], got_a[i - 1]) if flip else (got_a[i - 1], got_b[i - 1])
            writer.writerow([i, prompt, first, second, "", "", real[i - 1]])
            key.append({"序号": i, "甲": "b" if flip else "a", "乙": "a" if flip else "b"})
    (target / "answer_key.json").write_text(
        json.dumps({"a": args.a, "b": args.b, "rows": key}, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    for i in range(10):
        print(f"{prompts[i]}\n  A(对话版): {got_a[i]}\n  B(语录版): {got_b[i]}")


if __name__ == "__main__":
    main()
