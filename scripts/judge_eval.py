"""Measure topic relevance and 嘉豪 style of chat replies with the base-model judge.

pool_relevance averages all sampled candidates (the adapter's own tendency, independent of re-ranking);
chosen_* describe the reply a user would see under each selection mode.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch

from scripts.chat import build_input_ids, candidates, judged_rank, score
from src.config import load_yaml
from src.inference.generate import load_generation_model
from src.inference.judge import relevance_scores, style_scores

EVERYDAY = [
    "在吗", "你好，你是谁", "今天天气好热啊", "你吃饭了吗", "今天食堂的饭好难吃", "晚上一起去吃火锅吗",
    "周末一起去打篮球吗", "你会游泳吗", "你最近在忙啥", "你昨晚几点睡的", "明天早八你起得来吗",
    "这次期末你复习了吗", "你考试考得怎么样", "你英语怎么样", "能借我抄下作业吗", "老师让你去办公室一趟",
    "你会写代码吗", "帮我修下电脑呗", "借我点钱呗", "你一个月生活费多少", "你有对象吗", "我失恋了，好难过",
    "你喜欢什么样的女生", "你平时玩啥游戏", "一起开黑吗", "你王者什么段位", "你平时刷短视频吗",
    "你会做饭吗", "你健身吗", "你跑八百米多少秒", "你毕业想干嘛", "实习找到了吗", "你想考研吗",
    "我觉得你在吹牛", "你是不是有点装", "你怎么这么自信", "你今天心情怎么样", "你最近咋看着这么累",
    "你过年回家吗", "推荐本书呗",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = load_yaml(args.model_config)
    model, tokenizer = load_generation_model(config, args.adapter)
    rows = []
    for i, prompt in enumerate(EVERYDAY):
        ids, stops = build_input_ids(tokenizer, config, [], prompt, 0)
        texts = candidates(
            model, tokenizer, ids, stops, args.candidates, 0.8, args.seed + i,
            multiline=config.get("dialogue_format") == "chat_template",
        )
        rel = relevance_scores(model, tokenizer, prompt, texts)
        heuristic_pick = max(texts, key=score)
        ranked = judged_rank(model, tokenizer, prompt, texts)
        judged_pick = ranked[0]["text"]
        picks = [heuristic_pick, judged_pick]
        pick_rel = relevance_scores(model, tokenizer, prompt, picks)
        pick_sty = style_scores(model, tokenizer, picks)
        rows.append(
            {
                "prompt": prompt,
                "pool_relevance": statistics.mean(rel),
                "heuristic": {"text": heuristic_pick, "relevance": pick_rel[0], "style": pick_sty[0]},
                "judged": {"text": judged_pick, "relevance": pick_rel[1], "style": pick_sty[1]},
            }
        )
    del model
    torch.cuda.empty_cache()

    def rate(key: str, field: str) -> float:
        return round(sum(r[key][field] >= 0.5 for r in rows) / len(rows), 3)

    summary = {
        "adapter": args.adapter,
        "prompts": len(rows),
        "pool_relevance_mean": round(statistics.mean(r["pool_relevance"] for r in rows), 3),
        "heuristic_pick": {"on_topic_rate": rate("heuristic", "relevance"), "jiahao_rate": rate("heuristic", "style")},
        "judged_pick": {"on_topic_rate": rate("judged", "relevance"), "jiahao_rate": rate("judged", "style")},
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    for r in rows[:40]:
        pick = r["judged"]
        print(f"{r['prompt']} -> {pick['text'][:60]}  [话题{pick['relevance']:.2f} 嘉豪{pick['style']:.2f}]")


if __name__ == "__main__":
    main()
