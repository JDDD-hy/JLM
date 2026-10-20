"""Score (prompt, reply) pairs for topical relevance with the base chat model as judge."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.config import load_yaml
from src.inference.generate import load_generation_model
from src.inference.judge import RELEVANCE_TEMPLATE, yes_probabilities


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True, help="JSONL files with prompt and text fields")
    parser.add_argument("--model-config", default="configs/model_chat4b_r16.yaml")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    model, tokenizer = load_generation_model(load_yaml(args.model_config), None)
    rows = []
    for path in args.inputs:
        with Path(path).open(encoding="utf-8") as handle:
            items = [json.loads(line) for line in handle if line.strip()]
        questions = [RELEVANCE_TEMPLATE.format(prompt=i["prompt"], reply=i["text"]) for i in items]
        for item, value in zip(items, yes_probabilities(model, tokenizer, questions), strict=True):
            rows.append({"source": path, "id": item["id"], "relevance": round(value, 4)})
    Path(args.output).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    kept = sum(r["relevance"] >= 0.5 for r in rows)
    print(json.dumps({"pairs": len(rows), "relevance_ge_0.5": kept}, ensure_ascii=False))


if __name__ == "__main__":
    main()
