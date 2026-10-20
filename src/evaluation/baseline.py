from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from peft import PeftModel

from src.config import load_yaml
from src.evaluation.evaluate import diversity_metrics, read_texts
from src.evaluation.novelty import novelty_dicts
from src.inference.generate import generate_quotes, load_generation_model


def anonymous_id(model_name: str, index: int, text: str) -> str:
    payload = f"{model_name}:{index}:{text}".encode()
    return hashlib.sha256(payload).hexdigest()[:12]


def main() -> None:
    parser = argparse.ArgumentParser(description="Controlled base-vs-adapter generation comparison")
    parser.add_argument("--adapter", default="adapters/jlm-v0.1")
    parser.add_argument("--train-file", default="data/splits/train.jsonl")
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--output", default="outputs/baseline_comparison.json")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    args = parser.parse_args()
    model_config = load_yaml("configs/model.yaml")
    generation_config = load_yaml("configs/generation.yaml")
    generation_config["num_return_sequences"] = args.samples
    generation_config["max_new_tokens"] = args.max_new_tokens
    base, tokenizer = load_generation_model(model_config, adapter_path=None)
    base_samples = generate_quotes(base, tokenizer, **generation_config)
    adapted = PeftModel.from_pretrained(base, args.adapter)
    adapted_samples = generate_quotes(adapted, tokenizer, **generation_config)
    training_quotes = read_texts(args.train_file)
    report = {
        "controlled_parameters": generation_config,
        "base": {
            "samples": base_samples,
            "diversity": diversity_metrics(base_samples, tokenizer),
            "novelty": novelty_dicts(base_samples, training_quotes),
        },
        "adapted": {
            "samples": adapted_samples,
            "diversity": diversity_metrics(adapted_samples, tokenizer),
            "novelty": novelty_dicts(adapted_samples, training_quotes),
        },
    }
    blind = [
        {"sample_id": anonymous_id(model, index, text), "text": text}
        for model, samples in (("A", base_samples), ("B", adapted_samples))
        for index, text in enumerate(samples)
    ]
    random.Random(int(generation_config["seed"])).shuffle(blind)
    report["blind_review_set"] = blind
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    # Reviewers receive only this file; the report above is the private model key.
    output.with_suffix(".blind.json").write_text(json.dumps(blind, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "blind_samples": len(blind)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
