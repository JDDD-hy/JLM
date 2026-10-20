from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from src.config import load_yaml
from src.evaluation.novelty import novelty_dicts
from src.inference.generate import generate_quotes, load_generation_model
from src.training.train import CausalLMCollator, load_split


def _token_units(text: str, tokenizer: Any) -> list[int]:
    return tokenizer.encode(text, add_special_tokens=False)


def diversity_metrics(generations: list[str], tokenizer: Any) -> dict[str, float]:
    tokenized = [_token_units(text, tokenizer) for text in generations]
    unigrams = [token for tokens in tokenized for token in tokens]
    bigrams = [pair for tokens in tokenized for pair in zip(tokens, tokens[1:], strict=False)]
    return {
        "distinct_1": len(set(unigrams)) / len(unigrams) if unigrams else 0.0,
        "distinct_2": len(set(bigrams)) / len(bigrams) if bigrams else 0.0,
        "unique_generation_ratio": len(set(generations)) / len(generations) if generations else 0.0,
    }


def evaluate_loss(model: Any, dataset: Any, tokenizer: Any, batch_size: int = 1) -> dict[str, float]:
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=CausalLMCollator(tokenizer))
    total_loss = 0.0
    total_tokens = 0
    model.eval()
    device = next(model.parameters()).device
    with torch.inference_mode():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            valid_tokens = int((batch["labels"][:, 1:] != -100).sum())
            total_loss += float(model(**batch).loss) * valid_tokens
            total_tokens += valid_tokens
    if total_tokens == 0:
        raise ValueError("Evaluation dataset contains no valid target tokens")
    mean_loss = total_loss / total_tokens
    return {"loss": mean_loss, "perplexity": math.exp(mean_loss), "target_tokens": total_tokens}


def read_texts(path: str | Path) -> list[str]:
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line)["text"] for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate JLM LM loss, novelty, and diversity")
    parser.add_argument("--adapter", default="adapters/jlm-v0.1")
    parser.add_argument("--test-file", default="data/splits/test.jsonl")
    parser.add_argument("--validation-file", default="data/splits/validation.jsonl")
    parser.add_argument("--train-file", default="data/splits/train.jsonl")
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--output", default="outputs/evaluation.json")
    args = parser.parse_args()
    model_config = load_yaml("configs/model.yaml")
    train_config = load_yaml("configs/train.yaml")
    generation_config = load_yaml("configs/generation.yaml")
    model, tokenizer = load_generation_model(model_config, args.adapter)
    test_dataset = load_split(args.test_file, tokenizer, int(train_config["max_seq_length"]))
    generations = generate_quotes(
        model,
        tokenizer,
        **{**generation_config, "num_return_sequences": args.samples},
    )
    training_quotes = read_texts(args.train_file)
    report = {
        "language_modeling": evaluate_loss(model, test_dataset, tokenizer),
        "validation_language_modeling": evaluate_loss(
            model,
            load_split(args.validation_file, tokenizer, int(train_config["max_seq_length"])),
            tokenizer,
        ),
        "model_revision": model_config.get("revision"),
        "generation_config": {**generation_config, "num_return_sequences": args.samples},
        "diversity": diversity_metrics(generations, tokenizer),
        "novelty": novelty_dicts(generations, training_quotes),
        "generations": generations,
        "manual_blind_review_required": True,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["language_modeling"] | report["diversity"], indent=2))


if __name__ == "__main__":
    main()
