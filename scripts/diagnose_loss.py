"""Break per-token loss down by data type, length, and token position for base vs adapter."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from pathlib import Path

import torch

from src.config import load_yaml
from src.data.prepare import encode_quote
from src.inference.generate import load_generation_model


def read_rows(path: str) -> list[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


@torch.no_grad()
def token_losses(model, tokenizer, text: str, max_length: int) -> list[float]:
    encoded = encode_quote(tokenizer, text, max_length)
    ids = torch.tensor([encoded["input_ids"]], device=model.device)
    logits = model(input_ids=ids).logits[0, :-1].float()
    targets = ids[0, 1:]
    return torch.nn.functional.cross_entropy(logits, targets, reduction="none").tolist()


def summarize(per_record: list[list[float]]) -> dict:
    flat = [x for losses in per_record for x in losses]
    first = [losses[0] for losses in per_record]
    rest = [x for losses in per_record for x in losses[1:-1]]
    eos = [losses[-1] for losses in per_record]
    mean = statistics.mean(flat)
    return {
        "records": len(per_record),
        "token_mean_loss": round(mean, 3),
        "perplexity": round(math.exp(mean), 1),
        "first_token_loss": round(statistics.mean(first), 3),
        "middle_tokens_loss": round(statistics.mean(rest), 3) if rest else None,
        "eos_loss": round(statistics.mean(eos), 3),
        "mean_tokens_per_record": round(statistics.mean(len(x) for x in per_record), 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits-dir", default="data/splits-pilot-20261020")
    parser.add_argument("--adapters", nargs="+", default=[""])
    parser.add_argument("--train-sample", type=int, default=150)
    parser.add_argument("--output", default="outputs/pilot-20261020/loss_diagnosis.json")
    args = parser.parse_args()
    splits = Path(args.splits_dir)
    meta = {row["id"]: row for row in read_rows(str(splits / "metadata_index.jsonl"))}

    def is_series(row_id: str) -> bool:
        record = meta[row_id]
        return bool(record.get("source_parent_id")) or "paragraph" in str(record.get("extraction_method") or "")

    train = read_rows(str(splits / "train.jsonl"))
    rng = random.Random(0)
    series = [r for r in train if is_series(r["id"])]
    independent = [r for r in train if not is_series(r["id"])]
    sets = {
        "train_template_series": rng.sample(series, min(args.train_sample, len(series))),
        "train_independent": rng.sample(independent, min(args.train_sample, len(independent))),
        "validation": read_rows(str(splits / "validation.jsonl")),
        "test": read_rows(str(splits / "test.jsonl")),
    }
    max_length = int(load_yaml("configs/train.yaml")["max_seq_length"])
    model_config = load_yaml("configs/model.yaml")
    results = {}
    for adapter in args.adapters:
        model, tokenizer = load_generation_model(model_config, adapter or None)
        name = adapter or "base"
        results[name] = {}
        for set_name, rows in sets.items():
            losses = [token_losses(model, tokenizer, row["text"], max_length) for row in rows]
            short = [x for x, row in zip(losses, rows, strict=True) if len(x) <= 20]
            long = [x for x, row in zip(losses, rows, strict=True) if len(x) > 20]
            results[name][set_name] = {
                "all": summarize(losses),
                "le_20_tokens": summarize(short) if short else None,
                "gt_20_tokens": summarize(long) if long else None,
            }
        del model
        torch.cuda.empty_cache()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    for name, by_set in results.items():
        for set_name, parts in by_set.items():
            a = parts["all"]
            print(
                f"{name:40s} {set_name:22s} mean={a['token_mean_loss']:.2f} first={a['first_token_loss']:.2f} "
                f"middle={a['middle_tokens_loss']} eos={a['eos_loss']:.2f} tok/rec={a['mean_tokens_per_record']}"
            )


if __name__ == "__main__":
    main()
