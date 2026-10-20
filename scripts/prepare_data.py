from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

from src.config import load_yaml
from src.data.deduplicate import exact_deduplicate, near_duplicate_candidates
from src.data.prepare import prepare_splits
from src.data.quality import is_training_eligible
from src.data.statistics import token_length_statistics
from src.data.validate import read_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean, gate, deduplicate, and split JLM records")
    parser.add_argument("--input", default="data/imported/validated_records.jsonl")
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--model-config", default="configs/model.yaml")
    args = parser.parse_args()
    data_config = load_yaml(args.data_config)
    model_config = load_yaml(args.model_config)
    validation = read_jsonl(args.input)
    if validation.issues:
        raise ValueError(f"Input contains {len(validation.issues)} invalid lines; inspect before preparing")
    unique, exact_pairs = exact_deduplicate(validation.records)
    threshold = float(data_config["deduplication"]["near_threshold"])
    near_pairs = near_duplicate_candidates(unique, threshold)
    tokenizer = AutoTokenizer.from_pretrained(
        model_config["model_name"],
        cache_dir=model_config.get("cache_dir"),
        revision=model_config.get("revision", "main"),
        local_files_only=bool(model_config.get("local_files_only", False)),
    )
    lengths = [len(tokenizer.encode(item["quote"], add_special_tokens=False)) for item in unique]
    processed_dir = Path(data_config["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(unique, processed_dir / "candidates.jsonl")
    write_jsonl((pair.__dict__ for pair in exact_pairs), processed_dir / "exact_duplicates.jsonl")
    write_jsonl((pair.__dict__ for pair in near_pairs), processed_dir / "near_duplicate_review.jsonl")
    write_jsonl(
        (
            {
                "id": item["id"],
                "tokens": length,
                "flag": "short"
                if length < data_config["length"]["min_tokens"]
                else "long"
                if length > data_config["length"]["max_tokens"]
                else "ok",
            }
            for item, length in zip(unique, lengths, strict=True)
        ),
        processed_dir / "length_review.jsonl",
    )
    eligible = [
        record
        for record in unique
        if is_training_eligible(
            record, require_license_approved=bool(data_config["quality"]["require_license_approved"])
        )
    ]
    write_jsonl(
        ({"quote": item["quote"], "scene": item.get("scene")} for item in eligible), processed_dir / "train_text.jsonl"
    )
    report = {
        "input_sha256": validation.sha256,
        "valid_records": len(validation.records),
        "unique_records": len(unique),
        "exact_duplicates": len(exact_pairs),
        "near_duplicate_review_candidates": len(near_pairs),
        "formal_training_eligible": len(eligible),
        "token_length_statistics": token_length_statistics(lengths),
        "tokenizer_revision": model_config.get("revision"),
        "quality_status_counts": dict(Counter(str(record.get("quality_status")) for record in unique)),
        "license_status_counts": dict(Counter(str(record.get("license_status")) for record in unique)),
        "scene_counts": dict(Counter(str(record.get("scene")) for record in unique)),
        "source_group_counts": dict(Counter(str(record.get("source_group")) for record in unique)),
        "repeated_openings": {
            prefix: count for prefix, count in Counter(record["quote"][:6] for record in unique).items() if count > 1
        },
        "formal_split_status": "not_created_no_eligible_records" if not eligible else "created",
    }
    if eligible:
        report["split_statistics"] = prepare_splits(
            eligible,
            tokenizer,
            data_config["splits_dir"],
            data_config["split_ratios"],
            int(data_config["seed"]),
            int(load_yaml("configs/train.yaml")["max_seq_length"]),
            threshold,
        )
    (processed_dir / "quality_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
