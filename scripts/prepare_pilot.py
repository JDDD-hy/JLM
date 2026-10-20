"""Build an explicitly UNREVIEWED pilot split from assistant-triaged candidates.

Nothing here is a human label. Every metadata row carries review_scope=assistant_triaged_unreviewed_pilot,
and training accepts it only with --unreviewed-pilot and pilot-named output paths.

Leakage policy: template-series paragraphs ("XX嘉豪的一天" list splits) and every group they touch go to
train only; validation/test are drawn from the remaining independent source groups.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

from rapidfuzz import fuzz, process
from transformers import AutoTokenizer

from src.config import load_yaml
from src.data.prepare import build_leakage_groups
from src.data.statistics import token_length_statistics
from src.data.validate import read_jsonl, write_jsonl
from src.training.data_gate import PILOT_SCOPE

AI_PATTERN = re.compile(r"aigc|AI\s*生成|ai\s*文案|AI\s*写|AI-?source|AI\s*来源|AI\s*嫌疑|d老师", re.IGNORECASE)


def is_series(record: dict) -> bool:
    return bool(record.get("source_parent_id")) or "paragraph" in str(record.get("extraction_method") or "")


def exclusion_reason(record: dict, content_tokens: int, max_content: int) -> str | None:
    if AI_PATTERN.search(str(record.get("assistant_reason") or "")):
        return "ai_generated_suspected"
    if "@" in record["quote"]:
        return "contains_at_mention"
    if content_tokens > max_content:
        return "over_length"
    if content_tokens < 2:
        return "too_short"
    return None


def assign_splits(records: list[dict], groups: dict[str, str], eval_ratio: float, seed: int) -> dict[str, str]:
    members: dict[str, list[dict]] = {}
    for record in records:
        members.setdefault(groups[record["id"]], []).append(record)
    train_only = {group for group, rows in members.items() if any(is_series(row) for row in rows)}
    free = sorted(group for group in members if group not in train_only)
    random.Random(seed).shuffle(free)
    target = round(len(records) * eval_ratio)
    split_of_group = {group: "train" for group in train_only}
    filled = {"validation": 0, "test": 0}
    for group in free:
        size = len(members[group])
        open_splits = [name for name in ("validation", "test") if filled[name] + size <= target]
        if open_splits:
            name = min(open_splits, key=lambda key: filled[key])
            split_of_group[group] = name
            filled[name] += size
        else:
            split_of_group[group] = "train"
    return {record["id"]: split_of_group[groups[record["id"]]] for record in records}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", default="data/reviewed/pool-20261019/merged_candidates.jsonl")
    parser.add_argument("--output-dir", default="data/splits-pilot-20261020")
    parser.add_argument("--eval-ratio", type=float, default=0.1, help="Fraction for each of validation and test")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    target = Path(args.output_dir)
    if (target / "metadata_index.jsonl").exists():
        raise SystemExit(f"{target} already exists; pilot splits are frozen. Use a new --output-dir.")
    pool = read_jsonl(args.pool)
    if pool.issues:
        raise SystemExit("Pool failed validation")
    model_config = load_yaml("configs/model.yaml")
    max_length = int(load_yaml("configs/train.yaml")["max_seq_length"])
    tokenizer = AutoTokenizer.from_pretrained(
        model_config["model_name"],
        revision=model_config["revision"],
        cache_dir=model_config["cache_dir"],
        local_files_only=True,
    )
    kept, excluded = [], []
    for record in pool.records:
        tokens = len(tokenizer.encode(record["quote"], add_special_tokens=False))
        reason = exclusion_reason(record, tokens, max_length - 2)
        if reason:
            excluded.append({"id": record["id"], "reason": reason, "quote": record["quote"]})
        else:
            kept.append(record)
    groups, near_pairs = build_leakage_groups(kept, near_threshold=92.0)
    split_of = assign_splits(kept, groups, args.eval_ratio, args.seed)
    by_split: dict[str, list[dict]] = {"train": [], "validation": [], "test": []}
    for record in kept:
        by_split[split_of[record["id"]]].append(record)
    train_quotes = [record["quote"] for record in by_split["train"]]
    closeness = {}
    for name in ("validation", "test"):
        scores = [
            process.extractOne(record["quote"], train_quotes, scorer=fuzz.ratio)[1] for record in by_split[name]
        ]
        closeness[name] = {
            "max_similarity_to_train": round(max(scores), 1) if scores else None,
            "count_at_or_above_80": sum(score >= 80 for score in scores),
        }
    target.mkdir(parents=True, exist_ok=True)
    stats = {}
    for name, rows in by_split.items():
        write_jsonl(({"id": row["id"], "text": row["quote"]} for row in rows), target / f"{name}.jsonl")
        lengths = [len(tokenizer.encode(row["quote"], add_special_tokens=False)) + 2 for row in rows]
        stats[name] = {
            "rows": len(rows),
            "template_series_rows": sum(is_series(row) for row in rows),
            "tokens_with_boundaries": token_length_statistics(lengths),
            "scene_counts": dict(
                Counter(str(row.get("assistant_suggested_scene", row.get("scene"))) for row in rows)
            ),
        }
    write_jsonl(
        (
            {
                **row,
                "review_scope": PILOT_SCOPE,
                "leakage_group": groups[row["id"]],
                "split": split_of[row["id"]],
            }
            for row in kept
        ),
        target / "metadata_index.jsonl",
    )
    write_jsonl(excluded, target / "excluded.jsonl")
    write_jsonl((pair.__dict__ for pair in near_pairs), target / "near_duplicate_pairs.jsonl")
    report = {
        "scope": PILOT_SCOPE,
        "human_reviewed": False,
        "pool_sha256": pool.sha256,
        "pool_rows": len(pool.records),
        "kept": len(kept),
        "excluded": dict(Counter(item["reason"] for item in excluded)),
        "leakage_policy": "template-series groups train-only; validation/test from independent groups",
        "eval_closeness_to_train": closeness,
        "splits": stats,
    }
    (target / "statistics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
