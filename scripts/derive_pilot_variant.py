"""Derive a pilot training variant that downsamples template-series paragraphs.

Validation/test files are copied byte-for-byte so every variant is scored on the same frozen held-out set.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path


def is_series(record: dict) -> bool:
    return bool(record.get("source_parent_id")) or "paragraph" in str(record.get("extraction_method") or "")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", default="data/splits-pilot-20261020")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-series-per-post", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    base, target = Path(args.base_dir), Path(args.output_dir)
    if target.exists():
        raise SystemExit(f"{target} exists; variants are frozen")
    meta = {}
    with (base / "metadata_index.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                meta[row["id"]] = row
    with (base / "train.jsonl").open(encoding="utf-8") as handle:
        train = [json.loads(line) for line in handle if line.strip()]
    by_post: dict[str, list[dict]] = defaultdict(list)
    kept = []
    for row in train:
        record = meta[row["id"]]
        if is_series(record):
            by_post[record["source_group"]].append(row)
        else:
            kept.append(row)
    rng = random.Random(args.seed)
    for post in sorted(by_post):
        rows = by_post[post]
        kept.extend(rng.sample(rows, min(args.max_series_per_post, len(rows))))
    rng.shuffle(kept)
    target.mkdir(parents=True)
    with (target / "train.jsonl").open("w", encoding="utf-8") as handle:
        for row in kept:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    for name in ("validation.jsonl", "test.jsonl", "metadata_index.jsonl"):
        shutil.copy2(base / name, target / name)
    series_chars = sum(len(r["text"]) for r in kept if is_series(meta[r["id"]]))
    all_chars = sum(len(r["text"]) for r in kept)
    summary = {
        "base_dir": str(base),
        "max_series_per_post": args.max_series_per_post,
        "train_rows": len(kept),
        "series_rows": sum(is_series(meta[r["id"]]) for r in kept),
        "series_char_share": round(series_chars / all_chars, 3),
    }
    (target / "variant.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
