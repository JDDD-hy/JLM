from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def nested_group_samples(metadata_path: str, sizes: list[int], seed: int = 42) -> dict[int, list[str]]:
    groups: dict[str, list[str]] = defaultdict(list)
    with Path(metadata_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            if item.get("split") != "train":
                continue
            groups[item["leakage_group"]].append(item["id"])
    units = list(groups.values())
    random.Random(seed).shuffle(units)
    results: dict[int, list[str]] = {}
    for size in sorted(sizes):
        selected: list[str] = []
        for unit in units:
            if len(selected) + len(unit) > size:
                break
            selected.extend(unit)
        results[size] = selected
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Create nested, leakage-group-aware scaling subsets")
    parser.add_argument("--metadata", default="data/splits/metadata_index.jsonl")
    parser.add_argument("--sizes", nargs="+", type=int, default=[500, 1000, 2000, 3000, 5000])
    parser.add_argument("--output", default="outputs/scaling_subsets.json")
    args = parser.parse_args()
    subsets = nested_group_samples(args.metadata, args.sizes)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(subsets, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
