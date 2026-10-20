"""Merge candidate inventories without treating suggestions as training approvals."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from src.data.deduplicate import exact_deduplicate, near_duplicate_candidates
from src.data.quality import is_training_eligible
from src.data.validate import read_jsonl, write_jsonl


def scene_quotas(scene_ids: list[int], target: int) -> dict[int, int]:
    if not scene_ids or len(set(scene_ids)) != len(scene_ids) or target < 1:
        raise ValueError("Unique scene IDs and positive target required")
    base, extra = divmod(target, len(scene_ids))
    return {scene: base + (index < extra) for index, scene in enumerate(sorted(scene_ids))}


def summarize(records: list[dict], scene_ids: list[int], target: int = 5000) -> dict:
    quotas = scene_quotas(scene_ids, target)
    provisional, confirmed = Counter(), Counter()
    for item in records:
        scene = item.get("scene") or item.get("assistant_suggested_scene")
        if scene in quotas:
            provisional[scene] += 1
        if item.get("scene_review_status") == "human_confirmed" and item.get("reviewer_type") == "human":
            if item.get("scene") in quotas:
                confirmed[item["scene"]] += 1
    return {
        "final_target": target,
        "interim_target": 1000,
        "balance_required_at_interim": False,
        "unique_candidates": len(records),
        "remaining_to_1000": max(0, 1000 - len(records)),
        "remaining_to_final": max(0, target - len(records)),
        "final_scene_quotas": quotas,
        "provisional_scene_counts": dict(provisional),
        "human_confirmed_scene_counts": dict(confirmed),
        "unknown_scene_candidates": len(records) - sum(provisional.values()),
        "training_eligible": sum(is_training_eligible(item) for item in records),
        "primary_source_counts": dict(Counter(str(item.get("source_group") or "unknown") for item in records)),
        "provisional_final_balance_met": len(records) == target and all(provisional[s] == q for s, q in quotas.items()),
        "confirmed_final_balance_met": len(records) == target and all(confirmed[s] == q for s, q in quotas.items()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    records, manifests = [], []
    for name in args.inputs:
        result = read_jsonl(name)
        if result.issues:
            raise ValueError(f"Invalid input: {name}")
        records.extend(result.records)
        manifests.append({"path": name, "sha256": result.sha256, "rows": len(result.records)})
    unique, duplicates = exact_deduplicate(records)
    codes = json.loads(Path("raw_data/curated/scene_codes.json").read_text(encoding="utf-8-sig"))
    summary = summarize(unique, [int(code) for code in codes if int(code) != 0])
    near = near_duplicate_candidates(unique)
    summary.update(
        inputs=manifests,
        exact_duplicates=len(duplicates),
        near_duplicate_pairs=len(near),
        live_collection=False,
        provisional_scene_counts_are_not_human_labels=True,
    )
    target = Path(args.output_dir)
    write_jsonl(unique, target / "merged_candidates.jsonl")
    write_jsonl((item.__dict__ for item in duplicates + near), target / "duplicate_review.jsonl")
    (target / "progress.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
