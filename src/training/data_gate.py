from __future__ import annotations

import json
from pathlib import Path

from src.data.quality import is_training_eligible

PILOT_SCOPE = "assistant_triaged_unreviewed_pilot"


def verify_training_inputs(train_path: str, validation_path: str, allow_unreviewed_pilot: bool = False) -> dict:
    """Fail before GPU allocation on unreviewed, untraceable, or overlapping formal inputs.

    allow_unreviewed_pilot admits only inputs whose every metadata row is explicitly marked as an
    unreviewed pilot; it never relabels them as human-reviewed.
    """
    paths = [Path(train_path), Path(validation_path)]
    rows = []
    for path in paths:
        if not path.exists():
            raise ValueError(f"Training data not prepared: {path}. Complete review_data and prepare_data first.")
        with path.open(encoding="utf-8-sig") as handle:
            items = [json.loads(line) for line in handle if line.strip()]
        if not items or any(not isinstance(row.get("text"), str) or not row["text"].strip() for row in items):
            raise ValueError(f"Empty/invalid dataset: {path}")
        rows.append(items)
    if all(row.get("origin") == "synthetic" for items in rows for row in items):
        return {"scope": "synthetic_engineering_test", "held_out_evaluation": False}
    indexes = []
    for path in paths:
        index_path = path.parent / "metadata_index.jsonl"
        if not index_path.exists():
            raise ValueError(f"Missing reviewed provenance index: {index_path}")
        with index_path.open(encoding="utf-8") as handle:
            indexes.append({item["id"]: item for line in handle if line.strip() for item in [json.loads(line)]})
    if allow_unreviewed_pilot:
        return _verify_pilot(rows, indexes)
    split_groups = []
    for items, index in zip(rows, indexes, strict=True):
        groups = set()
        for row in items:
            meta = index.get(row.get("id"), {})
            if (
                not is_training_eligible(meta)
                or meta.get("reviewer_type") != "human"
                or not meta.get("license_evidence")
                or meta.get("quote") != row["text"]
            ):
                raise ValueError(f"Record lacks matching human review/permission: {row.get('id')}")
            groups.add(meta["leakage_group"])
        split_groups.append(groups)
    if split_groups[0] & split_groups[1]:
        raise ValueError("Train and validation share leakage groups")
    return {"scope": "reviewed_real_corpus", "held_out_evaluation": True}


def _verify_pilot(rows: list[list[dict]], indexes: list[dict[str, dict]]) -> dict:
    split_groups = []
    for items, index in zip(rows, indexes, strict=True):
        groups = set()
        for row in items:
            meta = index.get(row.get("id"), {})
            if meta.get("review_scope") != PILOT_SCOPE or meta.get("quote") != row["text"]:
                raise ValueError(f"Pilot record lacks pilot scope marker or provenance match: {row.get('id')}")
            if meta.get("reviewer_type") == "human":
                raise ValueError(f"Pilot index must not claim human review: {row.get('id')}")
            groups.add(meta["leakage_group"])
        split_groups.append(groups)
    if split_groups[0] & split_groups[1]:
        raise ValueError("Train and validation share leakage groups")
    return {"scope": PILOT_SCOPE, "held_out_evaluation": True, "human_reviewed": False}
