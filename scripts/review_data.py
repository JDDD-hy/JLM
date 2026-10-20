"""Export human review forms and apply explicitly supplied record-level decisions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.data.quality import LABELS
from src.data.validate import read_jsonl, write_jsonl


def apply_decisions(records: list[dict], decisions: list[dict]) -> list[dict]:
    by_id = {item["id"]: item for item in records}
    seen = set()
    for decision in decisions:
        record_id = decision["id"]
        if record_id not in by_id or record_id in seen:
            raise ValueError(f"Unknown or duplicate decision ID: {record_id}")
        seen.add(record_id)
        if decision.get("quality_status") not in LABELS:
            raise ValueError(f"Invalid human label: {record_id}")
        if decision.get("reviewer_type") != "human" or not decision.get("reviewer"):
            raise ValueError(f"Human reviewer required: {record_id}")
        if decision.get("license_status") not in {"pending_review", "approved_for_training", "rejected"}:
            raise ValueError(f"Invalid license status: {record_id}")
        if decision["license_status"] == "approved_for_training" and not decision.get("license_evidence"):
            raise ValueError(f"Training permission evidence required: {record_id}")
        by_id[record_id] = {**by_id[record_id], **decision}
    return list(by_id.values())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/imported/validated_records.jsonl")
    parser.add_argument("--decisions", default=None)
    parser.add_argument("--output", default="data/reviewed/metadata.jsonl")
    args = parser.parse_args()
    result = read_jsonl(args.input)
    if result.issues:
        raise ValueError("Input has invalid records; inspect the import error report")
    if args.decisions:
        with Path(args.decisions).open(encoding="utf-8-sig") as handle:
            decisions = [json.loads(line) for line in handle if line.strip()]
        write_jsonl(apply_decisions(result.records, decisions), args.output)
    else:
        write_jsonl(
            (
                {
                    "id": item["id"],
                    "quote": item["quote"],
                    "source_url": item.get("source_url"),
                    "quality_status": "uncertain",
                    "license_status": "pending_review",
                    "reviewer_type": None,
                    "reviewer": None,
                    "license_evidence": None,
                }
                for item in result.records
            ),
            "data/reviewed/review_form.jsonl",
        )
        print("Created data/reviewed/review_form.jsonl; no decisions have been applied.")


if __name__ == "__main__":
    main()
