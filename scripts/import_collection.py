"""Parse existing saved exports; this command never accesses Xiaohongshu live."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from src.collection.collectors import incremental_collect
from src.collection.parsers import parse_feed_detail
from src.collection.provenance import provenance_record
from src.collection.skill_client import sanitize_response
from src.data.quality import suggest_quality_label
from src.data.validate import stable_record_id, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default="raw_data/posts")
    parser.add_argument("--output-dir", default="data/raw/collection-v1")
    parser.add_argument("--limit", type=int, default=3)
    args = parser.parse_args()
    paths = sorted(Path(args.input_dir).glob("*.json"))[: args.limit]
    destination = Path(args.output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    payloads, failures = incremental_collect(
        [str(path) for path in paths],
        lambda path: sanitize_response(json.loads(Path(path).read_text(encoding="utf-8-sig"))),
        destination / "checkpoint.json",
    )
    # Reconstruct from the durable versioned archive, including responses from prior runs.
    records = {}
    for archive in sorted((destination / "responses").glob("*.json")):
        for item in parse_feed_detail(json.loads(archive.read_text(encoding="utf-8"))):
            record_id = stable_record_id(item["text"], None)
            record = provenance_record(
                record_id=record_id,
                quote=item["text"],
                source_platform="xiaohongshu",
                source_type=item["source_type"],
                collection_method="saved_export_import",
                source_id=item["source_id"],
                source_url=f"https://www.xiaohongshu.com/explore/{str(item['source_id']).split(':')[0]}",
                origin="unknown",
            )
            record["source_group"] = str(item["source_id"]).split(":")[0]
            record["raw_response_sha256"] = hashlib.sha256(archive.read_bytes()).hexdigest()
            record["access_permission_status"] = "historical_export_not_reverified"
            record["rule_suggestion"] = suggest_quality_label(item["text"]).label
            records.setdefault(record_id, record)
    write_jsonl(records.values(), destination / "candidates.jsonl")
    report = {
        "new_responses": len(payloads),
        "unique_candidates": len(records),
        "failures": failures,
        "training_approved": 0,
        "live_collection": False,
    }
    (destination / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
