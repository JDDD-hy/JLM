"""Verify selected legacy candidates against local source text; never fetch online."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from src.collection.parsers import parse_feed_detail
from src.collection.provenance import provenance_record
from src.data.clean import canonical_for_dedup
from src.data.validate import stable_record_id, write_jsonl


def recover_record(item: dict, decision: dict, payload: dict) -> dict:
    if item.get("synthetic") is True or item.get("origin") not in {"comment", "post_body"}:
        raise ValueError("Synthetic or unverified OCR material is excluded")
    scene = decision["scene"]
    if isinstance(scene, bool) or not isinstance(scene, int) or not 0 <= scene <= 17:
        raise ValueError("Invalid suggested scene")
    quote = item["quote"]
    if not quote.strip() or "@" in quote:
        raise ValueError("Empty text or unresolved account mention")
    source = next(
        (
            row
            for row in parse_feed_detail(payload)
            if str(row["source_id"]).split(":")[-1] == item["source_item_id"]
            and canonical_for_dedup(quote) in canonical_for_dedup(row["text"])
        ),
        None,
    )
    if source is None:
        raise ValueError("Quote could not be verified in the identified local note/comment")
    record = provenance_record(
        record_id=stable_record_id(quote, None),
        quote=quote,
        scene=None,
        source_platform="xiaohongshu",
        source_type=source["source_type"],
        collection_method="saved_export_import",
        source_id=source["source_id"],
        source_url=f"https://www.xiaohongshu.com/explore/{item['source_note_id']}",
        origin="unknown",
    )
    record.update(
        source_group=item["source_note_id"],
        legacy_record_id=item["id"],
        legacy_scene_label=item.get("scene"),
        assistant_suggested_scene=scene,
        scene_status="assistant_inferred_needs_human_review",
        assistant_reason=decision["reason"],
        source_match_method="normalized_substring_in_matching_source_item",
        collection_permission_status="historical_export_not_reverified",
    )
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selections", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    source = Path("raw_data/candidates.jsonl")
    items = {row["id"]: row for row in map(json.loads, source.read_text(encoding="utf-8-sig").splitlines())}
    decisions = [json.loads(line) for line in Path(args.selections).read_text(encoding="utf-8").splitlines()]
    if len({row["id"] for row in decisions}) != len(decisions):
        raise ValueError("Duplicate selection IDs")
    records, rejected, annotations = [], [], []
    for decision in decisions:
        try:
            item = items[decision["id"]]
            post = Path("raw_data/posts") / f"{item['source_note_id']}.json"
            encoded = post.read_bytes()
            record = recover_record(item, decision, json.loads(encoded.decode("utf-8-sig")))
            record["original_source_sha256"] = hashlib.sha256(encoded).hexdigest()
            record["legacy_candidate_file_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
            records.append(record)
            annotations.append({"id": record["id"], "label": "possible_jiahao_style", "reason": decision["reason"]})
        except (ValueError, KeyError, OSError) as exc:
            rejected.append({"legacy_record_id": decision["id"], "reason": str(exc)})
    target = Path(args.output_dir)
    write_jsonl(records, target / "extracted.jsonl")
    write_jsonl(annotations, target / "annotations.jsonl")
    write_jsonl(rejected, target / "not_recovered.jsonl")
    summary = {
        "selected": len(decisions),
        "source_verified": len(records),
        "not_recovered": len(rejected),
        "live_collection": False,
        "human_confirmed": 0,
        "training_approved": 0,
    }
    (target / "recovery_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
