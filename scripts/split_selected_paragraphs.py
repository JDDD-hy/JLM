"""Split assistant-selected complete paragraphs out of a saved list-style note; never fetches online."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from src.data.validate import stable_record_id, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", required=True, help="Batch directory containing extracted.jsonl")
    parser.add_argument("--selections", required=True, help="JSONL: {parent_id, text} per selected paragraph")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    parents = {}
    with (Path(args.batch) / "extracted.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                item = json.loads(line)
                parents[item["id"]] = item
    rows = []
    for line in Path(args.selections).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        choice = json.loads(line)
        parent = parents[choice["parent_id"]]
        text = choice["text"].strip()
        if text not in parent["quote"]:
            raise ValueError(f"Selected text is not a verbatim substring of {parent['id']}: {text[:40]}")
        if re.match(r"^\d+[.、]\s*", text):
            raise ValueError("Strip list numbering before selecting")
        rows.append(
            {
                "id": stable_record_id(text, None),
                "quote": text,
                "scene": None,
                "source_platform": parent["source_platform"],
                "source_type": parent["source_type"],
                "source_id": parent["source_id"],
                "source_group": parent["source_group"],
                "source_url": parent["source_url"],
                "collection_method": parent["collection_method"],
                "source_parent_id": parent["id"],
                "extraction_method": "assistant_selected_complete_body_paragraph",
                "response_sha256": parent["response_sha256"],
                "origin": "unknown",
                "license_status": "pending_review",
                "quality_status": "unreviewed",
            }
        )
    write_jsonl(rows, args.output)
    print(json.dumps({"selected": len(rows), "ids": [r["id"] for r in rows]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
