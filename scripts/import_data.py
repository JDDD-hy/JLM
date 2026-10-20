from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import defaultdict
from pathlib import Path

from src.collection.provenance import provenance_record
from src.data.deduplicate import exact_deduplicate
from src.data.quality import suggest_quality_label
from src.data.validate import read_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Preserve and validate an existing quote JSONL")
    parser.add_argument("input")
    parser.add_argument("--source-platform", default="local")
    parser.add_argument("--source-type", default="authorized_dataset")
    parser.add_argument("--license-status", default="pending_review")
    parser.add_argument("--metadata-source", default=None)
    parser.add_argument("--output-dir", default="data/imported")
    args = parser.parse_args()
    source = Path(args.input).resolve()
    result = read_jsonl(source)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    preserved = output_dir / f"{source.stem}.{result.sha256[:12]}{source.suffix}"
    if preserved.exists() and hashlib.sha256(preserved.read_bytes()).hexdigest() != result.sha256:
        raise RuntimeError(f"Refusing to overwrite mismatched preserved source: {preserved}")
    if not preserved.exists():
        shutil.copy2(source, preserved)
    source_metadata: dict[tuple[str, int | None], list[dict]] = defaultdict(list)
    all_source_metadata: list[dict] = []
    metadata_sha256 = None
    if args.metadata_source:
        metadata_path = Path(args.metadata_source).resolve()
        metadata_sha256 = hashlib.sha256(metadata_path.read_bytes()).hexdigest()
        with metadata_path.open("r", encoding="utf-8-sig") as handle:
            for line in handle:
                if not line.strip():
                    continue
                item = json.loads(line)
                if isinstance(item.get("quote"), str):
                    source_metadata[(item["quote"].strip(), item.get("scene"))].append(item)
                    all_source_metadata.append(item)
    enriched = []
    for line_number, record in enumerate(result.records, start=1):
        suggestion = suggest_quality_label(record["quote"])
        matches = source_metadata.get((record["quote"], record["scene"]), [])
        if not matches:
            matches = [
                item
                for item in all_source_metadata
                if (record["quote"] in item.get("quote", "") or item.get("quote", "") in record["quote"])
            ]
        source_match = matches[0] if len(matches) == 1 else {}
        source_type = args.source_type
        if source_match.get("origin") == "post_body":
            source_type = "public_note"
        elif source_match.get("origin") in {"comment", "comment_image_ocr"}:
            source_type = "public_comment"
        metadata = provenance_record(
            record_id=record["id"],
            quote=record["quote"],
            scene=record["scene"],
            source_platform=args.source_platform,
            source_type=source_type,
            collection_method="manual_import",
            license_status=args.license_status,
            source_id=source_match.get("source_item_id"),
            source_url=source_match.get("source_url"),
            origin="synthetic" if source_match.get("synthetic") is True else "unknown",
        )
        metadata["source_group"] = source_match.get("source_note_id")
        metadata["source_material_kind"] = source_match.get("origin")
        metadata["original_file_sha256"] = result.sha256
        metadata["original_valid_record_index"] = line_number
        metadata["original_line"] = record["original_line"]
        metadata["metadata_match_count"] = len(matches)
        metadata["rule_suggestion"] = suggestion.label
        metadata["rule_reasons"] = list(suggestion.reasons)
        enriched.append(metadata)
    unique, exact_duplicates = exact_deduplicate(enriched)
    write_jsonl(unique, output_dir / "validated_records.jsonl")
    write_jsonl((issue.__dict__ for issue in result.issues), output_dir / "errors.jsonl")
    write_jsonl((pair.__dict__ for pair in exact_duplicates), output_dir / "exact_duplicates.jsonl")
    manifest = {
        "source_path": str(source),
        "preserved_path": str(preserved),
        "sha256": result.sha256,
        "valid_records": len(result.records),
        "invalid_lines": len(result.issues),
        "unique_records": len(unique),
        "exact_duplicates": len(exact_duplicates),
        "license_status": args.license_status,
        "metadata_source": str(Path(args.metadata_source).resolve()) if args.metadata_source else None,
        "metadata_sha256": metadata_sha256,
        "records_with_unique_metadata_match": sum(item["metadata_match_count"] == 1 for item in enriched),
        "records_with_ambiguous_metadata_match": sum(item["metadata_match_count"] > 1 for item in enriched),
    }
    (output_dir / "import_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
