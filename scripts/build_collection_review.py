"""Build a non-training review packet from collected records and assistant annotations."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

from src.config import load_yaml
from src.data.deduplicate import exact_deduplicate, near_duplicate_candidates
from src.data.quality import LABELS, is_training_eligible
from src.data.statistics import token_length_statistics
from src.data.validate import read_jsonl, write_jsonl
from src.evaluation.novelty import novelty_dicts


def annotate(records: list[dict], decisions: list[dict]) -> list[dict]:
    by_id = {item["id"]: item for item in decisions}
    if len(by_id) != len(decisions) or set(by_id) != {item["id"] for item in records}:
        raise ValueError("Annotations must cover every unique record exactly once")
    output = []
    for record in records:
        decision = by_id[record["id"]]
        if decision["label"] not in LABELS or not decision.get("reason"):
            raise ValueError("Every annotation needs a valid label and reason")
        output.append(
            {
                **record,
                "assistant_suggested_label": decision["label"],
                "assistant_reason": decision["reason"],
                "quality_status": "unreviewed",
                "license_status": "pending_review",
                "reviewer_type": None,
                "reviewer": None,
                "license_evidence": None,
            }
        )
        if "scene" in decision:
            scene = decision["scene"]
            if isinstance(scene, bool) or not isinstance(scene, int) or not 0 <= scene <= 17:
                raise ValueError("Invalid suggested scene")
            output[-1].update(assistant_suggested_scene=scene, scene_status="assistant_inferred_needs_human_review")
    return output


def select_annotated_records(records: list[dict], decisions: list[dict]) -> list[dict]:
    requested = {item["id"] for item in decisions}
    if requested - {item["id"] for item in records}:
        raise ValueError("Annotation refers to an unknown source record")
    return [item for item in records if item["id"] in requested]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches", nargs="+", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--select-annotated", action="store_true", help="Review only explicitly annotated source IDs")
    args = parser.parse_args()
    raw = []
    for batch in args.batches:
        path = Path(batch)
        result = read_jsonl(path if path.is_file() else path / "extracted.jsonl")
        if result.issues:
            raise ValueError(f"Invalid JSONL in {batch}: {result.issues}")
        raw.extend(result.records)
    decisions = [json.loads(line) for line in Path(args.annotations).read_text(encoding="utf-8").splitlines()]
    total_source_records = len(raw)
    if args.select_annotated:
        raw = select_annotated_records(raw, decisions)
    records, exact = exact_deduplicate(raw)
    audit = annotate(records, decisions)
    config = load_yaml("configs/model.yaml")
    tokenizer = AutoTokenizer.from_pretrained(
        config["model_name"], revision=config["revision"], cache_dir=config["cache_dir"], local_files_only=True
    )
    for item in audit:
        item["token_count"] = len(tokenizer.encode(item["quote"], add_special_tokens=False)) + 1
        item["over_128_tokens"] = item["token_count"] > 128
    candidates = [
        item for item in audit if item["assistant_suggested_label"] in {"possible_jiahao_style", "genuine_jiahao_style"}
    ]
    assert not any(is_training_eligible(item) for item in audit)
    existing = read_jsonl("raw_data/curated/train_text.jsonl")
    if existing.issues:
        raise ValueError("Existing corpus failed validation")
    near = near_duplicate_candidates(records)
    target = Path(args.output_dir)
    write_jsonl(audit, target / "assistant_triage.jsonl")
    write_jsonl(candidates, target / "review_form.jsonl")
    write_jsonl((item.__dict__ for item in exact + near), target / "duplicate_review.jsonl")
    write_jsonl(
        novelty_dicts([item["quote"] for item in candidates], [item["quote"] for item in existing.records]),
        target / "nearest_existing.jsonl",
    )
    summary = {
        "source_posts": len({item["source_group"] for item in records}),
        "total_input_records": total_source_records,
        "not_in_this_review": total_source_records - len(raw),
        "extracted_fragments": len(raw),
        "unique_fragments": len(records),
        "exact_duplicates": len(exact),
        "near_duplicate_pairs": len(near),
        "assistant_label_counts": dict(Counter(item["assistant_suggested_label"] for item in audit)),
        "shortlisted_candidates": len(candidates),
        "over_128_token_candidates": sum(item["over_128_tokens"] for item in candidates),
        "candidate_token_lengths_including_eos": token_length_statistics([item["token_count"] for item in candidates]),
        "human_confirmed": 0,
        "training_approved": 0,
        "original_corpus_sha256": existing.sha256,
        "collection_methods": sorted({item.get("collection_method", "unknown") for item in records}),
    }
    (target / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# 语录候选待审包",
        "",
        json.dumps(summary, ensure_ascii=False, indent=2),
        "",
        "以下均为助手候选建议，不是人类正样本；访问/采集权限未独立核验，训练许可待审。",
        "review_form.jsonl 保留来源、审核字段和 Token 长度；长文本不静默截断。",
        "",
    ]
    for item in candidates:
        lines.extend(
            [
                f"## {item['id']}",
                "",
                item["quote"],
                "",
                item["assistant_reason"],
                "",
                f"[原帖]({item['source_url']}) · Tokens含EOS：{item['token_count']}",
                "",
            ]
        )
    (target / "REVIEW.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
