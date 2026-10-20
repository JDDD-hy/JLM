"""Apply the human review CSV to the candidate pool, producing records ready for prepare_data.py.

Rule of the simple sheet: blank 判定 = pass, x = reject, ? = unsure. Because blank means pass,
only rows with 序号 <= --upto are applied; later rows stay unreviewed no matter what they contain.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter
from pathlib import Path

from src.data.validate import read_jsonl, write_jsonl

DEFAULT_EVIDENCE = "data/reviewed/permissions/school-declaration-20261019.json"


def decide(mark: str) -> str:
    mark = mark.strip().lower()
    if mark == "":
        return "pass"
    if mark in {"x", "×", "✗", "n", "no", "不要"}:
        return "reject"
    if mark in {"?", "？", "存疑"}:
        return "unsure"
    raise ValueError(f"Unrecognized 判定 value: {mark!r} (use blank, x, or ?)")


def apply_review(pool: list[dict], sheet: list[dict], upto: int, reviewer: str, evidence: str) -> tuple[list, dict]:
    by_id = {row["id"]: row for row in sheet}
    stamp = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    counts: Counter = Counter()
    output = []
    for record in pool:
        row = by_id.get(record["id"])
        if row is None or int(row["序号"]) > upto:
            counts["not_reviewed"] += 1
            output.append(record)
            continue
        if row["语录"] != record["quote"]:
            raise ValueError(f"Sheet quote differs from pool for {record['id']}; re-export before importing")
        verdict = decide(row.get("判定", ""))
        scene_text = (row.get("场景") or "").strip()
        scene = int(scene_text) if scene_text.isdigit() and 1 <= int(scene_text) <= 17 else None
        updated = {
            **record,
            "reviewer_type": "human",
            "reviewer": reviewer,
            "reviewed_at": stamp,
            "human_note": (row.get("备注") or "").strip() or None,
            "scene": scene,
            "scene_review_status": "human_confirmed" if scene is not None else "human_unknown",
        }
        if verdict == "pass":
            updated.update(
                quality_status="genuine_jiahao_style",
                license_status="approved_for_training",
                license_evidence=evidence,
            )
        elif verdict == "reject":
            suggested = record.get("assistant_suggested_label")
            updated.update(
                quality_status="ordinary_text" if suggested in {None, "possible_jiahao_style"} else suggested,
                license_status="rejected",
            )
        else:
            updated.update(quality_status="uncertain", license_status="pending_review")
        counts[verdict] += 1
        output.append(updated)
    return output, dict(counts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sheet", default="人工审核表-20261020.csv")
    parser.add_argument("--pool", default="data/reviewed/pool-20261019/merged_candidates.jsonl")
    parser.add_argument("--upto", type=int, required=True, help="Last 序号 you actually reviewed")
    parser.add_argument("--reviewer", required=True, help="Your reviewer code, recorded per row")
    parser.add_argument("--license-evidence", default=DEFAULT_EVIDENCE)
    default_output = f"data/reviewed/human-review-{time.strftime('%Y%m%d')}/reviewed_records.jsonl"
    parser.add_argument("--output", default=default_output)
    args = parser.parse_args()
    if not Path(args.license_evidence).exists():
        raise SystemExit(f"License evidence not found: {args.license_evidence}")
    pool = read_jsonl(args.pool)
    if pool.issues:
        raise SystemExit("Pool failed validation")
    with Path(args.sheet).open(encoding="utf-8-sig", newline="") as handle:
        sheet = list(csv.DictReader(handle))
    records, counts = apply_review(pool.records, sheet, args.upto, args.reviewer, args.license_evidence)
    write_jsonl(records, args.output)
    summary = {
        "output": args.output,
        "applied_up_to": args.upto,
        "decisions": counts,
        "scene_counts_of_passed": dict(
            Counter(str(r.get("scene")) for r in records if r.get("quality_status") == "genuine_jiahao_style")
        ),
        "next": f"./.venv/Scripts/python.exe scripts/prepare_data.py --input {args.output}",
    }
    Path(args.output).with_name("import_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
