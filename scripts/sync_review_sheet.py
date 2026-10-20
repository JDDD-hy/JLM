"""Append new pool candidates to the human review CSV while preserving every human edit."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import time
from pathlib import Path

from scripts.export_review_sheet import SIMPLE_COLUMNS

HUMAN_COLUMNS = ("判定", "场景", "备注")


def build_simple_rows(export_csv: Path) -> list[dict]:
    with export_csv.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sheet", default="人工审核表-20261020.csv")
    parser.add_argument("--fresh-export", default="outputs/review-sheet-20261020/review_simple.csv")
    args = parser.parse_args()
    sheet = Path(args.sheet)
    fresh = {row["id"]: row for row in build_simple_rows(Path(args.fresh_export))}
    existing: list[dict] = []
    if sheet.exists():
        backup_dir = sheet.resolve().parent / "outputs" / "review-sheet-backups"
        backup_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(sheet, backup_dir / f"{sheet.stem}-{time.strftime('%Y%m%d-%H%M%S')}{sheet.suffix}")
        with sheet.open(encoding="utf-8-sig", newline="") as handle:
            existing = list(csv.DictReader(handle))
    known = {row["id"] for row in existing}
    next_no = max((int(row["序号"]) for row in existing), default=0) + 1
    removed = sorted(known - set(fresh))
    appended = 0
    for row_id, row in fresh.items():
        if row_id in known:
            continue
        row = dict(row)
        row["序号"] = next_no
        row["判定"] = ""
        row["备注"] = ""
        next_no += 1
        existing.append(row)
        appended += 1
    # refresh non-human columns (hints may gain near-duplicate flags) but never human ones
    for row in existing:
        latest = fresh.get(row["id"])
        if latest:
            for column in SIMPLE_COLUMNS:
                if column not in HUMAN_COLUMNS and column != "序号":
                    row[column] = latest[column]
    with sheet.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SIMPLE_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(existing)
    filled = sum(1 for row in existing if row.get("判定", "").strip())
    print(
        json.dumps(
            {"rows": len(existing), "appended": appended, "human_filled": filled, "ids_no_longer_in_pool": removed},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
