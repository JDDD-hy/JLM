from __future__ import annotations

import csv
import sys
from pathlib import Path

from scripts import sync_review_sheet
from scripts.export_review_sheet import SIMPLE_COLUMNS


def _write(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SIMPLE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _row(no: int, rid: str, **extra) -> dict:
    base = {column: "" for column in SIMPLE_COLUMNS}
    base.update({"序号": no, "id": rid, "语录": rid, "场景": "3"})
    base.update(extra)
    return base


def test_sync_preserves_human_edits_and_appends(tmp_path, monkeypatch):
    sheet, fresh = tmp_path / "sheet.csv", tmp_path / "fresh.csv"
    _write(sheet, [_row(1, "a", 判定="x", 场景="7", 备注="no")])
    _write(fresh, [_row(1, "a", 提示="新提示"), _row(2, "b")])
    monkeypatch.setattr(sys, "argv", ["sync", "--sheet", str(sheet), "--fresh-export", str(fresh)])
    sync_review_sheet.main()
    with sheet.open(encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    assert [r["id"] for r in rows] == ["a", "b"]
    assert rows[0]["判定"] == "x" and rows[0]["场景"] == "7" and rows[0]["备注"] == "no"
    assert rows[0]["提示"] == "新提示"
    assert rows[1]["序号"] == "2" and rows[1]["判定"] == ""
