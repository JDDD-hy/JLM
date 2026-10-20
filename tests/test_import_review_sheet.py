from __future__ import annotations

import pytest

from scripts.import_review_sheet import apply_review, decide


def _pool():
    return [
        {"id": "a", "quote": "qa", "assistant_suggested_label": "possible_jiahao_style"},
        {"id": "b", "quote": "qb", "assistant_suggested_label": "possible_jiahao_style"},
        {"id": "c", "quote": "qc", "assistant_suggested_label": "possible_jiahao_style"},
        {"id": "d", "quote": "qd", "assistant_suggested_label": "possible_jiahao_style"},
    ]


def _sheet():
    return [
        {"序号": "1", "id": "a", "语录": "qa", "判定": "", "场景": "3", "备注": ""},
        {"序号": "2", "id": "b", "语录": "qb", "判定": "x", "场景": "3", "备注": ""},
        {"序号": "3", "id": "c", "语录": "qc", "判定": "?", "场景": "", "备注": "hmm"},
        {"序号": "4", "id": "d", "语录": "qd", "判定": "", "场景": "3", "备注": ""},
    ]


def test_rows_after_upto_are_never_approved():
    records, counts = apply_review(_pool(), _sheet(), upto=3, reviewer="r1", evidence="e.json")
    by = {r["id"]: r for r in records}
    assert by["a"]["quality_status"] == "genuine_jiahao_style"
    assert by["a"]["license_status"] == "approved_for_training" and by["a"]["reviewer_type"] == "human"
    assert by["a"]["scene"] == 3
    assert by["b"]["license_status"] == "rejected"
    assert by["c"]["quality_status"] == "uncertain" and by["c"]["scene"] is None
    assert "reviewer_type" not in by["d"]
    assert counts == {"pass": 1, "reject": 1, "unsure": 1, "not_reviewed": 1}


def test_mismatched_quote_and_bad_mark_fail():
    sheet = _sheet()
    sheet[0]["语录"] = "changed"
    with pytest.raises(ValueError):
        apply_review(_pool(), sheet, upto=1, reviewer="r1", evidence="e.json")
    with pytest.raises(ValueError):
        decide("ok")
