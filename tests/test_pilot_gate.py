from __future__ import annotations

import json

import pytest

from src.training.data_gate import PILOT_SCOPE, verify_training_inputs


def _write(tmp_path, split, rows, meta):
    d = tmp_path / split
    d.mkdir()
    (d / "data.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    (d / "metadata_index.jsonl").write_text(
        "".join(json.dumps(m, ensure_ascii=False) + "\n" for m in meta), encoding="utf-8"
    )
    return str(d / "data.jsonl")


def _meta(rid, text, group, **extra):
    return {"id": rid, "quote": text, "leakage_group": group, "review_scope": PILOT_SCOPE, **extra}


def test_pilot_requires_flag_and_marker(tmp_path):
    train = _write(tmp_path, "t", [{"id": "a", "text": "qa"}], [_meta("a", "qa", "g1")])
    val = _write(tmp_path, "v", [{"id": "b", "text": "qb"}], [_meta("b", "qb", "g2")])
    with pytest.raises(ValueError):
        verify_training_inputs(train, val)
    scope = verify_training_inputs(train, val, allow_unreviewed_pilot=True)
    assert scope["scope"] == PILOT_SCOPE and scope["human_reviewed"] is False


def test_pilot_rejects_overlap_and_fake_human(tmp_path):
    train = _write(tmp_path, "t", [{"id": "a", "text": "qa"}], [_meta("a", "qa", "g1")])
    val = _write(tmp_path, "v", [{"id": "b", "text": "qb"}], [_meta("b", "qb", "g1")])
    with pytest.raises(ValueError):
        verify_training_inputs(train, val, allow_unreviewed_pilot=True)
    val2 = _write(tmp_path, "w", [{"id": "c", "text": "qc"}], [_meta("c", "qc", "g3", reviewer_type="human")])
    with pytest.raises(ValueError):
        verify_training_inputs(train, val2, allow_unreviewed_pilot=True)
