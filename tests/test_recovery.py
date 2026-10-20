import pytest

from scripts.recover_saved_candidates import recover_record
from src.collection.provenance import provenance_record
from src.data.quality import is_training_eligible


def test_unknown_authorship_is_default():
    record = provenance_record(
        record_id="synthetic1",
        quote="synthetic 测试文本",
        source_platform="local",
        source_type="test_fixture",
        collection_method="test",
    )
    assert record["origin"] == "unknown"


def test_recovery_requires_matching_source_and_keeps_review_gate():
    item = {
        "id": "legacy1",
        "quote": "synthetic 测试文本",
        "synthetic": False,
        "origin": "comment",
        "source_item_id": "c",
        "source_note_id": "n",
        "scene": "未知",
    }
    decision = {"scene": 5, "reason": "synthetic test only"}
    payload = {"note": {"noteId": "n"}, "comments": [{"id": "c", "content": "synthetic 测试文本"}]}
    row = recover_record(item, decision, payload)
    assert row["origin"] == "unknown" and row["scene"] is None
    assert row["assistant_suggested_scene"] == 5 and not is_training_eligible(row)
    with pytest.raises(ValueError, match="verified"):
        recover_record({**item, "source_item_id": "wrong"}, decision, payload)
    with pytest.raises(ValueError, match="Synthetic"):
        recover_record({**item, "synthetic": True}, decision, payload)
