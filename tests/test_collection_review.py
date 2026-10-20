import pytest

from scripts.build_collection_review import annotate, select_annotated_records
from src.data.quality import is_training_eligible


def test_assistant_annotation_cannot_approve_training():
    result = annotate(
        [
            {
                "id": "synthetic1",
                "quote": "synthetic 测试样本",
                "quality_status": "genuine_jiahao_style",
                "license_status": "approved_for_training",
            }
        ],
        [{"id": "synthetic1", "label": "genuine_jiahao_style", "reason": "synthetic test only"}],
    )
    assert result[0]["quality_status"] == "unreviewed"
    assert result[0]["license_status"] == "pending_review"
    assert result[0]["reviewer_type"] is None
    assert not is_training_eligible(result[0])


def test_annotation_must_cover_actual_records():
    with pytest.raises(ValueError, match="exactly once"):
        annotate([{"id": "synthetic1", "quote": "synthetic 测试样本"}], [])


def test_partial_review_is_explicit_and_does_not_invent_scene_truth():
    records = [
        {"id": "synthetic1", "quote": "synthetic 测试样本", "scene": None},
        {"id": "synthetic2", "quote": "synthetic 未审核样本", "scene": None},
    ]
    decisions = [{"id": "synthetic1", "label": "possible_jiahao_style", "scene": 5, "reason": "test only"}]
    selected = select_annotated_records(records, decisions)
    assert len(selected) == 1
    result = annotate(selected, decisions)
    assert result[0]["scene"] is None and result[0]["assistant_suggested_scene"] == 5
    assert not is_training_eligible(result[0])
    with pytest.raises(ValueError, match="unknown source"):
        select_annotated_records([], decisions)
    with pytest.raises(ValueError, match="suggested scene"):
        annotate(selected, [{**decisions[0], "scene": True}])
