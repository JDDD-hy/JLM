from src.data.quality import is_training_eligible, suggest_quality_label


def test_rules_do_not_promote_to_genuine() -> None:
    suggestion = suggest_quality_label("别人不懂我的实力，因为世界的规则尚未改变")
    assert suggestion.label != "genuine_jiahao_style"
    assert suggestion.requires_human_review


def test_training_gate_requires_human_and_license_status() -> None:
    record = {"quality_status": "genuine_jiahao_style", "license_status": "pending_review"}
    assert not is_training_eligible(record)
    record["license_status"] = "approved_for_training"
    assert is_training_eligible(record)
