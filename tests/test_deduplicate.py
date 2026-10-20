from src.data.deduplicate import exact_deduplicate, near_duplicate_candidates
from src.data.prepare import build_leakage_groups


def test_exact_and_near_duplicates_are_separate() -> None:
    records = [
        {"id": "a", "quote": "世界与我预期不同"},
        {"id": "b", "quote": "世界 与我预期不同"},
        {"id": "c", "quote": "世界与我的预期不同"},
    ]
    unique, exact = exact_deduplicate(records)
    near = near_duplicate_candidates(unique, threshold=80)
    assert [record["id"] for record in unique] == ["a", "c"]
    assert len(exact) == 1
    assert len(near) == 1


def test_duplicate_sources_remain_connected_after_exact_dedup():
    unique, _ = exact_deduplicate(
        [
            {"id": "a", "quote": "相同", "source_group": "postA"},
            {"id": "b", "quote": "相同", "source_group": "postB"},
            {"id": "c", "quote": "第三条不同的语录", "source_group": "postB"},
        ]
    )
    groups, _ = build_leakage_groups(unique)
    assert groups["a"] == groups["c"]
