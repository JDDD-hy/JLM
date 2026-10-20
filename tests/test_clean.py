from src.data.clean import canonical_for_dedup, normalize_text


def test_normalize_preserves_punctuation_and_spelling() -> None:
    assert normalize_text("  我&nbsp;没错！！\r\n展开全文  ") == "我 没错！！"


def test_canonical_dedup_ignores_whitespace_only() -> None:
    assert canonical_for_dedup("我 没错") == canonical_for_dedup("我没错")
