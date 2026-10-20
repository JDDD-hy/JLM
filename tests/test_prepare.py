import pytest

from src.data.prepare import encode_quote, split_grouped_records


class FakeTokenizer:
    bos_token_id = 1
    eos_token_id = 2
    pad_token_id = None
    eos_token = "<eos>"
    pad_token = None

    def encode(self, text, **kwargs):
        return list(range(10, 10 + len(text)))[: kwargs.get("max_length", len(text))]


def test_encode_quote_has_boundaries_and_no_padding() -> None:
    encoded = encode_quote(FakeTokenizer(), "测试文本", max_length=5)
    assert encoded["input_ids"][0] == 1
    assert encoded["input_ids"][-1] == 2
    assert encoded["labels"] == encoded["input_ids"]
    assert len(encoded["input_ids"]) == 5


def test_source_groups_never_cross_splits() -> None:
    records = [
        {"id": "a", "quote": "完全不同甲", "source_group": "post-1"},
        {"id": "b", "quote": "完全不同乙", "source_group": "post-1"},
        {"id": "c", "quote": "另一条文本", "source_group": "post-2"},
        {"id": "d", "quote": "第三条文本", "source_group": "post-3"},
    ]
    splits, _, _ = split_grouped_records(records, seed=42, near_threshold=99)
    locations = {record["id"]: split for split, values in splits.items() for record in values}
    assert locations["a"] == locations["b"]


def test_no_bos_uses_document_boundary_and_empty_quote_is_rejected():
    tokenizer = FakeTokenizer()
    tokenizer.bos_token_id = None
    encoded = encode_quote(tokenizer, "文本", 8)
    assert encoded["input_ids"] == [2, 10, 11, 2]
    with pytest.raises(ValueError, match="empty"):
        encode_quote(tokenizer, "  ", 8)
