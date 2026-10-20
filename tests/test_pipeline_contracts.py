import json

import pytest
import torch

from scripts.review_data import apply_decisions
from src.collection.collectors import incremental_collect
from src.collection.parsers import parse_feed_detail
from src.data.prepare import prepare_splits
from src.evaluation.evaluate import evaluate_loss
from src.evaluation.scaling import nested_group_samples
from src.training.data_gate import verify_training_inputs
from src.training.train import CausalLMCollator
from tests.test_prepare import FakeTokenizer


def test_padding_does_not_mask_real_eos():
    tokenizer = FakeTokenizer()
    tokenizer.pad_token_id = tokenizer.eos_token_id
    batch = CausalLMCollator(tokenizer)(
        [
            {"input_ids": [1, 7, 2], "attention_mask": [1, 1, 1], "labels": [1, 7, 2]},
            {"input_ids": [1, 8, 9, 2], "attention_mask": [1] * 4, "labels": [1, 8, 9, 2]},
        ]
    )
    assert batch["labels"][0, 2] == 2
    assert (batch["labels"][0, 3:] == -100).all()


def test_review_cannot_invent_authorization():
    with pytest.raises(ValueError, match="evidence"):
        apply_decisions(
            [{"id": "a"}],
            [
                {
                    "id": "a",
                    "quality_status": "genuine_jiahao_style",
                    "reviewer_type": "human",
                    "reviewer": "reviewer",
                    "license_status": "approved_for_training",
                }
            ],
        )


def test_frozen_test_set_cannot_be_silently_reassigned(tmp_path):
    records = [{"id": str(i), "quote": f"独立文本{i}", "source_id": str(i)} for i in range(12)]
    prepare_splits(records, FakeTokenizer(), tmp_path, {"train": 0.8, "validation": 0.1, "test": 0.1}, 42, 32, 99)
    with pytest.raises(ValueError, match="Frozen"):
        prepare_splits(
            records[:-1], FakeTokenizer(), tmp_path, {"train": 0.8, "validation": 0.1, "test": 0.1}, 42, 32, 99
        )


def test_incremental_archive_survives_checkpoint_restart(tmp_path):
    checkpoint = tmp_path / "checkpoint.json"
    calls = []

    def fetch(key):
        calls.append(key)
        return {"id": key, "text": "synthetic"}

    first, _ = incremental_collect(["one"], fetch, checkpoint)
    second, _ = incremental_collect(["one"], fetch, checkpoint)
    assert len(first) == 1 and second == [] and calls == ["one"]
    assert len(list((tmp_path / "responses").glob("*.json"))) == 1


def test_actual_export_shape_extracts_note_and_comment():
    result = parse_feed_detail({"note": {"noteId": "p", "desc": "笔记"}, "comments": [{"id": "c", "content": "评论"}]})
    assert result[0]["source_id"] == "p"
    assert result[1]["source_id"] == "p:c"


def test_data_gate_accepts_synthetic_and_rejects_missing_provenance(tmp_path):
    path = tmp_path / "train.jsonl"
    path.write_text(json.dumps({"text": "测试", "origin": "synthetic"}), encoding="utf-8")
    assert verify_training_inputs(str(path), str(path))["scope"] == "synthetic_engineering_test"
    path.write_text(json.dumps({"text": "测试"}), encoding="utf-8")
    with pytest.raises(ValueError, match="provenance"):
        verify_training_inputs(str(path), str(path))


def test_scaling_excludes_test_and_subsets_are_nested(tmp_path):
    rows = [{"id": str(i), "leakage_group": str(i // 2), "split": "train"} for i in range(8)]
    rows.append({"id": "held-out", "leakage_group": "test", "split": "test"})
    path = tmp_path / "index.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    subsets = nested_group_samples(str(path), [3, 5, 8])
    assert set(subsets[3]) <= set(subsets[5]) <= set(subsets[8])
    assert "held-out" not in subsets[8]


def test_loss_is_weighted_by_shifted_target_tokens():
    class Model:
        def parameters(self):
            yield torch.nn.Parameter(torch.zeros(1))

        def eval(self):
            pass

        def __call__(self, **batch):
            return type("Output", (), {"loss": batch["input_ids"][0, 1].float()})()

    tokenizer = FakeTokenizer()
    tokenizer.pad_token_id = 2
    dataset = [
        {"input_ids": [1, 2], "labels": [1, 2], "attention_mask": [1, 1]},
        {"input_ids": [1, 4, 2], "labels": [1, 4, 2], "attention_mask": [1, 1, 1]},
    ]
    metrics = evaluate_loss(Model(), dataset, tokenizer)
    assert metrics["loss"] == pytest.approx(10 / 3)
