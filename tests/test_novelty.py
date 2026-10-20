from src.evaluation.novelty import evaluate_novelty


def test_novelty_reports_exact_and_neighbor() -> None:
    results = evaluate_novelty(["同一句", "近似句子啊"], ["同一句", "近似句子"])
    assert results[0].exact_match
    assert results[1].nearest_training_quote == "近似句子"
    assert results[1].similarity > 80
