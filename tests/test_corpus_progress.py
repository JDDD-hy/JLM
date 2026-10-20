from scripts.corpus_progress import scene_quotas, summarize


def test_final_quota_is_5000_but_interim_has_no_balance_gate():
    quotas = scene_quotas(list(range(1, 18)), 5000)
    assert sum(quotas.values()) == 5000
    assert list(quotas.values()).count(295) == 2
    assert list(quotas.values()).count(294) == 15
    report = summarize(
        [
            {"scene": 0},
            {"scene": None, "assistant_suggested_scene": 5},
            {"scene": 3},
        ],
        list(range(1, 18)),
    )
    assert report["unknown_scene_candidates"] == 1
    assert report["provisional_scene_counts"] == {5: 1, 3: 1}
    assert report["human_confirmed_scene_counts"] == {}
    assert report["balance_required_at_interim"] is False
    assert report["training_eligible"] == 0
    assert report["primary_source_counts"] == {"unknown": 3}
    assert not report["confirmed_final_balance_met"]
