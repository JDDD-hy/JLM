from src.data.validate import read_jsonl, stable_record_id


def test_read_jsonl_reports_bad_lines_without_silent_drop(tmp_path) -> None:
    path = tmp_path / "input.jsonl"
    path.write_text(
        '\ufeff{"quote":"有效","scene":9}\n{"quote":"","scene":null}\nnot-json\n',
        encoding="utf-8",
    )
    result = read_jsonl(path)
    assert len(result.records) == 1
    assert [issue.code for issue in result.issues] == ["invalid_quote", "invalid_json"]
    assert result.sha256


def test_stable_id_is_content_based() -> None:
    assert stable_record_id("文本", 1) == stable_record_id("文本", 1)
    assert stable_record_id("文本", 1) != stable_record_id("文本", 2)
