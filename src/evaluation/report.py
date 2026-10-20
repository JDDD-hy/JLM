from __future__ import annotations

import argparse
import json
from pathlib import Path


def render_markdown(report: dict) -> str:
    lm = report.get("language_modeling", {})
    diversity = report.get("diversity", {})
    lines = [
        "# JLM v0.1 Evaluation Report",
        "",
        "## Language modeling",
        "",
        f"- Loss: {lm.get('loss', 'n/a')}",
        f"- Perplexity: {lm.get('perplexity', 'n/a')}",
        "",
        "## Diversity",
        "",
        f"- Distinct-1: {diversity.get('distinct_1', 'n/a')}",
        f"- Distinct-2: {diversity.get('distinct_2', 'n/a')}",
        f"- Unique generation ratio: {diversity.get('unique_generation_ratio', 'n/a')}",
        "",
        "## Blind style review",
        "",
        "Human reviewers must score shuffled base/adapted samples without seeing model identity.",
        "Suggested fields: style fidelity (1-5), naturalness (1-5), memorization concern, notes.",
        "",
        "## Samples and nearest training neighbors",
        "",
    ]
    for item in report.get("novelty", []):
        lines.extend(
            [
                f"### {item['generated']}",
                "",
                f"- Exact training match: {item['exact_match']}",
                f"- Nearest similarity: {item['similarity']:.2f}",
                f"- Nearest quote: {item['nearest_training_quote']}",
                "",
            ]
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", nargs="?", default="outputs/evaluation.json")
    parser.add_argument("--output", default="outputs/evaluation.md")
    args = parser.parse_args()
    report = json.loads(Path(args.input).read_text(encoding="utf-8"))
    Path(args.output).write_text(render_markdown(report), encoding="utf-8")


if __name__ == "__main__":
    main()
