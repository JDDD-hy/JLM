from __future__ import annotations

from dataclasses import asdict, dataclass

from rapidfuzz import process
from rapidfuzz.fuzz import ratio

from src.data.clean import canonical_for_dedup


@dataclass(frozen=True)
class NoveltyResult:
    generated: str
    exact_match: bool
    nearest_training_quote: str | None
    similarity: float


def evaluate_novelty(generations: list[str], training_quotes: list[str]) -> list[NoveltyResult]:
    canonical_training = [canonical_for_dedup(text) for text in training_quotes]
    lookup = dict(zip(canonical_training, training_quotes, strict=True))
    results: list[NoveltyResult] = []
    for generated in generations:
        key = canonical_for_dedup(generated)
        match = process.extractOne(key, canonical_training, scorer=ratio) if canonical_training else None
        nearest = lookup[match[0]] if match else None
        results.append(
            NoveltyResult(
                generated=generated,
                exact_match=key in lookup,
                nearest_training_quote=nearest,
                similarity=float(match[1]) if match else 0.0,
            )
        )
    return results


def novelty_dicts(generations: list[str], training_quotes: list[str]) -> list[dict]:
    return [asdict(result) for result in evaluate_novelty(generations, training_quotes)]
