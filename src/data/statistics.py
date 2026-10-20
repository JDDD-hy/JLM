from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from typing import Any

import numpy as np


def token_length_statistics(lengths: list[int]) -> dict[str, float | int]:
    if not lengths:
        return {key: 0 for key in ("count", "min", "max", "mean", "median", "p90", "p95", "p99")}
    values = np.asarray(lengths, dtype=np.int64)
    return {
        "count": int(values.size),
        "min": int(values.min()),
        "max": int(values.max()),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
    }


def categorical_counts(records: Iterable[dict[str, Any]], field: str) -> dict[str, int]:
    return dict(Counter(str(record.get(field)) for record in records))
