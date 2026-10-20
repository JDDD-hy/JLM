from __future__ import annotations

import time
from typing import Any

import torch
from transformers import TrainerCallback


class GpuMemoryCallback(TrainerCallback):
    """Inject peak/current CUDA allocation and elapsed time into Trainer logs."""

    def __init__(self) -> None:
        self.started_at = time.perf_counter()

    def on_train_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        self.started_at = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def on_log(self, args: Any, state: Any, control: Any, logs: dict | None = None, **kwargs: Any) -> None:
        if logs is None:
            return
        logs["elapsed_seconds"] = round(time.perf_counter() - self.started_at, 3)
        if torch.cuda.is_available():
            logs["gpu_memory_allocated_gib"] = round(torch.cuda.memory_allocated() / 2**30, 3)
            logs["gpu_memory_peak_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 3)
