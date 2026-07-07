"""
metrics.py
----------
Lightweight timing and performance utilities (FR-20).
No external dependencies — uses contextmanager + time.perf_counter.

Usage:
    from utils.metrics import Timer

    with Timer("retrieval") as t:
        results = hybrid_search(query)

    logger.info("Retrieval took {:.1f} ms", t.elapsed_ms)
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Generator


@dataclass
class TimerResult:
    name: str
    elapsed_sec: float = 0.0

    @property
    def elapsed_ms(self) -> float:
        return self.elapsed_sec * 1_000

    def __str__(self) -> str:
        if self.elapsed_ms < 1_000:
            return f"{self.name}: {self.elapsed_ms:.1f} ms"
        return f"{self.name}: {self.elapsed_sec:.2f} s"


@contextmanager
def Timer(name: str) -> Generator[TimerResult, None, None]:
    """
    Context manager that measures wall-clock time for a block.

    Args:
        name: Human-readable label for this measurement.

    Yields:
        TimerResult — populate .elapsed_sec after the block exits.
    """
    result = TimerResult(name=name)
    start = time.perf_counter()
    try:
        yield result
    finally:
        result.elapsed_sec = time.perf_counter() - start


@dataclass
class LatencyTracker:
    """
    Accumulates named timing observations across a single request lifecycle.
    Attach one to each ChatSession or request context.
    """

    _timings: dict[str, float] = field(default_factory=dict)

    def record(self, name: str, elapsed_sec: float) -> None:
        self._timings[name] = elapsed_sec

    @property
    def total_sec(self) -> float:
        return sum(self._timings.values())

    def summary(self) -> dict[str, str]:
        """Return human-readable timing dict, e.g. for CLI display."""
        out: dict[str, str] = {}
        for name, sec in self._timings.items():
            ms = sec * 1_000
            out[name] = f"{ms:.1f} ms" if ms < 1_000 else f"{sec:.2f} s"
        out["total"] = (
            f"{self.total_sec * 1_000:.1f} ms"
            if self.total_sec < 1
            else f"{self.total_sec:.2f} s"
        )
        return out

    def check_sla(
        self,
        retrieval_ms: float,
        generation_sec: float,
    ) -> list[str]:
        """
        Return a list of SLA violations as human-readable strings.
        Empty list means all targets met.
        """
        violations: list[str] = []
        actual_retrieval = self._timings.get("retrieval", 0) * 1_000
        actual_generation = self._timings.get("generation", 0)
        if actual_retrieval > retrieval_ms:
            violations.append(
                f"Retrieval SLA breach: {actual_retrieval:.1f} ms > {retrieval_ms} ms target"
            )
        if actual_generation > generation_sec:
            violations.append(
                f"Generation SLA breach: {actual_generation:.2f} s > {generation_sec} s target"
            )
        return violations