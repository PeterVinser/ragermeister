"""Thread-local cost meter for the evaluation harness.

The baselines run concurrently (one thread each) and SHARE some LLM/Embedder instances,
so a plain per-instance counter would mix their costs together. Instead we tag the current
thread with a baseline name and have the two leaf cost sources — ``Embedder`` and ``LLM`` —
report into a registry keyed by that name. Every caller (entity extractor, ER adjudicator,
resolver, knowledge base) is then counted automatically, attributed to whichever baseline's
thread issued the call. Calls made with no active context (e.g. the startup dimension probe)
are ignored.

Usage:
    cost_meter.reset()
    cost_meter.set_context("graph-only")   # at the start of a baseline's thread
    ...                                     # Embedder/LLM calls accrue under "graph-only"
    cost = cost_meter.snapshot()["graph-only"]
"""

from __future__ import annotations

import threading
from collections import defaultdict

_ctx = threading.local()
_lock = threading.Lock()
_registry: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))


def set_context(name: str | None) -> None:
    """Tag the calling thread so subsequent record() calls attribute to ``name``."""
    _ctx.name = name


def record(metric: str, amount: float = 1.0) -> None:
    """Add ``amount`` to ``metric`` for the calling thread's baseline (no-op if untagged)."""
    name = getattr(_ctx, "name", None)
    if name is None:
        return
    with _lock:
        _registry[name][metric] += amount


def snapshot() -> dict[str, dict[str, float]]:
    """Return a copy of the registry: {baseline: {metric: total}}."""
    with _lock:
        return {k: dict(v) for k, v in _registry.items()}


def reset() -> None:
    """Clear all accumulated counts (call once per shuffle before running baselines)."""
    with _lock:
        _registry.clear()
