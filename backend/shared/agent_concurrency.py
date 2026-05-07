"""
Shared guardrails for MiroFish agent execution fanout.

The UI may allow many total agents, but runtime LLM/OASIS calls must stay
bounded so a single run cannot exhaust the host.
"""

from __future__ import annotations

import os
from typing import Dict, Iterable, Iterator, List, TypeVar


T = TypeVar("T")

DEFAULT_AGENT_CONCURRENCY = 1
DEFAULT_AGENT_CONCURRENCY_MAX = 8
DEFAULT_PLATFORM_EXECUTION = "sequential"
VALID_PLATFORM_EXECUTIONS = {"sequential", "parallel"}


def _as_int(value: object, fallback: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def agent_concurrency_max(value: object = None) -> int:
    raw = value if value is not None else os.getenv("MIROFISH_AGENT_CONCURRENCY_MAX")
    return max(1, _as_int(raw, DEFAULT_AGENT_CONCURRENCY_MAX))


def clamp_agent_concurrency(value: object = None, *, fallback: object = None, maximum: object = None) -> int:
    limit = agent_concurrency_max(maximum)
    raw = value
    if raw is None:
        raw = fallback if fallback is not None else os.getenv("MIROFISH_AGENT_CONCURRENCY")
    return max(1, min(_as_int(raw, DEFAULT_AGENT_CONCURRENCY), limit))


def clamp_profile_concurrency(value: object = None) -> int:
    fallback = os.getenv("MIROFISH_PROFILE_CONCURRENCY", os.getenv("MIROFISH_AGENT_CONCURRENCY"))
    return clamp_agent_concurrency(value, fallback=fallback)


def platform_execution_mode(value: object = None) -> str:
    raw = str(value or os.getenv("MIROFISH_PLATFORM_EXECUTION", DEFAULT_PLATFORM_EXECUTION)).lower()
    return raw if raw in VALID_PLATFORM_EXECUTIONS else DEFAULT_PLATFORM_EXECUTION


def iter_chunks(items: Iterable[T], size: int) -> Iterator[List[T]]:
    batch: List[T] = []
    for item in items:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def iter_action_batches(actions: Dict[object, object], size: int) -> Iterator[Dict[object, object]]:
    for batch in iter_chunks(list(actions.items()), size):
        yield dict(batch)
