from shared.agent_concurrency import (
    clamp_agent_concurrency,
    iter_action_batches,
    iter_chunks,
    platform_execution_mode,
)


def test_agent_concurrency_defaults_to_one(monkeypatch):
    monkeypatch.delenv("MIROFISH_AGENT_CONCURRENCY", raising=False)
    assert clamp_agent_concurrency(None) == 1


def test_agent_concurrency_clamps_to_safe_max(monkeypatch):
    monkeypatch.setenv("MIROFISH_AGENT_CONCURRENCY_MAX", "4")
    assert clamp_agent_concurrency(100) == 4


def test_agent_batches_never_exceed_limit():
    batches = list(iter_chunks(list(range(10)), 4))
    assert [len(batch) for batch in batches] == [4, 4, 2]


def test_action_batches_never_exceed_limit():
    actions = {f"agent-{idx}": f"action-{idx}" for idx in range(10)}
    batches = list(iter_action_batches(actions, 1))
    assert len(batches) == 10
    assert all(len(batch) == 1 for batch in batches)


def test_platform_execution_defaults_to_sequential(monkeypatch):
    monkeypatch.delenv("MIROFISH_PLATFORM_EXECUTION", raising=False)
    assert platform_execution_mode(None) == "sequential"
    assert platform_execution_mode("invalid") == "sequential"
    assert platform_execution_mode("parallel") == "parallel"
