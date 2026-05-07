from __future__ import annotations

from typing import Any, Dict, Iterable, Iterator, List, Tuple, TypeVar

from .agent_concurrency import iter_chunks


T = TypeVar("T")


def normalize_model_routing(config: Dict[str, Any]) -> Dict[str, Any]:
    selection = list(config.get("agent_model_selection") or [])
    assignments = list(config.get("agent_model_assignments") or [])
    models_by_id = {item.get("model_id"): item for item in selection if item.get("model_id")}
    assignment_by_agent = {}
    for item in assignments:
        try:
            agent_id = int(item.get("agent_id"))
        except (TypeError, ValueError):
            continue
        model_id = item.get("model_id")
        if model_id:
            assignment_by_agent[agent_id] = item
    return {
        "selection": selection,
        "assignments": assignments,
        "models_by_id": models_by_id,
        "assignment_by_agent": assignment_by_agent,
    }


def model_for_agent(agent_id: int, routing: Dict[str, Any], fallback_model_id: str = "") -> Dict[str, Any]:
    assignment = routing.get("assignment_by_agent", {}).get(int(agent_id))
    if assignment:
        return assignment
    if fallback_model_id and fallback_model_id in routing.get("models_by_id", {}):
        return routing["models_by_id"][fallback_model_id]
    selection = routing.get("selection") or []
    return selection[0] if selection else {}


def concurrency_for_agent(agent_id: int, routing: Dict[str, Any], fallback: int) -> int:
    model = model_for_agent(agent_id, routing)
    model_id = model.get("model_id")
    configured = routing.get("models_by_id", {}).get(model_id, model)
    try:
        return max(1, int(configured.get("concurrency", model.get("concurrency", fallback))))
    except (TypeError, ValueError):
        return max(1, int(fallback))


def agent_id_from_object(agent: Any) -> int:
    for attr in ("social_agent_id", "agent_id", "id"):
        value = getattr(agent, attr, None)
        if value is not None:
            return int(value)
    raise ValueError(f"Unable to determine agent id for {agent!r}")


def iter_model_action_batches(
    actions: Dict[Any, Any],
    routing: Dict[str, Any],
    fallback_size: int,
) -> Iterator[Dict[Any, Any]]:
    grouped: Dict[str, List[Tuple[Any, Any]]] = {}
    sizes: Dict[str, int] = {}
    for agent, action in actions.items():
        agent_id = agent_id_from_object(agent)
        model = model_for_agent(agent_id, routing)
        model_id = str(model.get("model_id") or "default")
        grouped.setdefault(model_id, []).append((agent, action))
        sizes[model_id] = concurrency_for_agent(agent_id, routing, fallback_size)
    for model_id, items in grouped.items():
        for batch in iter_chunks(items, sizes.get(model_id, fallback_size)):
            yield dict(batch)


def iter_model_agent_batches(
    active_agents: Iterable[Tuple[int, Any]],
    routing: Dict[str, Any],
    fallback_size: int,
) -> Iterator[List[Tuple[int, Any]]]:
    grouped: Dict[str, List[Tuple[int, Any]]] = {}
    sizes: Dict[str, int] = {}
    for agent_id, agent in active_agents:
        model = model_for_agent(int(agent_id), routing)
        model_id = str(model.get("model_id") or "default")
        grouped.setdefault(model_id, []).append((agent_id, agent))
        sizes[model_id] = concurrency_for_agent(int(agent_id), routing, fallback_size)
    for model_id, items in grouped.items():
        yield from iter_chunks(items, sizes.get(model_id, fallback_size))
