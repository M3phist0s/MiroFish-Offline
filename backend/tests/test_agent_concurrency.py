from shared.agent_concurrency import (
    clamp_agent_concurrency,
    iter_action_batches,
    iter_chunks,
    platform_execution_mode,
)
from shared.agent_models import iter_model_action_batches, iter_model_agent_batches, normalize_model_routing


def _simulation_test_client(tmp_path, monkeypatch):
    from flask import Flask

    from app.api import simulation_bp
    from app.config import Config
    from app.services.simulation_manager import SimulationManager

    sim_root = tmp_path / "simulations"
    sim_root.mkdir()
    monkeypatch.setattr(Config, "OASIS_SIMULATION_DATA_DIR", str(sim_root))
    monkeypatch.setattr(SimulationManager, "SIMULATION_DATA_DIR", str(sim_root))

    app = Flask(__name__)
    app.register_blueprint(simulation_bp, url_prefix="/api/simulation")
    return app.test_client(), sim_root


def test_agent_concurrency_defaults_to_one(monkeypatch):
    monkeypatch.delenv("MIROFISH_AGENT_CONCURRENCY", raising=False)
    assert clamp_agent_concurrency(None) == 1


def test_agent_concurrency_default_max_allows_paid_routes(monkeypatch):
    monkeypatch.delenv("MIROFISH_AGENT_CONCURRENCY_MAX", raising=False)
    assert clamp_agent_concurrency(100) == 8


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


class _Agent:
    def __init__(self, agent_id):
        self.social_agent_id = agent_id


def test_model_action_batches_respect_per_model_limits():
    routing = normalize_model_routing(
        {
            "agent_model_selection": [
                {"model_id": "local", "concurrency": 1},
                {"model_id": "paid", "concurrency": 3},
            ],
            "agent_model_assignments": [
                {"agent_id": 0, "model_id": "local"},
                {"agent_id": 1, "model_id": "local"},
                {"agent_id": 2, "model_id": "paid"},
                {"agent_id": 3, "model_id": "paid"},
                {"agent_id": 4, "model_id": "paid"},
                {"agent_id": 5, "model_id": "paid"},
            ],
        }
    )
    actions = {_Agent(agent_id): f"action-{agent_id}" for agent_id in range(6)}

    batches = list(iter_model_action_batches(actions, routing, 8))
    sizes = [len(batch) for batch in batches]

    assert sizes == [1, 1, 3, 1]


def test_model_agent_batches_respect_per_model_limits():
    routing = normalize_model_routing(
        {
            "agent_model_selection": [
                {"model_id": "a", "concurrency": 1},
                {"model_id": "b", "concurrency": 2},
            ],
            "agent_model_assignments": [
                {"agent_id": 0, "model_id": "a"},
                {"agent_id": 1, "model_id": "a"},
                {"agent_id": 2, "model_id": "b"},
                {"agent_id": 3, "model_id": "b"},
                {"agent_id": 4, "model_id": "b"},
            ],
        }
    )
    active_agents = [(agent_id, _Agent(agent_id)) for agent_id in range(5)]

    batches = list(iter_model_agent_batches(active_agents, routing, 8))

    assert [len(batch) for batch in batches] == [1, 1, 2, 1]


def test_prepare_status_reports_failed_no_entity_task(tmp_path, monkeypatch):
    from app.models.task import TaskManager

    client, _ = _simulation_test_client(tmp_path, monkeypatch)
    task_manager = TaskManager()
    task_id = task_manager.create_task("simulation_prepare")
    task_manager.complete_task(
        task_id,
        {
            "simulation_id": "sim-empty",
            "project_id": "proj-empty",
            "graph_id": "graph-empty",
            "status": "failed",
            "entities_count": 0,
            "profiles_count": 0,
            "config_generated": False,
            "error": "No entities matching criteria found, check if graph is correctly constructed",
        },
    )

    response = client.post(
        "/api/simulation/prepare/status",
        json={"simulation_id": "sim-empty", "task_id": task_id},
    )
    payload = response.get_json()

    assert response.status_code == 200
    assert payload["success"] is True
    assert payload["data"]["status"] == "failed"
    assert payload["data"]["already_prepared"] is False
    assert payload["data"]["entities_count"] == 0
    assert payload["data"]["profiles_count"] == 0
    assert payload["data"]["config_generated"] is False
    assert "No entities" in payload["data"]["error"]


def test_start_rejects_unprepared_no_entity_simulation(tmp_path, monkeypatch):
    from app.services.simulation_manager import SimulationManager

    client, _ = _simulation_test_client(tmp_path, monkeypatch)
    manager = SimulationManager()
    state = manager.create_simulation("proj-empty", "graph-empty")
    state.status = type(state.status).FAILED
    state.error = "No entities matching criteria found, check if graph is correctly constructed"
    state.entities_count = 0
    state.profiles_count = 0
    state.config_generated = False
    manager._save_simulation_state(state)

    response = client.post(
        "/api/simulation/start",
        json={"simulation_id": state.simulation_id, "platform": "parallel"},
    )
    payload = response.get_json()

    assert response.status_code == 400
    assert payload["success"] is False
    assert "Simulation not ready" in payload["error"]
