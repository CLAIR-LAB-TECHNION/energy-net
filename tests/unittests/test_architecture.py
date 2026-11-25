"""
Enforce the layering of the prosumer stack by inspecting imports.

gym_envs (adapter) -> simulation -> grid_entities / dynamics / forecasting
-> foundation. The gym adapter holds no business logic, and nothing below
it depends on gymnasium.
"""

import ast
import pathlib

import pytest

PACKAGE = pathlib.Path(__file__).resolve().parents[2] / "energy_net"
GYM_ADAPTER = ["prosumer_env.py", "prosumer_observation.py", "prosumer_render.py"]
STDLIB_OK = {"typing", "dataclasses", "abc", "functools", "math", "collections"}


def _imports(path: pathlib.Path):
    """Absolute module names imported by ``path`` (relative imports resolved)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    package = ".".join(path.relative_to(PACKAGE.parent).with_suffix("").parts[:-1])
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package.split(".")[: len(package.split(".")) - node.level + 1]
                found.add(".".join(base + ([node.module] if node.module else [])))
            else:
                found.add(node.module)
    return found


def _modules(*parts):
    return sorted((PACKAGE.joinpath(*parts)).rglob("*.py"))


@pytest.mark.parametrize("name", GYM_ADAPTER)
def test_gym_adapter_only_talks_to_gym_numpy_and_the_simulation(name):
    allowed_prefixes = ("gymnasium", "numpy", "energy_net.simulation", "energy_net.gym_envs.prosumer_")
    for module in _imports(PACKAGE / "gym_envs" / name):
        top = module.split(".")[0]
        assert top in STDLIB_OK or module.startswith(allowed_prefixes), (
            f"{name} imports {module}; business logic belongs in energy_net.simulation"
        )


@pytest.mark.parametrize("layer", ["simulation", "grid_entities", "dynamics", "forecasting", "foundation"])
def test_core_layers_do_not_depend_on_gym_or_rl(layer):
    for path in _modules(layer):
        for module in _imports(path):
            assert not module.startswith(("gymnasium", "gym", "stable_baselines3")), (
                f"{path.relative_to(PACKAGE)} imports {module}"
            )
            assert not module.startswith("energy_net.gym_envs"), (
                f"{path.relative_to(PACKAGE)} imports the gym layer ({module})"
            )


@pytest.mark.parametrize("layer, forbidden", [
    ("foundation", ("energy_net.grid_entities", "energy_net.dynamics",
                    "energy_net.forecasting", "energy_net.simulation")),
    ("dynamics", ("energy_net.grid_entities", "energy_net.forecasting", "energy_net.simulation")),
    ("forecasting", ("energy_net.grid_entities", "energy_net.dynamics", "energy_net.simulation")),
    ("grid_entities", ("energy_net.dynamics", "energy_net.forecasting", "energy_net.simulation")),
])
def test_lower_layers_depend_only_on_foundation(layer, forbidden):
    for path in _modules(layer):
        for module in _imports(path):
            assert not module.startswith(forbidden), (
                f"{path.relative_to(PACKAGE)} imports {module}"
            )
