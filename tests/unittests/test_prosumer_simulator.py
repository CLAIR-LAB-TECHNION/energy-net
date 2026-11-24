"""Tests for the gym-free ProsumerSimulator and its services."""

import numpy as np
import pandas as pd
import pytest

from energy_net.dynamics import ConstantDynamics, CSVDynamics
from energy_net.forecasting import NoisyGroundTruthForecaster, TableForecaster
from energy_net.foundation.forecaster import BaseForecaster
from energy_net.foundation.noise import GaussianNoise
from energy_net.grid_entities.management.price_curve import (
    FixedPriceStrategy,
    PriceCurveStrategy,
    SineWavePriceStrategy,
)
from energy_net.simulation import (
    CALENDAR_FEATURE_COLUMNS,
    calendar_features,
    make_prosumer,
    make_prosumer_simulator,
    make_reward,
)
from energy_net.simulation.factory import steps_per_episode
from energy_net.simulation.prosumer_simulator import ProsumerSimulator
from energy_net.simulation.tariffs import TariffSchedule
from tests.golden.fixture import load_fixture
from tests.golden.scenarios import (
    SCENARIOS,
    RecordingDataDrivenStrategy,
    use_legacy_daily_observation,
)

DAYS = 4
TIMES = pd.date_range("2025-01-01", periods=DAYS * 48, freq="30min")


def _simulator(load=None, generation=None, strategy=None, horizon=None,
               load_forecaster=None, generation_forecaster=None, **kwargs):
    load = np.full(len(TIMES), 3.0) if load is None else load
    load_dynamics = CSVDynamics(pd.Series(load, index=TIMES), "consumption")
    if generation is None:
        generation_dynamics = ConstantDynamics(0.0, "generation")
    else:
        generation_dynamics = CSVDynamics(pd.Series(generation, index=TIMES), "generation")
    if load_forecaster is None:
        load_forecaster = TableForecaster(
            pd.DataFrame({"predicted_consumption": load}, index=TIMES),
            "predicted_consumption",
        )
    prosumer = make_prosumer(
        load_dynamics=load_dynamics,
        load_forecaster=load_forecaster,
        generation_dynamics=generation_dynamics,
        generation_forecaster=generation_forecaster,
        **kwargs,
    )
    return make_prosumer_simulator(
        prosumer=prosumer,
        price_strategy=strategy or FixedPriceStrategy(0.10),
        forecast_horizon=horizon,
    )


# ---------------------------------------------------- legacy equivalence

@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_simulator_reproduces_legacy_trajectories(name):
    """Every reward and physical/financial info value, bit for bit."""
    metadata, arrays = load_fixture()
    spec = SCENARIOS[name]
    recorded = metadata["scenarios"][name]
    options = dict(spec["env_kwargs"])
    reward_fn = make_reward(options.pop("include_shortage_penalty", False),
                            options.pop("shortage_penalty", 1.0))
    strategy = spec["strategy"]()
    simulator = make_prosumer_simulator(
        spec["load"], spec["forecast"], price_strategy=strategy,
        forecast_horizon=options.pop("prediction_horizon", None),
    )
    use_legacy_daily_observation(simulator)
    assert not options

    # Values the legacy env decoded from its float32 observation.
    approximate = {"forecast_at_decision_units": 1e-6}
    for start, episode in zip(spec["starts"], recorded["episodes"]):
        assert simulator.reset(start).timestamp.isoformat() == episode["start"]
        for step in episode["steps"]:
            result = simulator.step(step["action"])
            assert reward_fn(result).total == step["reward"]
            info = result.to_info()
            for key, expected in step["info"].items():
                if key not in info:
                    continue  # reward/gym keys are added by the env
                if key == "calendar_features_at_decision":
                    for column, value in expected.items():
                        assert info[key][column] == pytest.approx(value, abs=1e-6)
                elif key in approximate:
                    assert info[key] == pytest.approx(expected, rel=approximate[key])
                elif key in ("state_after", "terminal_state") and expected is not None:
                    assert {k: info[key][k] for k in expected} == expected
                else:
                    assert info[key] == expected, key

    if isinstance(strategy, RecordingDataDrivenStrategy):
        np.testing.assert_array_equal(
            np.stack(strategy.received), arrays[f"{name}/strategy_inputs"]
        )


# ------------------------------------------------------ lifecycle

def test_step_before_reset_and_after_termination_raise():
    simulator = _simulator()
    with pytest.raises(RuntimeError, match="call reset"):
        simulator.step(0.0)
    simulator.reset(TIMES[48])
    for _ in range(48):
        result = simulator.step(0.0)
    assert result.terminated and result.next_observation is None
    with pytest.raises(RuntimeError, match="terminated"):
        simulator.step(0.0)
    with pytest.raises(RuntimeError, match="call reset"):
        simulator.observe()


def test_reset_processes_no_load_and_observes_previous_interval():
    load = np.arange(len(TIMES), dtype=float) / 100.0
    simulator = _simulator(load=load)
    snapshot = simulator.reset(TIMES[50])
    assert snapshot.previous_load == load[49]
    assert snapshot.load_forecast[0] == load[50]
    assert simulator.metrics.total_consumption == 0.0
    result = simulator.step(0.0)
    assert result.transition.load == load[50]
    assert result.next_observation.previous_load == load[50]


def test_rejected_reset_changes_nothing():
    load = np.full(len(TIMES), 3.0)
    load[100] = np.nan
    simulator = _simulator(load=load)
    with pytest.raises(ValueError, match="Cannot start episode"):
        simulator.reset(TIMES[60])
    assert not simulator.has_episode  # failed first reset

    simulator.reset(TIMES[1])
    simulator.step(5.0)
    before = (simulator.current_time, simulator.step_index,
              simulator.prosumer.storage_level(), simulator.observe())
    with pytest.raises(ValueError, match="Non-finite or negative preceding/episode consumption"):
        simulator.reset(TIMES[60])
    assert (simulator.current_time, simulator.step_index,
            simulator.prosumer.storage_level(), simulator.observe()) == before


def test_automatic_reset_continues_then_scans_for_a_valid_start():
    simulator = _simulator()
    # The first timestamp has no previous interval, so the scan finds index 1.
    assert simulator.reset().timestamp == TIMES[1]
    for _ in range(48):
        simulator.step(0.0)
    assert simulator.reset().timestamp == TIMES[49]


def test_episode_length_must_be_whole_intervals_and_dt_half_hour():
    assert steps_per_episode(1, 0.5 / 24) == 48
    with pytest.raises(ValueError, match="whole number"):
        steps_per_episode(0.1, 0.5 / 24)
    prosumer = make_prosumer()
    tariffs = TariffSchedule(SineWavePriceStrategy())
    with pytest.raises(ValueError, match="30 minutes"):
        ProsumerSimulator(prosumer, tariffs, episode_steps=24, dt=pd.Timedelta(hours=1))


def test_initial_storage_defaults_to_the_battery_configuration():
    simulator = _simulator(battery_config={"min": 0.0, "max": 100.0, "init": 25.0})
    assert simulator.reset(TIMES[1]).storage == 25.0
    assert simulator.reset(TIMES[1], initial_storage=60.0).storage == 60.0


# --------------------------------------------------- (b) ground truth

def test_ground_truth_query_equals_what_is_realized():
    rng = np.random.default_rng(0)
    load = rng.uniform(0, 5, len(TIMES))
    generation = rng.uniform(0, 4, len(TIMES))
    simulator = _simulator(load=load, generation=generation)
    simulator.reset(TIMES[10])
    # Query the whole future first, in reverse order.
    future = TIMES[10:58]
    ahead = {
        unit: simulator.ground_truth(unit, future[::-1])[::-1]
        for unit in ("load", "generator")
    }
    for k in range(48):
        result = simulator.step(float(rng.uniform(-5, 5)))
        assert result.transition.load == ahead["load"][k]
        assert result.transition.generation == ahead["generator"][k]
    with pytest.raises(ValueError):
        simulator.ground_truth("load", TIMES[-1] + pd.Timedelta(minutes=30))
    with pytest.raises(ValueError, match="unit must be"):
        simulator.ground_truth("battery", TIMES[:1])


def test_sample_adds_noise_to_ground_truth():
    simulator = _simulator()
    exact = simulator.ground_truth("load", TIMES[:5])
    noisy = simulator.sample("load", TIMES[:5], np.random.default_rng(1), GaussianNoise(0.5))
    assert not np.array_equal(exact, noisy)


# --------------------------------------------------- (c) predictions

class _CountingStochasticForecaster(BaseForecaster):
    """Returns different values on every call; counts calls."""

    def __init__(self):
        self.calls = 0
        self.rng = np.random.default_rng(0)
        self.observed = []

    def forecast(self, issue_time, target_times):
        self.calls += 1
        return self.rng.uniform(0, 1, len(target_times))

    def observe(self, times, realized):
        self.observed.append((times[0], realized[0]))


def test_forecasts_are_fetched_once_per_decision():
    forecaster = _CountingStochasticForecaster()
    simulator = _simulator(generation_forecaster=forecaster)
    snapshot = simulator.reset(TIMES[1])
    assert forecaster.calls == 1
    assert simulator.observe() is snapshot
    np.testing.assert_array_equal(simulator.observe().generation_forecast,
                                  snapshot.generation_forecast)
    result = simulator.step(0.0)
    assert result.decision is snapshot
    assert forecaster.calls == 2


def test_learning_forecaster_only_sees_realized_past():
    forecaster = _CountingStochasticForecaster()
    simulator = _simulator(load_forecaster=forecaster)
    simulator.reset(TIMES[1])
    for k in range(5):
        decision_time = simulator.current_time
        simulator.step(0.0)
        seen_time, seen_value = forecaster.observed[-1]
        assert seen_time == decision_time < simulator.current_time
        assert seen_value == simulator.ground_truth("load", decision_time)[0]
    assert len(forecaster.observed) == 5


def test_swapping_a_forecaster_takes_effect_from_the_next_decision():
    simulator = _simulator()
    simulator.reset(TIMES[1])
    simulator.prosumer.generator.set_forecaster(
        NoisyGroundTruthForecaster(ConstantDynamics(7.0))
    )
    assert simulator.observe().generation_forecast[0] == 0.0
    assert simulator.step(0.0).next_observation.generation_forecast[0] == 7.0


def test_horizon_limits_and_episode_end_truncate_forecasts():
    simulator = _simulator(horizon=12)
    snapshot = simulator.reset(TIMES[1])
    assert len(snapshot.load_forecast) == 12
    for _ in range(40):
        snapshot = simulator.step(0.0).next_observation
    assert snapshot.steps_remaining == 8 and len(snapshot.load_forecast) == 8


def test_snapshot_arrays_are_read_only():
    snapshot = _simulator().reset(TIMES[1])
    with pytest.raises(ValueError):
        snapshot.load_forecast[0] = 1.0


# --------------------------------------------------- generation & export

def test_surplus_generation_is_exported_at_the_sell_tariff():
    load = np.full(len(TIMES), 1.0)
    generation = np.full(len(TIMES), 5.0)

    class _Spread(PriceCurveStrategy):
        requires_daily_observation = False

        def calculate_prosumer_tariffs(self, observation=None):
            return np.full(48, 0.30), np.full(48, 0.10)

    simulator = _simulator(load=load, generation=generation, strategy=_Spread(),
                           battery_config={"min": 0.0, "max": 100.0, "init": 99.0})
    simulator.reset(TIMES[1])
    result = simulator.step(0.0)
    t = result.transition
    assert t.surplus_to_storage == pytest.approx(1.0)
    assert t.exported_surplus == pytest.approx(3.0)
    assert result.money == pytest.approx(0.10 * 3.0)
    assert result.to_info()["exported_surplus_units"] == pytest.approx(3.0)
    assert result.metrics.total_generation == 5.0


# --------------------------------------------------- tariffs

class _BadStrategy(PriceCurveStrategy):
    requires_daily_observation = False

    def __init__(self, buy, sell):
        self.buy, self.sell = buy, sell

    def calculate_prosumer_tariffs(self, observation=None):
        return self.buy, self.sell


@pytest.mark.parametrize("buy, sell, message", [
    (np.full(48, np.nan), np.full(48, 0.1), "finite"),
    (np.full(48, -0.1), np.full(48, -0.2), "nonnegative"),
    (np.full(48, 0.1), np.full(48, 0.2), "sell tariff"),
    (np.full(24, 0.1), np.full(24, 0.1), "48-slot"),
])
def test_invalid_tariffs_are_rejected_without_side_effects(buy, sell, message):
    simulator = _simulator()
    simulator.reset(TIMES[1])
    before = (simulator.price_strategy, simulator.tariffs.cached_days(), simulator.observe())
    with pytest.raises(ValueError, match=message):
        simulator.set_price_strategy(_BadStrategy(buy, sell))
    assert (simulator.price_strategy, simulator.tariffs.cached_days(), simulator.observe()) == before


def test_set_price_strategy_refreshes_the_decision_atomically():
    simulator = _simulator(strategy=FixedPriceStrategy(0.10))
    assert simulator.set_price_strategy(FixedPriceStrategy(0.20)) is False
    simulator.reset(TIMES[1])
    old = simulator.observe()
    assert simulator.set_price_strategy(FixedPriceStrategy(0.30)) is True
    # FixedPriceStrategy produces float32 curves.
    assert old.buy_tariff == pytest.approx(0.20)
    assert simulator.observe().buy_tariff == pytest.approx(0.30)
    with pytest.raises(TypeError):
        simulator.set_price_strategy("cheap")


def test_data_driven_strategy_priced_once_per_day_at_reset_only():
    strategy = RecordingDataDrivenStrategy()
    simulator = _simulator(strategy=strategy)
    simulator.reset(TIMES[48 + 24])  # crosses midnight
    assert len(strategy.received) == 2
    for _ in range(48):
        simulator.step(0.0)
    assert len(strategy.received) == 2
    assert strategy.received[0].shape == (48 + 48 * 7,)


def test_data_driven_strategy_needs_full_day_forecasts():
    forecaster = TableForecaster(
        pd.DataFrame({"predicted_consumption": 3.0}, index=TIMES[:48 * 2 + 30]),
        "predicted_consumption",
    )
    simulator = _simulator(strategy=RecordingDataDrivenStrategy(), load_forecaster=forecaster)
    # Forecasts cover day 2 entirely but day 3 only until 14:30.
    assert simulator.episode_error(TIMES[48]) is None
    # This episode's own forecasts all exist (it ends at day 3, 11:30),
    # but pricing day 3 needs that whole day's forecasts.
    assert all(t in TIMES[:126] for t in simulator.episode_times(TIMES[72]))
    assert simulator.episode_error(TIMES[72]) == (
        "Missing forecast/calendar timestamp: 2025-01-03 15:00:00"
    )


# --------------------------------------------------- rewards & calendar

def test_reward_variants_and_validation():
    simulator = _simulator(load=np.full(len(TIMES), 3.0))
    simulator.reset(TIMES[1])
    result = simulator.step(0.0)  # empty battery: the grid supplies the load
    assert result.grid_supplied
    plain = make_reward()(result)
    shaped = make_reward(True, 2.0)(result)
    assert plain.total == result.money
    assert shaped.total == result.money - 2.0
    assert shaped.components == {"money": result.money, "grid_reliance": -2.0}
    with pytest.raises(ValueError, match="nonnegative"):
        make_reward(True, -1.0)


def test_calendar_features_match_the_prediction_package():
    from prediction import default_time_features

    stamps = pd.date_range("2024-12-28", periods=400, freq="30min")
    frame = default_time_features(pd.DataFrame({"Datetime": stamps}))
    np.testing.assert_allclose(
        calendar_features(stamps),
        frame[list(CALENDAR_FEATURE_COLUMNS)].to_numpy(dtype=float),
    )
