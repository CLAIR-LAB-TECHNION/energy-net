"""Console rendering of a ProsumerEnv. Pure formatting; never changes state."""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence


@dataclass(frozen=True)
class RenderView:
    """Read-only facts the renderer prints."""

    step: int
    max_steps: int
    datetime: Any  # pandas Timestamp
    dt_days: float
    storage: float
    capacity: float
    consumption: float
    generation: float
    price: Optional[float]
    strategy: str
    last_action: Optional[float]
    display_action: Optional[float]
    action_bound: float
    total_money: float
    total_penalty: float
    total_reward: float
    shortage_count: int
    available_charge: float
    available_discharge: float
    features: Dict[str, float] = field(default_factory=dict)
    next_load_forecasts: Sequence[float] = ()


def _action_type(action: float, upper: bool) -> str:
    if action > 0:
        label = "Charge"
    elif action < 0:
        label = "Discharge"
    else:
        label = "Idle"
    return label.upper() if upper else label


def render_prosumer(view: RenderView, verbosity: int) -> dict:
    """
    Print the view at the given verbosity and return its data.

    Levels: 0 silent, 1 episode summary, 2 condensed step (default),
    3 detailed step, 4 debug.
    """
    price_text = "n/a" if view.price is None else f"{view.price:.4f}"
    data = {
        "step": view.step,
        "max_steps": view.max_steps,
        "datetime": view.datetime,
        "storage": view.storage,
        "consumption": view.consumption,
        "generation": view.generation,
        "price": view.price,
        "strategy": view.strategy,
        "last_action": view.last_action,
        "total_money": view.total_money,
        "total_penalty": view.total_penalty,
        "total_reward": view.total_reward,
        "shortage_count": view.shortage_count,
    }
    if view.features:
        data["features"] = dict(view.features)

    if verbosity <= 0:
        return data

    when = view.datetime
    if verbosity == 1:
        if view.step == 0:
            print(f"\n[Prosumer Episode Start] Date: {when.strftime('%Y-%m-%d')}")
        elif view.step >= view.max_steps - 1:
            print(f"\n{'=' * 60}")
            print(f"[Prosumer Episode Summary] {when.strftime('%Y-%m-%d')}")
            print(f"{'=' * 60}")
            print(f"Total Steps:           {view.step + 1}")
            print(f"Money Earned:          ${view.total_money:.2f}")
            print(f"Shortage Penalties:    ${view.total_penalty:.2f}")
            print(f"Net Reward:            ${view.total_reward:.2f}")
            print(f"Shortage Count:        {view.shortage_count}")
            print(f"Final Storage:         {view.storage:.2f} units")
            print(f"Strategy Used:         {view.strategy}")
            print(f"{'=' * 60}\n")
        return data

    if verbosity == 2:
        print(f"\n{'=' * 60}")
        print(f"[Prosumer Step {view.step}/{view.max_steps}] {when.strftime('%Y-%m-%d %H:%M')}")
        print(f"{'=' * 60}")
        print(f"Storage:               {view.storage:.2f} / {view.capacity} units")
        print(f"Consumption:           {view.consumption:.2f} units/step")
        print(f"Generation:            {view.generation:.2f} units/step")
        print(f"Price ({view.strategy}): ${price_text}/unit")
        if view.display_action is not None:
            print(f"Battery Action:        {view.display_action:.2f} units")
        print("--- Episode Running Totals ---")
        print(f"Money:     ${view.total_money:.2f}")
        print(f"Penalties: ${view.total_penalty:.2f}")
        print(f"Net:       ${view.total_reward:.2f}")
        print(f"Shortages: {view.shortage_count}")
        print(f"{'=' * 60}")
        return data

    percent = 100.0 * view.storage / view.capacity
    if verbosity == 3:
        print(f"\n{'=' * 60}")
        print(f"[Prosumer DETAILED] Step {view.step}/{view.max_steps}")
        print(f"{'=' * 60}")
        print(f"Date/Time:             {when.strftime('%Y-%m-%d %H:%M')}")
        print(f"Storage Level:         {view.storage:.2f} / {view.capacity} units ({percent:.1f}%)")
        print(f"Consumption (Actual):  {view.consumption:.2f} units/step")
        print(f"Generation (Actual):   {view.generation:.2f} units/step")
        print(f"Price ({view.strategy}): ${price_text}/unit")
        if view.last_action is not None:
            print(f"Battery Action:        {view.last_action:.2f} units "
                  f"({_action_type(view.last_action, upper=False)})")
        if view.features:
            print("\n--- Environmental Features ---")
            for name, value in view.features.items():
                print(f"  {name:<20}: {value:.4f}")
        print("\n--- Episode Cumulative Metrics ---")
        print(f"Gross Money Earned:    ${view.total_money:.2f}")
        print(f"Shortage Penalties:    ${view.total_penalty:.2f}")
        print(f"Net Reward:            ${view.total_reward:.2f}")
        print(f"Shortage Occurrences:  {view.shortage_count}")
        print(f"{'=' * 60}")
        return data

    # verbosity >= 4: debug
    data["available_charge"] = view.available_charge
    data["available_discharge"] = view.available_discharge
    print(f"\n{'=' * 80}")
    print(f"[Prosumer DEBUG MODE] Step {view.step}/{view.max_steps}")
    print(f"{'=' * 80}")
    print(f"Date/Time:             {when.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Relative Time:         {view.step * view.dt_days:.4f} days")
    print("\n--- Storage System ---")
    print(f"Storage Level:         {view.storage:.4f} / {view.capacity} units ({percent:.2f}%)")
    print(f"Available Charge Cap:  {view.available_charge:.4f} units")
    print(f"Available Discharge:   {view.available_discharge:.4f} units")
    print("\n--- Consumption, Generation & Pricing ---")
    print(f"Consumption (Actual):  {view.consumption:.4f} units/step")
    print(f"Generation (Actual):   {view.generation:.4f} units/step")
    print(f"Pricing Strategy:      {view.strategy}")
    print(f"Current Price:         ${'n/a' if view.price is None else f'{view.price:.6f}'}/unit")
    if view.last_action is not None:
        print("\n--- Agent Action ---")
        print(f"Raw Action Value:      {view.last_action:.4f} units")
        print(f"Action Type:           {_action_type(view.last_action, upper=True)}")
        print(f"Action Bounds:         [{-view.action_bound}, {view.action_bound}]")
    if view.features:
        print(f"\n--- Environmental Features (All {len(view.features)}) ---")
        for name, value in view.features.items():
            print(f"  {name:<25}: {value:.6f}")
    if len(view.next_load_forecasts):
        print(f"\n--- Consumption Predictions (Next {len(view.next_load_forecasts)} steps) ---")
        for offset, value in enumerate(view.next_load_forecasts, start=1):
            print(f"  Step +{offset}: {value:.4f} units")
    print("\n--- Financial Metrics (Cumulative) ---")
    print(f"Gross Money Earned:    ${view.total_money:.6f}")
    print(f"Shortage Penalties:    ${view.total_penalty:.6f}")
    print(f"Net Reward (Total):    ${view.total_reward:.6f}")
    print(f"Shortage Count:        {view.shortage_count}")
    if view.step > 0:
        print(f"Average Reward/Step:   ${view.total_reward / view.step:.6f}")
    print(f"{'=' * 80}")
    return data
