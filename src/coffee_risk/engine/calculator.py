"""The calculator's arithmetic, in Python, on the calculator's own files.

The web page (built by the author in JavaScript) reads
`calculator_paths.json` and `scenarios.json` and applies the steps of
`docs/calculator_spec.md`. This module applies the same steps to the same
files, with the engine's cost and decision functions, and its results are
written to `parity.json`: the page is right when it reproduces them.

Nothing here touches the price data: the only inputs are the two exported
files and the user's inputs.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from coffee_risk.engine import cost as costs
from coffee_risk.engine import decision as dec
from coffee_risk.engine import stress
from coffee_risk.engine.inputs import (WEEKS_PER_YEAR, Inputs, Strategy, now, partial, payment_day, staged)
from coffee_risk.engine.simulator import Paths, to_levels

SERIES = ("usd_z", "arabica", "robusta")


# ------------------------------------------------------------------- inputs
def kit_inputs(kit: dict, **changes) -> Inputs:
    """Inputs with the three data-based values taken from the file. The
    premium share defaults to the mix of the two premiums in the file."""
    w = float(changes.get("arabica_share", Inputs.arabica_share))
    values = {"s0": float(kit["s0"]), "market_rate": float(kit["market_rate"]),
              "premium_share": w * kit["premium"]["arabica"] + (1.0 - w) * kit["premium"]["robusta"]}
    values.update(changes)
    return Inputs(**values)


def growth(kit: dict, inputs: Inputs, horizon: Optional[int] = None) -> np.ndarray:
    """Target mean of S_k / S_0, k = 0..horizon, for the drift of the inputs."""
    k = np.arange((kit["horizon"] if horizon is None else horizon) + 1)
    if inputs.drift == "zero":
        return np.ones(len(k))
    if inputs.drift == "neutral":
        return (1.0 + inputs.market_rate) ** (k / WEEKS_PER_YEAR)
    if inputs.drift == "historical":
        return (1.0 + float(kit["hist_mean_weekly_usd_return"])) ** k
    return (1.0 + float(inputs.drift)) ** (k / WEEKS_PER_YEAR)


# -------------------------------------------------------------------- paths
def decode(kit: dict, name: str) -> np.ndarray:
    """One stored array as floats: integers times the scale, path by path."""
    a = np.asarray(kit[name], dtype=float) * float(kit["scale"])
    return a if name == "demand_z" else a.reshape(kit["n_paths"], kit["horizon"])


def kit_paths(kit: dict, inputs: Inputs) -> Paths:
    """The page's paths: rescale the USD residuals by the volatility, cumulate
    the weekly log returns, and scale each week so that the mean over the
    paths is the target (USD: the drift; each coffee: 1)."""
    target = growth(kit, inputs)
    ones = np.ones(kit["horizon"] + 1)
    usd_returns = decode(kit, "usd_z") * float(kit["usd_vol_weekly"]) * inputs.usd_vol_scale
    return Paths(usd=to_levels(usd_returns, target), arabica=to_levels(decode(kit, "arabica"), ones),
                 robusta=to_levels(decode(kit, "robusta"), ones), growth=target,
                 info={"seed": 0, "source": "calculator kit"}, demand_z=decode(kit, "demand_z"))


# --------------------------------------------------------------- strategies
def named_strategies(payment_week: int = 13) -> dict:
    """The strategies of parity.json: now, staged, payment_day and h_0 .. h_100."""
    out = {"now": now(), "staged": staged(3, payment_week), "payment_day": payment_day()}
    out.update({f"h_{10 * k}": partial(k / 10) for k in range(11)})
    return out


def _metrics(cost: np.ndarray, rev: np.ndarray, c_plan: float, inputs: Inputs) -> dict:
    m = dec.metrics(cost, rev, c_plan, inputs)
    return {"mean": m["mean_cost_pct"], "median": m["median_cost_pct"], "var95": m["var_pct"],
            "cvar95": m["cvar_pct"], "p_over_budget": m["p_over_budget"],
            "p_margin_below_floor": m["p_margin_below_floor"], "margin_q05": m["margin_q"]}


# ----------------------------------------------------------------- scenarios
def scenario_paths(kit: dict, scenario: dict, inputs: Inputs) -> Paths:
    """One path from the aggregates of scenarios.json: levels are known at the
    staged weeks and at the payment week only, which is all the strategies
    touch. `inputs.payment_week` must be the scenario's. The plan grows as the
    drift of the inputs says, up to the scenario's payment week."""
    T = int(scenario["payment_week"])
    if inputs.payment_week != T:
        raise ValueError("price a scenario with payment_week set to the scenario's")
    level = {}
    for name in ("usd", "arabica", "robusta"):
        x = np.full((1, T + 1), np.nan)
        x[0, 0] = 1.0
        for week, change in zip(scenario["weeks"], scenario[name]):
            x[0, week] = 1.0 + change
        level[name] = x
    return Paths(usd=level["usd"], arabica=level["arabica"], robusta=level["robusta"],
                 growth=growth(kit, inputs, T), info={"seed": 0, "scenario": scenario["key"]})


def scenario_outcomes(kit: dict, scenario: dict, inputs: Inputs) -> dict:
    """Cost (share of the scenario's plan), margin and breach of now / staged
    / payment_day on a scenario replayed from today. Demand noise is left
    out: a scenario is one path."""
    T = int(scenario["payment_week"])
    i = inputs.with_(payment_week=T, demand_sd=0.0)
    p = scenario_paths(kit, scenario, i)
    table = stress.outcomes(p, i, [now(), staged(3, T), payment_day()])
    return {name: {"cost": float(table.loc[key, "cost_pct"]), "margin": float(table.loc[key, "margin"]),
                   "breaks": bool(table.loc[key, "breaks"])}
            for name, key in (("now", "now"), ("staged", "staged_3"), ("payment_day", "payment_day"))}


def default_stress_scenario(scenarios: Sequence[dict]) -> Optional[dict]:
    """D-053: the page's default scenario for the stress floor is the largest
    real 13-week USD jump seen up to the decision date. From the scenario
    file: among the non-hypothetical scenarios with payment week 13, the one
    with the largest USD change at the payment week (the first on a tie)."""
    real = [s for s in scenarios if not s.get("hypothetical", False) and int(s["payment_week"]) == 13]
    return max(real, key=lambda s: s["usd"][-1]) if real else None


def stress_floor(kit: dict, scenario: dict, inputs: Inputs) -> Optional[float]:
    """D-051: the share to buy at week 0 so that the margin stays at the floor
    if USD ends where the scenario ends (fixed-contract formula: coffee flat,
    no demand noise). None when even a full purchase does not survive."""
    T = int(scenario["payment_week"])
    i = inputs.with_(payment_week=T)
    return stress.surviving_share(i, 1.0 + float(scenario["usd"][-1]), float(growth(kit, i, T)[T]))


# ---------------------------------------------------------------- one case
def evaluate(kit: dict, inputs: Inputs, scenarios: Sequence[dict] = (),
             floor_scenarios: Sequence[str] = ("S1", "S2"),
             reverse_shares: Sequence[float] = (0.0, 0.25, 0.5, 0.75)) -> dict:
    """Everything parity.json holds for one set of inputs. Costs are shares
    of the planned cost; probabilities are shares of the paths."""
    T = inputs.payment_week
    p = kit_paths(kit, inputs)
    c_plan = costs.planned_cost(p, inputs)
    rev = costs.revenue(p, inputs)
    out = {"plan_growth": float(p.growth[T]), "plan_rial": float(c_plan),
           "planned_revenue_rial": float(c_plan / (1.0 - inputs.planned_margin)),
           "strategies": {name: _metrics(costs.cost_on_paths(p, s, inputs), rev, c_plan, inputs)
                          for name, s in named_strategies(T).items()}}
    curve = dec.hedge_curve(p, inputs)
    out["min_hedge"] = dec.minimum_hedge_ratio(curve, inputs)
    table = dec.evaluate(p, inputs)
    d = dec.decide(table, inputs, c_plan)
    rename = lambda n: None if n is None else ("staged" if n.startswith("staged") else
                                               n.replace("partial_", "h_") if n.startswith("partial_") else n)
    out.update({"qualifies": bool(d.qualifies), "chosen": rename(d.chosen),
                "best_probability": d.best_probability, "best_probability_strategy": rename(d.best_probability_strategy),
                "highest_floor": d.highest_floor, "highest_floor_strategy": rename(d.highest_floor_strategy)})
    by_key = {s["key"]: s for s in scenarios}
    out["stress_floor"] = {k: stress_floor(kit, by_key[k], inputs) for k in floor_scenarios if k in by_key}
    out["reverse_stress"] = {}
    for h in reverse_shares:
        x = stress.breaking_usd_level(inputs, h, float(p.growth[T]))
        out["reverse_stress"][f"h_{int(round(h * 100))}"] = None if np.isinf(x) else float(x)
    out["scenarios"] = {s["key"]: scenario_outcomes(kit, s, inputs) for s in scenarios}
    return out


__all__ = ["kit_inputs", "kit_paths", "growth", "decode", "named_strategies", "scenario_paths",
           "scenario_outcomes", "stress_floor", "default_stress_scenario", "evaluate", "Strategy"]
