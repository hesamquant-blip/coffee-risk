"""Stress scenarios: replay a real crisis path from today's position.

"Replay from today" means that the window's actual weekly returns (USD;
roll-excluded coffee, D-022) are applied to today's rate and today's coffee
prices. The funds rate, the premium and the plan are today's. Nothing is
simulated: a scenario is one path, priced by the same cost functions as the
simulated paths.

The reverse stress test asks the opposite question: how far must USD rise by
the payment week before the margin falls below the floor?
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from coffee_risk import eda
from coffee_risk.engine import cost as costs
from coffee_risk.engine.inputs import WEEKS_PER_YEAR, Inputs, Strategy
from coffee_risk.engine.simulator import RETURN_COLS, Paths, target_growth

SERIES = ("usd", "arabica", "robusta")


@dataclass(frozen=True)
class Scenario:
    key: str
    returns: pd.DataFrame                    # one row per week, columns usd, arabica, robusta (log returns)
    usd_window: tuple                        # (first Wednesday, last Wednesday) the USD path comes from
    coffee_window: tuple                     # the same for the two coffees
    hypothetical: bool = False               # True when the USD and coffee windows differ

    @property
    def horizon(self) -> int:
        return len(self.returns)

    def change(self, name: str) -> float:
        """Simple change of one series over the whole scenario."""
        return float(np.expm1(self.returns[name].sum()))

    def mix_change(self, arabica_share: float = 0.5) -> float:
        """Simple change of the coffee basket in USD: w * A_T/A_0 + (1 - w) * R_T/R_0 - 1."""
        a, r = np.exp(self.returns["arabica"].sum()), np.exp(self.returns["robusta"].sum())
        return float(arabica_share * a + (1.0 - arabica_share) * r - 1.0)


# ------------------------------------------------------------------ windows
def window_returns(weekly: pd.DataFrame, start, end, names: Sequence[str] = SERIES) -> pd.DataFrame:
    """Weekly log returns of the weeks after `start` up to and including `end`.
    Raises if a Wednesday is missing from the panel or a return is empty: a
    stress path is never filled."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    rows = weekly.loc[(weekly.index > start) & (weekly.index <= end), [RETURN_COLS[n] for n in names]]
    rows.columns = list(names)
    weeks = int(round((end - start).days / 7))
    if len(rows) != weeks or rows.isna().any().any():
        raise ValueError(f"window {start.date()} .. {end.date()}: {weeks} weeks expected, {len(rows)} in the "
                         f"panel, {int(rows.isna().any(axis=1).sum())} with an empty return")
    return rows


def stress_windows(weekly: pd.DataFrame) -> dict:
    """The windows of D-027: for each D-005 episode the 13-week window with the
    largest rise of the 50/50 rial cost of coffee (exactly as in section 7 of
    the D2 report), and the full war path from its first week to the last week
    of the panel. key -> (first Wednesday, last Wednesday)."""
    idx, stale, gaps = eda.log_index(weekly)
    ch = eda.horizon_changes(idx, stale, gaps)
    out = {}
    for key, _kind, first, last in eda.EPISODES:
        end = eda.best_window(ch["cost_mix"], first, last)
        out[key] = (eda.window_start(end), end)
    out["war_full"] = (pd.Timestamp(eda.WAR_START_WEEK), weekly.index[-1])
    return out


def scenarios(weekly: pd.DataFrame) -> list[Scenario]:
    """S1..S7 of spec D4."""
    w = stress_windows(weekly)
    out = []
    for key in ("fx_1397", "fx_1401_1402", "coffee_2021", "coffee_2024", "war", "war_full"):
        out.append(Scenario(key, window_returns(weekly, *w[key]), w[key], w[key]))
    usd = window_returns(weekly, *w["fx_1397"], names=("usd",))
    coffee = window_returns(weekly, *w["coffee_2024"], names=("arabica", "robusta"))
    storm = pd.DataFrame({"usd": usd["usd"].values, "arabica": coffee["arabica"].values,
                          "robusta": coffee["robusta"].values})
    out.append(Scenario("perfect_storm", storm, w["fx_1397"], w["coffee_2024"], hypothetical=True))
    return out


# -------------------------------------------------------------------- replay
def replay(returns: pd.DataFrame, inputs: Inputs) -> Paths:
    """One path that starts at today's levels (ratio 1) and follows `returns`.
    The plan grows as the inputs say (neutral: at today's market rate)."""
    horizon = len(returns)
    level = {n: np.concatenate([[1.0], np.exp(np.cumsum(returns[n].values.astype(float)))])[None, :]
             for n in SERIES}
    growth = target_growth(inputs.drift, horizon, inputs.market_rate)
    return Paths(usd=level["usd"], arabica=level["arabica"], robusta=level["robusta"], growth=growth,
                 info={"seed": 0, "replay": True})


def outcomes(paths: Paths, inputs: Inputs, strategies: Sequence[Strategy]) -> pd.DataFrame:
    """Cost, margin and floor breach of each strategy on a single path."""
    c_plan = costs.planned_cost(paths, inputs)
    rev = costs.revenue(paths, inputs)
    rows = {}
    for s in strategies:
        cost = costs.cost_on_paths(paths, s, inputs)
        m = costs.margin(cost, rev)
        rows[s.name] = {"share_at_week0": s.share_at_week0, "cost": float(cost[0]),
                        "cost_pct": float(cost[0] / c_plan), "margin": float(m[0]),
                        "breaks": bool(m[0] < inputs.margin_floor)}
    return pd.DataFrame.from_dict(rows, orient="index")


# ------------------------------------------------------------ reverse stress
def _reverse_terms(inputs: Inputs, growth_T: Optional[float]) -> tuple[float, float, float]:
    """(g, F, L) in units of usd_amount * s0 * (1 - discount):
    g  expected USD level at the payment week (the plan),
    F  what one unit bought at week 0 has cost by the payment week,
    L  cost allowed per unit of plan before the margin falls below the floor."""
    years = inputs.payment_week / WEEKS_PER_YEAR
    g = (1.0 + inputs.market_rate) ** years if growth_T is None else float(growth_T)
    F = (1.0 + inputs.cost_of_funds) ** years + inputs.fx_holding_cost * years
    L = (1.0 - inputs.margin_floor) / (1.0 - inputs.planned_margin)
    return g, F, L


def allowed_cost(inputs: Inputs, usd_level: float, growth_T: Optional[float] = None) -> float:
    """The highest cost (same units) that keeps the margin at the floor when
    USD ends at `usd_level` = S_T / S_0: L * g * (1 + pass_through * surprise),
    surprise = usd_level / g - 1, never below 0 in the up-only mode. Demand
    noise is left out: a stress test is one path."""
    g, _, L = _reverse_terms(inputs, growth_T)
    surprise = usd_level / g - 1.0
    if inputs.pass_through_mode == "up_only":
        surprise = max(surprise, 0.0)
    return L * g * (1.0 + inputs.pass_through * surprise)


def breaking_usd_level(inputs: Inputs, h: float, growth_T: Optional[float] = None) -> float:
    """The USD level S_T / S_0 at which the margin falls below the floor when
    the share h is bought at week 0 and the rest on the payment day.

    Fixed contract (or flat coffee). Demand noise is left out: a stress test
    is one path. With x = S_T / S_0:

        cost    = h * F + (1 - h) * x
        allowed = L * g * (1 + p * (x / g - 1))        p = pass-through

    No pass-through: the margin is below the floor for x > (L * g - h * F) / (1 - h).
    With funds = market rate and no holding cost, F = g and the rise above the
    expected level is (L - 1) / (1 - h): 0.2 / (1 - h) for a 25% plan and a
    10% floor. Returns inf for h = 1 (the cost is fixed) unless the floor is
    broken at any level (then 0).

    With pass-through the question is about a rise above the expected level g
    (there the two modes are the same). The result is the lowest x >= g at
    which the floor is broken: g itself if the plan already fails at the
    expected level; inf if 1 - h - L * p <= 0, because revenue then rises at
    least as fast as cost; otherwise (L * g * (1 - p) - h * F) / (1 - h - L * p).
    """
    g, F, L = _reverse_terms(inputs, growth_T)
    p = inputs.pass_through
    if p == 0.0:
        if h >= 1.0:
            return float("inf") if F <= L * g else 0.0
        return max((L * g - h * F) / (1.0 - h), 0.0)
    if h * F + (1.0 - h) * g > L * g:                   # broken at the expected level
        return g
    den = 1.0 - h - L * p
    if den <= 0.0:
        return float("inf")
    return (L * g * (1.0 - p) - h * F) / den


def surviving_share(inputs: Inputs, usd_level: float, growth_T: Optional[float] = None) -> Optional[float]:
    """The smallest share h bought at week 0 that keeps the margin at or above
    the floor if USD ends at `usd_level` = S_T / S_0: the stress floor of
    D-051. Fixed contract (or flat coffee), no demand noise.

        h * F + (1 - h) * x <= allowed(x)   =>   h >= (x - allowed) / (x - F)

    0 when no hedge is needed, None when even a full purchase at week 0 does
    not survive (F above the allowed cost).
    """
    _, F, _ = _reverse_terms(inputs, growth_T)
    allowed = allowed_cost(inputs, usd_level, growth_T)
    if usd_level <= allowed:
        return 0.0
    if F > allowed:
        return None
    return float(min((usd_level - allowed) / (usd_level - F), 1.0))


def historical_changes(weekly: pd.DataFrame, start, horizon: int = 13) -> pd.Series:
    """Simple USD change over `horizon` weeks for every weekly start from
    `start` whose outcome is in the panel (overlapping), indexed by start."""
    level = weekly["usd_free"]
    change = level.shift(-horizon) / level - 1.0
    return change[(change.index >= pd.Timestamp(start))].dropna()
