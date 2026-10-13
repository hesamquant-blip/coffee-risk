"""Risk metrics per strategy and the decision rule (D-035).

Rule: among all strategies, keep those with P(margin >= floor) >= confidence;
choose the lowest mean rial cost; tie-break on the lower CVaR. If none
qualifies, say so and report the best achievable probability and the highest
floor achievable at the chosen confidence.

Ties (D-037): mean costs that differ from the lowest by less than
`tie_tolerance` (0.25% of the planned cost) count as equal. Under the neutral
drift the means are equal by construction, and a difference below
Monte-Carlo noise should not pick the answer. The headline output is the
minimum hedge ratio; the chosen strategy is shown next to it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from coffee_risk.engine import cost as costs
from coffee_risk.engine.inputs import Inputs, Strategy, partial, strategy_set
from coffee_risk.engine.simulator import Paths

ROUNDING = 1e-9               # share of the plan: with tie_tolerance = 0, sums over paths still round
HEDGE_GRID = 101              # the minimum hedge ratio is searched on 0%, 1%, ..., 100%


def tail_mean(x: np.ndarray, confidence: float) -> float:
    """Mean of the worst (largest) 1 - confidence share of x: the CVaR."""
    x = np.sort(np.asarray(x, dtype=float))
    k = max(1, int(np.ceil(len(x) * (1.0 - confidence) - 1e-9)))
    return float(x[-k:].mean())


def metrics(cost: np.ndarray, rev: np.ndarray, c_plan: float, inputs: Inputs) -> dict:
    """Metrics of one strategy from its per-path cost and revenue."""
    n = len(cost)
    m = costs.margin(cost, rev)
    c = inputs.confidence
    below = m < inputs.margin_floor
    p_below = float(below.mean())
    return {
        "mean_cost": float(cost.mean()), "median_cost": float(np.median(cost)),
        "mean_cost_pct": float(cost.mean() / c_plan), "median_cost_pct": float(np.median(cost) / c_plan),
        "var": float(np.quantile(cost, c)), "cvar": tail_mean(cost, c),
        "var_pct": float(np.quantile(cost, c) / c_plan), "cvar_pct": tail_mean(cost, c) / c_plan,
        "p_over_budget": float((cost > inputs.budget_factor * c_plan).mean()),
        "margin_q": float(np.quantile(m, 1.0 - c)),          # the (1 - confidence) margin percentile
        "p_margin_below_floor": p_below,
        "se_mean_cost": float(cost.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0,
        "se_mean_cost_pct": float(cost.std(ddof=1) / np.sqrt(n) / c_plan) if n > 1 else 0.0,
        "se_p_margin_below_floor": float(np.sqrt(p_below * (1.0 - p_below) / n)),
        "n_paths": n,
    }


def evaluate(paths: Paths, inputs: Inputs,
             strategies: Optional[Sequence[Strategy]] = None) -> pd.DataFrame:
    """One row of metrics per strategy, in the order given."""
    if strategies is None:
        strategies = strategy_set(inputs.payment_week)
    c_plan = costs.planned_cost(paths, inputs)
    rev = costs.revenue(paths, inputs)
    rows = {}
    for s in strategies:
        row = metrics(costs.cost_on_paths(paths, s, inputs), rev, c_plan, inputs)
        row["share_at_week0"] = s.share_at_week0
        rows[s.name] = row
    return pd.DataFrame.from_dict(rows, orient="index")


@dataclass
class Decision:
    qualifies: bool
    chosen: Optional[str]                    # None when no strategy meets the constraint
    qualifying: list = field(default_factory=list)
    best_probability: float = np.nan         # highest P(margin >= floor) over the strategies
    best_probability_strategy: Optional[str] = None
    highest_floor: float = np.nan            # highest margin floor any strategy meets at the confidence
    highest_floor_strategy: Optional[str] = None


def decide(table: pd.DataFrame, inputs: Inputs, c_plan: float) -> Decision:
    """Apply D-035 and D-037 to a table from `evaluate` (or to hand-made
    metrics with the columns mean_cost, cvar, p_margin_below_floor, margin_q).
    `c_plan` is the planned cost, the unit of the tie tolerance."""
    p_ok = 1.0 - table["p_margin_below_floor"]
    best_p, best_floor = p_ok.idxmax(), table["margin_q"].idxmax()
    out = Decision(qualifies=False, chosen=None,
                   best_probability=float(p_ok[best_p]), best_probability_strategy=best_p,
                   highest_floor=float(table.loc[best_floor, "margin_q"]),
                   highest_floor_strategy=best_floor)
    ok = table[p_ok >= inputs.confidence - 1e-12]
    if ok.empty:
        return out
    gap = (ok["mean_cost"] - ok["mean_cost"].min()) / float(c_plan)
    tied = ok[gap < max(inputs.tie_tolerance, ROUNDING)]
    out.qualifies, out.qualifying = True, list(ok.index)
    out.chosen = tied["cvar"].idxmin()
    return out


def hedge_curve(paths: Paths, inputs: Inputs, n_grid: int = HEDGE_GRID) -> pd.DataFrame:
    """Metrics of the partial family on a fine grid: share h bought at week 0,
    the rest on the payment day. Index = h."""
    c_plan = costs.planned_cost(paths, inputs)
    rev = costs.revenue(paths, inputs)
    rows = {}
    for h in np.linspace(0.0, 1.0, n_grid):
        h = float(round(h, 6))
        rows[h] = metrics(costs.cost_on_paths(paths, partial(h), inputs), rev, c_plan, inputs)
    out = pd.DataFrame.from_dict(rows, orient="index")
    out.index.name = "share_at_week0"
    return out


def minimum_hedge_ratio(curve: pd.DataFrame, inputs: Inputs) -> Optional[float]:
    """Smallest share bought at week 0 (on the grid) whose P(margin >= floor)
    reaches the confidence; None when no share does."""
    ok = curve.index[(1.0 - curve["p_margin_below_floor"]) >= inputs.confidence - 1e-12]
    return float(ok.min()) if len(ok) else None


@dataclass
class Result:
    inputs: Inputs
    table: pd.DataFrame            # metrics per strategy
    decision: Decision
    curve: pd.DataFrame            # the partial family on a 1% grid
    min_hedge: Optional[float]
    c_plan: float
    info: dict


def run(paths: Paths, inputs: Inputs,
        strategies: Optional[Sequence[Strategy]] = None) -> Result:
    """Price the strategies on given paths and apply the rule."""
    table = evaluate(paths, inputs, strategies)
    c_plan = costs.planned_cost(paths, inputs)
    curve = hedge_curve(paths, inputs)
    return Result(inputs=inputs, table=table, decision=decide(table, inputs, c_plan),
                  curve=curve, min_hedge=minimum_hedge_ratio(curve, inputs),
                  c_plan=c_plan, info=dict(paths.info))
