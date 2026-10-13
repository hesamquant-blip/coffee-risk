"""Rial cost of a purchase strategy on simulated paths, and the margin.

All rial flows are valued at the payment week T. Everything is vectorised
over the paths; the same functions price bootstrap paths, a stress path
(n_paths = 1) or a historical window in the backtest.

Limit stated in the report: the official channel is modelled as the free rate
times (1 - channel_discount), with the same dynamics. A managed rate that
moves in steps is not modelled.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from coffee_risk.engine.inputs import WEEKS_PER_YEAR, Inputs, Strategy
from coffee_risk.engine.simulator import Paths

DEMAND_STREAM = 7919          # keeps the demand draws apart from the path draws


def coffee_factor(paths: Paths, inputs: Inputs) -> np.ndarray:
    """USD needed at T as a multiple of usd_amount.

    fixed     1: the invoice was priced at signing.
    floating  (1 - p) * (w * A_T/A_0 + (1 - w) * R_T/R_0) + p: the futures
              part of the invoice moves with the two coffees, the physical
              premium p (a share of the invoice) was fixed at signing.
    """
    T = inputs.payment_week
    if inputs.contract == "fixed":
        return np.ones(paths.n_paths)
    w, p = inputs.arabica_share, inputs.premium_share
    return (1.0 - p) * (w * paths.arabica[:, T] + (1.0 - w) * paths.robusta[:, T]) + p


def usd_needed(paths: Paths, inputs: Inputs) -> np.ndarray:
    return inputs.usd_amount * coffee_factor(paths, inputs)


def cost_on_paths(paths: Paths, strategy: Strategy, inputs: Inputs) -> np.ndarray:
    """Rial cost at T of paying the invoice, per path.

    Each tranche buys its planned share of usd_amount at week t at
    S_t * (1 - discount). A rial spent at week t costs (1 + funds_rate)^((T-t)/52)
    at T, plus fx_holding_cost * (T - t)/52 of the rial value at purchase.
    At T the shortfall is bought at S_T * (1 - discount); an excess is sold at
    S_T * (1 - discount) * (1 - sellback_spread).

    On a realised path that carries `paths.funds` (the value of one rial of
    funds week by week), the rial spent at week t grows by what the funds
    actually earned from t to T, not by the constant rate of the inputs.
    """
    T = inputs.payment_week
    if T > paths.horizon:
        raise ValueError(f"payment week {T} is beyond the simulated horizon {paths.horizon}")
    rate = inputs.s0 * (1.0 - inputs.channel_discount) * paths.usd      # rial per USD, each week
    funds = inputs.cost_of_funds
    cost = np.zeros(paths.n_paths)
    bought = 0.0
    for week, share in strategy.tranches:
        if not 0 <= week < T:
            raise ValueError(f"tranche week {week} must be in 0..{T - 1}")
        usd = share * inputs.usd_amount
        years = (T - week) / WEEKS_PER_YEAR
        grown = (1.0 + funds) ** years if paths.funds is None else paths.funds[:, T] / paths.funds[:, week]
        cost += usd * rate[:, week] * (grown + inputs.fx_holding_cost * years)
        bought += usd
    gap = usd_needed(paths, inputs) - bought                # > 0: still to buy; < 0: to sell back
    cost += np.where(gap >= 0, gap, gap * (1.0 - inputs.sellback_spread)) * rate[:, T]
    return cost


# ------------------------------------------------------------ revenue, margin
def planned_cost(paths: Paths, inputs: Inputs) -> float:
    """C_plan: the invoice at the expected rate of the payment week."""
    g_T = float(paths.growth[inputs.payment_week])
    return inputs.usd_amount * inputs.s0 * (1.0 - inputs.channel_discount) * g_T


def demand_normals(n: int, seed: int = 0) -> np.ndarray:
    """Standard-normal draws for the demand factor, from a stream of their own
    so that they are independent of the price paths."""
    return np.random.default_rng([DEMAND_STREAM, int(seed)]).standard_normal(n)


def demand_factor(n: int, sd: float, seed: int = 0, z: Optional[np.ndarray] = None) -> np.ndarray:
    """Lognormal factor with mean 1 and standard deviation `sd`:
    D = exp(-s2 / 2 + sqrt(s2) * z), s2 = ln(1 + sd^2), z standard normal
    (given, or drawn from the demand stream of `seed`)."""
    if sd <= 0:
        return np.ones(n)
    s2 = np.log1p(sd ** 2)
    z = demand_normals(n, seed) if z is None else np.asarray(z, dtype=float)
    return np.exp(-0.5 * s2 + np.sqrt(s2) * z)


def fx_surprise(paths: Paths, inputs: Inputs) -> np.ndarray:
    """The unexpected USD move that reaches selling prices: S_T / (S_0 * g_T) - 1,
    and in the "up_only" mode never below zero (D-043: prices rise with an
    unexpected increase but do not fall when USD ends below its expected level)."""
    T = inputs.payment_week
    surprise = paths.usd[:, T] / paths.growth[T] - 1.0
    return np.maximum(surprise, 0.0) if inputs.pass_through_mode == "up_only" else surprise


def revenue(paths: Paths, inputs: Inputs, seed: Optional[int] = None) -> np.ndarray:
    """R = R_plan * (1 + pass_through * surprise) * D, with
    R_plan = C_plan / (1 - planned_margin) and the surprise of `fx_surprise`."""
    r_plan = planned_cost(paths, inputs) / (1.0 - inputs.planned_margin)
    surprise = fx_surprise(paths, inputs)
    if seed is None:
        seed = int(paths.info.get("seed", 0))
    return r_plan * (1.0 + inputs.pass_through * surprise) * demand_factor(
        paths.n_paths, inputs.demand_sd, seed, paths.demand_z)


def margin(cost: np.ndarray, rev: np.ndarray) -> np.ndarray:
    return 1.0 - cost / rev
