"""Rolling backtest on real history, past-only (D5).

Two questions:

  A. If the firm had followed each simple strategy every week, what would the
     invoice have cost and how often would the margin have broken the floor?
     No model is involved: realised USD levels, realised coffee returns
     (roll-excluded, D-022) and the fund's realised growth as the cost of
     rial funds.
  B. Did the model's promise hold? At each date the engine is run on data up
     to that date only; the strategy it picks is then priced on what really
     happened next.
  C. Calibration: how often reality fell outside the simulated 5%-95% range,
     split by volatility regime, with a width-aware score.

D5b adds the stress floor of D-051 as an illustration: the hedge is the
larger of the model's minimum hedge ratio and the share needed to survive the
largest 13-week USD rise seen up to that date.

One row per decision date. A date whose outcome cannot be built is kept in
the table with the reason; nothing is filled or dropped silently.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

from coffee_risk.dates import jalali_year
from coffee_risk.engine import cost as costs
from coffee_risk.engine import decision as dec
from coffee_risk.engine import stress
from coffee_risk.engine.calibration import BACKTEST_START, binomial_range, independent_windows
from coffee_risk.engine.inputs import (Inputs, Strategy, market_rate, now, partial, partial_family,
                                       payment_day, premium_by_coffee, staged, strategy_set)
from coffee_risk.engine.simulator import RETURN_COLS, Paths, simulate, target_growth

CONTRACTS = ("fixed", "floating")
METHODS = ("fhs", "plain")
MAIN = ("now", "staged_3", "payment_day")
LEVELS = (0.05, 0.95)
TARGETS = ("usd", "pd_fixed", "pd_floating")      # what the calibration compares with its simulated range


def fixed_strategies(payment_week: int = 13) -> list[Strategy]:
    """All today, staged, all on the payment day and the partial family in 10% steps."""
    return [now(), staged(3, payment_week), payment_day()] + partial_family()


# ----------------------------------------------------------- realised paths
def realized_paths(weekly: pd.DataFrame, t, inputs: Inputs, horizon: int = 13):
    """What really happened from week `t`: (Paths with one path, "") or
    (None, reason).

    usd      realised level ratios S_k / S_0
    coffee   exp of the summed weekly returns, roll days out, ICO weeks in (D-022)
    funds    the fund's index ratio: what one rial of idle cash became
    growth   the plan made at t: neutral drift at that week's market rate
    """
    t = pd.Timestamp(t)
    k0 = weekly.index.get_loc(t)
    if k0 + horizon >= len(weekly.index):
        return None, "outcome not known yet"
    rows = weekly.iloc[k0:k0 + horizon + 1]
    if rows.index[-1] != t + pd.Timedelta(weeks=horizon):
        return None, "a week is missing from the panel"
    usd = rows["usd_free"].values.astype(float)
    old = rows["usd_free_old"].fillna(False).astype(bool).values
    if np.isnan(usd).any() or old[0] or old[-1]:
        return None, "USD level missing or too old"
    coffee = rows[[RETURN_COLS["arabica"], RETURN_COLS["robusta"]]].values[1:].astype(float)
    if np.isnan(coffee).any():
        return None, "a coffee week has no return"
    fund = rows["fund_tr_index"].values.astype(float)
    if np.isnan(fund).any():
        return None, "fund level missing"
    if np.isnan(inputs.market_rate):
        return None, "no market rate"
    level = np.concatenate([np.zeros((1, 2)), np.cumsum(coffee, axis=0)])
    return Paths(usd=(usd / usd[0])[None, :], arabica=np.exp(level[:, 0])[None, :],
                 robusta=np.exp(level[:, 1])[None, :],
                 growth=target_growth(inputs.drift, horizon, inputs.market_rate),
                 info={"seed": 0, "realized": True}, funds=(fund / fund[0])[None, :]), ""


def realized_outcome(real: Paths, strategy: Strategy, inputs: Inputs) -> tuple[float, float]:
    """(cost as a share of the plan, margin) of one strategy on a realised path."""
    cost = costs.cost_on_paths(real, strategy, inputs)
    m = costs.margin(cost, costs.revenue(real, inputs))
    return float(cost[0] / costs.planned_cost(real, inputs)), float(m[0])


# ------------------------------------------------------------- the backtest
def inputs_at(weekly: pd.DataFrame, t, daily: Optional[pd.DataFrame], imf: Optional[pd.DataFrame],
              premium_share: Optional[float] = None) -> tuple[Inputs, int]:
    """Base inputs as of `t` (s0, market rate, premium with the IMF lag), and
    the number of months behind the premium (0 when it is given)."""
    t = pd.Timestamp(t)
    if premium_share is not None:
        return Inputs(s0=float(weekly.loc[t, "usd_free"]), market_rate=market_rate(weekly, t),
                      premium_share=premium_share), 0
    prem = premium_by_coffee(daily, imf, t)
    months = min(prem["arabica"]["months"], prem["robusta"]["months"])
    return Inputs.from_data(t, weekly, daily, imf), months


def run_backtest(weekly: pd.DataFrame, daily: Optional[pd.DataFrame] = None,
                 imf: Optional[pd.DataFrame] = None, n_paths: int = 2000, seed: int = 0,
                 horizon: int = 13, start=BACKTEST_START, end=None,
                 methods: Sequence[str] = METHODS, premium_share: Optional[float] = None) -> pd.DataFrame:
    """One row per Wednesday from `start` (to `end`, or to the last one whose
    outcome is known).

    Columns (c = fixed | floating, s = strategy name, m = fhs | plain):
      usable, reason, end, jalali_year, s0, market_rate, premium_share, plan_growth, sigma
      usd_change, coffee_need, fund_growth          what happened
      c.s.cost, c.s.margin                          realised, cost as a share of the plan
      m.h_star, m.h_promise                         minimum hedge ratio (fixed) and its simulated P(margin >= floor)
      m.fixed.cost, m.fixed.margin                  realised outcome of buying h_star today
      m.floating.choice / .feasible / .promise      the floating decision; if none qualifies, the
      m.floating.cost, m.floating.margin            highest-probability strategy, marked infeasible
      m.<target>.q05 / .q95 / .real                 simulated range and realised value; targets: usd
                                                    (simple change), pd_fixed, pd_floating (cost of
                                                    "all on the payment day" as a share of the plan)
      stress.max_rise, stress.floor                 D-051: the largest `horizon`-week USD rise in the data
      fhs.floor_hedge, fhs.floor.cost / .margin     up to the date, the share that survives it (fixed
                                                    contract), and max(h_star, that share) with its
                                                    realised outcome
    Nothing after a date enters the decision at that date; only the realised
    columns look ahead.
    """
    index = weekly.index
    late = payment_day()
    by_name = {s.name: s for s in strategy_set(horizon)}
    fund_weeks = sorted({w for w, _ in staged(3, horizon).tranches} | {0, horizon})
    fund_old = (weekly["fund_tr_index_old"].fillna(False).astype(bool) if "fund_tr_index_old" in weekly
                else pd.Series(False, index=index))
    # the largest rise over `horizon` weeks among the windows that end on or before each week
    max_rise = (weekly["usd_free"] / weekly["usd_free"].shift(horizon) - 1.0).cummax()
    rows = []
    for k, t in enumerate(index):
        if t < pd.Timestamp(start) or (end is not None and t > pd.Timestamp(end)):
            continue
        if end is None and k + horizon >= len(index):
            continue
        row = {"start": t, "end": t + pd.Timedelta(weeks=horizon), "jalali_year": jalali_year(t.date())}
        base, months = inputs_at(weekly, t, daily, imf, premium_share)
        inputs = {"floating": base.with_(payment_week=horizon),
                  "fixed": base.with_(payment_week=horizon, contract="fixed")}
        real, reason = realized_paths(weekly, t, base, horizon)
        row.update({"usable": real is not None, "reason": reason, "s0": base.s0,
                    "market_rate": base.market_rate, "premium_share": base.premium_share,
                    "premium_months": months})
        if real is None:
            rows.append(row)
            continue
        row.update({"plan_growth": float(real.growth[horizon]), "usd_change": float(real.usd[0, horizon] - 1.0),
                    "coffee_need": float(costs.coffee_factor(real, inputs["floating"])[0]),
                    "fund_growth": float(real.funds[0, horizon]),
                    # a purchase week or the payment week falls on a fund level older than 5 days
                    # (market holidays): the last published level is used, and the date is counted
                    "fund_level_old": bool(fund_old.iloc[[k + w for w in fund_weeks]].any())})
        # ---- A. fixed strategies, no model
        for c in CONTRACTS:
            for s in fixed_strategies(horizon):
                row[f"{c}.{s.name}.cost"], row[f"{c}.{s.name}.margin"] = realized_outcome(real, s, inputs[c])
        # ---- B and C. the model, on data up to t only
        for m in methods:
            opts = {c: inputs[c].with_(usd_vol=m) for c in CONTRACTS}
            sim_paths = simulate(t, n_paths, horizon, seed + k, opts["fixed"], weekly)
            if m == "fhs" or "sigma" not in row:
                row["sigma"] = sim_paths.info.get("sigma_as_of_raw", np.nan)
            c_plan = costs.planned_cost(sim_paths, opts["fixed"])
            curve = dec.hedge_curve(sim_paths, opts["fixed"])
            h = dec.minimum_hedge_ratio(curve, opts["fixed"])
            row[f"{m}.h_star"] = np.nan if h is None else h
            if h is not None:
                row[f"{m}.h_promise"] = float(1.0 - curve.loc[h, "p_margin_below_floor"])
                row[f"{m}.fixed.cost"], row[f"{m}.fixed.margin"] = realized_outcome(real, partial(h), inputs["fixed"])
            if m == "fhs" and h is not None:
                # D-051: never hedge less than what survives the worst rise seen so far
                rise = float(max_rise.loc[t]) if not np.isnan(max_rise.loc[t]) else 0.0
                share = stress.surviving_share(inputs["fixed"], 1.0 + max(rise, 0.0))
                share = 1.0 if share is None else share           # nothing survives it: buy everything
                hedge = max(h, share)
                row.update({"stress.max_rise": rise, "stress.floor": share, "fhs.floor_hedge": hedge})
                row["fhs.floor.cost"], row["fhs.floor.margin"] = realized_outcome(real, partial(hedge), inputs["fixed"])
            table = dec.evaluate(sim_paths, opts["floating"])
            d = dec.decide(table, opts["floating"], c_plan)
            choice = d.chosen if d.qualifies else d.best_probability_strategy
            row[f"{m}.floating.choice"] = choice
            row[f"{m}.floating.feasible"] = bool(d.qualifies)
            row[f"{m}.floating.promise"] = float(1.0 - table.loc[choice, "p_margin_below_floor"])
            row[f"{m}.floating.cost"], row[f"{m}.floating.margin"] = \
                realized_outcome(real, by_name[choice], inputs["floating"])
            sims = {"usd": sim_paths.usd[:, horizon] - 1.0,
                    "pd_fixed": costs.cost_on_paths(sim_paths, late, opts["fixed"]) / c_plan,
                    "pd_floating": costs.cost_on_paths(sim_paths, late, opts["floating"]) / c_plan}
            reals = {"usd": row["usd_change"], "pd_fixed": row["fixed.payment_day.cost"],
                     "pd_floating": row["floating.payment_day.cost"]}
            for target in TARGETS:
                lo, hi = np.quantile(sims[target], LEVELS)
                row.update({f"{m}.{target}.q05": float(lo), f"{m}.{target}.q95": float(hi),
                            f"{m}.{target}.real": float(reals[target])})
        rows.append(row)
    return pd.DataFrame(rows).set_index("start")


def scenario_floor(bt: pd.DataFrame, weekly: pd.DataFrame, rise: float, available_from=None,
                   horizon: int = 13) -> pd.DataFrame:
    """D-051 with one fixed scenario instead of "the largest rise seen so
    far": on the fixed contract, hedge = max(fhs.h_star, the share that
    survives a USD rise of `rise` over the horizon, with the inputs of the
    date).

    Before `available_from` the scenario has not happened yet: the floor is
    zero there and the hedge is the model's alone. Without `available_from`
    the scenario is applied at every date, which looks ahead for the dates
    before it happened. One row per usable date: floor, hedge, cost (share
    of the plan) and margin, priced on the realised path like everything else.
    """
    rows = {}
    for t, r in bt[bt["usable"]].iterrows():
        fixed = Inputs(s0=float(r["s0"]), market_rate=float(r["market_rate"]),
                       premium_share=float(r["premium_share"]), contract="fixed", payment_week=horizon)
        share = stress.surviving_share(fixed, 1.0 + rise)
        share = 1.0 if share is None else share
        if available_from is not None and t < pd.Timestamp(available_from):
            share = 0.0
        hedge = max(float(r["fhs.h_star"]), share)
        real, reason = realized_paths(weekly, t, fixed, horizon)
        if real is None:
            raise ValueError(f"{t.date()}: marked usable but {reason}")
        cost, margin = realized_outcome(real, partial(hedge), fixed)
        rows[t] = {"floor": share, "hedge": hedge, "cost": cost, "margin": margin}
    return pd.DataFrame.from_dict(rows, orient="index")


# ---------------------------------------------------------------- summaries
def tail_mean(x, confidence: float = 0.95) -> float:
    return dec.tail_mean(np.asarray(x, dtype=float), confidence)


def strategy_summary(bt: pd.DataFrame, contract: str, floor: float, names: Sequence[str]) -> pd.DataFrame:
    """Per strategy: mean and median realised cost (share of plan), CVaR95 of
    the realised costs, and the share of dates with the margin below the floor."""
    ok = bt[bt["usable"]]
    rows = {}
    for n in names:
        cost, margin = ok[f"{contract}.{n}.cost"], ok[f"{contract}.{n}.margin"]
        rows[n] = {"mean_cost": float(cost.mean()), "median_cost": float(cost.median()),
                   "cvar": tail_mean(cost), "breach": float((margin < floor).mean()),
                   "worst_margin": float(margin.min())}
    return pd.DataFrame.from_dict(rows, orient="index")


def cheapest_share(bt: pd.DataFrame, contract: str, names: Sequence[str] = MAIN) -> pd.Series:
    """How often each of `names` was the cheapest of them (ties go to the first)."""
    ok = bt[bt["usable"]]
    cost = ok[[f"{contract}.{n}.cost" for n in names]]
    best = cost.values.argmin(axis=1)
    return pd.Series({n: float((best == i).mean()) for i, n in enumerate(names)})


def by_year(bt: pd.DataFrame, columns: dict, floor: float, horizon: int = 13) -> pd.DataFrame:
    """Breach rate by Jalali year of the decision date and overall (row
    'all'). `columns`: label -> margin column."""
    ok = bt[bt["usable"]]

    def one(part):
        out = {"starts": len(part), "independent": independent_windows(part.index, horizon)}
        for label, col in columns.items():
            out[label] = float((part[col] < floor).mean())
        return out

    rows = {int(y): one(part) for y, part in ok.groupby("jalali_year")}
    rows["all"] = one(ok)
    return pd.DataFrame.from_dict(rows, orient="index")


def regime(bt: pd.DataFrame) -> pd.Series:
    """'high' where the EWMA USD volatility at the decision date is above the
    median of its values over all usable start dates, else 'low' (D-045; the
    boundary was fixed before any result was seen)."""
    ok = bt[bt["usable"]]
    return pd.Series(np.where(ok["sigma"] > ok["sigma"].median(), "high", "low"), index=ok.index)


def pinball(y, q, tau: float) -> np.ndarray:
    """Quantile loss: tau * (y - q) when y is above q, (1 - tau) * (q - y) when below."""
    d = np.asarray(y, dtype=float) - np.asarray(q, dtype=float)
    return np.where(d >= 0, tau * d, (tau - 1.0) * d)


def calibration(bt: pd.DataFrame, target: str, methods: Sequence[str] = METHODS,
                horizon: int = 13) -> pd.DataFrame:
    """Exceedance rates and pinball losses of one target, overall and by
    volatility regime. Index: (subset, method)."""
    ok = bt[bt["usable"]]
    reg = regime(bt)
    rows = {}
    for subset in ("all", "high", "low"):
        part = ok if subset == "all" else ok[(reg == subset).values]
        n_ind = independent_windows(part.index, horizon)
        lo, hi = binomial_range(n_ind) if n_ind else (np.nan, np.nan)
        for m in methods:
            real, q05, q95 = part[f"{m}.{target}.real"], part[f"{m}.{target}.q05"], part[f"{m}.{target}.q95"]
            rows[subset, m] = {"starts": len(part), "independent": n_ind, "band_lo": lo, "band_hi": hi,
                               "above": float((real > q95).mean()), "below": float((real < q05).mean()),
                               "pinball05": float(pinball(real, q05, 0.05).mean()),
                               "pinball95": float(pinball(real, q95, 0.95).mean()),
                               "width": float((q95 - q05).median())}
    out = pd.DataFrame.from_dict(rows, orient="index")
    out["pinball"] = (out["pinball05"] + out["pinball95"]) / 2.0
    out.index = pd.MultiIndex.from_tuples(out.index, names=["subset", "method"])
    return out
