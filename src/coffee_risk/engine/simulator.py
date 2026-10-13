"""Joint simulation of 13-week paths for USD/IRR, arabica and robusta.

Method (decisions D-024..D-029):

  pool      weeks up to and including the decision date that have a weekly
            return for all three series. The week ending on the decision date
            is in (D-038): its return is known on that date. No later week is
            ever touched.
  sampling  stationary bootstrap: each path starts at a random pool week and,
            every week, continues with the next pool week with probability
            1 - 1/mean_block or jumps to a new random week; the pool wraps
            around. The same sampled weeks are used for the three series, so
            their joint behaviour is kept.
  USD       filtered historical simulation: each pool week's USD return is
            divided by the volatility estimated just before that week (EWMA
            of earlier weeks only), the standardised series is demeaned, and
            a sampled value is multiplied by the volatility estimated on the
            decision date from all weeks up to it: the forecast for the first
            simulated week. The shape comes from history, the size from today.
            Option: plain bootstrap of the returns themselves.
  coffee    plain bootstrap of the roll-excluded weekly returns.
  drift     each week's simulated levels are scaled by one number so that the
            average of S_k/S_0 over the paths equals the target growth to
            week k, and the average of F_k/F_0 equals 1 for each coffee. This
            sets the mean exactly and leaves the shape untouched.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np
import pandas as pd

from coffee_risk.engine.inputs import WEEKS_PER_YEAR, Inputs

RETURN_COLS = {"usd": "usd_free_ret", "arabica": "arabica_usd_kg_ret",
               "robusta": "robusta_usd_kg_ret"}
MIN_HISTORY = 13          # earlier weeks needed before a volatility estimate exists
VOL_FLOOR_Q = 0.05        # past-only quantile used as the floor of the EWMA volatility


@dataclass
class Paths:
    """Simulated level ratios, shape (n_paths, horizon + 1); column 0 is 1."""
    usd: np.ndarray
    arabica: np.ndarray
    robusta: np.ndarray
    growth: np.ndarray                 # target mean of usd[:, k], k = 0..horizon
    info: dict = field(default_factory=dict)
    # Realised paths only: value of one rial of funds at week k (week 0 = 1).
    # When given, a rial spent at week k costs funds[:, T] / funds[:, k] at T
    # instead of growing at the constant funds rate of the inputs.
    funds: Optional[np.ndarray] = None
    # Standard-normal draws behind the demand factor, one per path. When given
    # (paths read back from the calculator's file) they are used as they are;
    # otherwise the draws come from the demand stream of the seed.
    demand_z: Optional[np.ndarray] = None

    @property
    def n_paths(self) -> int:
        return self.usd.shape[0]

    @property
    def horizon(self) -> int:
        return self.usd.shape[1] - 1


# ------------------------------------------------------------------ the pool
def weekly_returns(weekly: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({name: weekly[col] for name, col in RETURN_COLS.items()})


def pool_weeks(returns: pd.DataFrame, as_of) -> pd.DatetimeIndex:
    """Weeks up to and including `as_of` with a return for every series
    (D-024, D-038)."""
    ok = returns.notna().all(axis=1) & (returns.index <= pd.Timestamp(as_of))
    return returns.index[ok]


# ------------------------------------------------------------- EWMA volatility
def ewma_vol(r: pd.Series, half_life: float = 13.0, min_history: int = MIN_HISTORY) -> pd.Series:
    """Volatility estimate for each week from EARLIER weeks only.

    sigma_t^2 = weighted mean of r_j^2 over the weeks j < t that have a
    return, with weights halving every `half_life` weeks. No mean is removed
    (weekly means are small next to weekly moves). Empty until `min_history`
    earlier returns exist. The value at week t never uses r_t.
    """
    lam = 0.5 ** (1.0 / half_life)
    r2 = r.astype(float) ** 2
    # pandas skips empty weeks and keeps the weights tied to calendar distance
    upto = r2.ewm(alpha=1 - lam, adjust=True, ignore_na=False).mean()   # weeks <= t ...
    count = r.notna().cumsum()
    est = upto.shift(1)                                                 # ... so shift: weeks < t
    return np.sqrt(est.where(count.shift(1) >= min_history))


def floored_vol(sigma: pd.Series, q: float = VOL_FLOOR_Q) -> tuple[pd.Series, pd.Series]:
    """(volatility used, floor). The floor at week t is the q-quantile of the
    estimates up to week t; all of them use data before week t."""
    floor = sigma.expanding(min_periods=1).quantile(q)
    return sigma.clip(lower=floor), floor


# ------------------------------------------------------------------- sampling
def stationary_indices(rng: np.random.Generator, n_pool: int, n_paths: int, horizon: int,
                       mean_block: float) -> np.ndarray:
    """Pool positions for every path and week, shape (n_paths, horizon)."""
    idx = np.empty((n_paths, horizon), dtype=np.int64)
    idx[:, 0] = rng.integers(0, n_pool, n_paths)
    jump = rng.random((n_paths, horizon)) < 1.0 / mean_block
    fresh = rng.integers(0, n_pool, (n_paths, horizon))
    for k in range(1, horizon):
        idx[:, k] = np.where(jump[:, k], fresh[:, k], (idx[:, k - 1] + 1) % n_pool)
    return idx


# ---------------------------------------------------------------------- drift
def target_growth(drift: Union[str, float], horizon: int, market_rate: Optional[float],
                  pool_usd_returns: Optional[np.ndarray] = None) -> np.ndarray:
    """Target mean of S_k/S_0 for k = 0..horizon (D-028).

    neutral     rials growing at the market rate: (1 + market_rate)^(k/52)
    historical  the mean simple weekly USD return of the pool, compounded
    zero        1
    a number    an annual simple rate chosen by the user
    """
    k = np.arange(horizon + 1)
    if drift == "zero":
        return np.ones(horizon + 1)
    if drift == "neutral":
        if market_rate is None:
            raise ValueError("neutral drift needs market_rate")
        return (1.0 + market_rate) ** (k / WEEKS_PER_YEAR)
    if drift == "historical":
        if pool_usd_returns is None:
            raise ValueError("historical drift needs the pool's USD returns")
        return (1.0 + float(np.expm1(pool_usd_returns).mean())) ** k
    return (1.0 + float(drift)) ** (k / WEEKS_PER_YEAR)


def to_levels(log_returns: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Cumulate weekly log returns into level ratios and scale each week so
    the mean over the paths equals `target` (one scale number per week)."""
    levels = np.exp(np.cumsum(log_returns, axis=1))
    levels *= target[1:] / levels.mean(axis=0)
    return np.concatenate([np.ones((levels.shape[0], 1)), levels], axis=1)


# ------------------------------------------------------------------- simulate
def prepare(weekly: pd.DataFrame, as_of, options: Inputs) -> dict:
    """Everything that depends on the data and the decision date but not on
    the random draws: the pool, the returns to sample, the volatility facts."""
    as_of = pd.Timestamp(as_of)
    rets = weekly_returns(weekly)
    weeks = pool_weeks(rets, as_of)
    usd_all = rets["usd"]
    info = {"as_of": as_of, "usd_vol": options.usd_vol, "half_life": options.half_life,
            "mean_block": options.mean_block, "drift": options.drift}
    if options.usd_vol == "fhs":
        # One empty row for the week after the decision date: the estimate there
        # is made from every week up to the decision date and from nothing later.
        known = usd_all[usd_all.index <= as_of]
        ahead = as_of + pd.Timedelta(weeks=1)
        sigma = ewma_vol(pd.Series(np.append(known.values, np.nan), index=known.index.append(
            pd.DatetimeIndex([ahead]).astype(known.index.dtype))), options.half_life)
        used, floor = floored_vol(sigma)
        if np.isnan(sigma.loc[ahead]):
            raise ValueError(f"fewer than {MIN_HISTORY} USD returns up to {as_of.date()}")
        weeks = weeks[sigma.reindex(weeks).notna().values]            # drop the warm-up weeks
        r_pool, s_pool = usd_all.reindex(weeks).values, used.reindex(weeks).values
        z = np.divide(r_pool, s_pool, out=np.zeros(len(weeks)), where=s_pool > 0)
        z_mean = float(z.mean())
        sigma_now = float(used.loc[ahead])
        usd_z = z - z_mean                                            # what the calculator's file stores
        usd_sample = usd_z * sigma_now * options.usd_vol_scale
        hit = sigma.reindex(weeks) < floor.reindex(weeks)
        info.update({"sigma_as_of": sigma_now, "sigma_as_of_raw": float(sigma.loc[ahead]),
                     "sigma_floor_as_of": float(floor.loc[ahead]),
                     "floor_hits": int(hit.sum()), "z_std": float(z.std(ddof=1)),
                     "floor_hit_first": hit.index[hit.values].min() if hit.any() else None,
                     "floor_hit_last": hit.index[hit.values].max() if hit.any() else None,
                     "z_abs_max": float(np.abs(z).max()),
                     "z_mean": z_mean, "sigma_pool_median": float(sigma.reindex(weeks).median()),
                     "last_week_in_vol": known.dropna().index.max()})
    else:
        usd_z = None
        usd_sample = usd_all.reindex(weeks).values.astype(float) * options.usd_vol_scale
    info.update({"pool_weeks": len(weeks), "pool_first": weeks.min(), "pool_last": weeks.max(),
                 "pool_usd_std": float(usd_all.reindex(weeks).std(ddof=1)),
                 "sample_usd_std": float(np.std(usd_sample, ddof=1))})
    return {"weeks": weeks, "usd": usd_sample, "usd_z": usd_z,
            "usd_raw": usd_all.reindex(weeks).values.astype(float),
            "arabica": rets["arabica"].reindex(weeks).values.astype(float),
            "robusta": rets["robusta"].reindex(weeks).values.astype(float), "info": info}


def simulate(as_of, n_paths: int, horizon: int = 13, seed: int = 0,
             options: Optional[Inputs] = None, weekly: Optional[pd.DataFrame] = None) -> Paths:
    """Simulate joint paths as of a decision date.

    `options` carries the simulation choices (drift, usd_vol, half_life,
    mean_block) and the market rate; `weekly` is the weekly panel (loaded from
    data/processed if not given). Only weeks up to `as_of` are used.
    """
    if options is None:
        raise ValueError("simulate needs options (an Inputs) for the drift and volatility choices")
    if weekly is None:
        from coffee_risk.data import weekly as weekly_panel
        weekly = weekly_panel.load()
    prep = prepare(weekly, as_of, options)
    n_pool = len(prep["weeks"])
    if n_pool < 2 * horizon:
        raise ValueError(f"pool has only {n_pool} weeks before {pd.Timestamp(as_of).date()}")
    rng = np.random.default_rng(seed)
    idx = stationary_indices(rng, n_pool, n_paths, horizon, options.mean_block)
    growth = target_growth(options.drift, horizon, options.market_rate, prep["usd_raw"])
    ones = np.ones(horizon + 1)
    info = dict(prep["info"])
    info.update({"n_paths": n_paths, "horizon": horizon, "seed": seed,
                 "first_sampled": prep["weeks"][idx.min()], "last_sampled": prep["weeks"][idx.max()]})
    return Paths(usd=to_levels(prep["usd"][idx], growth),
                 arabica=to_levels(prep["arabica"][idx], ones),
                 robusta=to_levels(prep["robusta"][idx], ones),
                 growth=growth, info=info)


def frozen_usd(paths: Paths) -> Paths:
    """The same paths with USD following its target growth exactly."""
    usd = np.tile(paths.growth, (paths.n_paths, 1))
    return Paths(usd=usd, arabica=paths.arabica, robusta=paths.robusta, growth=paths.growth,
                 info=dict(paths.info, usd="frozen"), funds=paths.funds, demand_z=paths.demand_z)
