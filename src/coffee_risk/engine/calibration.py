"""Calibration preview: do realised USD moves fall outside the simulated range
as often as they should?

For each weekly start, the engine simulates as of that week (data up to that
week only) and the realised change of USD over the horizon is compared with
the simulated 5th and 95th percentiles. A calibrated simulator is exceeded
about 5% of the time on each side.

Information only: nothing here is tuned to the result. Windows that start one
week apart share most of their weeks, so the number of independent
(non-overlapping) windows is reported next to every rate.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from coffee_risk.dates import jalali_year
from coffee_risk.engine.inputs import Inputs, market_rate
from coffee_risk.engine.simulator import simulate

BACKTEST_START = pd.Timestamp("2015-06-17")     # D-041: 1394-03-27, the first week with a 13-week fund return
LEVELS = (0.05, 0.95)


def usd_exceedance(weekly: pd.DataFrame, n_paths: int = 2000, seed: int = 0, horizon: int = 13,
                   start=BACKTEST_START, methods: Sequence[str] = ("fhs", "plain"),
                   drift="neutral") -> pd.DataFrame:
    """One row per weekly start from `start` whose `horizon`-week outcome is in
    the panel.

    Columns: end, jalali_year (of the start), realized (simple USD change),
    usable (False if a level is too old or the market rate is missing; such
    starts are kept in the table and counted, not dropped), and for each
    method m: m_q05, m_q95, m_above (realized > q95), m_below (realized < q05).
    """
    level, old = weekly["usd_free"], weekly["usd_free_old"].fillna(False).astype(bool)
    index = weekly.index
    rows = []
    for k, t in enumerate(index):
        if t < pd.Timestamp(start) or k + horizon >= len(index):
            continue
        end = index[k + horizon]
        row = {"start": t, "end": end, "jalali_year": jalali_year(t.date()),
               "realized": float(level.loc[end] / level.loc[t] - 1.0), "reason": ""}
        rate = market_rate(weekly, t) if drift == "neutral" else 0.0
        if end != t + pd.Timedelta(weeks=horizon):
            row["reason"] = "a week is missing from the panel"
        elif old.loc[t] or old.loc[end] or np.isnan(row["realized"]):
            row["reason"] = "USD level too old at one end"
        elif np.isnan(rate):
            row["reason"] = "no market rate"
        row["usable"] = not row["reason"]
        if row["usable"]:
            for m in methods:
                options = Inputs(s0=float(level.loc[t]), market_rate=rate, premium_share=0.0,
                                 usd_vol=m, drift=drift)
                change = simulate(t, n_paths, horizon, seed + k, options, weekly).usd[:, horizon] - 1.0
                lo, hi = np.quantile(change, LEVELS)
                row.update({f"{m}_q05": float(lo), f"{m}_q95": float(hi),
                            f"{m}_above": bool(row["realized"] > hi),
                            f"{m}_below": bool(row["realized"] < lo)})
        rows.append(row)
    return pd.DataFrame(rows).set_index("start")


def independent_windows(starts: pd.DatetimeIndex, horizon: int = 13) -> int:
    """How many of the windows can be chosen so that no two share a week:
    take the first start, then the next one at least `horizon` weeks later,
    and so on."""
    n, last = 0, None
    for t in sorted(starts):
        if last is None or t >= last + pd.Timedelta(weeks=horizon):
            n, last = n + 1, t
    return n


def summarize(table: pd.DataFrame, methods: Sequence[str] = ("fhs", "plain"),
              horizon: int = 13) -> pd.DataFrame:
    """Exceedance rates by Jalali year of the start and overall (row 'all')."""
    ok = table[table["usable"]]

    def one(part: pd.DataFrame) -> dict:
        out = {"starts": len(part), "independent": independent_windows(part.index, horizon)}
        for m in methods:
            out[f"{m}_above"] = float(part[f"{m}_above"].mean())
            out[f"{m}_below"] = float(part[f"{m}_below"].mean())
        return out

    rows = {int(y): one(part) for y, part in ok.groupby("jalali_year")}
    rows["all"] = one(ok)
    return pd.DataFrame.from_dict(rows, orient="index")


def binomial_range(n: int, p: float = 0.05, coverage: float = 0.95) -> tuple[float, float]:
    """The range of exceedance rates a calibrated simulator would show with
    `n` independent windows: central `coverage` range of Binomial(n, p) / n."""
    from math import comb
    cdf, lo, hi = 0.0, None, None
    for k in range(n + 1):
        cdf += comb(n, k) * p ** k * (1 - p) ** (n - k)
        if lo is None and cdf >= (1 - coverage) / 2:
            lo = k
        if hi is None and cdf >= 1 - (1 - coverage) / 2:
            hi = k
            break
    return lo / n, hi / n
