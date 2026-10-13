"""Data-quality checks. Nothing here changes data: every function returns a
flag or a table that the panel stores and the report counts."""
from __future__ import annotations

import datetime as dt
import math

import numpy as np
import pandas as pd

from coffee_risk.dates import from_jalali, jalali_year

USD_EXTREME = 0.10        # |log return| above this is an extreme daily move
COFFEE_EXTREME = 0.08
STALE_RUN = 5             # same close on this many consecutive observations
NEAR_STALE_STEP = 0.0005  # |log change| below 0.05% counts as "did not move"
GAP_DAYS = 5              # a missing stretch is a gap of more than this many calendar days
_LN10, _LN10K = math.log(10), math.log(10_000)


def log_returns(s: pd.Series) -> pd.Series:
    """Log return between consecutive OBSERVATIONS of one series."""
    x = s.dropna().astype(float)
    return np.log(x).diff()


def extreme_flag(s: pd.Series, threshold: float) -> pd.Series:
    r = log_returns(s)
    return (r.abs() > threshold).reindex(s.index, fill_value=False)


def stale_flag(s: pd.Series, run: int = STALE_RUN) -> pd.Series:
    """True on every observation that belongs to a run of `run` or more
    consecutive observations with the same value."""
    x = s.dropna()
    block = (x != x.shift()).cumsum()
    size = x.groupby(block).transform("size")
    return (size >= run).reindex(s.index, fill_value=False)


def stale_runs(s: pd.Series, run: int = STALE_RUN) -> pd.DataFrame:
    x = s.dropna()
    block = (x != x.shift()).cumsum()
    g = x.groupby(block)
    out = pd.DataFrame({"first_date": g.apply(lambda v: v.index[0]),
                        "last_date": g.apply(lambda v: v.index[-1]),
                        "observations": g.size(), "value": g.first()})
    return out[out["observations"] >= run].reset_index(drop=True)


def _near_stale_blocks(s: pd.Series, step: float) -> tuple[pd.Series, pd.Series]:
    """(observations, block id): neighbours whose |log change| is below
    `step` share a block."""
    x = s.dropna().astype(float)
    small = np.log(x).diff().abs() < step          # False for the first observation
    return x, (~small).cumsum()


def near_stale_flag(s: pd.Series, step: float = NEAR_STALE_STEP,
                    run: int = STALE_RUN) -> pd.Series:
    """True on every observation inside a run of `run` or more consecutive
    observations in which each |log change| from the previous observation is
    below `step`. An exact repeat is the special case of a zero change; this
    also catches quotes that drift by a few hundred rials while the market
    moves (a frozen or managed quote)."""
    x, block = _near_stale_blocks(s, step)
    size = x.groupby(block).transform("size")
    return (size >= run).reindex(s.index, fill_value=False)


def near_stale_runs(s: pd.Series, step: float = NEAR_STALE_STEP,
                    run: int = STALE_RUN) -> pd.DataFrame:
    """One row per near-stale run: first and last date, number of
    observations, first and last value, and the largest step inside it."""
    x, block = _near_stale_blocks(s, step)
    steps = np.log(x).diff().abs()
    rows = []
    for _, v in x.groupby(block):
        if len(v) >= run:
            rows.append({"first_date": v.index[0], "last_date": v.index[-1],
                         "observations": len(v), "first_value": v.iloc[0],
                         "last_value": v.iloc[-1],
                         "max_abs_step": float(steps.loc[v.index[1:]].max())})
    return pd.DataFrame(rows, columns=["first_date", "last_date", "observations",
                                       "first_value", "last_value", "max_abs_step"])


def covers_nowruz(a: dt.date, b: dt.date) -> bool:
    """Does the open interval (a, b) contain a day of the Nowruz holidays
    (1-13 Farvardin) of some year?"""
    for jy in {jalali_year(a), jalali_year(b), jalali_year(b) + 1}:
        start, end = from_jalali(jy, 1, 1), from_jalali(jy, 1, 13)
        if a < end and start < b and max(a + dt.timedelta(days=1), start) <= min(
                b - dt.timedelta(days=1), end):
            return True
    return False


def covers_year_end(a: dt.date, b: dt.date) -> bool:
    """Does the open interval (a, b) contain 25 December or 1 January?"""
    for y in {a.year, b.year}:
        for day in (dt.date(y, 12, 25), dt.date(y, 1, 1)):
            if a < day < b:
                return True
    return False


def gaps(s: pd.Series, min_days: int = GAP_DAYS, iran: bool = True,
         max_holiday_days: int = 21) -> pd.DataFrame:
    """Stretches of more than `min_days` calendar days with no observation.

    Each is labelled by the holiday that explains it: `nowruz` for a series
    from Iran (`iran=True`), `year_end` for an exchange series, else `other`.
    A stretch longer than `max_holiday_days` is never called a holiday.
    """
    idx = [d.date() if hasattr(d, "date") else d for d in s.dropna().index]
    rows = []
    for a, b in zip(idx[:-1], idx[1:]):
        n = (b - a).days
        if n > min_days:
            kind = "other"
            if n <= max_holiday_days:
                if iran and covers_nowruz(a, b):
                    kind = "nowruz"
                elif not iran and covers_year_end(a, b):
                    kind = "year_end"
            rows.append({"last_before": a, "first_after": b, "calendar_days": n, "kind": kind})
    return pd.DataFrame(rows, columns=["last_before", "first_after", "calendar_days", "kind"])


def level_breaks(s: pd.Series, tol: float = 0.35) -> pd.DataFrame:
    """Observations where the level jumps by about x10 or x10,000 (a
    toman/rial mix-up or a redenomination looks like this)."""
    r = log_returns(s)
    near10 = (r.abs() - _LN10).abs() < tol
    near10k = (r.abs() - _LN10K).abs() < tol
    hit = r[near10 | near10k]
    return pd.DataFrame({"date": hit.index, "log_return": hit.values,
                         "factor": np.where(near10k.reindex(hit.index), 10_000, 10)})
