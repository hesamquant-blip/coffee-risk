"""Rial opportunity cost: a listed fixed-income fund's total-return index.

Candidates are all funds TSETMC classes as fixed-income ETFs. The fund with
the most traded days among those still trading is chosen (= the longest
continuous history).

Total return. TSETMC's "previous close" is NOT restated when a fund pays a
cash distribution (checked on funds that pay monthly: each payment shows as a
price drop of 1.5-2.5%), so a price series is a total-return series only for a
fund that does not distribute. `distribution_like_drops` counts the days on
which the closing price fell by more than 0.4%: a monthly payer has about 12
a year, an accumulating fund almost none. The chosen fund must pass that
check; otherwise NAV data would be needed and the question goes to the author.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from coffee_risk.data import tsetmc

DROP_THRESHOLD = -0.004
MAX_DROPS_PER_YEAR = 2.0
STILL_TRADING_DAYS = 30


def traded(ins_code: str) -> pd.DataFrame:
    """Days with at least one trade (TSETMC also lists zero-volume days)."""
    d = tsetmc.load_daily(ins_code)
    return d[d["volume"] > 0].reset_index(drop=True)


def candidates() -> pd.DataFrame:
    rows = []
    for _, f in tsetmc.fixed_income_funds().iterrows():
        if not tsetmc.has_daily(f["ins_code"]):
            continue
        all_rows = tsetmc.load_daily(f["ins_code"])
        t = all_rows[all_rows["volume"] > 0]
        if t.empty:
            continue
        dates = list(t["date"])
        gaps = [(b - a).days for a, b in zip(dates[:-1], dates[1:])] or [0]
        ret = t["close"] / t["close"].shift() - 1
        years = max((dates[-1] - dates[0]).days / 365.25, 1e-9)
        drops = int((ret < DROP_THRESHOLD).sum())
        rows.append({
            "ticker": f["ticker"], "name": f["name"], "ins_code": f["ins_code"],
            "first_traded": dates[0], "last_traded": dates[-1],
            "traded_days": len(t), "zero_volume_rows": int(len(all_rows) - len(t)),
            "max_gap_days": int(max(gaps)),
            "distribution_like_drops": drops, "drops_per_year": drops / years,
            "median_drop": float(ret[ret < DROP_THRESHOLD].median()) if drops else float("nan"),
            "first_close": float(t["close"].iloc[0]), "last_close": float(t["close"].iloc[-1]),
        })
    return pd.DataFrame(rows).sort_values("first_traded", kind="stable").reset_index(drop=True)


def choose(cands: pd.DataFrame) -> pd.Series:
    """Most traded days among funds still trading that do not distribute."""
    newest = cands["last_traded"].max()
    alive = cands[cands["last_traded"] >= newest - dt.timedelta(days=STILL_TRADING_DAYS)]
    clean = alive[alive["drops_per_year"] <= MAX_DROPS_PER_YEAR]
    if clean.empty:
        raise RuntimeError("no accumulating fixed-income fund found; NAV data needed")
    return clean.sort_values(["traded_days", "first_traded"],
                             ascending=[False, True], kind="stable").iloc[0]


def total_return_index(ins_code: str) -> pd.DataFrame:
    """-> date, close (rial), tr_index (100 on the first traded day).
    Price-based: valid for an accumulating fund (see module docstring)."""
    t = traded(ins_code)[["date", "close"]].copy()
    t["tr_index"] = 100.0 * t["close"] / t["close"].iloc[0]
    return t


def annual_returns(index: pd.DataFrame) -> pd.DataFrame:
    """Return per Jalali year, from the last observation of the previous year
    (the first observation, for the first year) to the last of this year,
    annualised over the calendar days actually covered."""
    from coffee_risk.dates import jalali_year
    s = index.set_index("date")["tr_index"]
    jy = pd.Series([jalali_year(d) for d in s.index], index=s.index)
    rows = []
    for y in sorted(jy.unique()):
        seg, prev = s[jy == y], s[jy < y]
        start_date = prev.index[-1] if len(prev) else seg.index[0]
        start = prev.iloc[-1] if len(prev) else seg.iloc[0]
        days = (seg.index[-1] - start_date).days
        total = seg.iloc[-1] / start - 1
        rows.append({"jalali_year": int(y), "from": start_date, "to": seg.index[-1],
                     "observations": len(seg), "calendar_days": days, "total_return": total,
                     "annualised": (1 + total) ** (365.0 / days) - 1 if days >= 30 else np.nan})
    return pd.DataFrame(rows)
