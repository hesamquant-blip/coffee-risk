"""Weekly panel (D-011): one row per Wednesday.

Level. For each series the level of a week is its last observation on or
before that Wednesday, so a closed Wednesday falls back to the last earlier
observation and nothing is filled.

Weekly log return, Wednesday to Wednesday:
  * USD and the fund: log(level this Wednesday / level last Wednesday);
  * coffee: the sum of the daily log returns dated inside the week
    (last Wednesday, this Wednesday], leaving out roll days, so the jump of a
    futures roll inside a week does not leak into the weekly return. The plain
    log change of the level is kept next to it as `*_ret_raw`.

Missing returns (D-018). A weekly return is left empty when it is not a real
one-week move:
  * the level at the end of the week is stale: its observation is more than
    MAX_AGE_DAYS calendar days older than the Wednesday (a week inside a data
    hole, where the return would be a false zero);
  * the week holds the first observation after a hole of more than
    MAX_AGE_DAYS calendar days (that return would carry the whole hole).
`*_ret_raw` is never emptied, so the level change across a hole stays visible.

Robusta gap fill (D-020). Where the robusta return is empty for one of those
reasons, it is taken from the ICO Robustas indicator (same Wednesday rule,
same D-018 test on the ICO series), provided the two series pass the
agreement test fixed in the spec. ICO has no futures roll, so nothing is
excluded in those weeks. `robusta_return_source` says "tgju" or "ico". Levels
are never taken from ICO: it is a physical price, not the futures price.

Dates are calendar dates as each source publishes them (Tehran for the dollar
and the fund, the exchange's date for coffee).
"""
from __future__ import annotations

import datetime as dt
from typing import Mapping

import numpy as np
import pandas as pd

from coffee_risk import paths
from coffee_risk.dates import to_jalali

WEEKLY = paths.PROCESSED / "panel_weekly.parquet"
WEDNESDAY = 2                 # datetime.weekday(): Monday = 0
MAX_AGE_DAYS = 5              # staleness limit and hole length of D-018, calendar days

# series -> how its weekly return is built
RATIO_SERIES = ["usd_free", "usd_tgju", "fund_tr_index"]
COFFEE_SERIES = {"arabica_usd_kg": "arabica_roll", "robusta_usd_kg": "robusta_roll"}
NEAR_STALE = {"usd_free": "usd_free_near_stale", "usd_tgju": "usd_tgju_near_stale",
              "fund_tr_index": "fund_near_stale", "arabica_usd_kg": "arabica_near_stale",
              "robusta_usd_kg": "robusta_near_stale"}

# ICO agreement test (spec "Robusta gap fill", step 3): fixed before any result was seen
ICO_COL = "robusta_ico_clb"
ICO_TEST_FROM, ICO_TEST_TO = dt.date(2025, 1, 1), dt.date(2026, 9, 30)
ICO_MIN_CORR = 0.90
ICO_VOL_RATIO = (0.80, 1.25)


def wednesdays(first: dt.date, last: dt.date) -> pd.DatetimeIndex:
    """Every Wednesday from the first one on or after `first` to the last one
    on or before `last`."""
    start = first + dt.timedelta(days=(WEDNESDAY - first.weekday()) % 7)
    end = last - dt.timedelta(days=(last.weekday() - WEDNESDAY) % 7)
    return pd.date_range(start, end, freq="7D", name="date")


def last_observation(s: pd.Series, weds: pd.DatetimeIndex) -> pd.DataFrame:
    """For each Wednesday: the last observation on or before it (`level`) and
    its date (`obs_date`); empty before the series starts."""
    x = s.dropna()
    pos = x.index.searchsorted(weds, side="right") - 1
    ok = pos >= 0
    level = np.where(ok, x.values[np.clip(pos, 0, None)] if len(x) else np.nan, np.nan)
    dates = pd.Series(pd.NaT, index=weds, dtype="datetime64[ns]")
    if len(x):
        dates[ok] = x.index[pos[ok]]
    return pd.DataFrame({"level": level.astype(float), "obs_date": dates.values}, index=weds)


def ratio_return(level: pd.Series) -> pd.Series:
    return np.log(level).diff()


def summed_return(s: pd.Series, weds: pd.DatetimeIndex,
                  exclude: pd.Series | None = None) -> pd.Series:
    """Sum of the daily log returns dated in (previous Wednesday, this
    Wednesday]. A daily return is dated at its later observation; returns on
    dates where `exclude` is True are left out. Empty until the series has an
    observation on or before the previous Wednesday."""
    x = s.dropna().astype(float)
    r = np.log(x).diff().dropna()
    if exclude is not None:
        r = r[~exclude.reindex(r.index).fillna(False).astype(bool)]
    r = r[(r.index > weds[0]) & (r.index <= weds[-1])]          # the first row has no return
    week = weds[weds.searchsorted(r.index, side="left")]       # first Wednesday on or after
    out = r.groupby(week).sum().reindex(weds, fill_value=0.0)
    started = pd.Series(weds, index=weds).shift(1) >= x.index[0] if len(x) else False
    return out.where(started)


def stale_weeks(obs_date: pd.Series, max_age: int = MAX_AGE_DAYS) -> pd.Series:
    """Weeks whose level is more than `max_age` calendar days old."""
    age = (pd.Series(obs_date.index, index=obs_date.index) - obs_date).dt.days
    return (age > max_age).fillna(False)


def after_hole_weeks(s: pd.Series, weds: pd.DatetimeIndex,
                     max_gap: int = MAX_AGE_DAYS) -> pd.Series:
    """Weeks that hold the first observation after a hole of more than
    `max_gap` calendar days between two consecutive observations."""
    x = s.dropna()
    out = pd.Series(False, index=weds)
    if len(x) < 2:
        return out
    gap = pd.Series(x.index[1:] - x.index[:-1], index=x.index[1:]).dt.days
    first_after = gap.index[gap > max_gap]
    first_after = first_after[(first_after > weds[0]) & (first_after <= weds[-1])]
    out.loc[weds[weds.searchsorted(first_after, side="left")]] = True
    return out


def missing_weeks(s: pd.Series, weds: pd.DatetimeIndex) -> pd.Series:
    """D-018: weeks whose return is not a real one-week move."""
    lo = last_observation(s, weds)
    return stale_weeks(lo["obs_date"]) | after_hole_weeks(s, weds)


def ico_weekly_return(ico: pd.Series, weds: pd.DatetimeIndex) -> pd.Series:
    """Weekly log return of the ICO indicator, with the same D-018 rule: empty
    where either end of the week is stale or a hole ends inside the week."""
    lo = last_observation(ico, weds)
    stale = stale_weeks(lo["obs_date"])
    bad = stale | stale.shift(1, fill_value=False) | after_hole_weeks(ico, weds)
    return ratio_return(lo["level"]).where(~bad)


def ico_agreement(tgju_ret: pd.Series, ico_ret: pd.Series, roll_week: pd.Series) -> dict:
    """The agreement test of the gap-fill spec, on weekly log returns.

    Weeks: Wednesdays from ICO_TEST_FROM to ICO_TEST_TO where both returns
    exist (both series fresh at both ends) and no TGJU roll day lies inside
    the week. Pass = correlation >= ICO_MIN_CORR and the ratio of weekly
    volatilities (ICO / TGJU) inside ICO_VOL_RATIO.
    """
    both = pd.concat([tgju_ret.rename("tgju"), ico_ret.rename("ico")], axis=1)
    inside = (both.index >= pd.Timestamp(ICO_TEST_FROM)) & (both.index <= pd.Timestamp(ICO_TEST_TO))
    x = both[inside & ~roll_week.reindex(both.index).fillna(False).values].dropna()
    out = {"weeks": len(x), "corr": np.nan, "vol_ico": np.nan, "vol_tgju": np.nan,
           "vol_ratio": np.nan, "passed": False,
           "first_week": x.index.min() if len(x) else None,
           "last_week": x.index.max() if len(x) else None}
    if len(x) >= 3:
        out["corr"] = float(x["tgju"].corr(x["ico"]))
        out["vol_ico"], out["vol_tgju"] = float(x["ico"].std()), float(x["tgju"].std())
        out["vol_ratio"] = out["vol_ico"] / out["vol_tgju"]
        out["passed"] = bool(out["corr"] >= ICO_MIN_CORR
                             and ICO_VOL_RATIO[0] <= out["vol_ratio"] <= ICO_VOL_RATIO[1])
    return out


def roll_weeks(roll_flag: pd.Series, weds: pd.DatetimeIndex) -> pd.Series:
    """Weeks with a roll-flagged day inside (previous Wednesday, this Wednesday]."""
    days = roll_flag.index[roll_flag.fillna(False).astype(bool)]
    days = days[(days > weds[0]) & (days <= weds[-1])]
    out = pd.Series(False, index=weds)
    out.loc[weds[weds.searchsorted(days, side="left")]] = True
    return out


def robusta_inputs(daily: pd.DataFrame, weds: pd.DatetimeIndex) -> dict:
    """TGJU weekly robusta return (D-018 applied), ICO weekly return, the
    weeks with a roll, and the result of the agreement test."""
    s = daily["robusta_usd_kg"]
    tgju_ret = summed_return(s, weds, exclude=daily["robusta_roll"]).where(~missing_weeks(s, weds))
    has_ico = ICO_COL in daily.columns and daily[ICO_COL].notna().any()
    ico_ret = (ico_weekly_return(daily[ICO_COL], weds) if has_ico
               else pd.Series(np.nan, index=weds))
    rolls = roll_weeks(daily["robusta_roll"], weds)
    return {"tgju_ret": tgju_ret, "ico_ret": ico_ret, "roll_week": rolls,
            "test": ico_agreement(tgju_ret, ico_ret, rolls)}


def build(daily: pd.DataFrame, save: bool = True,
          coffee: Mapping[str, str] = COFFEE_SERIES) -> pd.DataFrame:
    """Weekly panel from the daily panel.

    Per series X: `X` (level), `X_obs_date`, `X_ret`, `X_old` (the observation
    is more than MAX_AGE_DAYS calendar days older than the Wednesday) and
    `X_near_stale` (the daily near-stale flag at that observation). Coffee
    series also get `X_ret_raw`. `robusta_return_source` names the source of
    each robusta return.
    """
    first = daily["usd_free"].dropna().index.min().date()
    weds = wednesdays(first, daily.index.max().date())
    w = pd.DataFrame(index=weds)
    w["jdate"] = [to_jalali(d.date()) for d in weds]
    for col in RATIO_SERIES + list(coffee):
        s = daily[col]
        lo = last_observation(s, weds)
        w[col] = lo["level"]
        w[f"{col}_obs_date"] = lo["obs_date"]
        missing = missing_weeks(s, weds)
        if col in coffee:
            w[f"{col}_ret"] = summed_return(s, weds, exclude=daily[coffee[col]]).where(~missing)
            w[f"{col}_ret_raw"] = ratio_return(lo["level"])
        else:
            w[f"{col}_ret"] = ratio_return(lo["level"]).where(~missing)
        w[f"{col}_old"] = stale_weeks(lo["obs_date"])
        flag = daily[NEAR_STALE[col]]
        w[f"{col}_near_stale"] = [bool(flag.get(d, False)) if pd.notna(d) else False
                                  for d in lo["obs_date"]]

    if "robusta_usd_kg" in coffee:
        rb = robusta_inputs(daily, weds)
        source = pd.Series(np.where(w["robusta_usd_kg_ret"].notna(), "tgju", None),
                           index=weds, dtype=object)
        if rb["test"]["passed"]:
            fill = w["robusta_usd_kg_ret"].isna() & rb["ico_ret"].notna() \
                & w["robusta_usd_kg"].notna()
            w.loc[fill, "robusta_usd_kg_ret"] = rb["ico_ret"][fill]
            source[fill] = "ico"
        w["robusta_return_source"] = source
    if save:
        paths.PROCESSED.mkdir(parents=True, exist_ok=True)
        w.to_parquet(WEEKLY)
    return w


def load() -> pd.DataFrame:
    return pd.read_parquet(WEEKLY)
