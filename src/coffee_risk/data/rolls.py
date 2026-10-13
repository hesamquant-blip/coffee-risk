"""Futures roll days of the continuous coffee series.

A continuous front-month series follows one contract and then switches to the
next. On the switch day the price jumps by the spread between the two
contracts. That jump is not a price change for a buyer, so the return on that
day is excluded later; levels stay raw.

What D1 found (evidence in reports/d1_data):
  * Yahoo KC=F (arabica) keeps the expiring contract until its last trading
    day and shows the next contract from the following trading day. ICE sets
    the last trading day of Coffee "C" 8 business days before the last
    business day of the delivery month (Mar, May, Jul, Sep, Dec), so the
    switch day is the trading day with exactly 7 trading days left in that
    month.
  * TGJU's London-coffee series (robusta) behaves the same way: ICE robusta
    stops trading 4 business days before the last business day of the delivery
    month (Jan, Mar, May, Jul, Sep, Nov), and the switch day is the trading
    day with exactly 3 trading days left in that month.

The calendar is known in advance (it comes from contract rules, not from
prices), so flagging a roll day uses no future price information.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RollRule:
    name: str
    months: tuple[int, ...]      # delivery months
    days_left: int               # trading days left in the month on the switch day


ARABICA = RollRule("arabica", (3, 5, 7, 9, 12), 7)
ROBUSTA = RollRule("robusta", (1, 3, 5, 7, 9, 11), 3)


def trading_calendar(own_dates: Iterable[dt.date],
                     ref_dates: Optional[Iterable[dt.date]] = None,
                     hole_days: int = 5) -> list[dt.date]:
    """Business days of a series: the weekdays on which it has a value.

    If the series has a hole longer than `hole_days` calendar days, the hole
    is filled FOR THE CALENDAR ONLY (never for prices) with the weekdays of a
    reference series, so a roll day inside the hole is still found. The last
    month is completed with its remaining weekdays (holidays ignored), so the
    current month can be flagged without waiting for it to end.
    """
    own = sorted({d for d in own_dates if d.weekday() < 5})
    cal = set(own)
    if ref_dates is not None:
        ref = sorted({d for d in ref_dates if d.weekday() < 5})
        for a, b in zip(own[:-1], own[1:]):
            if (b - a).days > hole_days:
                cal.update(d for d in ref if a < d < b)
    if own:
        d = own[-1] + dt.timedelta(days=1)
        while d.month == own[-1].month:
            if d.weekday() < 5:
                cal.add(d)
            d += dt.timedelta(days=1)
    return sorted(cal)


def days_left_in_month(calendar: list[dt.date]) -> dict[dt.date, int]:
    """For each calendar day: how many calendar days follow it in its month."""
    by_month: dict[tuple[int, int], list[dt.date]] = {}
    for d in calendar:
        by_month.setdefault((d.year, d.month), []).append(d)
    return {d: len(days) - 1 - i for days in by_month.values() for i, d in enumerate(days)}


def switch_days(calendar: list[dt.date], rule: RollRule) -> list[dt.date]:
    """Days on which the continuous series first shows the next contract."""
    left = days_left_in_month(calendar)
    return [d for d in calendar if d.month in rule.months and left[d] == rule.days_left]


def roll_flags(obs_dates: Iterable[dt.date], switches: Iterable[dt.date]) -> dict[dt.date, bool]:
    """Flag for each observed date. A switch day with no observation (holiday
    or data hole) moves its flag to the next observed date, because that is
    where the jump shows up in the series."""
    obs = sorted(set(obs_dates))
    flags = {d: False for d in obs}
    arr = np.array(obs, dtype="datetime64[D]")
    for s in switches:
        i = int(np.searchsorted(arr, np.datetime64(s), side="left"))
        if 0 < i < len(obs):          # needs an earlier observation to jump from
            flags[obs[i]] = True
    return flags


# ------------------------------------------------------------------ evidence
def volume_evidence(df: pd.DataFrame, switches: list[dt.date]) -> pd.DataFrame:
    """Arabica: does volume come back on the day before each switch day?

    In KC=F the volume of an expiring contract falls to a few lots during the
    delivery month. On its last trading day the series already reports the
    next contract's volume; the price follows one trading day later. A volume
    jump on the day before the rule's switch day therefore confirms the rule
    without any single-contract data.
    """
    s = df.set_index("date")
    dates = list(s.index)
    pos = {d: i for i, d in enumerate(dates)}
    logret = np.log(s["close"]).diff()
    rows = []
    for sw in switches:
        i = pos.get(sw)
        if i is None or i < 6:
            continue
        before = float(np.median(s["volume"].iloc[i - 6:i - 1]))
        last_day = float(s["volume"].iloc[i - 1])
        rows.append({"switch_day": sw, "last_trading_day": dates[i - 1],
                     "median_volume_5d_before": before, "volume_last_trading_day": last_day,
                     "volume_jump": bool(last_day >= 1000 and last_day >= 20 * max(before, 1.0)),
                     "switch_day_return": float(logret.iloc[i])})
    return pd.DataFrame(rows)


def contract_evidence(continuous: pd.DataFrame, contract: pd.DataFrame,
                      switches: list[dt.date]) -> pd.DataFrame:
    """Arabica: compare the continuous series with one listed contract.

    gap = continuous return - contract return on the switch day. The listed
    contract does not roll, so the gap is the jump caused by the switch.
    `same_before` / `same_on` say whether the two closes are equal the day
    before / on the switch day (equal on, not before = the continuous series
    switched INTO this contract that day).
    """
    j = (continuous.set_index("date")[["close"]]
         .join(contract.set_index("date")[["close"]], rsuffix="_contract", how="inner"))
    r = np.log(j).diff()
    same = (j["close"] - j["close_contract"]).abs() < 1e-6
    dates = list(j.index)
    pos = {d: i for i, d in enumerate(dates)}
    rows = []
    for sw in switches:
        i = pos.get(sw)
        if i is None or i < 1:
            continue
        rows.append({"switch_day": sw,
                     "continuous_return": float(r["close"].iloc[i]),
                     "contract_return": float(r["close_contract"].iloc[i]),
                     "gap": float(r["close"].iloc[i] - r["close_contract"].iloc[i]),
                     "same_before": bool(same.iloc[i - 1]), "same_on": bool(same.iloc[i])})
    return pd.DataFrame(rows)


def days_left_evidence(target: pd.Series, control: pd.Series, calendar: list[dt.date],
                       rule: RollRule, eras: list[tuple[int, int]],
                       beta: float = 0.5, max_left: int = 8) -> pd.DataFrame:
    """Robusta: no single-contract data and no volume, so the test is
    statistical. Take the part of the daily robusta return that arabica does
    not explain (return - beta * arabica return) and group delivery-month days
    by the number of trading days left in the month. A roll day shows up as a
    larger spread of that residual and, when near contracts are dearer than
    far ones (backwardation), a negative mean.
    """
    left = days_left_in_month(calendar)
    rt = np.log(target).diff()
    rc = np.log(control).diff()
    res = (rt - beta * rc.reindex(rt.index)).dropna()
    res = res[[d in left for d in res.index]]
    rows = []
    for a, b in eras:
        in_era = [a <= d.year <= b for d in res.index]
        era = res[in_era]
        for k in range(max_left + 1):
            x = era[[d.month in rule.months and left[d] == k for d in era.index]]
            rows.append({"era": f"{a}-{b}", "days_left": k, "n": len(x),
                         "std": float(x.std()) if len(x) > 1 else np.nan,
                         "mean": float(x.mean()) if len(x) else np.nan})
        rows.append({"era": f"{a}-{b}", "days_left": -1, "n": len(era),
                     "std": float(era.std()), "mean": float(era.mean())})   # all days
    return pd.DataFrame(rows)


def expiry_window_noise(continuous: pd.DataFrame, contract: pd.DataFrame,
                        switches: list[dt.date], window: int = 13) -> dict:
    """Arabica: how far does the continuous series drift from a liquid listed
    contract in the `window` trading days before each switch day, when it is
    still following the thinly traded expiring contract?

    Returns the std of (continuous return - contract return) inside those
    windows and on all other days; switch days themselves are left out.
    """
    j = (continuous.set_index("date")[["close"]]
         .join(contract.set_index("date")[["close"]], rsuffix="_contract", how="inner"))
    diff = (np.log(j["close"]).diff() - np.log(j["close_contract"]).diff()).dropna()
    dates = list(diff.index)
    pos = {d: i for i, d in enumerate(dates)}
    inside = set()
    for sw in switches:
        i = pos.get(sw)
        if i is not None:
            inside.update(dates[max(0, i - window):i])
    sw_set = set(switches)
    is_in = np.array([d in inside for d in dates])
    is_sw = np.array([d in sw_set for d in dates])
    # after the last switch the two series are the same contract: leave it out
    same = (j["close"] - j["close_contract"]).abs().reindex(diff.index) < 1e-6
    keep = ~is_sw & ~same.values
    return {"first": dates[0], "last": dates[-1],
            "n_window": int((is_in & keep).sum()), "n_other": int((~is_in & keep).sum()),
            "std_window": float(diff[is_in & keep].std()),
            "std_other": float(diff[~is_in & keep].std()),
            "max_abs_window": float(diff[is_in & keep].abs().max())}
