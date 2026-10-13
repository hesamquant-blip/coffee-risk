"""Returns on a common calendar.

Iran's FX market (Saturday-Thursday, Iranian holidays) and ICE (Monday-Friday,
US/UK holidays) are open on different days. Nothing is forward-filled. For a
chosen set of columns we keep only the dates on which ALL of them are observed
and take log returns between consecutive such dates. No move is lost: what
happened to one series while another was closed lands on the next common date.
"""
from __future__ import annotations

from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd


def common_dates(panel: pd.DataFrame, cols: Sequence[str],
                 unobserved: Optional[Mapping[str, str]] = None) -> pd.DatetimeIndex:
    """Dates on which every column in `cols` is observed.

    `unobserved` maps a column to a boolean flag column: where the flag is
    True the value is treated as not observed (used for rows a data check
    marked as unusable, e.g. weekend-dated robusta rows).
    """
    ok = panel[list(cols)].notna()
    for col, flag in (unobserved or {}).items():
        if col in ok.columns:
            ok[col] &= ~panel[flag].fillna(False).astype(bool)
    return panel.index[ok.all(axis=1)]


def common_calendar_returns(panel: pd.DataFrame, cols: Sequence[str],
                            exclude: Optional[Mapping[str, str]] = None,
                            unobserved: Optional[Mapping[str, str]] = None) -> pd.DataFrame:
    """Log returns between consecutive common dates.

    `exclude` maps a column to a boolean flag column (e.g. a futures roll
    flag). A return is set to missing when a flagged day lies in the interval
    it covers, (previous common date, this common date]; other returns are
    untouched. Levels are never changed.

    The result is indexed by the later date of each pair and carries
    `days` = calendar days the return covers.
    """
    cols = list(cols)
    dates = common_dates(panel, cols, unobserved)
    levels = panel.loc[dates, cols].astype(float)
    ret = np.log(levels).diff().iloc[1:]
    for col, flag in (exclude or {}).items():
        if col not in ret.columns:
            continue
        hits = panel[flag].fillna(False).astype(int).cumsum().loc[dates].diff().iloc[1:]
        ret.loc[hits > 0, col] = np.nan
    ret["days"] = pd.Series(dates[1:] - dates[:-1], index=dates[1:]).dt.days
    return ret


def window_counts(dates: pd.DatetimeIndex, window_days: int = 91) -> pd.Series:
    """Number of dates in every `window_days`-calendar-day window that fits
    inside the span of `dates` (one window per starting calendar day)."""
    if len(dates) == 0:
        return pd.Series(dtype=float)
    marks = pd.Series(1, index=dates).reindex(
        pd.date_range(dates.min(), dates.max(), freq="D"), fill_value=0)
    counts = marks.rolling(window_days).sum().dropna()
    return counts
