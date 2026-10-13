import datetime as dt

import numpy as np
import pandas as pd

from coffee_risk.data import quality, rolls


def _weekdays(start, end):
    d, out = start, []
    while d <= end:
        if d.weekday() < 5:
            out.append(d)
        d += dt.timedelta(days=1)
    return out


def test_switch_days_from_contract_rules():
    """September 2026: last business day is Wed 30. Arabica stops 8 business
    days earlier (Fri 18), robusta 4 earlier (Thu 24); the continuous series
    shows the next contract on the following trading day."""
    cal = _weekdays(dt.date(2026, 8, 3), dt.date(2026, 10, 30))
    assert rolls.switch_days(cal, rolls.ARABICA) == [dt.date(2026, 9, 21)]
    assert rolls.switch_days(cal, rolls.ROBUSTA) == [dt.date(2026, 9, 25)]


def test_holiday_inside_the_month_moves_the_switch_day():
    cal = [d for d in _weekdays(dt.date(2026, 9, 1), dt.date(2026, 9, 30))
           if d != dt.date(2026, 9, 28)]                    # one closed day late in the month
    assert rolls.switch_days(cal, rolls.ARABICA) == [dt.date(2026, 9, 18)]


def test_calendar_fills_a_hole_and_completes_the_last_month():
    ref = _weekdays(dt.date(2026, 6, 1), dt.date(2026, 9, 30))
    own = [d for d in ref if not (dt.date(2026, 7, 20) < d < dt.date(2026, 8, 24))]
    own = [d for d in own if d <= dt.date(2026, 9, 10)]     # series ends mid-month
    cal = rolls.trading_calendar(own, ref_dates=ref)
    assert dt.date(2026, 7, 28) in cal                      # inside the hole
    assert cal[-1] == dt.date(2026, 9, 30)                  # month completed
    sw = rolls.switch_days(cal, rolls.ROBUSTA)
    assert sw == [dt.date(2026, 7, 28), dt.date(2026, 9, 25)]
    flags = rolls.roll_flags(own, sw)
    # the July switch day has no observation: its flag lands on the first
    # observation after the hole; the September one lies after the last
    # observation and flags nothing yet
    assert [d for d, f in flags.items() if f] == [dt.date(2026, 8, 24)]


def test_weekend_rows_do_not_enter_the_calendar():
    own = _weekdays(dt.date(2026, 9, 1), dt.date(2026, 9, 30)) + [dt.date(2026, 9, 26)]
    assert dt.date(2026, 9, 26) not in rolls.trading_calendar(own)


def test_stale_flag_and_runs():
    idx = pd.date_range("2024-01-01", periods=12, freq="D")
    s = pd.Series([1, 2, 2, 2, 2, 2, 3, 4, 4, 4, 4, 5], index=idx, dtype=float)
    f = quality.stale_flag(s)
    assert f.tolist() == [False] + [True] * 5 + [False] * 6     # run of 5, not the run of 4
    runs = quality.stale_runs(s)
    assert len(runs) == 1 and runs["observations"].iloc[0] == 5 and runs["value"].iloc[0] == 2
    s2 = s.copy()
    s2.iloc[3] = np.nan                                         # now only 4 observations of "2"
    assert quality.stale_flag(s2).sum() == 0
    s3 = pd.Series([1, 2, 2, 2, np.nan, 2, 2, 3, 4, 4, 4, 5], index=idx, dtype=float)
    f3 = quality.stale_flag(s3)                                 # a missing day inside the run
    assert f3.sum() == 5 and not f3.iloc[4]                     # does not break it


def test_extreme_flag_uses_consecutive_observations():
    idx = pd.date_range("2024-01-01", periods=5, freq="D")
    s = pd.Series([100.0, np.nan, 112.0, 113.0, 100.0], index=idx)
    f = quality.extreme_flag(s, 0.10)
    assert f.tolist() == [False, False, True, False, True]


def test_gaps_are_labelled():
    days = [dt.date(2024, 3, 16), dt.date(2024, 3, 17), dt.date(2024, 3, 25),   # Nowruz 1403
            dt.date(2024, 3, 26), dt.date(2024, 6, 1), dt.date(2024, 6, 10),    # other
            dt.date(2024, 12, 23), dt.date(2025, 1, 2)]                         # year end
    s = pd.Series(1.0, index=pd.to_datetime(days))

    def short(g):
        return g[g["calendar_days"] < 60]

    iran = short(quality.gaps(s))                     # a series from Iran
    assert iran["kind"].tolist() == ["nowruz", "other", "other"]
    assert iran["calendar_days"].tolist() == [8, 9, 10]
    exchange = short(quality.gaps(s, iran=False))     # an exchange series
    assert exchange["kind"].tolist() == ["other", "other", "year_end"]


def test_a_long_gap_is_never_called_a_holiday():
    s = pd.Series(1.0, index=pd.to_datetime([dt.date(2015, 3, 17), dt.date(2015, 5, 17)]))
    assert quality.gaps(s)["kind"].tolist() == ["other"]      # 61 days across Nowruz


def test_level_breaks_find_a_times_ten_jump():
    idx = pd.date_range("2024-01-01", periods=6, freq="D")
    s = pd.Series([42000, 42100, 4210, 4200, 42050, 42000], index=idx, dtype=float)
    b = quality.level_breaks(s)
    assert list(b["date"]) == [idx[2], idx[4]] and (b["factor"] == 10).all()
    assert quality.level_breaks(pd.Series([100, 118, 95, 130.0], index=idx[:4])).empty


def test_near_stale_catches_a_quote_that_barely_moves():
    idx = pd.date_range("2026-02-28", periods=12, freq="D")
    # five quotes within a few hundred rials of each other, then a real move,
    # then four quotes that barely move (too short to count)
    s = pd.Series([1_663_600, 1_663_400, 1_663_800, 1_663_500, 1_663_600,
                   1_470_000, 1_490_000,
                   1_520_000, 1_520_100, 1_520_000, 1_520_200, 1_560_000],
                  index=idx, dtype=float)
    f = quality.near_stale_flag(s)
    assert f.tolist() == [True] * 5 + [False] * 7
    assert not quality.stale_flag(s).any()                    # the exact-repeat check misses it
    runs = quality.near_stale_runs(s)
    assert len(runs) == 1
    r = runs.iloc[0]
    assert (r["first_date"], r["last_date"], r["observations"]) == (idx[0], idx[4], 5)
    assert (r["first_value"], r["last_value"]) == (1_663_600, 1_663_600)
    assert 0 < r["max_abs_step"] < quality.NEAR_STALE_STEP


def test_near_stale_includes_exact_repeats_and_skips_missing_days():
    idx = pd.date_range("2024-01-01", periods=9, freq="D")
    s = pd.Series([100, 100, np.nan, 100, 100, 100, 101, 101, 101], index=idx, dtype=float)
    f = quality.near_stale_flag(s)
    assert f.sum() == 5 and not f.iloc[2]                     # 5 observations of 100; the gap is not one
    assert quality.near_stale_runs(s)["max_abs_step"].iloc[0] == 0
    assert quality.stale_flag(s).sum() == 5                   # the exact check agrees here


def test_near_stale_threshold_is_on_each_step_not_on_the_whole_run():
    idx = pd.date_range("2024-01-01", periods=6, freq="D")
    drift = pd.Series(100 * np.exp(np.arange(6) * 0.0004), index=idx)     # 0.04% a day, 0.2% in total
    assert quality.near_stale_flag(drift).all()
    faster = pd.Series(100 * np.exp(np.arange(6) * 0.0006), index=idx)    # 0.06% a day
    assert not quality.near_stale_flag(faster).any()

