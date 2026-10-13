import datetime as dt

import numpy as np
import pandas as pd
import pytest

from coffee_risk.data import weekly


def test_wednesdays_first_on_or_after_last_on_or_before():
    w = weekly.wednesdays(dt.date(2012, 10, 9), dt.date(2012, 10, 30))     # Tue .. Tue
    assert [d.date() for d in w] == [dt.date(2012, 10, 10), dt.date(2012, 10, 17),
                                     dt.date(2012, 10, 24)]
    w = weekly.wednesdays(dt.date(2012, 10, 10), dt.date(2012, 10, 24))    # Wed .. Wed
    assert len(w) == 3 and all(d.weekday() == 2 for d in w)


def test_closed_wednesday_falls_back_to_the_last_earlier_observation():
    weds = weekly.wednesdays(dt.date(2024, 1, 3), dt.date(2024, 1, 24))
    s = pd.Series({pd.Timestamp("2024-01-02"): 10.0,       # Tue
                   pd.Timestamp("2024-01-08"): 11.0,       # Mon; Wed 10 Jan is closed
                   pd.Timestamp("2024-01-11"): 99.0,       # Thu: belongs to the next week
                   pd.Timestamp("2024-01-17"): 12.0})      # Wed
    lo = weekly.last_observation(s, weds)
    assert lo["level"].tolist() == [10.0, 11.0, 12.0, 12.0]
    assert [d.date() for d in lo["obs_date"]] == [dt.date(2024, 1, 2), dt.date(2024, 1, 8),
                                                  dt.date(2024, 1, 17), dt.date(2024, 1, 17)]


def test_no_level_before_the_series_starts():
    weds = weekly.wednesdays(dt.date(2024, 1, 3), dt.date(2024, 1, 17))
    s = pd.Series({pd.Timestamp("2024-01-09"): 5.0})
    lo = weekly.last_observation(s, weds)
    assert np.isnan(lo["level"].iloc[0]) and pd.isna(lo["obs_date"].iloc[0])
    assert lo["level"].tolist()[1:] == [5.0, 5.0]


def _daily(seed=11, days=140):
    """A daily panel with the columns weekly.build needs. The dollar has a
    value every day; coffee is closed at weekends and on two Wednesdays."""
    idx = pd.date_range("2024-01-02", periods=days, freq="D", name="date")      # starts on a Tuesday
    rng = np.random.default_rng(seed)
    walk = lambda scale, start: start * np.exp(np.cumsum(rng.normal(0, scale, days)))
    p = pd.DataFrame(index=idx)
    p["usd_free"] = walk(0.01, 600_000)
    p["usd_tgju"] = walk(0.01, 598_000)
    p["fund_tr_index"] = 100 * np.exp(np.arange(days) * 0.0008)
    for name in ("arabica_usd_kg", "robusta_usd_kg"):
        s = pd.Series(walk(0.02, 5.0), index=idx)
        s[idx.weekday >= 5] = np.nan
        s[[pd.Timestamp("2024-01-17"), pd.Timestamp("2024-03-06")]] = np.nan   # closed Wednesdays
        p[name] = s
    for flag in ("arabica_roll", "robusta_roll", *weekly.NEAR_STALE.values()):
        p[flag] = False
    return p


def test_weekly_usd_returns_telescope():
    p = _daily()
    w = weekly.build(p, save=False)
    assert w.index[0].date() == dt.date(2024, 1, 3)           # first Wednesday on or after the start
    assert all(d.weekday() == 2 for d in w.index)
    assert np.isnan(w["usd_free_ret"].iloc[0])                # the first row has no return
    for col in ("usd_free", "usd_tgju", "fund_tr_index"):
        assert w[f"{col}_ret"].sum() == pytest.approx(np.log(w[col].iloc[-1] / w[col].iloc[0]))
    # the dollar is observed every day, so the level is the Wednesday's own value
    assert (w["usd_free_obs_date"] == w.index).all()
    assert w["usd_free"].tolist() == p.loc[w.index, "usd_free"].tolist()


def test_coffee_week_without_a_roll_equals_the_raw_level_change():
    w = weekly.build(_daily(), save=False)
    np.testing.assert_allclose(w["arabica_usd_kg_ret"].dropna(), w["arabica_usd_kg_ret_raw"].dropna())
    # closed Wednesday 17 Jan: the level is Tuesday's, and the week is not flagged (1 day old)
    row = w.loc["2024-01-17"]
    assert row["arabica_usd_kg_obs_date"].date() == dt.date(2024, 1, 16)
    assert not row["arabica_usd_kg_old"]


def test_roll_day_inside_a_week_is_left_out_of_that_week_only():
    p = _daily()
    roll_day = pd.Timestamp("2024-02-12")                      # a Monday, in the week ending Wed 14 Feb
    p.loc[roll_day, "arabica_roll"] = True
    w = weekly.build(p, save=False)
    daily_r = np.log(p["arabica_usd_kg"].dropna()).diff()
    week = pd.Timestamp("2024-02-14")
    assert w.loc[week, "arabica_usd_kg_ret"] == pytest.approx(
        w.loc[week, "arabica_usd_kg_ret_raw"] - daily_r[roll_day])
    others = w.index != week
    np.testing.assert_allclose(w.loc[others, "arabica_usd_kg_ret"].dropna(),
                               w.loc[others, "arabica_usd_kg_ret_raw"].dropna())
    np.testing.assert_allclose(w["robusta_usd_kg_ret"].dropna(),                 # the other series
                               w["robusta_usd_kg_ret_raw"].dropna())             # is untouched
    assert (p["arabica_usd_kg"].dropna() == _daily()["arabica_usd_kg"].dropna()).all()   # levels as they were


def test_roll_on_a_wednesday_belongs_to_the_week_that_ends_that_day():
    p = _daily()
    roll_day = pd.Timestamp("2024-02-21")                      # a Wednesday
    p.loc[roll_day, "robusta_roll"] = True
    w = weekly.build(p, save=False)
    daily_r = np.log(p["robusta_usd_kg"].dropna()).diff()
    assert w.loc[roll_day, "robusta_usd_kg_ret"] == pytest.approx(
        w.loc[roll_day, "robusta_usd_kg_ret_raw"] - daily_r[roll_day])
    nxt = roll_day + pd.Timedelta(days=7)
    assert w.loc[nxt, "robusta_usd_kg_ret"] == pytest.approx(w.loc[nxt, "robusta_usd_kg_ret_raw"])


def test_returns_around_a_hole_are_missing_not_zero():
    """D-018: inside a hole the level is stale (the return would be a false
    zero) and the first return after it would carry the whole hole."""
    p = _daily()
    hole = (p.index >= "2024-03-01") & (p.index <= "2024-03-20")
    p.loc[hole, "robusta_usd_kg"] = np.nan                     # last observation: Thu 29 Feb
    w = weekly.build(p, save=False)
    old = w["robusta_usd_kg_old"]
    assert not old.loc["2024-02-28"]
    assert old.loc["2024-03-06"] and old.loc["2024-03-13"] and old.loc["2024-03-20"]   # 6, 13, 20 days
    assert not old.loc["2024-03-27"]
    r = w["robusta_usd_kg_ret"]
    assert r.loc[["2024-03-06", "2024-03-13", "2024-03-20"]].isna().all()     # stale level
    assert np.isnan(r.loc["2024-03-27"])                                         # first after the hole
    assert not np.isnan(r.loc["2024-02-28"]) and not np.isnan(r.loc["2024-04-03"])
    # the raw level change is kept, so the move across the hole stays visible
    raw = w["robusta_usd_kg_ret_raw"]
    assert raw.loc["2024-03-13"] == 0.0
    assert raw.loc["2024-03-27"] == pytest.approx(
        np.log(w.loc["2024-03-27", "robusta_usd_kg"] / w.loc["2024-03-20", "robusta_usd_kg"]))
    assert w["arabica_usd_kg_ret"].iloc[1:].notna().all()                        # other series untouched


def test_a_short_hole_between_two_fresh_wednesdays_still_empties_the_return():
    """Observations on Tue 31 Mar and Wed 8 Apr: both Wednesdays have a fresh
    level, but the 8-day hole makes the second return span more than a week."""
    weds = weekly.wednesdays(dt.date(2026, 3, 18), dt.date(2026, 4, 22))
    days = pd.bdate_range("2026-03-16", "2026-04-22")
    days = days[(days <= "2026-03-31") | (days >= "2026-04-08")]
    s = pd.Series(np.linspace(3.5, 3.3, len(days)), index=days)
    after = weekly.after_hole_weeks(s, weds)
    assert after[after].index.tolist() == [pd.Timestamp("2026-04-08")]
    lo = weekly.last_observation(s, weds)
    assert not weekly.stale_weeks(lo["obs_date"]).any()                          # ages 1 and 0 days
    assert weekly.missing_weeks(s, weds)[lambda m: m].index.tolist() == [pd.Timestamp("2026-04-08")]
    five = pd.Series(1.0, index=pd.to_datetime(["2026-03-27", "2026-04-01"]))    # 5 days: not a hole
    assert not weekly.after_hole_weeks(five, weds).any()


# ------------------------------------------------------------ ICO gap fill
def _with_ico(p, noise, seed=5):
    """Add an ICO-like indicator: a different level (a physical price) whose
    daily returns equal robusta's plus noise; quoted on every weekday."""
    rng = np.random.default_rng(seed)
    base = _daily()["robusta_usd_kg"].ffill()                  # the full path, before any hole
    r = np.log(base).diff().fillna(0.0) + rng.normal(0, noise, len(base))
    ico = 180.0 * np.exp(r.cumsum())
    ico[ico.index.weekday >= 5] = np.nan
    p[weekly.ICO_COL] = ico
    return p


@pytest.fixture
def ico_window(monkeypatch):
    monkeypatch.setattr(weekly, "ICO_TEST_FROM", dt.date(2024, 1, 1))
    monkeypatch.setattr(weekly, "ICO_TEST_TO", dt.date(2024, 12, 31))


def test_ico_fills_the_missing_robusta_returns_when_the_test_passes(ico_window):
    p = _with_ico(_daily(), noise=0.002)
    hole = (p.index >= "2024-03-01") & (p.index <= "2024-03-20")
    p.loc[hole, "robusta_usd_kg"] = np.nan
    w = weekly.build(p, save=False)
    weds = w.index
    rb = weekly.robusta_inputs(p, weds)
    assert rb["test"]["passed"] and rb["test"]["corr"] > 0.9
    gap = ["2024-03-06", "2024-03-13", "2024-03-20", "2024-03-27"]
    assert w.loc[gap, "robusta_return_source"].tolist() == ["ico"] * 4
    np.testing.assert_allclose(w.loc[gap, "robusta_usd_kg_ret"], rb["ico_ret"].loc[gap])
    assert (w["robusta_return_source"].drop(index=pd.to_datetime(gap)).iloc[1:] == "tgju").all()
    # levels are never taken from ICO: the level column keeps its hole
    assert (w.loc[gap[:3], "robusta_usd_kg_obs_date"] == pd.Timestamp("2024-02-29")).all()
    assert w.loc[gap[:3], "robusta_usd_kg_old"].all()
    assert w["robusta_usd_kg"].max() < 50                      # still USD/kg, not US cents/lb
    # weeks that had a TGJU return keep it
    plain = weekly.build(_daily(), save=False)
    keep = ~w.index.isin(pd.to_datetime(gap))
    np.testing.assert_allclose(w.loc[keep, "robusta_usd_kg_ret"].iloc[1:],
                               plain.loc[keep, "robusta_usd_kg_ret"].iloc[1:])


def test_weeks_stay_missing_when_the_test_fails(ico_window):
    p = _with_ico(_daily(), noise=0.05)                        # ICO barely related to TGJU
    hole = (p.index >= "2024-03-01") & (p.index <= "2024-03-20")
    p.loc[hole, "robusta_usd_kg"] = np.nan
    w = weekly.build(p, save=False)
    assert not weekly.robusta_inputs(p, w.index)["test"]["passed"]
    gap = ["2024-03-06", "2024-03-13", "2024-03-20", "2024-03-27"]
    assert w.loc[gap, "robusta_usd_kg_ret"].isna().all()
    assert w.loc[gap, "robusta_return_source"].isna().all()
    assert "ico" not in set(w["robusta_return_source"].dropna())


def test_agreement_test_rules():
    idx = weekly.wednesdays(dt.date(2025, 1, 1), dt.date(2025, 12, 31))
    rng = np.random.default_rng(1)
    t = pd.Series(rng.normal(0, 0.04, len(idx)), index=idx)
    no_roll = pd.Series(False, index=idx)
    same = weekly.ico_agreement(t, t * 0.95, no_roll)
    assert same["passed"] and same["weeks"] == len(idx) and same["vol_ratio"] == pytest.approx(0.95)
    assert not weekly.ico_agreement(t, t * 0.7, no_roll)["passed"]            # volatility ratio too low
    assert not weekly.ico_agreement(t, t * 1.3, no_roll)["passed"]            # ... too high
    noisy = t + pd.Series(rng.normal(0, 0.04, len(idx)), index=idx)
    assert not weekly.ico_agreement(t, noisy, no_roll)["passed"]              # correlation too low
    # weeks with a roll, and weeks where either return is missing, are left out
    rolls = no_roll.copy()
    rolls.iloc[:5] = True
    ico = (t * 0.95).copy()
    ico.iloc[10:13] = np.nan
    assert weekly.ico_agreement(t, ico, rolls)["weeks"] == len(idx) - 5 - 3
    # weeks outside the fixed test window are left out
    late = t.copy()
    late.index = late.index + pd.Timedelta(weeks=200)
    assert weekly.ico_agreement(late, late, pd.Series(False, index=late.index))["weeks"] == 0


def test_ico_month_missing_makes_its_weeks_unavailable():
    """A rejected month leaves a hole in the ICO series: the weekly ICO return
    is empty for every week that touches the hole."""
    weds = weekly.wednesdays(dt.date(2026, 6, 3), dt.date(2026, 8, 26))
    days = pd.bdate_range("2026-06-01", "2026-08-31")
    ico = pd.Series(180.0 + np.arange(len(days)) * 0.1, index=days)
    ico = ico[(ico.index.month != 7)]                           # July rejected
    r = weekly.ico_weekly_return(ico, weds)
    assert not np.isnan(r.loc["2026-07-01"])                    # level of Wed 1 Jul = Tue 30 Jun, fresh
    assert r.loc["2026-07-08":"2026-08-05"].isna().all()        # stale end, stale start, or end of the hole
    assert not np.isnan(r.loc["2026-08-12"])


def test_near_stale_flag_follows_the_observation_used():
    p = _daily()
    p.loc["2024-02-07", "usd_free_near_stale"] = True          # a Wednesday
    p.loc["2024-02-08", "usd_tgju_near_stale"] = True          # a Thursday: not a level day
    w = weekly.build(p, save=False)
    assert w["usd_free_near_stale"].sum() == 1 and w.loc["2024-02-07", "usd_free_near_stale"]
    assert w["usd_tgju_near_stale"].sum() == 0


def test_a_series_that_starts_later_is_empty_until_then():
    p = _daily()
    p.loc[p.index < "2024-02-05", "fund_tr_index"] = np.nan    # starts on a Monday
    w = weekly.build(p, save=False)
    assert w.loc[:"2024-01-31", "fund_tr_index"].isna().all()
    assert w.loc["2024-02-07", "fund_tr_index"] > 0
    assert np.isnan(w.loc["2024-02-07", "fund_tr_index_ret"])  # first level: no return yet
    assert not np.isnan(w.loc["2024-02-14", "fund_tr_index_ret"])
