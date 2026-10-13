import datetime as dt

import numpy as np
import pandas as pd
import pytest

from coffee_risk import eda
from coffee_risk.data import weekly


# ------------------------------------------------------------- mix index
def test_mix_return_is_the_log_of_the_weighted_gross_returns():
    ra, rr = np.log(1.10), np.log(0.90)                      # arabica +10%, robusta -10%
    assert eda.mix_return(ra, rr, 0.5) == pytest.approx(np.log(0.5 * 1.10 + 0.5 * 0.90))
    assert eda.mix_return(ra, rr, 0.5) == pytest.approx(0.0)  # a 50/50 basket is flat
    assert eda.mix_return(ra, rr, 1.0) == ra and eda.mix_return(ra, rr, 0.0) == rr
    assert eda.mix_return(ra, rr, 0.25) == pytest.approx(np.log(0.25 * 1.10 + 0.75 * 0.90))
    # it is not the average of the two log returns
    assert eda.mix_return(np.log(2.0), np.log(0.5), 0.5) == pytest.approx(np.log(1.25))


def test_rial_cost_return_adds_the_dollar_and_the_coffee_parts():
    idx = pd.date_range("2024-01-03", periods=4, freq="7D")
    r = pd.DataFrame({"usd": [0.02, -0.01, 0.05, 0.0], "arabica": [0.03, 0.01, -0.04, np.nan],
                      "robusta": [-0.01, 0.02, 0.01, 0.03], "fund": 0.004}, index=idx)
    c = eda.rial_cost_returns(r)
    np.testing.assert_allclose(c["cost_arabica"].iloc[:3], r["usd"].iloc[:3] + r["arabica"].iloc[:3])
    np.testing.assert_allclose(c["cost_robusta"], r["usd"] + r["robusta"])
    expected = r["usd"] + np.log(0.5 * np.exp(r["arabica"]) + 0.5 * np.exp(r["robusta"]))
    np.testing.assert_allclose(c["cost_mix"].iloc[:3], expected.iloc[:3])
    assert np.isnan(c["cost_mix"].iloc[3]) and np.isnan(c["cost_arabica"].iloc[3])   # a missing leg
    assert not np.isnan(c["cost_robusta"].iloc[3])


# ----------------------------------------------- non-overlapping selection
def _changes(values, start="2020-01-01"):
    return pd.Series(values, index=pd.date_range(start, periods=len(values), freq="7D"), dtype=float)


def test_top_nonoverlapping_windows_do_not_share_a_week():
    v = np.zeros(60)
    v[20], v[22], v[23] = 0.50, 0.90, 0.80         # one cluster: only its best window may be taken
    v[40] = 0.60
    v[55] = 0.30
    ch = _changes(v)
    top = eda.top_nonoverlapping(ch, k=3, weeks=13)
    assert top.tolist() == [0.90, 0.60, 0.30]
    ends = list(top.index)
    for i, a in enumerate(ends):
        for b in ends[i + 1:]:
            assert abs((a - b).days) >= 13 * 7       # windows (end-13w, end] are disjoint
    # exactly 13 weeks apart is allowed: the two windows only touch
    v2 = np.zeros(40)
    v2[10], v2[23] = 0.9, 0.8
    assert eda.top_nonoverlapping(_changes(v2), k=2, weeks=13).tolist() == [0.9, 0.8]
    v2[22] = 0.85                                   # 12 weeks apart: overlaps the first, is skipped
    assert eda.top_nonoverlapping(_changes(v2), k=2, weeks=13).tolist() == [0.9, 0.8]


def test_top_nonoverlapping_skips_missing_and_can_take_the_smallest():
    ch = _changes([0.1, np.nan, 0.4, -0.3, 0.2, np.nan, -0.5, 0.0])
    assert eda.top_nonoverlapping(ch, k=2, weeks=2).tolist() == [0.4, 0.2]
    assert eda.top_nonoverlapping(ch, k=2, weeks=2, largest=False).tolist() == [-0.5, -0.3]
    assert len(eda.top_nonoverlapping(ch, k=10, weeks=4)) < 10      # runs out of disjoint windows


# ------------------------------------------------------ stress-window search
def test_best_window_returns_the_true_maximum_inside_the_range():
    """Small fixture: a price path with one planted 13-week surge. Brute force
    over every window gives the answer the search must return."""
    weeks = 13
    idx = pd.date_range("2018-01-03", periods=120, freq="7D")
    rng = np.random.default_rng(42)
    steps = rng.normal(0, 0.01, len(idx))
    steps[40:53] += 0.06                             # the surge: weeks 40..52
    level = pd.Series(np.cumsum(steps), index=idx)
    ch = level - level.shift(weeks)
    first, last = idx[10].date(), idx[100].date()
    best = eda.best_window(ch, first, last, weeks)
    brute = max(((level.iloc[e] - level.iloc[e - weeks], idx[e]) for e in range(weeks, len(idx))
                 if idx[e - weeks] >= pd.Timestamp(first) and idx[e] <= pd.Timestamp(last)))
    assert best == brute[1] == idx[52]               # the window that covers exactly the surge
    assert eda.window_start(best, weeks) == idx[39]


def test_best_window_must_lie_wholly_inside_the_range():
    weeks = 13
    idx = pd.date_range("2018-01-03", periods=80, freq="7D")
    level = pd.Series(0.0, index=idx)
    level.iloc[20:] += 1.0                           # a jump at week 20: the best windows straddle it
    level.iloc[60:] += 0.2                           # a smaller one at week 60
    ch = level - level.shift(weeks)
    # range starts after the big jump: only the small one can be found
    best = eda.best_window(ch, idx[30].date(), idx[79].date(), weeks)
    assert ch[best] == pytest.approx(0.2) and eda.window_start(best, weeks) >= idx[30]
    # a range too short for any 13-week window
    assert eda.best_window(ch, idx[30].date(), idx[40].date(), weeks) is None
    # an open-ended range (the war episode) runs to the last week
    assert eda.best_window(ch, idx[50].date(), None, weeks) is not None


# -------------------------------------------------- indices built on the panel
def _panel(weeks=60, seed=3):
    """A weekly panel with the columns eda needs; coffee returns = raw level changes."""
    idx = weekly.wednesdays(dt.date(2024, 1, 3), dt.date(2024, 1, 3) + dt.timedelta(weeks=weeks - 1))
    rng = np.random.default_rng(seed)
    w = pd.DataFrame(index=idx)
    for col, vol, start in (("usd_free", 0.03, 600_000.0), ("fund_tr_index", 0.001, 100.0),
                            ("arabica_usd_kg", 0.04, 5.0), ("robusta_usd_kg", 0.03, 4.0)):
        level = start * np.exp(np.cumsum(rng.normal(0.002, vol, len(idx))))
        w[col] = level
        w[f"{col}_ret"] = np.log(w[col]).diff()
        w[f"{col}_old"] = False
        if col.endswith("usd_kg"):
            w[f"{col}_ret_raw"] = w[f"{col}_ret"]
    w["robusta_return_source"] = np.where(w["robusta_usd_kg_ret"].notna(), "tgju", None)
    return w


def test_thirteen_week_changes_equal_level_changes_when_nothing_is_missing():
    w = _panel()
    idx, stale, gaps = eda.log_index(w)
    ch = eda.horizon_changes(idx, stale, gaps, 13)
    for name, col in eda.SERIES.items():
        expected = np.log(w[col] / w[col].shift(13))
        np.testing.assert_allclose(ch[name].dropna(), expected.dropna())
    np.testing.assert_allclose(ch["cost_arabica"].dropna(), (ch["usd"] + ch["arabica"]).dropna())
    assert ch.iloc[:13].isna().all().all() and ch.iloc[13:].notna().all().all()


def test_a_roll_week_stays_out_of_the_coffee_index():
    w = _panel()
    week = w.index[20]
    w.loc[week, "arabica_usd_kg_ret"] -= 0.05        # the roll-excluded return differs from the raw one
    idx, stale, gaps = eda.log_index(w)
    ch = eda.horizon_changes(idx, stale, gaps, 13)
    raw = np.log(w["arabica_usd_kg"] / w["arabica_usd_kg"].shift(13))
    assert ch.loc[w.index[25], "arabica"] == pytest.approx(raw[w.index[25]] - 0.05)   # window covers it
    assert ch.loc[w.index[40], "arabica"] == pytest.approx(raw[w.index[40]])          # window after it


def _with_hole(w, fills):
    """Robusta hole: weeks 30-32 have a stale level, week 33 is the first
    after the hole. D-018 emptied those four returns; `fills` maps a week
    position to the ICO return put back in."""
    hole, back = w.index[30:33], w.index[33]
    w.loc[hole, "robusta_usd_kg"] = w["robusta_usd_kg"].iloc[29]      # the level stays at the last observation
    w.loc[hole, "robusta_usd_kg_old"] = True
    w["robusta_usd_kg_ret_raw"] = np.log(w["robusta_usd_kg"]).diff()
    w.loc[list(hole) + [back], "robusta_usd_kg_ret"] = np.nan         # D-018
    w["robusta_return_source"] = np.where(w["robusta_usd_kg_ret"].notna(), "tgju", None)
    for pos, value in fills.items():
        w.loc[w.index[pos], "robusta_usd_kg_ret"] = value
        w.loc[w.index[pos], "robusta_return_source"] = "ico"
    return w


def test_coffee_changes_are_sums_of_weekly_returns_including_ico_weeks():
    """D-022: a multi-week coffee change is the sum of the panel's weekly
    returns, roll days out and ICO-filled weeks in. The level of the source
    across the hole (which would let a roll jump back in) is not used."""
    fills = {30: 0.01, 31: -0.02, 32: 0.03, 33: -0.04}
    w = _with_hole(_panel(), fills)
    idx, stale, gaps = eda.log_index(w)
    ch = eda.horizon_changes(idx, stale, gaps, 13)
    assert not stale["robusta"].any() and gaps["robusta"].iloc[-1] == 0       # nothing missing now
    for end_pos in (31, 33, 40, 45):
        end = w.index[end_pos]
        expected = w["robusta_usd_kg_ret"].iloc[end_pos - 12:end_pos + 1].sum()
        assert ch.loc[end, "robusta"] == pytest.approx(expected)              # incl. windows ending in the hole
    raw_change = np.log(w["robusta_usd_kg"].iloc[40] / w["robusta_usd_kg"].iloc[27])
    assert ch.loc[w.index[40], "robusta"] != pytest.approx(raw_change)        # not the level endpoints
    # the baskets use the same weekly returns
    mix = eda.mix_return(w["arabica_usd_kg_ret"], w["robusta_usd_kg_ret"], 0.5)
    assert ch.loc[w.index[40], "coffee_mix"] == pytest.approx(mix.iloc[28:41].sum())
    assert ch.loc[w.index[40], "cost_mix"] == pytest.approx(
        ch.loc[w.index[40], "usd"] + ch.loc[w.index[40], "coffee_mix"])


def test_a_week_without_a_return_empties_every_window_that_contains_it():
    w = _with_hole(_panel(), {31: -0.02, 32: 0.03, 33: -0.04})                # week 30 stays empty
    idx, stale, gaps = eda.log_index(w)
    ch = eda.horizon_changes(idx, stale, gaps, 13)
    assert ch["robusta"].iloc[30:43].isna().all()                             # windows (e-13, e] containing week 30
    assert ch["robusta"].iloc[[29, 43]].notna().all()                         # the ones just before and after
    assert ch["cost_mix"].iloc[30:43].isna().all() and ch["cost_robusta"].iloc[30:43].isna().all()
    assert ch["usd"].iloc[30:43].notna().all() and ch["cost_arabica"].iloc[30:43].notna().all()
    path = eda.path_change(idx, stale, gaps, w.index[20], w.index[50])
    assert np.isnan(path["robusta"]) and not np.isnan(path["arabica"])
    path = eda.path_change(idx, stale, gaps, w.index[31], w.index[50])        # starts after the empty week
    assert path["robusta"] == pytest.approx(w["robusta_usd_kg_ret"].iloc[32:51].sum())


def test_stale_level_empties_dollar_and_fund_changes_only_at_the_endpoints():
    w = _panel()
    w.loc[w.index[25], "fund_tr_index_old"] = True
    idx, stale, gaps = eda.log_index(w)
    ch = eda.horizon_changes(idx, stale, gaps, 13)
    assert np.isnan(ch.loc[w.index[25], "fund"]) and np.isnan(ch.loc[w.index[38], "fund"])
    assert ch["fund"].iloc[26:38].notna().all()                               # windows that only pass over it


def test_jensen_gap_is_about_half_the_variance():
    rng = np.random.default_rng(2)
    x = pd.Series(rng.normal(0.05, 0.20, 200_000))                            # 13-week log changes
    j = eda.jensen_gap(x)
    assert j["mean_simple"] > j["from_mean_log"]
    assert j["gap"] == pytest.approx(np.exp(0.05) * (np.exp(0.02) - 1), rel=0.03)   # lognormal value
    assert j["half_variance"] == pytest.approx(0.02, rel=0.02)
    flat = eda.jensen_gap(pd.Series([0.03] * 50))                             # no spread, no gap
    assert flat["gap"] == pytest.approx(0.0, abs=1e-12)


def test_carry_is_indexed_by_start_week_and_signed_as_usd_minus_fund():
    w = _panel()
    idx, stale, gaps = eda.log_index(w)
    c = eda.carry(idx, stale, gaps, 13)
    t = w.index[5]
    usd = np.log(w["usd_free"].iloc[18] / w["usd_free"].iloc[5])
    fund = np.log(w["fund_tr_index"].iloc[18] / w["fund_tr_index"].iloc[5])
    assert c.loc[t, "carry"] == pytest.approx(usd - fund)
    assert c.index[-1] == w.index[-14]                                # last start with 13 weeks ahead
    d = eda.describe_carry(c["carry"])
    assert d["n"] == len(c) and 0.0 <= d["share_positive"] <= 1.0


def test_backtest_sample_starts_in_the_week_of_1394_02_30():
    assert eda.BACKTEST_START == dt.date(2015, 5, 20) and eda.BACKTEST_START.weekday() == 2
    idx = weekly.wednesdays(dt.date(2015, 1, 7), dt.date(2015, 12, 30))
    s = pd.Series(1.0, index=idx)
    assert eda.sample(s, "backtest").index[0] == pd.Timestamp("2015-05-20")
    # a 13-week window belongs to the backtest sample only if it starts inside it
    assert eda.sample_windows(s, "backtest").index[0] == pd.Timestamp("2015-08-19")
    assert len(eda.sample(s, "full")) == len(s)


def test_war_start_week_is_the_last_wednesday_before_9_esfand_1404():
    from coffee_risk.dates import from_jalali
    war = from_jalali(1404, 12, 9)
    assert eda.WAR_START_WEEK.weekday() == 2
    assert 0 < (war - eda.WAR_START_WEEK).days <= 7


def test_variance_ratio_is_one_for_independent_weeks_and_higher_for_trends():
    rng = np.random.default_rng(0)
    iid = pd.Series(rng.normal(0, 0.03, 5000))
    assert eda.variance_ratio(iid, 13)["variance_ratio"] == pytest.approx(1.0, abs=0.15)
    e = rng.normal(0, 0.03, 5000)
    ar = np.zeros(5000)
    for i in range(1, 5000):
        ar[i] = 0.5 * ar[i - 1] + e[i]
    assert eda.variance_ratio(pd.Series(ar), 13)["variance_ratio"] > 2.0


def test_acf_ignores_gaps_without_shifting_lags():
    s = pd.Series(np.tile([1.0, -1.0], 50))                           # perfect alternation
    a = eda.acf(s, 4)
    assert a[1] == pytest.approx(-1.0) and a[2] == pytest.approx(1.0)
    s.iloc[[10, 11, 40]] = np.nan
    b = eda.acf(s, 4)
    assert b[1] == pytest.approx(-1.0) and b[2] == pytest.approx(1.0)
