import numpy as np
import pandas as pd
import pytest

from coffee_risk.data.common_calendar import (common_calendar_returns, common_dates,
                                              window_counts)


@pytest.fixture
def panel():
    """60 calendar days; `fx` is closed on Fridays, `coffee` on weekends and
    on two holidays, so the two calendars only partly overlap."""
    idx = pd.date_range("2024-01-01", periods=60, freq="D")
    rng = np.random.default_rng(7)
    fx = pd.Series(50_000 * np.exp(np.cumsum(rng.normal(0, 0.01, 60))), index=idx)
    coffee = pd.Series(4.0 * np.exp(np.cumsum(rng.normal(0, 0.02, 60))), index=idx)
    fx[idx.weekday == 4] = np.nan
    coffee[idx.weekday >= 5] = np.nan
    coffee.loc[["2024-01-15", "2024-02-19"]] = np.nan
    p = pd.DataFrame({"fx": fx, "coffee": coffee})
    p["coffee_roll"] = False
    p["coffee_bad"] = False
    return p


def test_common_dates_need_every_column(panel):
    d = common_dates(panel, ["fx", "coffee"])
    assert panel.loc[d, ["fx", "coffee"]].notna().all().all()
    assert not any(x.weekday() >= 4 for x in d)            # no Fri/Sat/Sun
    assert pd.Timestamp("2024-01-15") not in d


def test_returns_telescope(panel):
    """Sum of log returns from the first to the last common date equals
    log(last / first): no move is lost by skipping closed days."""
    cols = ["fx", "coffee"]
    r = common_calendar_returns(panel, cols)
    d = common_dates(panel, cols)
    for c in cols:
        assert r[c].sum() == pytest.approx(np.log(panel.loc[d[-1], c] / panel.loc[d[0], c]))
    assert r["days"].sum() == (d[-1] - d[0]).days
    assert r.index.equals(d[1:])


def test_no_forward_fill_and_levels_untouched(panel):
    before = panel.copy()
    common_calendar_returns(panel, ["fx", "coffee"], exclude={"coffee": "coffee_roll"})
    pd.testing.assert_frame_equal(panel, before)


def test_roll_exclusion_leaves_other_returns_unchanged(panel):
    cols = ["fx", "coffee"]
    base = common_calendar_returns(panel, cols)
    roll_day = pd.Timestamp("2024-01-23")                   # a common date
    panel.loc[roll_day, "coffee_roll"] = True
    r = common_calendar_returns(panel, cols, exclude={"coffee": "coffee_roll"})
    assert np.isnan(r.loc[roll_day, "coffee"])
    assert r["coffee"].isna().sum() == 1
    others = r.index != roll_day
    np.testing.assert_array_equal(r.loc[others, "coffee"].values,
                                  base.loc[others, "coffee"].values)
    np.testing.assert_array_equal(r["fx"].values, base["fx"].values)   # fx never touched


def test_roll_on_a_non_common_day_hits_the_return_that_spans_it(panel):
    cols = ["fx", "coffee"]
    base = common_calendar_returns(panel, cols)
    panel.loc["2024-01-26", "coffee_roll"] = True           # a Friday: fx is closed
    r = common_calendar_returns(panel, cols, exclude={"coffee": "coffee_roll"})
    nxt = common_dates(panel, cols)
    nxt = nxt[nxt > pd.Timestamp("2024-01-26")][0]          # next common date
    assert np.isnan(r.loc[nxt, "coffee"])
    assert r["coffee"].isna().sum() == 1
    assert r["coffee"].dropna().equals(base["coffee"].drop(index=nxt))


def test_unobserved_flag_removes_the_date_not_the_move(panel):
    cols = ["fx", "coffee"]
    d0 = common_dates(panel, cols)
    bad = d0[10]
    panel.loc[bad, "coffee_bad"] = True
    d1 = common_dates(panel, cols, unobserved={"coffee": "coffee_bad"})
    assert bad not in d1 and len(d1) == len(d0) - 1
    r = common_calendar_returns(panel, cols, unobserved={"coffee": "coffee_bad"})
    assert r["coffee"].sum() == pytest.approx(
        np.log(panel.loc[d1[-1], "coffee"] / panel.loc[d1[0], "coffee"]))


def test_window_counts():
    d = pd.date_range("2024-01-01", periods=200, freq="D")
    assert (window_counts(d, 91) == 91).all()
    assert (window_counts(d[::2], 91).isin([45, 46])).all()
