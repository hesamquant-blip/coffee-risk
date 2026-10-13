import numpy as np
import pandas as pd
import pytest

from coffee_risk.engine import cost as costs
from coffee_risk.engine import inputs as inp
from coffee_risk.engine import stress
from coffee_risk.engine.inputs import Inputs

NO_FRICTION = dict(sellback_spread=0.0, fx_holding_cost=0.0, channel_discount=0.0)


def base_inputs(**changes):
    values = dict(s0=500_000.0, market_rate=0.30, premium_share=0.10)
    values.update(changes)
    return Inputs(**values)


def weekly_panel(n=80, level0=40_000.0, seed=2):
    """A weekly panel with a USD level far away from any s0 used in the tests."""
    idx = pd.date_range("2018-01-03", periods=n, freq="7D")
    rng = np.random.default_rng(seed)
    usd = rng.normal(0.01, 0.04, n)
    usd[0] = np.nan
    df = pd.DataFrame({"usd_free_ret": usd, "arabica_usd_kg_ret": rng.normal(0, 0.04, n),
                       "robusta_usd_kg_ret": rng.normal(0, 0.03, n)}, index=idx)
    df["usd_free"] = level0 * np.exp(np.nan_to_num(usd).cumsum())
    return df


def returns_frame(usd, arabica=None, robusta=None):
    usd = np.asarray(usd, dtype=float)
    zero = np.zeros_like(usd)
    return pd.DataFrame({"usd": usd, "arabica": zero if arabica is None else arabica,
                         "robusta": zero if robusta is None else robusta})


# ------------------------------------------------------------------- replay
def test_replaying_a_flat_path_gives_the_zero_volatility_numbers():
    i = base_inputs(contract="fixed", drift="zero", funds_rate=1.07 ** 4 - 1, **NO_FRICTION)
    p = stress.replay(returns_frame(np.zeros(13)), i)
    assert p.n_paths == 1 and p.horizon == 13 and np.all(p.usd == 1.0) and np.all(p.growth == 1.0)
    out = stress.outcomes(p, i, [inp.now(), inp.staged(), inp.payment_day()])
    assert [round(v, 3) for v in out["cost"] / (i.usd_amount * i.s0)] == [1.070, 1.046, 1.000]
    assert [round(v, 3) for v in out["cost_pct"]] == [1.070, 1.046, 1.000]         # the plan is 1.000
    np.testing.assert_allclose(out["margin"], 1 - 0.75 * out["cost_pct"])
    assert not out["breaks"].any()


def test_a_scenario_starts_from_todays_rate_not_from_the_historical_level():
    wk = weekly_panel()
    start, end = wk.index[20], wk.index[33]
    r = stress.window_returns(wk, start, end)
    assert len(r) == 13 and list(r.columns) == ["usd", "arabica", "robusta"]
    assert r.index[0] == wk.index[21] and r.index[-1] == end              # the weeks after the start
    np.testing.assert_allclose(np.exp(r["usd"].sum()), wk.loc[end, "usd_free"] / wk.loc[start, "usd_free"])
    i = base_inputs(contract="fixed", funds_rate=0.0, **NO_FRICTION)
    p = stress.replay(r, i)
    assert p.usd[0, 0] == 1.0
    late = stress.outcomes(p, i, [inp.payment_day()]).loc["payment_day", "cost"]
    assert late == pytest.approx(100_000 * 500_000.0 * np.exp(r["usd"].sum()))  # s0 = 500,000, not ~40,000
    # the historical level plays no part: move it and nothing changes; move s0 and the cost scales
    shifted = wk.copy()
    shifted["usd_free"] *= 7.0
    pd.testing.assert_frame_equal(stress.window_returns(shifted, start, end), r)
    doubled = i.with_(s0=1_000_000.0)
    assert stress.outcomes(stress.replay(r, doubled), doubled, [inp.payment_day()]).loc["payment_day", "cost"] \
        == pytest.approx(2 * late)
    # the plan is today's: neutral growth at today's market rate
    np.testing.assert_allclose(p.growth, 1.30 ** (np.arange(14) / 52))


def test_a_window_with_a_missing_week_is_refused_not_filled():
    wk = weekly_panel()
    holed = wk.copy()
    holed.loc[wk.index[25], "robusta_usd_kg_ret"] = np.nan
    with pytest.raises(ValueError):
        stress.window_returns(holed, wk.index[20], wk.index[33])
    assert len(stress.window_returns(holed, wk.index[20], wk.index[33], names=("usd", "arabica"))) == 13
    with pytest.raises(ValueError):
        stress.window_returns(wk.drop(index=wk.index[25]), wk.index[20], wk.index[33])


def test_floating_contract_follows_the_scenarios_coffee_path():
    usd = np.r_[np.zeros(12), np.log(1.5)]
    a, r = np.r_[np.zeros(12), np.log(1.4)], np.r_[np.zeros(12), np.log(1.2)]
    i = base_inputs(arabica_share=0.5, premium_share=0.0, funds_rate=0.0, drift="zero", **NO_FRICTION)
    sc = stress.Scenario("x", returns_frame(usd, a, r), (None, None), (None, None))
    assert sc.change("usd") == pytest.approx(0.5) and sc.mix_change() == pytest.approx(0.3)
    out = stress.outcomes(stress.replay(sc.returns, i), i, [inp.now(), inp.payment_day()])
    assert out.loc["payment_day", "cost_pct"] == pytest.approx(1.3 * 1.5)          # more dollars, each dearer
    assert out.loc["now", "cost_pct"] == pytest.approx(1.0 + 0.3 * 1.5)            # only the extra 30% at 1.5
    fixed = i.with_(contract="fixed")
    assert stress.outcomes(stress.replay(sc.returns, fixed), fixed, [inp.now()]).loc["now", "cost_pct"] == 1.0


def test_a_longer_scenario_moves_the_payment_week_and_the_tranches():
    r = returns_frame(np.full(32, np.log(1.01)))
    i = base_inputs(contract="fixed", payment_week=32, **NO_FRICTION)
    p = stress.replay(r, i)
    assert p.horizon == 32 and [t for t, _ in inp.staged(3, 32).tranches] == [0, 11, 21]
    assert costs.planned_cost(p, i) == pytest.approx(100_000 * 500_000.0 * 1.30 ** (32 / 52))
    out = stress.outcomes(p, i, [inp.now(), inp.staged(3, 32), inp.payment_day()])
    assert out.loc["now", "cost_pct"] == pytest.approx(1.0)                         # funds = market rate
    assert out.loc["payment_day", "cost_pct"] == pytest.approx(1.01 ** 32 / 1.30 ** (32 / 52))
    staged = np.mean([1.01 ** k * 1.30 ** ((32 - k) / 52) for k in (0, 11, 21)]) / 1.30 ** (32 / 52)
    assert out.loc["staged_3", "cost_pct"] == pytest.approx(staged)


# ------------------------------------------------------------ reverse stress
def test_breaking_level_is_the_closed_form_of_the_spec():
    i = base_inputs(contract="fixed", **NO_FRICTION)                   # plan 25%, floor 10%, funds = market
    g = 1.30 ** 0.25
    for h in (0.0, 0.25, 0.5, 0.57, 0.75):
        assert stress.breaking_usd_level(i, h) / g - 1 == pytest.approx(0.2 / (1 - h))
    assert stress.breaking_usd_level(i, 1.0) == float("inf")
    dear = i.with_(funds_rate=0.80)                                    # dearer funds leave less room
    assert stress.breaking_usd_level(dear, 0.5) < stress.breaking_usd_level(i, 0.5)
    assert stress.breaking_usd_level(dear, 0.0) == stress.breaking_usd_level(i, 0.0)
    assert stress.breaking_usd_level(i.with_(demand_sd=0.1), 0.5) == stress.breaking_usd_level(i, 0.5)


def _margins_on_a_grid(i, h, grid):
    T = i.payment_week
    usd = np.ones((len(grid), T + 1))
    usd[:, T] = grid
    ones = np.ones_like(usd)
    growth = (1 + i.market_rate) ** (np.arange(T + 1) / 52)
    p = stress.Paths(usd=usd, arabica=ones, robusta=ones, growth=growth)
    return costs.margin(costs.cost_on_paths(p, inp.partial(h), i), costs.revenue(p, i))


@pytest.mark.parametrize("changes", [dict(), dict(funds_rate=0.60), dict(fx_holding_cost=0.03),
                                     dict(margin_floor=0.05, planned_margin=0.30), dict(payment_week=32),
                                     dict(funds_rate=3.0)])
def test_breaking_level_matches_a_brute_force_search(changes):
    i = base_inputs(contract="fixed", sellback_spread=0.0, **changes)
    grid = np.arange(0.05, 6.0, 0.0005)                                # candidate S_T / S_0
    for h in (0.0, 0.25, 0.5, 0.57, 0.75, 0.9):
        m = _margins_on_a_grid(i, h, grid)
        broken = grid[m < i.margin_floor]
        closed = stress.breaking_usd_level(i, h)
        if len(broken) == len(grid):
            assert closed <= grid[0]
        elif len(broken):
            assert broken.min() == pytest.approx(closed, abs=0.0006)
            assert np.all(m[grid < closed - 0.0006] >= i.margin_floor)
            assert np.all(m[grid > closed + 0.0006] < i.margin_floor)
        else:
            assert closed > grid.max()


@pytest.mark.parametrize("changes", [dict(pass_through=0.3), dict(pass_through=0.5),
                                     dict(pass_through=0.5, pass_through_mode="symmetric"),
                                     dict(pass_through=0.2, funds_rate=0.60, pass_through_mode="symmetric"),
                                     dict(pass_through=0.5, funds_rate=3.0),
                                     dict(pass_through=0.3, funds_rate=3.0, pass_through_mode="symmetric")])
def test_breaking_level_with_pass_through_is_the_lowest_broken_level_above_the_plan(changes):
    i = base_inputs(contract="fixed", sellback_spread=0.0, **changes)
    g = (1 + i.market_rate) ** (i.payment_week / 52)
    grid = np.arange(g, 12.0, 0.0005)                                  # rises above the expected level
    for h in (0.0, 0.1, 0.25, 0.5, 0.57, 0.75, 0.9):
        m = _margins_on_a_grid(i, h, grid)
        broken = grid[m < i.margin_floor - 1e-12]
        closed = stress.breaking_usd_level(i, h)
        if len(broken):
            assert broken.min() == pytest.approx(closed, abs=0.0006)
        else:
            assert closed > grid.max()


@pytest.mark.parametrize("changes", [dict(), dict(funds_rate=0.60), dict(fx_holding_cost=0.03),
                                     dict(margin_floor=0.05), dict(pass_through=0.5),
                                     dict(pass_through=0.5, pass_through_mode="symmetric")])
def test_stress_floor_is_the_smallest_share_that_survives_the_scenario(changes):
    """D-051: the share from the closed form keeps the margin at the floor on
    the scenario's path, and one point less does not."""
    i = base_inputs(contract="fixed", sellback_spread=0.0, **changes)
    T = i.payment_week
    growth = (1 + i.market_rate) ** (np.arange(T + 1) / 52)
    for rise in (0.10, 0.30, 0.59, 1.41):
        usd = np.ones((1, T + 1))
        usd[0, T] = 1 + rise
        p = stress.Paths(usd=usd, arabica=np.ones_like(usd), robusta=np.ones_like(usd), growth=growth)

        def margin(h):
            return costs.margin(costs.cost_on_paths(p, inp.partial(h), i), costs.revenue(p, i))[0]

        h = stress.surviving_share(i, 1 + rise)
        if h == 0.0:
            assert margin(0.0) >= i.margin_floor - 1e-12
        else:
            assert 0 < h <= 1 and margin(h) == pytest.approx(i.margin_floor, abs=1e-9)
            assert margin(h - 0.01) < i.margin_floor < margin(min(h + 0.01, 1.0)) + 1e-12
        # the inverse of the reverse stress: at that share the breaking level is the scenario's level
        if 0 < h < 1:
            assert stress.breaking_usd_level(i, h) == pytest.approx(1 + rise)
    assert stress.surviving_share(base_inputs(contract="fixed", funds_rate=3.0), 2.4) is None   # funds too dear
    no_pt = base_inputs(contract="fixed")
    g = 1.30 ** 0.25                                         # by hand: (x - 1.2 g) / (x - g)
    assert stress.surviving_share(no_pt, 2.405) == pytest.approx((2.405 - 1.2 * g) / (2.405 - g))
    assert stress.surviving_share(no_pt, 1.2 * g) == 0.0


def test_historical_changes_are_start_dated_and_need_a_known_outcome():
    wk = weekly_panel(40)
    ch = stress.historical_changes(wk, wk.index[10])
    assert ch.index[0] == wk.index[10] and ch.index[-1] == wk.index[40 - 13 - 1] and len(ch) == 17
    t = wk.index[15]
    assert ch.loc[t] == pytest.approx(wk["usd_free"].loc[t + pd.Timedelta(weeks=13)] / wk["usd_free"].loc[t] - 1)
