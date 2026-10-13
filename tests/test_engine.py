import json

import numpy as np
import pandas as pd
import pytest

from coffee_risk.engine import cost as costs
from coffee_risk.engine import decision as dec
from coffee_risk.engine import inputs as inp
from coffee_risk.engine import simulator as sim
from coffee_risk.engine.inputs import Inputs
from coffee_risk.engine.simulator import Paths

RET = list(sim.RETURN_COLS.values())


# ------------------------------------------------------------------- helpers
def make_weekly(n=320, seed=1, zero=False):
    """A synthetic weekly panel: USD with volatility clusters and a positive
    drift, two correlated coffees, a fund growing 0.5% a week."""
    idx = pd.date_range("2018-01-03", periods=n, freq="7D")
    rng = np.random.default_rng(seed)
    cluster = 1.0 + 0.8 * np.sin(np.arange(n) / 15.0)
    usd = 0.004 + 0.03 * cluster * rng.standard_normal(n)
    common = rng.standard_normal(n)
    arabica = 0.04 * (0.8 * common + 0.6 * rng.standard_normal(n))
    robusta = 0.035 * (0.8 * common + 0.6 * rng.standard_normal(n))
    if zero:
        usd, arabica, robusta = np.zeros(n), np.zeros(n), np.zeros(n)
    usd[0] = np.nan                                            # the first week has no return
    df = pd.DataFrame({"usd_free_ret": usd, "arabica_usd_kg_ret": arabica,
                       "robusta_usd_kg_ret": robusta}, index=idx)
    df["usd_free"] = 500_000.0 * np.exp(np.nan_to_num(usd).cumsum())
    df["fund_tr_index"] = 10_000.0 * 1.005 ** np.arange(n)
    return df


def base_inputs(**changes):
    values = dict(s0=500_000.0, market_rate=0.30, premium_share=0.10)
    values.update(changes)
    return Inputs(**values)


def hand_paths(usd, arabica=None, robusta=None):
    usd = np.asarray(usd, dtype=float)
    ones = np.ones_like(usd)
    return Paths(usd=usd, arabica=ones if arabica is None else np.asarray(arabica, dtype=float),
                 robusta=ones if robusta is None else np.asarray(robusta, dtype=float),
                 growth=usd.mean(axis=0), info={"seed": 0})


def flat(n=4, horizon=13):
    return np.ones((n, horizon + 1))


NO_FRICTION = dict(sellback_spread=0.0, fx_holding_cost=0.0, channel_discount=0.0)


# ---------------------------------------------------------------- strategies
def test_strategies_buy_when_the_spec_says():
    assert inp.now().tranches == ((0, 1.0),)
    assert inp.payment_day().tranches == ()
    assert [t for t, _ in inp.staged(3, 13).tranches] == [0, 4, 9]
    assert [s for _, s in inp.staged(3, 13).tranches] == pytest.approx([1 / 3] * 3)
    assert inp.partial(0.3).tranches == ((0, 0.3),) and inp.partial(0.0).tranches == ()
    assert [round(s.share_at_week0, 2) for s in inp.partial_family()] == [i / 10 for i in range(11)]
    names = [s.name for s in inp.strategy_set()]
    assert names[:3] == ["now", "staged_3", "payment_day"] and len(names) == 12
    assert "partial_0" not in names and "partial_100" not in names      # these are payment_day and now


def test_inputs_reject_unknown_choices():
    for bad in (dict(contract="spot"), dict(usd_vol="garch"), dict(drift="up"), dict(arabica_share=1.2)):
        with pytest.raises(ValueError):
            base_inputs(**bad)
    assert base_inputs().cost_of_funds == 0.30                      # default = the market rate
    assert base_inputs(funds_rate=0.40).cost_of_funds == 0.40


# --------------------------------------------------- 1. the zero-volatility check
def test_zero_volatility_costs_are_the_hand_values():
    """Returns all zero, USD flat, fixed contract, funds at 7% a quarter, no
    frictions: buying now costs the 7% of a quarter's funding, the staged plan
    (weeks 0, 4, 9) costs 4.6%, buying on the payment day costs nothing extra."""
    i = base_inputs(contract="fixed", drift="zero", funds_rate=1.07 ** 4 - 1, **NO_FRICTION)
    p = hand_paths(flat())
    unit = i.usd_amount * i.s0
    got = {s.name: costs.cost_on_paths(p, s, i) / unit for s in (inp.now(), inp.staged(), inp.payment_day())}
    np.testing.assert_allclose(got["now"], 1.07, rtol=0, atol=1e-12)
    np.testing.assert_allclose(got["staged_3"], (1.07 + 1.07 ** (9 / 13) + 1.07 ** (4 / 13)) / 3, atol=1e-12)
    assert [round(float(got[k][0]), 3) for k in ("now", "staged_3", "payment_day")] == [1.070, 1.046, 1.000]


@pytest.mark.parametrize("usd_vol", ["plain", "fhs"])
def test_zero_volatility_through_the_simulator(usd_vol):
    wk = make_weekly(zero=True)
    i = base_inputs(contract="fixed", drift="zero", usd_vol=usd_vol, funds_rate=1.07 ** 4 - 1, **NO_FRICTION)
    p = sim.simulate(wk.index[-1], 50, seed=5, options=i, weekly=wk)
    assert np.all(p.usd == 1.0) and np.all(p.arabica == 1.0) and np.all(p.robusta == 1.0)
    unit = i.usd_amount * i.s0
    got = [round(float(costs.cost_on_paths(p, s, i).mean() / unit), 3)
           for s in (inp.now(), inp.staged(), inp.payment_day())]
    assert got == [1.070, 1.046, 1.000]


# ------------------------------------------------------------------- 2. cost
def test_fixed_contract_cost_ignores_the_coffee_paths():
    wk = make_weekly()
    i = base_inputs(contract="fixed")
    p = sim.simulate(wk.index[-1], 400, seed=2, options=i, weekly=wk)
    wild = Paths(usd=p.usd, arabica=p.arabica * 3.0, robusta=p.robusta[::-1] * 0.2, growth=p.growth)
    for s in inp.strategy_set():
        np.testing.assert_array_equal(costs.cost_on_paths(p, s, i), costs.cost_on_paths(wild, s, i))
    floating = i.with_(contract="floating")
    assert not np.allclose(costs.cost_on_paths(p, inp.now(), floating),
                           costs.cost_on_paths(wild, inp.now(), floating))


def test_floating_contract_needs_more_dollars_when_coffee_rises():
    usd, a, r = flat(2), flat(2), flat(2)
    a[0, 13], r[0, 13] = 1.20, 1.10                       # path 0: arabica +20%, robusta +10%
    a[1, 13], r[1, 13] = 0.80, 0.90
    i = base_inputs(arabica_share=0.25, premium_share=0.10, **NO_FRICTION, drift="zero")
    p = hand_paths(usd, a, r)
    f = costs.coffee_factor(p, i)
    assert f[0] == pytest.approx(0.9 * (0.25 * 1.20 + 0.75 * 1.10) + 0.10)
    assert f[1] == pytest.approx(0.9 * (0.25 * 0.80 + 0.75 * 0.90) + 0.10)
    np.testing.assert_allclose(costs.usd_needed(p, i), 100_000 * f)
    # payment day: everything is bought at S_T
    np.testing.assert_allclose(costs.cost_on_paths(p, inp.payment_day(), i), 100_000 * f * 500_000.0)


def test_shortfall_is_bought_and_excess_sold_back_at_the_spread():
    usd, a, r = flat(2), flat(2), flat(2)
    usd[:, 13] = 1.30
    a[0, 13] = r[0, 13] = 1.50                             # needs more dollars than planned
    a[1, 13] = r[1, 13] = 0.50                             # needs fewer: the excess is sold
    i = base_inputs(premium_share=0.0, funds_rate=0.0, sellback_spread=0.02, drift="zero")
    p = hand_paths(usd, a, r)
    c = costs.cost_on_paths(p, inp.now(), i) / (i.usd_amount * i.s0)
    assert c[0] == pytest.approx(1.0 + 0.5 * 1.30)                    # 50,000 more at S_T
    assert c[1] == pytest.approx(1.0 - 0.5 * 1.30 * (1 - 0.02))       # 50,000 sold at S_T less 2%


def test_funding_holding_cost_and_channel_discount():
    p = hand_paths(flat(1))
    unit = 100_000 * 500_000.0
    hold = base_inputs(contract="fixed", funds_rate=0.0, fx_holding_cost=0.04, drift="zero")
    assert costs.cost_on_paths(p, inp.now(), hold)[0] / unit == pytest.approx(1.0 + 0.04 * 13 / 52)
    assert costs.cost_on_paths(p, inp.payment_day(), hold)[0] / unit == pytest.approx(1.0)
    half = costs.cost_on_paths(p, inp.partial(0.5), hold.with_(funds_rate=0.20))[0] / unit
    assert half == pytest.approx(0.5 * (1.20 ** 0.25 + 0.01) + 0.5)
    disc = hold.with_(channel_discount=0.10, fx_holding_cost=0.0)
    assert costs.cost_on_paths(p, inp.staged(), disc)[0] / unit == pytest.approx(0.90)
    assert costs.planned_cost(p, disc) == pytest.approx(0.90 * unit)
    with pytest.raises(ValueError):
        costs.cost_on_paths(p, inp.Strategy("late", ((13, 1.0),)), hold)


# ----------------------------------------------------------- 3. neutral drift
def test_neutral_drift_makes_now_and_payment_day_cost_the_same_on_average():
    wk = make_weekly()
    i = base_inputs(drift="neutral", funds_rate=None, **NO_FRICTION)       # funds = market rate
    p = sim.simulate(wk.index[-1], 4000, seed=11, options=i, weekly=wk)
    now, late = costs.cost_on_paths(p, inp.now(), i), costs.cost_on_paths(p, inp.payment_day(), i)
    se = np.sqrt(now.var(ddof=1) / len(now) + late.var(ddof=1) / len(late))
    assert abs(now.mean() - late.mean()) < 3 * se
    assert abs(now.mean() - late.mean()) < 3 * (now - late).std(ddof=1) / np.sqrt(len(now))
    fixed = i.with_(contract="fixed")                     # and the same holds for every partial share
    means = [costs.cost_on_paths(p, s, fixed).mean() for s in inp.partial_family()]
    np.testing.assert_allclose(means, means[0], rtol=1e-12)


# ----------------------------------------------------- 4. drift normalisation
@pytest.mark.parametrize("drift", ["neutral", "historical", "zero", 0.60])
@pytest.mark.parametrize("usd_vol", ["fhs", "plain"])
def test_drift_normalisation_hits_the_target_mean_every_week(drift, usd_vol):
    wk = make_weekly()
    as_of = wk.index[-1]
    i = base_inputs(drift=drift, usd_vol=usd_vol)
    p = sim.simulate(as_of, 3000, seed=4, options=i, weekly=wk)
    k = np.arange(14)
    if drift == "neutral":
        target = 1.30 ** (k / 52)
    elif drift == "zero":
        target = np.ones(14)
    elif drift == "historical":
        pool = sim.prepare(wk, as_of, i)["usd_raw"]
        target = (1 + np.expm1(pool).mean()) ** k
    else:
        target = 1.60 ** (k / 52)
    np.testing.assert_allclose(p.growth, target, rtol=1e-12)
    np.testing.assert_allclose(p.usd.mean(axis=0), target, rtol=1e-12)
    np.testing.assert_allclose(p.arabica.mean(axis=0), 1.0, rtol=1e-12)
    np.testing.assert_allclose(p.robusta.mean(axis=0), 1.0, rtol=1e-12)
    assert np.all(p.usd[:, 0] == 1.0) and p.usd.shape == (3000, 14)


def test_scaling_a_week_keeps_the_shape_of_the_distribution():
    r = np.random.default_rng(0).normal(0.01, 0.05, (500, 13))
    a, b = sim.to_levels(r, np.ones(14)), sim.to_levels(r, 1.5 ** (np.arange(14) / 52))
    ratio = b[:, 1:] / a[:, 1:]
    np.testing.assert_allclose(ratio, np.broadcast_to(ratio[0], ratio.shape), rtol=1e-12)


# ---------------------------------------------------------------- 5. past only
def test_last_week_is_in_and_nothing_later_is_touched():
    """D-038: the week ending on the decision date is in the pool and in the
    volatility estimate; every later week is invisible."""
    wk = make_weekly()
    as_of = wk.index[200]
    for usd_vol in ("fhs", "plain"):
        i = base_inputs(usd_vol=usd_vol, drift="historical")
        a = sim.simulate(as_of, 800, seed=3, options=i, weekly=wk)
        poisoned = wk.copy()
        poisoned.loc[poisoned.index > as_of, RET] = 9.99           # absurd returns after as_of
        b = sim.simulate(as_of, 800, seed=3, options=i, weekly=poisoned)
        cut_off = sim.simulate(as_of, 800, seed=3, options=i, weekly=wk[wk.index <= as_of])
        for name in ("usd", "arabica", "robusta"):
            np.testing.assert_array_equal(getattr(a, name), getattr(b, name))
            np.testing.assert_array_equal(getattr(a, name), getattr(cut_off, name))
        prep = sim.prepare(wk, as_of, i)
        assert prep["weeks"].max() == as_of                         # the last week is in the pool
        assert a.info["pool_last"] == as_of and a.info["last_sampled"] <= as_of
        if usd_vol == "fhs":
            assert a.info["last_week_in_vol"] == as_of
            assert prep["info"]["sigma_as_of"] == sim.prepare(poisoned, as_of, i)["info"]["sigma_as_of"]
    # the return of the last week moves the volatility used for every simulated week
    i = base_inputs()
    loud = wk.copy()
    loud.loc[as_of, "usd_free_ret"] = 0.40
    quiet, noisy = sim.prepare(wk, as_of, i)["info"], sim.prepare(loud, as_of, i)["info"]
    assert noisy["sigma_as_of"] > 1.5 * quiet["sigma_as_of"]
    r = wk.loc[wk.index <= as_of, "usd_free_ret"]                  # and it is the EWMA of all weeks up to it
    lam = 0.5 ** (1 / 13.0)
    w = lam ** np.arange(len(r) - 1, -1, -1.0)
    keep = r.notna().values
    assert quiet["sigma_as_of_raw"] == pytest.approx(np.sqrt((w[keep] * r.values[keep] ** 2).sum() / w[keep].sum()))
    # a week earlier in the pool matters too
    changed = wk.copy()
    changed.loc[wk.index[150], "usd_free_ret"] = 0.5
    c = sim.simulate(as_of, 800, seed=3, options=i, weekly=changed)
    d = sim.simulate(as_of, 800, seed=3, options=i, weekly=wk)
    assert not np.array_equal(c.usd, d.usd)


def test_pool_holds_only_complete_weeks_up_to_the_date():
    wk = make_weekly(60)
    wk.loc[wk.index[10], "robusta_usd_kg_ret"] = np.nan
    weeks = sim.pool_weeks(sim.weekly_returns(wk), wk.index[40])
    assert wk.index[10] not in weeks and wk.index[0] not in weeks      # a missing leg, the first week
    assert weeks.max() == wk.index[40] and len(weeks) == 39
    assert wk.index[41] not in weeks


def test_ewma_volatility_uses_earlier_weeks_only():
    r = pd.Series([0.01, np.nan, -0.03, 0.02, np.nan, 0.05, 0.01])
    lam = 0.5 ** (1 / 2.0)
    got = sim.ewma_vol(r, half_life=2.0, min_history=1)
    assert np.isnan(got[0])
    for t in range(1, len(r)):
        js = [j for j in range(t) if not np.isnan(r[j])]
        w = np.array([lam ** (t - 1 - j) for j in js])
        assert got[t] == pytest.approx(np.sqrt((w * r[js].values ** 2).sum() / w.sum()))
    # changing week t and later leaves the estimate at t unchanged
    wk = make_weekly()
    x = wk["usd_free_ret"]
    y = x.copy()
    y.iloc[150:] = 5.0
    a, b = sim.ewma_vol(x), sim.ewma_vol(y)
    pd.testing.assert_series_equal(a.iloc[:151], b.iloc[:151])
    assert b.iloc[151] > 10 * a.iloc[151]
    assert a.iloc[:14].isna().all() and a.iloc[14:].notna().all()      # 13 earlier returns needed


def test_volatility_floor_is_the_past_only_fifth_percentile():
    s = pd.Series(np.r_[np.linspace(0.05, 0.01, 100), 0.0001, 0.02])
    used, floor = sim.floored_vol(s)
    assert floor.iloc[100] == pytest.approx(np.quantile(s.iloc[:101], 0.05))
    assert used.iloc[100] == floor.iloc[100] > s.iloc[100]             # the tiny value is floored
    assert used.iloc[101] == s.iloc[101]
    changed = s.copy()
    changed.iloc[101] = 9.0
    assert sim.floored_vol(changed)[1].iloc[:101].equals(floor.iloc[:101])   # no later value used


def test_fhs_rescales_history_to_todays_volatility():
    wk = make_weekly()
    as_of = wk.index[-1]
    prep = sim.prepare(wk, as_of, base_inputs())
    info = prep["info"]
    assert prep["usd"].mean() == pytest.approx(0.0, abs=1e-15)         # demeaned
    assert np.std(prep["usd"], ddof=1) == pytest.approx(info["sigma_as_of"] * info["z_std"])
    plain = sim.prepare(wk, as_of, base_inputs(usd_vol="plain"))
    np.testing.assert_array_equal(plain["usd"], plain["usd_raw"])
    assert plain["info"]["pool_weeks"] == info["pool_weeks"] + sim.MIN_HISTORY   # the warm-up weeks
    np.testing.assert_array_equal(prep["arabica"], wk["arabica_usd_kg_ret"].reindex(prep["weeks"]).values)


def test_simulator_matches_a_slow_reference():
    """The same paths built week by week with plain loops."""
    wk = make_weekly(120, seed=3)
    as_of = wk.index[100]
    i = base_inputs(half_life=8.0, mean_block=5.0)
    n, h = 40, 13
    fast = sim.simulate(as_of, n, h, seed=21, options=i, weekly=wk)

    r = wk[RET].values                                       # usd, arabica, robusta
    t_as_of = wk.index.get_loc(as_of)
    lam = 0.5 ** (1 / 8.0)
    sigma = np.full(t_as_of + 2, np.nan)                     # the last entry: the week after as_of
    for t in range(t_as_of + 2):
        js = [j for j in range(t) if not np.isnan(r[j, 0])]
        if len(js) >= sim.MIN_HISTORY:
            w = np.array([lam ** (t - 1 - j) for j in js])
            sigma[t] = np.sqrt((w * r[js, 0] ** 2).sum() / w.sum())
    used = np.array([max(sigma[t], np.nanquantile(sigma[:t + 1], 0.05)) if not np.isnan(sigma[t]) else np.nan
                     for t in range(t_as_of + 2)])
    pool = [t for t in range(t_as_of + 1) if not np.isnan(r[t]).any() and not np.isnan(sigma[t])]
    assert pool[-1] == t_as_of                               # D-038
    z = np.array([r[t, 0] / used[t] for t in pool])
    usd_pool = (z - z.mean()) * used[t_as_of + 1]

    rng = np.random.default_rng(21)                          # the same three draws, in the same order
    start, jump = rng.integers(0, len(pool), n), rng.random((n, h)) < 1 / 5.0
    fresh = rng.integers(0, len(pool), (n, h))
    logs = np.zeros((3, n, h))
    for path in range(n):
        pos = start[path]
        for k in range(h):
            if k > 0:
                pos = fresh[path, k] if jump[path, k] else (pos + 1) % len(pool)
            logs[:, path, k] = usd_pool[pos], r[pool[pos], 1], r[pool[pos], 2]
    levels = np.exp(np.cumsum(logs, axis=2))
    target = 1.30 ** (np.arange(1, h + 1) / 52)
    np.testing.assert_allclose(fast.usd[:, 1:], levels[0] / levels[0].mean(axis=0) * target, rtol=1e-10)
    np.testing.assert_allclose(fast.arabica[:, 1:], levels[1] / levels[1].mean(axis=0), rtol=1e-10)
    np.testing.assert_allclose(fast.robusta[:, 1:], levels[2] / levels[2].mean(axis=0), rtol=1e-10)


# ------------------------------------------------------------------- sampling
def test_stationary_bootstrap_walks_in_blocks_and_wraps():
    rng = np.random.default_rng(0)
    idx = sim.stationary_indices(rng, 50, 20_000, 13, mean_block=13.0)
    assert idx.min() == 0 and idx.max() == 49
    step = (idx[:, 1:] - idx[:, :-1]) % 50
    assert 1 - 1 / 13 - 0.01 < (step == 1).mean() < 1 - 1 / 13 + 0.01 + 1 / 50
    assert ((idx[:, :-1] == 49) & (idx[:, 1:] == 0)).any()             # the pool wraps around
    short = sim.stationary_indices(np.random.default_rng(0), 50, 20_000, 13, mean_block=1.0)
    assert abs(((short[:, 1:] - short[:, :-1]) % 50 == 1).mean() - 1 / 50) < 0.01   # every step jumps


# ------------------------------------------------------------- 7. same seed
def test_same_seed_gives_identical_results():
    wk = make_weekly()
    i = base_inputs(demand_sd=0.10)
    a = sim.simulate(wk.index[-1], 600, seed=9, options=i, weekly=wk)
    b = sim.simulate(wk.index[-1], 600, seed=9, options=i, weekly=wk)
    c = sim.simulate(wk.index[-1], 600, seed=10, options=i, weekly=wk)
    for name in ("usd", "arabica", "robusta"):
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name))
    assert not np.array_equal(a.usd, c.usd)
    ra, rb = dec.run(a, i), dec.run(b, i)
    pd.testing.assert_frame_equal(ra.table, rb.table)
    pd.testing.assert_frame_equal(ra.curve, rb.curve)
    assert ra.decision == rb.decision and ra.min_hedge == rb.min_hedge


# ---------------------------------------------------------- revenue, margin
def test_demand_factor_has_mean_one_and_the_asked_spread():
    d = costs.demand_factor(400_000, 0.10, seed=1)
    assert d.mean() == pytest.approx(1.0, abs=5e-4) and d.std() == pytest.approx(0.10, abs=5e-4)
    assert np.all(costs.demand_factor(5, 0.0) == 1.0)
    np.testing.assert_array_equal(d[:10], costs.demand_factor(400_000, 0.10, seed=1)[:10])
    assert not np.array_equal(d[:10], costs.demand_factor(400_000, 0.10, seed=2)[:10])


def test_revenue_plan_and_symmetric_pass_through():
    usd = flat(2)
    usd[0, 13], usd[1, 13] = 1.40, 0.80                    # mean 1.10 = the plan's growth
    p = hand_paths(usd)
    i = base_inputs(contract="fixed", planned_margin=0.25, funds_rate=0.0, pass_through_mode="symmetric")
    c_plan = costs.planned_cost(p, i)
    assert c_plan == pytest.approx(100_000 * 500_000.0 * 1.10)
    np.testing.assert_allclose(costs.revenue(p, i), c_plan / 0.75)              # no pass-through
    half = costs.revenue(p, i.with_(pass_through=0.5))
    np.testing.assert_allclose(half, c_plan / 0.75 * (1 + 0.5 * (usd[:, 13] / 1.10 - 1)))
    # full pass-through: buying on the payment day keeps the planned margin on every path
    full = i.with_(pass_through=1.0)
    m = costs.margin(costs.cost_on_paths(p, inp.payment_day(), full), costs.revenue(p, full))
    np.testing.assert_allclose(m, 0.25)


def test_up_only_pass_through_does_not_cut_revenue_when_usd_ends_below_plan():
    """D-043: selling prices rise with an unexpected USD increase and do not
    fall when USD ends below S_0 * g_T."""
    usd = flat(4)
    usd[:, 13] = [1.40, 1.12, 1.08, 0.80]                  # mean 1.10 = g_T; two paths above, two below
    p = hand_paths(usd)
    assert base_inputs().pass_through_mode == "up_only"                # the default once pass_through > 0
    up = base_inputs(contract="fixed", pass_through=0.5)
    sym = up.with_(pass_through_mode="symmetric")
    r_plan = costs.planned_cost(p, up) / 0.75
    r_up, r_sym = costs.revenue(p, up), costs.revenue(p, sym)
    below = usd[:, 13] < 1.10
    np.testing.assert_array_equal(r_up[below], np.full(2, r_plan))     # unchanged below the plan
    np.testing.assert_array_equal(r_up[~below], r_sym[~below])         # the same as symmetric above it
    np.testing.assert_allclose(r_up[~below], r_plan * (1 + 0.5 * (usd[~below, 13] / 1.10 - 1)))
    assert np.all(r_sym[below] < r_plan)
    np.testing.assert_array_equal(costs.fx_surprise(p, up), np.maximum(usd[:, 13] / 1.10 - 1, 0.0))
    # with no pass-through the mode changes nothing
    np.testing.assert_array_equal(costs.revenue(p, up.with_(pass_through=0.0)),
                                  costs.revenue(p, sym.with_(pass_through=0.0)))
    with pytest.raises(ValueError):
        base_inputs(pass_through_mode="down_only")


def test_funds_spread_is_added_to_the_market_rate():
    """D-039: the funds rate as a spread over the market rate."""
    i = base_inputs(funds_spread=0.10)                     # market rate 30%
    assert i.cost_of_funds == pytest.approx(0.40)
    assert base_inputs(funds_rate=0.55).cost_of_funds == 0.55          # a level still works
    with pytest.raises(ValueError):
        base_inputs(funds_rate=0.40, funds_spread=0.10)                # but not both
    p = hand_paths(flat(1))
    fixed = dict(contract="fixed", drift="zero", **NO_FRICTION)
    now = costs.cost_on_paths(p, inp.now(), base_inputs(funds_spread=0.10, **fixed))[0]
    assert now / (100_000 * 500_000.0) == pytest.approx(1.40 ** 0.25)
    assert costs.cost_on_paths(p, inp.now(), base_inputs(**fixed))[0] / 5e10 == pytest.approx(1.30 ** 0.25)


# ---------------------------------------------------------------- 5. metrics
def test_metrics_on_a_hand_made_distribution():
    cost = np.arange(1.0, 101.0)                           # 1, 2, ..., 100
    rev = np.full(100, 125.0)
    i = base_inputs(margin_floor=0.24, confidence=0.95, budget_factor=1.10)
    m = dec.metrics(cost, rev, c_plan=80.0, inputs=i)
    assert m["mean_cost"] == 50.5 and m["median_cost"] == 50.5
    assert m["mean_cost_pct"] == pytest.approx(50.5 / 80)
    assert m["var"] == pytest.approx(np.quantile(cost, 0.95)) and m["cvar"] == 98.0   # mean of 96..100
    assert m["p_over_budget"] == pytest.approx(12 / 100)                # cost > 88
    assert m["p_margin_below_floor"] == pytest.approx(5 / 100)          # margin < 24% <=> cost > 95
    assert m["margin_q"] == pytest.approx(np.quantile(1 - cost / 125, 0.05))
    assert m["se_mean_cost"] == pytest.approx(cost.std(ddof=1) / 10)
    assert m["se_p_margin_below_floor"] == pytest.approx(np.sqrt(0.05 * 0.95 / 100))
    assert dec.tail_mean(np.array([5.0, 1.0, 3.0]), 0.95) == 5.0        # never an empty tail


# ----------------------------------------------------------- 6. decision rule
def _table(rows):
    return pd.DataFrame(rows, columns=["mean_cost", "cvar", "p_margin_below_floor", "margin_q"],
                        index=list("abcd"))


def test_decision_rule_keeps_the_safe_ones_then_takes_the_cheapest():
    i = base_inputs(confidence=0.95)
    t = _table([[100.0, 110.0, 0.01, 0.20],        # a: safe, dearest
                [99.0, 120.0, 0.04, 0.11],         # b: safe, cheap
                [98.0, 150.0, 0.20, -0.10],        # c: cheapest but fails the constraint
                [99.5, 115.0, 0.05, 0.10]])        # d: exactly at the confidence -> qualifies
    d = dec.decide(t, i, c_plan=100.0)
    assert d.qualifies and d.chosen == "b" and d.qualifying == ["a", "b", "d"]
    assert d.best_probability == pytest.approx(0.99) and d.best_probability_strategy == "a"
    assert d.highest_floor == pytest.approx(0.20) and d.highest_floor_strategy == "a"


def test_mean_costs_within_a_quarter_percent_of_plan_are_a_tie():
    """D-037: a difference below 0.25% of the plan is a tie, broken by the
    lower CVaR; 0.25% or more is a real difference."""
    i = base_inputs(confidence=0.95)
    assert i.tie_tolerance == 0.0025

    def chosen(cost_d, inputs=i):
        t = _table([[100.0, 110.0, 0.01, 0.20], [99.0, 120.0, 0.04, 0.11],
                    [98.0, 150.0, 0.20, -0.10], [cost_d, 115.0, 0.05, 0.10]])
        return dec.decide(t, inputs, c_plan=100.0).chosen

    assert chosen(99.0) == "d"                      # equal means: the lower CVaR wins
    assert chosen(99.2) == "d"                      # 0.20% of plan dearer: still a tie
    assert chosen(99.249) == "d"
    assert chosen(99.25) == "b"                     # 0.25% is not below 0.25%
    assert chosen(99.3) == "b"                      # a real difference: the cheaper one wins
    assert chosen(98.9) == "d"                      # d cheapest; b is within the tolerance but has the higher CVaR
    assert chosen(98.7) == "d"
    # the tolerance is a share of the plan, not of the cost
    t = _table([[100.0, 110.0, 0.01, 0.20], [99.0, 120.0, 0.04, 0.11],
                [98.0, 150.0, 0.20, -0.10], [99.2, 115.0, 0.05, 0.10]])
    assert dec.decide(t, i, c_plan=50.0).chosen == "b"                 # 0.2 is 0.4% of a plan of 50
    # the fail-safe strategy never joins a tie, however cheap
    assert dec.decide(t, i, c_plan=1000.0).chosen == "a"               # a, b, d all tie: the lowest CVaR
    # with the tolerance switched off only rounding noise ties
    literal = i.with_(tie_tolerance=0.0)
    assert chosen(99.2, literal) == "b" and chosen(99.0 + 1e-12, literal) == "d"


def test_decision_rule_says_so_when_none_qualifies():
    i = base_inputs(confidence=0.95)
    t = _table([[100.0, 110.0, 0.07, 0.08], [99.0, 120.0, 0.12, 0.03],
                [98.0, 150.0, 0.20, -0.10], [99.0, 115.0, 0.06, 0.07]])
    d = dec.decide(t, i, c_plan=100.0)
    assert not d.qualifies and d.chosen is None and d.qualifying == []
    assert d.best_probability == pytest.approx(0.94) and d.best_probability_strategy == "d"
    assert d.highest_floor == pytest.approx(0.08) and d.highest_floor_strategy == "a"


def test_rule_and_minimum_hedge_ratio_on_a_hand_made_case():
    """USD is flat until the payment week, then jumps 45% on one path in ten.
    Fixed contract, funding costs 8% for the quarter. Plan = 1.045, revenue =
    1.045 / 0.75. A 10% floor allows a cost of 0.9 * revenue = 1.254. On the
    jump paths the cost is 1.08 h + 1.45 (1 - h), so h must be at least
    (1.45 - 1.254) / 0.37 = 0.5297: the minimum hedge ratio is 53%."""
    usd = flat(100)
    usd[:10, 13] = 1.45
    p = hand_paths(usd)
    i = base_inputs(contract="fixed", funds_rate=1.08 ** 4 - 1, **NO_FRICTION)
    res = dec.run(p, i)
    assert res.c_plan == pytest.approx(100_000 * 500_000.0 * 1.045)
    t = res.table
    assert t.loc["payment_day", "p_margin_below_floor"] == pytest.approx(0.10)
    assert t.loc["partial_50", "p_margin_below_floor"] == pytest.approx(0.10)
    assert t.loc["partial_60", "p_margin_below_floor"] == 0.0
    assert res.min_hedge == pytest.approx(0.53)
    staged = (1.08 + 1.08 ** (9 / 13) + 1.08 ** (4 / 13)) / 3
    assert t.loc["staged_3", "mean_cost_pct"] == pytest.approx(staged / 1.045)
    assert t.loc["partial_60", "mean_cost_pct"] == pytest.approx((0.6 * 1.08 + 0.4 * 1.045) / 1.045)
    # safe: now, staged, partial 60..90; the cheapest of them on average is the staged plan
    assert res.decision.qualifying == ["now", "staged_3", "partial_60", "partial_70", "partial_80", "partial_90"]
    assert res.decision.chosen == "staged_3"
    # with a 30% floor nothing qualifies: the best path of all (cost 1.0) leaves 28.2%.
    # The highest floor that can be met is the staged plan's sure margin.
    none = dec.run(p, i.with_(margin_floor=0.30))
    assert not none.decision.qualifies and none.decision.chosen is None and none.min_hedge is None
    assert none.decision.best_probability == 0.0
    assert none.decision.highest_floor == pytest.approx(1 - staged * 0.75 / 1.045)
    assert none.decision.highest_floor_strategy == "staged_3"


def test_hedge_curve_ends_are_payment_day_and_now():
    wk = make_weekly()
    i = base_inputs()
    p = sim.simulate(wk.index[-1], 500, seed=1, options=i, weekly=wk)
    res = dec.run(p, i)
    assert len(res.curve) == 101 and res.curve.index[0] == 0.0 and res.curve.index[-1] == 1.0
    for h, name in ((0.0, "payment_day"), (1.0, "now"), (0.3, "partial_30")):
        for col in ("mean_cost", "cvar", "p_margin_below_floor", "margin_q"):
            assert res.curve.loc[h, col] == pytest.approx(res.table.loc[name, col], rel=1e-12)


# ----------------------------------------------------------- data defaults
def test_data_based_defaults_use_the_decision_date():
    wk = make_weekly()
    as_of = wk.index[100]
    assert inp.spot(wk, as_of) == wk.loc[as_of, "usd_free"]
    assert inp.market_rate(wk, as_of) == pytest.approx(1.005 ** 52 - 1)
    i = Inputs.from_data(as_of, wk, premium_share=0.2)
    assert (i.s0, i.premium_share) == (wk.loc[as_of, "usd_free"], 0.2)
    with pytest.raises(ValueError):
        Inputs.from_data(as_of, wk)                        # the premium needs the daily and IMF data


def test_premium_default_is_one_minus_the_median_ratio_and_respects_the_imf_delay():
    days = pd.bdate_range("2020-01-01", "2024-06-28")
    month = days.to_period("M").to_timestamp()
    late = month >= pd.Timestamp("2024-04-01")             # months not yet published at the decision date
    daily = pd.DataFrame({"arabica_usd_kg": np.where(late, 9.0, 4.5),
                          "robusta_usd_kg": np.where(late, 9.0, 2.4)}, index=days)
    months = pd.date_range("2020-01-01", "2024-06-01", freq="MS")
    from coffee_risk.units import cents_lb_to_usd_kg
    per_kg = float(cents_lb_to_usd_kg(1.0))
    imf = pd.DataFrame({"month": months, "arabica_imf_clb": 5.0 / per_kg, "robusta_imf_clb": 3.0 / per_kg})
    as_of = pd.Timestamp("2024-06-26")
    p = inp.premium_by_coffee(daily, imf, as_of)
    assert p["arabica"]["premium"] == pytest.approx(0.10) and p["robusta"]["premium"] == pytest.approx(0.20)
    assert p["arabica"]["months"] == 36 and p["arabica"]["last"] == pd.Timestamp("2024-03-01")
    assert inp.premium_share(daily, imf, as_of, 0.25) == pytest.approx(0.25 * 0.10 + 0.75 * 0.20)
    no_lag = inp.premium_by_coffee(daily, imf, as_of, months=3, lag_months=1)
    assert no_lag["arabica"]["premium"] == pytest.approx(1 - 9.0 / 5.0)      # would see the late months


# --------------------------------------------------------------- calibration
def test_calibration_compares_the_realised_change_with_past_only_percentiles():
    from coffee_risk.engine import calibration as cal
    wk = make_weekly(150, seed=5)
    wk["usd_free_old"] = False
    start = wk.index[60]
    tab = cal.usd_exceedance(wk, n_paths=300, seed=4, start=start)
    assert tab.index[0] == start and tab.index[-1] == wk.index[150 - 13 - 1]    # outcome must be known
    assert len(tab) == 150 - 13 - 60 and tab["usable"].all()
    t = tab.index[10]
    level = wk["usd_free"]
    assert tab.loc[t, "end"] == t + pd.Timedelta(weeks=13)
    assert tab.loc[t, "realized"] == pytest.approx(level.loc[tab.loc[t, "end"]] / level.loc[t] - 1)
    k = wk.index.get_loc(t)
    for m in ("fhs", "plain"):                               # the percentiles are the engine's own
        options = Inputs(s0=float(level.loc[t]), market_rate=inp.market_rate(wk, t), premium_share=0.0, usd_vol=m)
        change = sim.simulate(t, 300, 13, 4 + k, options, wk).usd[:, 13] - 1
        assert tab.loc[t, f"{m}_q95"] == pytest.approx(np.quantile(change, 0.95))
        assert tab.loc[t, f"{m}_q05"] == pytest.approx(np.quantile(change, 0.05))
        assert tab.loc[t, f"{m}_above"] == (tab.loc[t, "realized"] > tab.loc[t, f"{m}_q95"])
        assert tab.loc[t, f"{m}_below"] == (tab.loc[t, "realized"] < tab.loc[t, f"{m}_q05"])
    # past-only: what happens after a start changes its outcome, never its percentiles
    later = wk.copy()
    later.loc[later.index > t, "usd_free_ret"] = 0.3
    later.loc[later.index > t, "usd_free"] *= 5.0
    tab2 = cal.usd_exceedance(later, n_paths=300, seed=4, start=start)
    for col in ("fhs_q05", "fhs_q95", "plain_q05", "plain_q95"):
        assert tab2.loc[t, col] == tab.loc[t, col]
    assert tab2.loc[t, "realized"] > tab.loc[t, "realized"] and tab2.loc[t, "fhs_above"]


def test_calibration_flags_unusable_starts_and_counts_independent_windows():
    from coffee_risk.engine import calibration as cal
    wk = make_weekly(120, seed=5)
    wk["usd_free_old"] = False
    wk.loc[wk.index[70], "usd_free_old"] = True              # a level that is too old
    tab = cal.usd_exceedance(wk, n_paths=100, seed=1, start=wk.index[50])
    bad = tab[~tab["usable"]]
    assert list(bad.index) == [wk.index[57], wk.index[70]]   # the windows ending and starting there
    assert set(bad["reason"]) == {"USD level too old at one end"}
    assert np.isnan(bad["fhs_q95"]).all()
    summary = cal.summarize(tab)
    assert summary.loc["all", "starts"] == len(tab) - 2
    assert summary.loc["all", "fhs_above"] == pytest.approx(tab.loc[tab["usable"], "fhs_above"].mean())
    assert summary.drop(index="all")["starts"].sum() == len(tab) - 2
    weekly_starts = pd.date_range("2020-01-01", periods=40, freq="7D")
    assert cal.independent_windows(weekly_starts) == 4       # weeks 0, 13, 26, 39
    assert cal.independent_windows(weekly_starts[:13]) == 1 and cal.independent_windows(weekly_starts[:14]) == 2
    assert cal.independent_windows(weekly_starts[[0, 5, 20, 30, 39]]) == 3
    lo, hi = cal.binomial_range(45)
    assert lo == 0.0 and hi == pytest.approx(5 / 45)         # P(X <= 4) = 0.936, P(X <= 5) = 0.978
    assert cal.BACKTEST_START == pd.Timestamp("2015-06-17")  # D-041: 1394-03-27
