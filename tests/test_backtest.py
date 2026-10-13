import numpy as np
import pandas as pd
import pytest

from coffee_risk.engine import backtest as bt
from coffee_risk.engine import cost as costs
from coffee_risk.engine import inputs as inp
from coffee_risk.engine.inputs import Inputs

RET = ["usd_free_ret", "arabica_usd_kg_ret", "robusta_usd_kg_ret"]


def make_weekly(n=140, seed=1):
    idx = pd.date_range("2018-01-03", periods=n, freq="7D")
    rng = np.random.default_rng(seed)
    usd = 0.004 + 0.03 * (1.0 + 0.8 * np.sin(np.arange(n) / 15.0)) * rng.standard_normal(n)
    common = rng.standard_normal(n)
    df = pd.DataFrame({"usd_free_ret": usd,
                       "arabica_usd_kg_ret": 0.04 * (0.8 * common + 0.6 * rng.standard_normal(n)),
                       "robusta_usd_kg_ret": 0.035 * (0.8 * common + 0.6 * rng.standard_normal(n))}, index=idx)
    df.iloc[0, :3] = np.nan
    df["usd_free"] = 500_000.0 * np.exp(np.nan_to_num(usd).cumsum())
    df["usd_free_old"] = False
    df["fund_tr_index"] = 10_000.0 * np.cumprod(1.0 + 0.004 + 0.002 * rng.random(n))    # an uneven weekly return
    df["fund_tr_index_old"] = False
    return df


def run(wk, **kw):
    args = dict(n_paths=200, seed=7, start=wk.index[60], premium_share=0.10)
    args.update(kw)
    return bt.run_backtest(wk, **args)


# ------------------------------------------------- realised cost = cost function
def test_realised_cost_is_the_cost_function_on_the_realised_path():
    wk = make_weekly()
    t = wk.index[70]
    i = Inputs(s0=float(wk.loc[t, "usd_free"]), market_rate=inp.market_rate(wk, t), premium_share=0.10)
    real, reason = bt.realized_paths(wk, t, i)
    assert reason == "" and real.n_paths == 1 and real.horizon == 13
    rows = wk.iloc[70:84]
    S, F = rows["usd_free"].values / rows["usd_free"].values[0], rows["fund_tr_index"].values / rows["fund_tr_index"].values[0]
    np.testing.assert_allclose(real.usd[0], S)
    np.testing.assert_allclose(real.funds[0], F)
    A, R = np.exp(rows["arabica_usd_kg_ret"].values[1:].sum()), np.exp(rows["robusta_usd_kg_ret"].values[1:].sum())
    assert real.arabica[0, 13] == pytest.approx(A) and real.robusta[0, 13] == pytest.approx(R)
    assert real.arabica[0, 0] == 1.0 and real.growth[13] == pytest.approx((1 + i.market_rate) ** 0.25)

    unit = i.usd_amount * i.s0
    fixed = i.with_(contract="fixed")
    # by hand: a rial spent at week k grows by what the fund earned from k to 13
    hand = {"now": F[13] / F[0], "payment_day": S[13],
            "staged_3": (S[0] * F[13] / F[0] + S[4] * F[13] / F[4] + S[9] * F[13] / F[9]) / 3,
            "partial_30": 0.3 * F[13] + 0.7 * S[13]}
    for s in (inp.now(), inp.payment_day(), inp.staged(), inp.partial(0.3)):
        assert costs.cost_on_paths(real, s, fixed)[0] / unit == pytest.approx(hand[s.name])
    need = 0.9 * (0.5 * A + 0.5 * R) + 0.1                           # floating: the dollars really needed
    gap = need - 1.0
    assert costs.cost_on_paths(real, inp.now(), i)[0] / unit == \
        pytest.approx(F[13] + (gap if gap >= 0 else gap * 0.99) * S[13])
    assert costs.cost_on_paths(real, inp.payment_day(), i)[0] / unit == pytest.approx(need * S[13])

    # and the backtest table holds exactly these numbers, as shares of the plan made at t
    table = run(wk, start=t, end=t)
    assert list(table.index) == [t] and bool(table.loc[t, "usable"])
    plan = (1 + i.market_rate) ** 0.25
    for name, value in hand.items():
        assert table.loc[t, f"fixed.{name}.cost"] == pytest.approx(value / plan)
        assert table.loc[t, f"fixed.{name}.margin"] == pytest.approx(1 - 0.75 * value / plan)
    for s in bt.fixed_strategies():
        cost, margin = bt.realized_outcome(real, s, i)
        assert table.loc[t, f"floating.{s.name}.cost"] == pytest.approx(cost)
        assert table.loc[t, f"floating.{s.name}.margin"] == pytest.approx(margin)
    assert table.loc[t, "usd_change"] == pytest.approx(S[13] - 1) and table.loc[t, "fund_growth"] == pytest.approx(F[13])
    assert table.loc[t, "coffee_need"] == pytest.approx(need)
    # the model's own strategies are priced with the same function
    h = table.loc[t, "fhs.h_star"]
    assert table.loc[t, "fhs.fixed.cost"] == pytest.approx((h * F[13] + (1 - h) * S[13]) / plan)
    assert table.loc[t, "fhs.pd_fixed.real"] == table.loc[t, "fixed.payment_day.cost"]
    assert table.loc[t, "fhs.usd.real"] == table.loc[t, "usd_change"]


def test_constant_funds_rate_is_used_only_when_no_realised_funds_are_given():
    usd = np.ones((1, 14))
    i = Inputs(s0=1.0, market_rate=0.30, premium_share=0.0, contract="fixed", usd_amount=1.0)
    from coffee_risk.engine.simulator import Paths
    plain = Paths(usd=usd, arabica=usd, robusta=usd, growth=np.ones(14))
    assert costs.cost_on_paths(plain, inp.now(), i)[0] == pytest.approx(1.30 ** 0.25)
    funds = np.linspace(1.0, 1.13, 14)[None, :]
    real = Paths(usd=usd, arabica=usd, robusta=usd, growth=np.ones(14), funds=funds)
    assert costs.cost_on_paths(real, inp.now(), i)[0] == pytest.approx(1.13)
    assert costs.cost_on_paths(real, inp.staged(), i)[0] == pytest.approx(
        (1.13 + 1.13 / funds[0, 4] + 1.13 / funds[0, 9]) / 3)


# ----------------------------------------------------------------- past only
def test_nothing_after_a_date_enters_the_decision_at_that_date():
    wk = make_weekly()
    a = run(wk)
    t = a.index[20]
    later = wk.copy()
    after = later.index > t
    later.loc[after, RET] = later.loc[after, RET] * 3.0 + 0.02
    later.loc[after, "usd_free"] *= 1.7
    later.loc[after, "fund_tr_index"] *= 1.2
    b = run(later)
    decision_cols = [c for c in a.columns if c.split(".")[-1] in ("h_star", "h_promise", "choice", "feasible",
                                                                 "promise", "q05", "q95", "max_rise",
                                                                 "floor", "floor_hedge")]
    assert len(decision_cols) == 2 * (2 + 3 + 6) + 3
    for c in decision_cols + ["s0", "market_rate", "sigma", "plan_growth"]:
        assert a.loc[t, c] == b.loc[t, c], c
        pd.testing.assert_series_equal(a.loc[:t, c], b.loc[:t, c])          # and at every earlier date
    # what happened afterwards does change, so the realised columns differ
    assert a.loc[t, "usd_change"] != b.loc[t, "usd_change"]
    assert a.loc[t, "fixed.payment_day.cost"] != b.loc[t, "fixed.payment_day.cost"]
    assert a.loc[t, "fhs.usd.real"] != b.loc[t, "fhs.usd.real"]
    # a change before the date does reach the decision
    earlier = wk.copy()
    earlier.loc[wk.index[70], "usd_free_ret"] = 0.6
    c = run(earlier)
    assert c.loc[t, "fhs.usd.q95"] != a.loc[t, "fhs.usd.q95"]


def test_only_dates_with_a_known_outcome_are_decision_dates():
    wk = make_weekly(100)
    table = run(wk)
    assert table.index[0] == wk.index[60] and table.index[-1] == wk.index[100 - 13 - 1]
    assert (table["end"] == table.index + pd.Timedelta(weeks=13)).all() and table["usable"].all()
    real, reason = bt.realized_paths(wk, wk.index[90], Inputs(s0=1.0, market_rate=0.3, premium_share=0.1))
    assert real is None and reason == "outcome not known yet"


def test_unusable_dates_are_kept_and_named():
    wk = make_weekly()
    wk.loc[wk.index[75], "robusta_usd_kg_ret"] = np.nan          # a coffee week with no return
    wk.loc[wk.index[100], "usd_free_old"] = True                 # a USD level that is too old
    table = run(wk, methods=("plain",))
    bad = table[~table["usable"]]
    coffee = [t for t in table.index if t < wk.index[75] <= t + pd.Timedelta(weeks=13)]
    usd = [wk.index[87], wk.index[100]]
    assert sorted(bad.index) == sorted(set(coffee) | set(usd))
    assert set(bad.loc[coffee, "reason"]) == {"a coffee week has no return"}
    assert set(bad.loc[usd, "reason"]) == {"USD level missing or too old"}
    assert bad["fixed.now.cost"].isna().all() and len(table) == 140 - 13 - 60
    summary = bt.strategy_summary(table, "fixed", 0.10, ["now", "payment_day"])
    ok = table[table["usable"]]
    assert summary.loc["payment_day", "mean_cost"] == pytest.approx(ok["fixed.payment_day.cost"].mean())
    assert summary.loc["payment_day", "breach"] == pytest.approx((ok["fixed.payment_day.margin"] < 0.10).mean())


# ------------------------------------------------------------------ same seed
def test_same_seed_gives_the_same_backtest():
    wk = make_weekly(110)
    a, b, c = run(wk), run(wk), run(wk, seed=8)
    pd.testing.assert_frame_equal(a, b)
    assert not a["fhs.usd.q95"].equals(c["fhs.usd.q95"])
    pd.testing.assert_series_equal(a["fixed.staged_3.cost"], c["fixed.staged_3.cost"])   # realised: no seed


# --------------------------------------------------------------- stress floor
def test_stress_floor_uses_the_largest_rise_seen_so_far_and_never_lowers_the_hedge():
    """D-051, the past-only variant of D5b."""
    from coffee_risk.engine import stress
    wk = make_weekly()
    wk.loc[wk.index[50]:, "usd_free"] *= 1.6                          # one big jump early in the history
    table = run(wk)
    t = table.index[25]
    k = wk.index.get_loc(t)
    level = wk["usd_free"].values
    rises = [level[j] / level[j - 13] - 1 for j in range(13, k + 1)]  # windows ending on or before t
    assert table.loc[t, "stress.max_rise"] == pytest.approx(max(rises))
    assert max(rises) > 0.5                                           # the jump is in it
    fixed = Inputs(s0=float(wk.loc[t, "usd_free"]), market_rate=inp.market_rate(wk, t), premium_share=0.10,
                   contract="fixed")
    share = stress.surviving_share(fixed, 1 + max(rises))
    assert 0 < share < 1 and table.loc[t, "stress.floor"] == pytest.approx(share)
    assert (table["fhs.floor_hedge"] >= table["fhs.h_star"] - 1e-12).all()
    assert (table["fhs.floor_hedge"] >= table["stress.floor"] - 1e-12).all()
    np.testing.assert_allclose(table["fhs.floor_hedge"], np.maximum(table["fhs.h_star"], table["stress.floor"]))
    real, _ = bt.realized_paths(wk, t, fixed)
    cost, margin = bt.realized_outcome(real, inp.partial(table.loc[t, "fhs.floor_hedge"]), fixed)
    assert table.loc[t, "fhs.floor.cost"] == pytest.approx(cost)
    assert table.loc[t, "fhs.floor.margin"] == pytest.approx(margin)
    # a rise that happens after a date never enters the floor of that date
    later = wk.copy()
    later.loc[later.index > t, "usd_free"] *= 3.0
    assert run(later).loc[t, "stress.floor"] == table.loc[t, "stress.floor"]
    assert "plain.floor_hedge" not in table.columns                  # the floor is combined with FHS only


def test_a_fixed_scenario_floor_applies_only_once_the_scenario_has_happened():
    """The sensitivity of D5b: a floor from one named scenario, at every date
    (which looks ahead) or only from the date it had happened."""
    from coffee_risk.engine import stress
    wk = make_weekly()
    table = run(wk)
    rise, since = 0.46, table.index[30]
    always = bt.scenario_floor(table, wk, rise)
    past_only = bt.scenario_floor(table, wk, rise, available_from=since)
    assert list(always.index) == list(table.index) == list(past_only.index)
    before, after = table.index < since, table.index >= since
    # before the scenario exists the floor is zero: the hedge and the outcome are the model's alone
    assert (past_only.loc[before, "floor"] == 0).all()
    np.testing.assert_array_equal(past_only.loc[before, "hedge"], table.loc[before, "fhs.h_star"])
    np.testing.assert_allclose(past_only.loc[before, "cost"], table.loc[before, "fhs.fixed.cost"], rtol=1e-12)
    np.testing.assert_allclose(past_only.loc[before, "margin"], table.loc[before, "fhs.fixed.margin"], rtol=1e-12)
    # from then on the two variants are the same
    pd.testing.assert_frame_equal(past_only[after], always[after])
    assert (always["floor"] > 0).all() and (always["hedge"] >= table["fhs.h_star"] - 1e-12).all()
    # one date by hand
    t = table.index[40]
    fixed = Inputs(s0=float(wk.loc[t, "usd_free"]), market_rate=inp.market_rate(wk, t), premium_share=0.10,
                   contract="fixed")
    share = stress.surviving_share(fixed, 1 + rise)
    assert always.loc[t, "floor"] == pytest.approx(share)
    hedge = max(table.loc[t, "fhs.h_star"], share)
    real, _ = bt.realized_paths(wk, t, fixed)
    cost, margin = bt.realized_outcome(real, inp.partial(hedge), fixed)
    assert always.loc[t, "hedge"] == pytest.approx(hedge)
    assert (always.loc[t, "cost"], always.loc[t, "margin"]) == (pytest.approx(cost), pytest.approx(margin))
    # fixed contract: the cost is a straight line between "payment day" and "all today"
    np.testing.assert_allclose(always["cost"], always["hedge"] * table["fixed.now.cost"]
                               + (1 - always["hedge"]) * table["fixed.payment_day.cost"], rtol=1e-12)
    # a rise nobody needs to hedge against changes nothing
    none = bt.scenario_floor(table, wk, 0.0)
    np.testing.assert_array_equal(none["hedge"], table["fhs.h_star"])


# ------------------------------------------------------------------ summaries
def test_model_choice_is_recorded_with_its_promise():
    wk = make_weekly()
    table = run(wk)
    names = {s.name for s in inp.strategy_set()}
    for m in ("fhs", "plain"):
        assert table[f"{m}.floating.choice"].isin(names).all()
        feasible = table[f"{m}.floating.feasible"].astype(bool)
        assert (table.loc[feasible, f"{m}.floating.promise"] >= 0.95 - 1e-12).all()
        assert (table.loc[~feasible, f"{m}.floating.promise"] < 0.95).all()      # the best available, not enough
        assert (table[f"{m}.h_promise"].dropna() >= 0.95 - 1e-12).all()
        assert table[f"{m}.h_star"].between(0, 1).all()


def test_pinball_loss_by_hand():
    y = np.array([10.0, 10.0, 10.0])
    q = np.array([8.0, 10.0, 13.0])
    np.testing.assert_allclose(bt.pinball(y, q, 0.95), [0.95 * 2, 0.0, 0.05 * 3])
    np.testing.assert_allclose(bt.pinball(y, q, 0.05), [0.05 * 2, 0.0, 0.95 * 3])
    # a band that is never exceeded still pays for its width
    assert bt.pinball(0.0, 5.0, 0.95) == pytest.approx(0.25) and bt.pinball(0.0, 1.0, 0.95) == pytest.approx(0.05)


def test_calibration_splits_by_the_median_volatility():
    wk = make_weekly()
    table = run(wk)
    reg = bt.regime(table)
    assert set(reg) == {"high", "low"} and abs((reg == "high").mean() - 0.5) <= 1 / len(reg)
    assert (table.loc[reg[reg == "high"].index, "sigma"] > table["sigma"].median()).all()
    cal = bt.calibration(table, "usd")
    assert list(cal.index.get_level_values("subset").unique()) == ["all", "high", "low"]
    ok = table[table["usable"]]
    assert cal.loc[("all", "fhs"), "above"] == pytest.approx((ok["fhs.usd.real"] > ok["fhs.usd.q95"]).mean())
    high = ok[(reg == "high").values]
    assert cal.loc[("high", "plain"), "below"] == pytest.approx((high["plain.usd.real"] < high["plain.usd.q05"]).mean())
    assert cal.loc[("all", "fhs"), "starts"] == cal.loc[("high", "fhs"), "starts"] + cal.loc[("low", "fhs"), "starts"]
    assert cal.loc[("all", "fhs"), "pinball"] == pytest.approx(
        (bt.pinball(ok["fhs.usd.real"], ok["fhs.usd.q05"], 0.05).mean()
         + bt.pinball(ok["fhs.usd.real"], ok["fhs.usd.q95"], 0.95).mean()) / 2)
    # fixed contract, payment day: the cost is the USD change in other units, so the same hits
    fixed = bt.calibration(table, "pd_fixed")
    np.testing.assert_allclose(fixed["above"], cal["above"])
    np.testing.assert_allclose(fixed["below"], cal["below"])


def test_breach_rates_by_year_and_cheapest_share():
    wk = make_weekly()
    table = run(wk, methods=("plain",))
    by = bt.by_year(table, {"late": "fixed.payment_day.margin", "now": "fixed.now.margin"}, 0.10)
    assert by.drop(index="all")["starts"].sum() == by.loc["all", "starts"] == len(table)
    assert by.loc["all", "late"] == pytest.approx((table["fixed.payment_day.margin"] < 0.10).mean())
    share = bt.cheapest_share(table, "fixed")
    assert share.sum() == pytest.approx(1.0) and list(share.index) == list(bt.MAIN)
    costs_ = table[[f"fixed.{n}.cost" for n in bt.MAIN]].values
    assert share["now"] == pytest.approx((costs_.argmin(axis=1) == 0).mean())
