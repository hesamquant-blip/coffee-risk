"""The calculator kit: the files, and the arithmetic the page must reproduce."""
import json
import re

import numpy as np
import pandas as pd
import pytest

from coffee_risk.engine import calculator as calc
from coffee_risk.engine import cost as costs
from coffee_risk.engine import export
from coffee_risk.engine import inputs as inp
from coffee_risk.engine import simulator as sim
from coffee_risk.engine import stress
from coffee_risk.engine.inputs import Inputs
from coffee_risk.units import cents_lb_to_usd_kg

N, SEED = 600, 11


def make_weekly(n=320, seed=1):
    idx = pd.date_range("2018-01-03", periods=n, freq="7D")
    rng = np.random.default_rng(seed)
    usd = 0.004 + 0.03 * (1.0 + 0.8 * np.sin(np.arange(n) / 15.0)) * rng.standard_normal(n)
    common = rng.standard_normal(n)
    df = pd.DataFrame({"usd_free_ret": usd,
                       "arabica_usd_kg_ret": 0.04 * (0.8 * common + 0.6 * rng.standard_normal(n)),
                       "robusta_usd_kg_ret": 0.035 * (0.8 * common + 0.6 * rng.standard_normal(n))}, index=idx)
    df.iloc[0, :3] = np.nan
    df["usd_free"] = 500_000.0 * np.exp(np.nan_to_num(usd).cumsum())
    df["fund_tr_index"] = 10_000.0 * 1.005 ** np.arange(n)
    return df


@pytest.fixture(scope="module")
def world():
    wk = make_weekly()
    as_of = wk.index[-1]
    days = pd.bdate_range("2020-01-01", as_of)
    daily = pd.DataFrame({"arabica_usd_kg": 4.5, "robusta_usd_kg": 2.4}, index=days)
    per_kg = float(cents_lb_to_usd_kg(1.0))
    imf = pd.DataFrame({"month": pd.date_range("2020-01-01", as_of, freq="MS"),
                        "arabica_imf_clb": 5.0 / per_kg, "robusta_imf_clb": 3.0 / per_kg})
    base = Inputs.from_data(as_of, wk, daily, imf)
    kit = export.paths_kit(as_of, N, SEED, options=base, weekly=wk, daily=daily, imf=imf)
    kit = json.loads(json.dumps(kit))                       # exactly what a reader of the file gets
    scen = {"scenarios": [
        {"key": "S1", "payment_week": 13, "weeks": [4, 9, 13], "usd": [0.18, 0.39, 1.41],
         "arabica": [-0.08, -0.18, -0.23], "robusta": [-0.01, -0.07, -0.09]},
        {"key": "S2", "payment_week": 13, "weeks": [4, 9, 13], "usd": [0.12, 0.24, 0.46],
         "arabica": [0.03, -0.01, 0.22], "robusta": [0.05, 0.10, 0.23]},
        {"key": "S6", "payment_week": 32, "weeks": [11, 21, 32], "usd": [0.08, 0.16, 0.60],
         "arabica": [0.02, 0.16, 0.15], "robusta": [-0.01, 0.06, -0.05]}]}
    return {"wk": wk, "as_of": as_of, "daily": daily, "imf": imf, "base": base, "kit": kit, "scen": scen}


# ------------------------------------------------------------------ the file
def test_paths_file_holds_undated_samples_and_the_settings(world):
    kit = world["kit"]
    assert kit["n_paths"] == N and kit["horizon"] == 13 and kit["scale"] == 1e-5 and kit["seed"] == SEED
    for name in ("usd_z", "arabica", "robusta"):
        assert len(kit[name]) == N * 13 and all(isinstance(v, int) for v in kit[name])
    assert len(kit["demand_z"]) == N and all(isinstance(v, int) for v in kit["demand_z"])
    assert kit["s0"] == world["base"].s0 and kit["market_rate"] == world["base"].market_rate
    assert kit["premium"] == {"arabica": pytest.approx(0.10), "robusta": pytest.approx(0.20)}
    prep = sim.prepare(world["wk"], world["as_of"], world["base"])
    assert kit["usd_vol_weekly"] == prep["info"]["sigma_as_of"]
    assert kit["hist_mean_weekly_usd_return"] == pytest.approx(np.expm1(prep["usd_raw"]).mean())
    assert kit["half_life"] == 13 and kit["mean_block"] == 13
    # the residuals are the pool's, demeaned: the sample mean is near zero, the spread near one
    z = calc.decode(kit, "usd_z")
    assert abs(z.mean()) < 0.1 and 0.8 < z.std() < 1.3
    # nothing dated but the decision date, and no levels
    text = json.dumps(kit)
    assert re.findall(r"\d{4}-\d{2}-\d{2}", text) == [str(world["as_of"].date())]
    assert not any(k in kit for k in ("weeks", "dates", "usd_free", "prices", "levels", "usd"))


def test_the_same_weeks_are_sampled_for_the_three_series(world):
    """Path p, week k of each array comes from one pool week: rebuild the
    indices the way simulate() draws them and compare."""
    kit, wk, base = world["kit"], world["wk"], world["base"]
    prep = sim.prepare(wk, world["as_of"], base)
    idx = sim.stationary_indices(np.random.default_rng(SEED), len(prep["weeks"]), N, 13, base.mean_block)
    np.testing.assert_allclose(calc.decode(kit, "arabica"), prep["arabica"][idx], atol=0.5e-5 + 1e-12)
    np.testing.assert_allclose(calc.decode(kit, "robusta"), prep["robusta"][idx], atol=0.5e-5 + 1e-12)
    np.testing.assert_allclose(calc.decode(kit, "usd_z"), prep["usd_z"][idx], atol=0.5e-5 + 1e-12)
    np.testing.assert_allclose(calc.decode(kit, "demand_z"), costs.demand_normals(N, SEED), atol=0.5e-5 + 1e-12)


@pytest.mark.parametrize("changes", [dict(), dict(usd_vol_scale=0.6), dict(drift="zero"), dict(drift="historical"),
                                     dict(drift=0.8), dict(usd_vol_scale=1.5, drift="historical")])
def test_page_paths_are_the_engines_paths(world, changes):
    """The page's steps on the file give the paths simulate() gives with the
    same seed, up to the rounding of the file."""
    kit = world["kit"]
    engine = sim.simulate(world["as_of"], N, 13, SEED, world["base"].with_(**changes), world["wk"])
    page = calc.kit_paths(kit, calc.kit_inputs(kit, **changes))
    np.testing.assert_allclose(page.growth, engine.growth, rtol=1e-12)
    for name in ("usd", "arabica", "robusta"):
        np.testing.assert_allclose(getattr(page, name), getattr(engine, name), rtol=2e-4)
        np.testing.assert_allclose(getattr(page, name).mean(axis=0), getattr(engine, name).mean(axis=0), rtol=1e-12)


def test_volatility_scale_widens_the_usd_paths_only(world):
    kit = world["kit"]
    one, half = calc.kit_paths(kit, calc.kit_inputs(kit)), calc.kit_paths(kit, calc.kit_inputs(kit, usd_vol_scale=0.5))
    assert np.log(half.usd[:, 13]).std() == pytest.approx(0.5 * np.log(one.usd[:, 13]).std(), rel=1e-9)
    np.testing.assert_array_equal(one.arabica, half.arabica)
    np.testing.assert_allclose(half.usd.mean(axis=0), one.usd.mean(axis=0), rtol=1e-12)     # same drift
    engine = sim.prepare(world["wk"], world["as_of"], world["base"].with_(usd_vol_scale=0.5))
    np.testing.assert_allclose(engine["usd"], 0.5 * sim.prepare(world["wk"], world["as_of"], world["base"])["usd"])
    with pytest.raises(ValueError):
        world["base"].with_(usd_vol_scale=-1.0)


def test_demand_factor_uses_the_stored_draws(world):
    kit = world["kit"]
    i = calc.kit_inputs(kit, demand_sd=0.10)
    page = calc.kit_paths(kit, i)
    engine = sim.simulate(world["as_of"], N, 13, SEED, world["base"].with_(demand_sd=0.10), world["wk"])
    assert page.demand_z is not None and engine.demand_z is None
    s2 = np.log1p(0.10 ** 2)
    by_hand = costs.planned_cost(page, i) / 0.75 * np.exp(-s2 / 2 + np.sqrt(s2) * calc.decode(kit, "demand_z"))
    np.testing.assert_allclose(costs.revenue(page, i), by_hand, rtol=1e-12)
    np.testing.assert_allclose(costs.revenue(page, i), costs.revenue(engine, world["base"].with_(demand_sd=0.10)),
                               rtol=1e-5)
    assert costs.demand_factor(5, 0.0, z=np.arange(5.0)).tolist() == [1.0] * 5


# --------------------------------------------- the spec, followed to the letter
def spec_quantile(x, q):
    y = np.sort(np.asarray(x, dtype=float))
    pos = q * (len(y) - 1)
    i = int(np.floor(pos))
    return float(y[i]) if i == len(y) - 1 else float(y[i] + (pos - i) * (y[i + 1] - y[i]))


def spec_growth(kit, c, k):
    if c["drift"] == "neutral":
        return (1 + c["market_rate"]) ** (k / 52)
    if c["drift"] == "zero":
        return 1.0
    if c["drift"] == "historical":
        return (1 + kit["hist_mean_weekly_usd_return"]) ** k
    return (1 + c["drift"]) ** (k / 52)


def spec_outputs(kit, scenarios, c):
    """docs/calculator_spec.md sections 3-11 with nothing from the engine:
    `c` is the input record a case of parity.json carries."""
    n_paths, H, sc = kit["n_paths"], kit["horizon"], kit["scale"]
    T = c["payment_week"]

    def levels(name, factor=1.0):
        r = np.array(kit[name], dtype=float).reshape(n_paths, H) * sc * factor
        return np.exp(np.cumsum(r, axis=1))

    g = np.array([spec_growth(kit, c, k) for k in range(H + 1)])
    LS, LA, LR = levels("usd_z", kit["usd_vol_weekly"] * c["usd_vol_scale"]), levels("arabica"), levels("robusta")
    one = np.ones((n_paths, 1))
    S = np.hstack([one, LS * g[1:] / LS.mean(axis=0)])
    A, R = np.hstack([one, LA / LA.mean(axis=0)]), np.hstack([one, LR / LR.mean(axis=0)])
    w, prem = c["arabica_share"], c["premium_share"]
    need = c["usd_amount"] * (1.0 if c["contract"] == "fixed" else (1 - prem) * (w * A[:, T] + (1 - w) * R[:, T]) + prem)
    rate = c["s0"] * (1 - c["channel_discount"]) * S

    def cost_of(tranches):
        cost, bought = np.zeros(n_paths), 0.0
        for t, q in tranches:
            usd = q * c["usd_amount"]
            years = (T - t) / 52
            cost = cost + usd * rate[:, t] * ((1 + c["funds_rate"]) ** years + c["fx_holding_cost"] * years)
            bought += usd
        gap = need - bought
        return cost + rate[:, T] * np.where(gap >= 0, gap, gap * (1 - c["sellback_spread"]))

    c_plan = c["usd_amount"] * c["s0"] * (1 - c["channel_discount"]) * g[T]
    x = S[:, T] / g[T] - 1
    if c["pass_through_mode"] == "up_only":
        x = np.maximum(x, 0)
    s2 = np.log(1 + c["demand_sd"] ** 2)
    D = 1.0 if c["demand_sd"] == 0 else np.exp(-s2 / 2 + np.sqrt(s2) * np.array(kit["demand_z"]) * sc)
    revenue = c_plan / (1 - c["planned_margin"]) * (1 + c["pass_through"] * x) * D
    alpha, floor = c["confidence"], c["margin_floor"]

    def metrics(tranches):
        cost = cost_of(tranches)
        margin = 1 - cost / revenue
        k_tail = max(1, int(np.ceil(n_paths * (1 - alpha) - 1e-9)))
        return {"mean": cost.mean() / c_plan, "median": spec_quantile(cost, 0.5) / c_plan,
                "var95": spec_quantile(cost, alpha) / c_plan, "cvar95": np.sort(cost)[-k_tail:].mean() / c_plan,
                "p_over_budget": float((cost > c["budget_factor"] * c_plan).mean()),
                "p_margin_below_floor": float((margin < floor).mean()), "margin_q05": spec_quantile(margin, 1 - alpha)}

    staged = [(int(round(k * T / 3)), 1 / 3) for k in range(3)]
    tranches = {"now": [(0, 1.0)], "staged": staged, "payment_day": []}
    tranches.update({f"h_{10 * k}": ([(0, k / 10)] if k else []) for k in range(11)})
    out = {"plan_growth": g[T], "plan_rial": c_plan, "strategies": {n: metrics(t) for n, t in tranches.items()}}

    order = ["now", "staged", "payment_day"] + [f"h_{10 * k}" for k in range(1, 10)]
    m = out["strategies"]
    prob = {n: 1 - m[n]["p_margin_below_floor"] for n in order}
    out["best_probability"] = max(prob.values())
    out["best_probability_strategy"] = next(n for n in order if prob[n] == out["best_probability"])
    out["highest_floor"] = max(m[n]["margin_q05"] for n in order)
    out["highest_floor_strategy"] = next(n for n in order if m[n]["margin_q05"] == out["highest_floor"])
    ok = [n for n in order if prob[n] >= alpha - 1e-12]
    out["qualifies"], out["chosen"] = bool(ok), None
    if ok:
        best = min(m[n]["mean"] for n in ok)
        tied = [n for n in ok if m[n]["mean"] - best < max(c["tie_tolerance"], 1e-9)]
        low = min(m[n]["cvar95"] for n in tied)
        out["chosen"] = next(n for n in tied if m[n]["cvar95"] == low)
    out["min_hedge"] = None
    for i in range(101):
        h = i / 100
        if 1 - metrics([(0, h)] if h else [])["p_margin_below_floor"] >= alpha - 1e-12:
            out["min_hedge"] = h
            break

    L = (1 - floor) / (1 - c["planned_margin"])
    rho = c["pass_through"]

    def terms(T_):
        return spec_growth(kit, c, T_), (1 + c["funds_rate"]) ** (T_ / 52) + c["fx_holding_cost"] * T_ / 52

    def breaking(h):
        g_, F = terms(T)
        if rho == 0:
            return (L * g_ - h * F) / (1 - h)
        if h * F + (1 - h) * g_ > L * g_:
            return g_
        return None if 1 - h - L * rho <= 0 else (L * g_ * (1 - rho) - h * F) / (1 - h - L * rho)

    out["reverse_stress"] = {f"h_{int(round(h * 100))}": breaking(h) for h in (0.0, 0.25, 0.5, 0.75)}
    out["stress_floor"], out["scenarios"] = {}, {}
    for s in scenarios:
        Ts = s["payment_week"]
        g_, F = terms(Ts)
        x_end = 1 + s["usd"][-1]
        surprise = x_end / g_ - 1
        if c["pass_through_mode"] == "up_only":
            surprise = max(surprise, 0)
        allowed = L * g_ * (1 + rho * surprise)
        if s["key"] in ("S1", "S2"):
            out["stress_floor"][s["key"]] = (0.0 if x_end <= allowed else None if F > allowed
                                             else min((x_end - allowed) / (x_end - F), 1.0))
        # replay: one path, levels at the staged weeks and at the payment week
        w1, w2, _ = s["weeks"]
        Su = {0: 1.0, w1: 1 + s["usd"][0], w2: 1 + s["usd"][1], Ts: 1 + s["usd"][2]}
        need_s = c["usd_amount"] * (1.0 if c["contract"] == "fixed" else
                                    (1 - prem) * (w * (1 + s["arabica"][2]) + (1 - w) * (1 + s["robusta"][2])) + prem)
        plan = c["usd_amount"] * c["s0"] * (1 - c["channel_discount"]) * g_
        rev = plan / (1 - c["planned_margin"]) * (1 + rho * surprise)
        res = {}
        for name, tr in (("now", [(0, 1.0)]), ("staged", [(0, 1 / 3), (w1, 1 / 3), (w2, 1 / 3)]), ("payment_day", [])):
            cost, bought = 0.0, 0.0
            for t, q in tr:
                usd = q * c["usd_amount"]
                years = (Ts - t) / 52
                cost += usd * c["s0"] * (1 - c["channel_discount"]) * Su[t] * (
                    (1 + c["funds_rate"]) ** years + c["fx_holding_cost"] * years)
                bought += usd
            gap = need_s - bought
            cost += c["s0"] * (1 - c["channel_discount"]) * Su[Ts] * (gap if gap >= 0 else gap * (1 - c["sellback_spread"]))
            res[name] = {"cost": cost / plan, "margin": 1 - cost / rev, "breaks": bool(1 - cost / rev < floor)}
        out["scenarios"][s["key"]] = res
    return out


def _same(a, b, path=""):
    if isinstance(a, dict):
        assert set(a) <= set(b), f"{path}: {set(a) - set(b)}"
        for k in a:
            _same(a[k], b[k], f"{path}.{k}")
    elif a is None or isinstance(a, (str, bool)):
        assert a == b, f"{path}: {a!r} != {b!r}"
    else:
        assert b == pytest.approx(a, rel=1e-9, abs=1e-12), f"{path}: {a!r} != {b!r}"


@pytest.mark.parametrize("name", list(export.CASES))
def test_the_spec_followed_literally_gives_the_parity_numbers(world, name):
    """Sections 3-11 of docs/calculator_spec.md, written again with plain
    arrays, against what parity.json holds for every case."""
    kit, scen = world["kit"], world["scen"]
    parity = export.parity_kit(kit, scen, {name: export.CASES[name]})["cases"][0]
    by_spec = spec_outputs(kit, scen["scenarios"], parity["inputs"])
    _same(by_spec, json.loads(json.dumps(parity["outputs"])), name)


def test_parity_holds_the_twelve_cases_of_the_spec(world):
    parity = export.parity_kit(world["kit"], world["scen"])
    assert [c["name"] for c in parity["cases"]] == [
        "base", "fixed", "funds_plus_10pp", "fx_holding_3pct", "drift_zero", "drift_historical",
        "arabica_share_0", "arabica_share_1", "pass_through_0.5_up_only", "demand_sd_10pct", "usd_vol_x0.6",
        "margin_floor_5pct"]
    json.dumps(parity, allow_nan=False)                         # null, never NaN
    by = {c["name"]: c for c in parity["cases"]}
    base = by["base"]
    assert set(base["outputs"]["strategies"]) == {"now", "staged", "payment_day"} | {f"h_{10 * k}" for k in range(11)}
    assert base["outputs"]["strategies"]["h_0"] == base["outputs"]["strategies"]["payment_day"]
    assert base["outputs"]["strategies"]["h_100"] == base["outputs"]["strategies"]["now"]
    assert set(base["outputs"]["stress_floor"]) == {"S1", "S2"}
    assert by["funds_plus_10pp"]["inputs"]["funds_rate"] == pytest.approx(base["inputs"]["market_rate"] + 0.10)
    assert by["arabica_share_0"]["inputs"]["premium_share"] == pytest.approx(world["kit"]["premium"]["robusta"])
    assert by["arabica_share_1"]["inputs"]["premium_share"] == pytest.approx(world["kit"]["premium"]["arabica"])
    assert by["pass_through_0.5_up_only"]["inputs"]["pass_through_mode"] == "up_only"
    assert by["usd_vol_x0.6"]["outputs"]["strategies"]["payment_day"]["cvar95"] \
        < base["outputs"]["strategies"]["payment_day"]["cvar95"]
    fixed = by["fixed"]["outputs"]
    assert fixed["strategies"]["now"]["cvar95"] == pytest.approx(1.0) and fixed["min_hedge"] is not None
    # the page's engine is the report's engine: the decision of dec.run on the same paths
    from coffee_risk.engine import decision as dec
    i = calc.kit_inputs(world["kit"], contract="fixed")
    res = dec.run(calc.kit_paths(world["kit"], i), i)
    assert fixed["min_hedge"] == res.min_hedge and fixed["highest_floor"] == res.decision.highest_floor


# ----------------------------------------------------------------- scenarios
def test_scenario_file_holds_aggregates_only(monkeypatch):
    rng = np.random.default_rng(3)

    def fake(_weekly):
        out = []
        for key, T in (("fx_1397", 13), ("war_full", 32)):
            r = pd.DataFrame(rng.normal(0.01, 0.04, (T, 3)), columns=["usd", "arabica", "robusta"])
            window = (pd.Timestamp("2018-06-27"), pd.Timestamp("2018-06-27") + pd.Timedelta(weeks=T))
            out.append(stress.Scenario(key, r, window, window))
        return out

    monkeypatch.setattr(stress, "scenarios", fake)
    doc = export.scenarios_kit(None)
    s1, s6 = doc["scenarios"]
    assert (s1["key"], s1["payment_week"], s1["weeks"]) == ("S1", 13, [4, 9, 13])
    assert (s6["key"], s6["payment_week"], s6["weeks"]) == ("S6", 32, [11, 21, 32])
    for s in (s1, s6):
        assert all(len(s[n]) == 3 for n in ("usd", "arabica", "robusta"))       # three numbers, no weekly series
        assert s["usd_window"][0] == "2018-06-27" and not s["hypothetical"]
    json.dumps(doc, allow_nan=False)


def test_a_scenario_priced_from_its_aggregates_equals_the_full_replay(world):
    """The three numbers per series are all a strategy needs: the outcome is
    the same as replaying the full weekly path (D4)."""
    kit = world["kit"]
    rng = np.random.default_rng(8)
    for T in (13, 32):
        r = pd.DataFrame(rng.normal(0.01, 0.04, (T, 3)), columns=["usd", "arabica", "robusta"])
        full = stress.Scenario("x", r, (None, None), (None, None))
        weeks = [w for w, _ in inp.staged(3, T).tranches if w > 0] + [T]
        agg = {"key": "X", "payment_week": T, "weeks": weeks}
        for n in ("usd", "arabica", "robusta"):
            lvl = np.concatenate([[1.0], np.exp(np.cumsum(r[n].values))])
            agg[n] = [float(lvl[w] - 1) for w in weeks]
        for changes in (dict(), dict(contract="fixed"), dict(pass_through=0.5), dict(drift="zero")):
            i = calc.kit_inputs(kit, **changes).with_(payment_week=T)
            want = stress.outcomes(stress.replay(full.returns, i), i,
                                   [inp.now(), inp.staged(3, T), inp.payment_day()])
            got = calc.scenario_outcomes(kit, agg, calc.kit_inputs(kit, **changes))
            for name, key in (("now", "now"), ("staged", "staged_3"), ("payment_day", "payment_day")):
                assert got[name]["cost"] == pytest.approx(want.loc[key, "cost_pct"], rel=1e-12)
                assert got[name]["margin"] == pytest.approx(want.loc[key, "margin"], rel=1e-9, abs=1e-12)


def test_stress_floor_of_a_scenario_is_the_share_that_survives_its_usd_rise(world):
    kit, scen = world["kit"], world["scen"]["scenarios"]
    i = calc.kit_inputs(kit, contract="fixed", sellback_spread=0.0)
    for s in scen:
        h = calc.stress_floor(kit, s, i)
        T = s["payment_week"]
        only_usd = dict(s, arabica=[0, 0, 0], robusta=[0, 0, 0], weeks=s["weeks"])
        p = calc.scenario_paths(kit, only_usd, i.with_(payment_week=T))
        j = i.with_(payment_week=T)
        m = costs.margin(costs.cost_on_paths(p, inp.partial(h), j), costs.revenue(p, j))[0]
        assert 0 < h < 1 and m == pytest.approx(j.margin_floor, abs=1e-9)
        assert stress.breaking_usd_level(j, h, float(p.growth[T])) == pytest.approx(1 + s["usd"][-1])
    assert calc.stress_floor(kit, dict(scen[0], usd=[0, 0, 0.05]), i) == 0.0


def test_default_stress_scenario_is_the_largest_real_13_week_jump(world):
    """D-053."""
    scen = world["scen"]["scenarios"]
    assert calc.default_stress_scenario(scen)["key"] == "S1"                       # +141% against +46%
    bigger_but_long = scen + [dict(scen[2], key="S6b", usd=[0.5, 1.0, 3.0])]       # 32 weeks: not a 13-week jump
    assert calc.default_stress_scenario(bigger_but_long)["key"] == "S1"
    storm = dict(scen[0], key="S7", hypothetical=True, usd=[0.5, 1.0, 5.0])        # never happened
    assert calc.default_stress_scenario(scen + [storm])["key"] == "S1"
    new_jump = dict(scen[1], key="S8", usd=[0.3, 0.9, 1.9])                        # a bigger real jump takes over
    assert calc.default_stress_scenario(scen + [new_jump])["key"] == "S8"
    tie = dict(scen[0], key="S9")
    assert calc.default_stress_scenario(scen + [tie])["key"] == "S1"               # the first on a tie
    assert calc.default_stress_scenario([s for s in scen if s["payment_week"] != 13]) is None


def test_files_round_trip(tmp_path, world):
    facts = export.write_json(world["kit"], tmp_path / "calculator_paths.json")
    assert facts["bytes"] == (tmp_path / "calculator_paths.json").stat().st_size and facts["gzip_bytes"] < facts["bytes"]
    assert export.read_json(tmp_path / "calculator_paths.json") == world["kit"]
