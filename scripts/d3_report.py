#!/usr/bin/env python
"""Build the D3 report: reports/d3_engine/README.md (Persian) and its figures.

FROZEN since D6-prep. This script produced the D3 report the author reviewed
(rebuilt in D3b), at the commit tagged `d3b`. Its last step wrote the
calculator's path file in the D3 format; D6-prep replaced that file by the
calculator kit (`scripts/d6_kit.py`), so running this script from a later
commit would no longer describe what is on disk. To reproduce the report:

    git checkout d3b && .venv/bin/python scripts/d3_report.py

D3 is the engine: simulate joint 13-week paths, price each FX-purchase
strategy in rials, measure the risk, apply the decision rule. This script runs
the base case as of the latest week of the weekly panel, the sensitivities and
the calibration preview (D3b). Every number, table and figure in the report
is computed here.
"""
from __future__ import annotations

import sys

if __name__ == "__main__":
    sys.exit("scripts/d3_report.py is frozen (see its docstring): "
             "check out the tag `d3b` to reproduce the D3 report.")

import datetime as dt
import re
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from coffee_risk import eda, paths
from coffee_risk.data import fred, panel, weekly
from coffee_risk.engine import calibration as cal
from coffee_risk.engine import cost as costs
from coffee_risk.engine import decision as dec
from coffee_risk.engine import export as exporter
from coffee_risk.engine import inputs as I
from coffee_risk.engine import settings
from coffee_risk.engine import simulator as sim
from coffee_risk.engine.inputs import Inputs
from coffee_risk.reporting import (AQUA, BLUE, INK, INK2, ORANGE, fa, jd, jg, num, pct, style,
                                   table)
from coffee_risk.reporting import billions as bn, strategy_fa as strat_fa

OUT = paths.REPORTS / "d3_engine"
N_PATHS = settings.REPORT_PATHS
N_EXPORT = settings.EXPORT_PATHS
N_CALIBRATION = settings.BACKTEST_PATHS
SEED = settings.REPORT_SEED
T = 13
ANN = np.sqrt(I.WEEKS_PER_YEAR)
PREMIUM_MIN_DAYS = 15            # the default of inputs.premium_by_coffee
MAIN = ("now", "staged_3", "payment_day")
EN = {"now": "all today", "staged_3": "staged (weeks 0, 4, 9)", "payment_day": "all on the payment day"}
COLOR = {"now": BLUE, "staged_3": AQUA, "payment_day": ORANGE}


# ------------------------------------------------------------------ wording
def share_fa(h) -> str:
    return "هیچ" if h is None else pct(h, 0)


def _save(fig, name: str) -> str:
    fig.savefig(OUT / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return name


# ------------------------------------------------------------------- running
class Runner:
    """Runs a case once per distinct Inputs (same seed everywhere, so the
    cases share their random draws and differ only by the change made)."""

    def __init__(self, as_of, wk):
        self.as_of, self.wk, self.cache = as_of, wk, {}

    def __call__(self, inputs: Inputs):
        if inputs not in self.cache:
            p = sim.simulate(self.as_of, N_PATHS, T, SEED, inputs, self.wk)
            self.cache[inputs] = (p, dec.run(p, inputs))
        return self.cache[inputs]


def max_hedge(res: dec.Result):
    c = res.curve
    ok = c.index[(1.0 - c["p_margin_below_floor"]) >= res.inputs.confidence - 1e-12]
    return float(ok.max()) if len(ok) else None


def hedge_band(res: dec.Result, k: float = 2.0):
    """The minimum hedge ratio if every probability moved k Monte-Carlo
    standard errors down / up."""
    c, allowed = res.curve, 1.0 - res.inputs.confidence
    p, se = c["p_margin_below_floor"], c["se_p_margin_below_floor"]
    lo, hi = c.index[p - k * se <= allowed], c.index[p + k * se <= allowed]
    return (float(lo.min()) if len(lo) else None, float(hi.min()) if len(hi) else None)


def groups(base: Inputs, premium) -> list:
    """The one-at-a-time changes of the spec: (key, Persian name, English
    name, [(Persian label, English label, Inputs, is_base)])."""
    def v(fa_label, en_label, **ch):
        return (fa_label, en_label, base.with_(**ch), not ch)
    return [
        ("drift", "رانش دلار", "USD drift",
         [v("خنثی", "neutral"), v("میانگین تاریخی (فقط گذشته)", "historical", drift="historical"),
          v("صفر", "zero", drift="zero")]),
        ("funds_rate", "نرخ تأمین مالی بنگاه", "funds rate",
         [v("برابر نرخ بازار", "market"), v("نرخ بازار + ۱۰ واحد درصد", "market + 10 pp", funds_spread=0.10)]),
        ("fx_holding_cost", "هزینهٔ نگهداری ارز", "FX holding cost",
         [v("صفر", "0"), v("٪۳ در سال", "3%", fx_holding_cost=0.03)]),
        ("contract", "قرارداد قهوه", "contract",
         [v("شناور", "floating"), v("ثابت", "fixed", contract="fixed")]),
        ("arabica_share", "سهم عربیکا", "arabica share",
         [v("۰", "0", arabica_share=0.0, premium_share=premium(0.0)), v("۰.۵", "0.5"),
          v("۱", "1", arabica_share=1.0, premium_share=premium(1.0))]),
        ("usd_vol", "نوسان دلار", "USD volatility",
         [v("فیلترشده، نیمه‌عمر ۱۳", "FHS hl 13"), v("فیلترشده، نیمه‌عمر ۲۶", "FHS hl 26", half_life=26.0),
          v("بوت‌استرپ ساده", "plain", usd_vol="plain")]),
        ("mean_block", "میانگین طول بلوک", "mean block",
         [v("۴ هفته", "4", mean_block=4.0), v("۸ هفته", "8", mean_block=8.0), v("۱۳ هفته", "13")]),
        ("pass_through", "انتقال به قیمت فروش", "pass-through",
         [v("صفر", "0"), v("۰.۵، فقط بالا", "0.5 up-only", pass_through=0.5),
          v("۰.۵، متقارن", "0.5 sym.", pass_through=0.5, pass_through_mode="symmetric")]),
        ("demand_sd", "نااطمینانی فروش", "demand sd",
         [v("صفر", "0"), v("٪۱۰", "10%", demand_sd=0.10)]),
        ("margin_floor", "کف حاشیهٔ سود", "margin floor",
         [v("٪۵", "5%", margin_floor=0.05), v("٪۱۰", "10%"), v("٪۱۵", "15%", margin_floor=0.15)]),
    ]


# -------------------------------------------------------------------- figures
def fig_costs(cost_pct: dict, table_: pd.DataFrame, limit: float) -> str:
    fig, axes = plt.subplots(3, 1, figsize=(9.5, 6.4), sharex=True, sharey=True)
    bins = np.linspace(40, 240, 81)
    for ax, name in zip(axes, MAIN):
        x = cost_pct[name] * 100
        ax.hist(np.clip(x, bins[0], bins[-1]), bins=bins, color=COLOR[name], alpha=0.85, lw=0,
                weights=np.full(len(x), 100.0 / len(x)))
        ax.axvline(100, color=INK2, lw=0.9)
        ax.axvline(limit * 100, color=INK, lw=1.1, ls="--")
        cvar = table_.loc[name, "cvar_pct"] * 100
        ax.axvline(cvar, color=COLOR[name], lw=1.6, ls=":")
        ax.set_title(f"{EN[name]}:  mean {table_.loc[name, 'mean_cost_pct'] * 100:.1f}%,  "
                     f"CVaR95 {cvar:.0f}% (dotted),  P(margin < floor) "
                     f"{table_.loc[name, 'p_margin_below_floor'] * 100:.1f}%",
                     fontsize=9.5, fontweight="normal", color=INK, pad=4)
        ax.set_ylabel("% of paths")
    fig.suptitle("Rial cost at the payment week, base case (floating contract)", x=0.01, ha="left",
                 fontsize=11, fontweight="bold")
    axes[-1].set_xlabel(f"cost, % of the planned cost  (solid line = plan, dashed = cost at which the "
                        f"margin hits the floor, {limit * 100:.0f}%; last bin holds everything above 240%)")
    fig.tight_layout()
    return _save(fig, "fig1_cost_distributions.png")


def fig_insurance(curves: list, confidence: float, min_h) -> str:
    fig, (a, b) = plt.subplots(2, 1, figsize=(9.5, 6.2), sharex=True)
    for label, color, c in curves:
        h = c.index.values * 100
        a.plot(h, c["mean_cost_pct"] * 100, color=color, lw=1.8, label=label)
        b.plot(h, c["p_margin_below_floor"] * 100, color=color, lw=1.8, label=label)
    a.set_ylim(98.5, 101.5)
    a.set_ylabel("mean cost, % of plan")
    a.set_title("Price of insurance: what buying a share today costs and what it buys", pad=26)
    a.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, borderaxespad=0.2)
    allowed = (1 - confidence) * 100
    b.axhline(allowed, color=INK, lw=1.0, ls="--")
    b.text(1, allowed + 0.5, f"allowed: {allowed:.0f}%", fontsize=8.5, color=INK, va="bottom")
    if min_h is not None:
        b.axvline(min_h * 100, color=INK2, lw=0.9, ls=":")
        b.text(min_h * 100 + 1, b.get_ylim()[1] * 0.92, f"fixed contract: minimum {min_h * 100:.0f}%",
               fontsize=8.5, color=INK2, va="top")
    b.set_ylim(bottom=0)
    b.set_ylabel("P(margin < floor), %")
    b.set_xlabel("share of the USD bought today (the rest on the payment day), %")
    fig.tight_layout()
    return _save(fig, "fig2_price_of_insurance.png")


def fig_tornado(left: list, left_base, right: list, right_base, floor: float) -> str:
    """left/right: [(group label, [(variant label, value or None, is_base), ...])]."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(11.0, 5.2))

    def draw(ax, rows, base, top, unit, none_text):
        def value(x):
            return (top if x is None else x) * 100
        rows = sorted(rows, key=lambda r: max(value(x) for _, x, _ in r[1]) - min(value(x) for _, x, _ in r[1]))
        for y, (label, vals) in enumerate(rows):
            xs = [value(x) for _, x, _ in vals]
            lo, hi = min(xs), max(xs)
            ax.barh(y, hi - lo, left=lo, height=0.55, color=BLUE if ax is a else ORANGE, alpha=0.85, lw=0)
            if hi - lo < 1e-9:
                ax.text(hi + 0.8 * unit, y, "no change", ha="left", va="center", fontsize=8, color=INK2)
                continue
            for end, side, shift in ((lo, "right", -0.8), (hi, "left", 0.8)):
                at_end = [v for v, at in zip(vals, xs) if at == end]
                if not any(is_base for _, _, is_base in at_end):       # the base end is the vertical line
                    text = " & ".join(t + (none_text if x is None else "") for t, x, _ in at_end)
                    ax.text(end + shift * unit, y, text, ha=side, va="center", fontsize=8, color=INK2)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([r[0] for r in rows])
        ax.axvline(base * 100, color=INK, lw=1.0)
        ax.grid(axis="y", visible=False)

    draw(a, left, left_base, 1.0, 1.0, " (none)")
    a.set_xlim(-62, 122)
    a.set_xticks(range(0, 101, 20))
    a.set_xlabel("minimum share bought today, %")
    a.set_title(f"Minimum hedge ratio, fixed contract\n(line = {left_base * 100:.0f}%)", fontsize=10)
    draw(b, right, right_base, 0.0, 0.25, "")
    b.axvline(floor * 100, color=INK, lw=1.0, ls="--")
    b.set_xlim(-6, 33)
    b.set_xlabel("highest margin floor met at 95%, margin %")
    b.set_title(f"Highest floor any strategy meets, floating contract\n"
                f"(line = {right_base * 100:.1f}%, dashed = the {floor * 100:.0f}% floor)", fontsize=10)
    fig.suptitle("One change at a time: which assumption moves the answer", x=0.01, ha="left",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    return _save(fig, "fig3_tornado.png")


def fig_split(rows: list) -> str:
    fig, ax = plt.subplots(figsize=(9.5, 2.9))
    labels = [r[0] for r in rows][::-1]
    vals = [(r[1] - 1) * 100 for r in rows][::-1]
    colors = [r[2] for r in rows][::-1]
    ax.barh(labels, vals, color=colors, alpha=0.9, lw=0, height=0.6)
    for y, v in enumerate(vals):
        ax.text(v + 0.8, y, f"+{v:.0f}%", va="center", fontsize=9, color=INK)
    ax.set_xlim(0, max(vals) * 1.15)
    ax.set_xlabel("CVaR95 of the payment-day cost above the plan, % of plan")
    ax.set_title("Where the risk comes from: buying everything on the payment day", pad=10)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return _save(fig, "fig4_risk_split.png")


def fig_calibration(tab: pd.DataFrame, summary: pd.DataFrame) -> str:
    ok = tab[tab["usable"]]
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 6.6), sharex=True, sharey=True)
    for ax, (m, name, color) in zip(axes, (("fhs", "filtered historical simulation", BLUE),
                                           ("plain", "plain bootstrap", ORANGE))):
        ax.fill_between(ok.index, ok[f"{m}_q05"] * 100, ok[f"{m}_q95"] * 100, color=color, alpha=0.28, lw=0,
                        label="simulated 5th to 95th percentile")
        ax.plot(ok.index, ok["realized"] * 100, color=INK, lw=1.0, label="realised 13-week change")
        out = ok[ok[f"{m}_above"] | ok[f"{m}_below"]]
        ax.plot(out.index, out["realized"] * 100, ls="", marker="o", ms=3.2, color=color, label="outside the range")
        ax.axhline(0, color=INK2, lw=0.6)
        ax.set_title(f"{name}:  above the 95th percentile in {summary.loc['all', m + '_above'] * 100:.1f}% of "
                     f"starts,  below the 5th in {summary.loc['all', m + '_below'] * 100:.1f}%",
                     fontsize=9.5, fontweight="normal", color=INK, pad=4)
        ax.set_ylabel("13-week USD change, %")
    axes[0].legend(loc="lower right", ncol=3, fontsize=8)
    axes[-1].set_xlabel("start week")
    fig.suptitle("Calibration preview: realised USD change against the range simulated at each start",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout()
    return _save(fig, "fig5_calibration.png")


# --------------------------------------------------------------------- report
def run_tests() -> dict:
    out = subprocess.run([sys.executable, "-m", "pytest", "-q", str(paths.ROOT / "tests")],
                         capture_output=True, text=True, cwd=paths.ROOT).stdout
    line = out.strip().splitlines()[-1] if out.strip() else ""
    engine = subprocess.run([sys.executable, "-m", "pytest", "-q", str(paths.ROOT / "tests" / "test_engine.py")],
                            capture_output=True, text=True, cwd=paths.ROOT).stdout.strip().splitlines()[-1]
    def count(word, s):
        m = re.search(rf"(\d+) {word}", s)
        return int(m.group(1)) if m else 0
    return {"passed": count("passed", line), "failed": count("failed", line) + count("error", line),
            "engine_passed": count("passed", engine)}


def build() -> str:
    OUT.mkdir(parents=True, exist_ok=True)
    style(plt)
    wk, daily, imf = weekly.load(), panel.load(), fred.load()
    as_of = wk.index[-1]
    prem = I.premium_by_coffee(daily, imf, as_of, min_days=PREMIUM_MIN_DAYS)
    last_imf = prem["arabica"]["last"]
    july_robusta_days = int(daily.loc[(daily.index >= last_imf)
                                      & (daily.index < last_imf + pd.DateOffset(months=1)),
                                      "robusta_usd_kg"].notna().sum())

    def premium(w):
        return I.premium_share(daily, imf, as_of, w)

    base = Inputs.from_data(as_of, wk, daily, imf)
    fixed = base.with_(contract="fixed")
    run = Runner(as_of, wk)
    P, R = run(base)
    Pf, Rf = run(fixed)
    info = P.info
    tb, d = R.table, R.decision
    c_plan = R.c_plan
    g_T = float(P.growth[T])
    limit = (1 - base.margin_floor) / (1 - base.planned_margin)      # cost / plan where margin = floor
    assert not d.qualifies, "the text below is written for a base case in which no strategy qualifies"
    assert Rf.decision.qualifies and Rf.min_hedge is not None

    # ---- the D3 rule for the last week (before D-038): hide its returns and run again
    wk_d3 = wk.copy()
    wk_d3.loc[as_of, list(sim.RETURN_COLS.values())] = np.nan
    run_d3 = Runner(as_of, wk_d3)
    Po, Ro = run_d3(base)
    _, Rfo = run_d3(fixed)
    plain_pool_d3 = run_d3(base.with_(usd_vol="plain"))[0].info["pool_weeks"]

    # ---- simulated against historical 13-week changes
    hist = {k: np.expm1(wk.loc[wk.index <= as_of, col].rolling(T, min_periods=T).sum().dropna())
            for k, col in sim.RETURN_COLS.items()}
    Pp, Rp = run(base.with_(usd_vol="plain"))
    qs = [0.05, 0.25, 0.50, 0.75, 0.95]

    def qrow(label, x):
        x = np.asarray(x)
        return [label, pct(x.mean(), 1, True)] + [pct(v, 1, True) for v in np.quantile(x, qs)]

    q_rows = [qrow("دلار، شبیه‌سازی (فیلترشده، پایه)", P.usd[:, T] - 1),
              qrow("دلار، شبیه‌سازی (بوت‌استرپ ساده)", Pp.usd[:, T] - 1),
              qrow(f"دلار، تاریخ ({fa(len(hist['usd']))} پنجره)", hist["usd"]),
              qrow("عربیکا، شبیه‌سازی", P.arabica[:, T] - 1), qrow("عربیکا، تاریخ", hist["arabica"]),
              qrow("روبوستا، شبیه‌سازی", P.robusta[:, T] - 1), qrow("روبوستا، تاریخ", hist["robusta"])]
    sim_spread = np.quantile(P.usd[:, T], 0.95) - np.quantile(P.usd[:, T], 0.05)
    hist_spread = np.quantile(hist["usd"], 0.95) - np.quantile(hist["usd"], 0.05)
    r_usd = wk["usd_free_ret"]

    # ---- sensitivities
    G = groups(base, premium)
    Gf = [(k, f_, e_, [(a, b, i.with_(contract="fixed"), is_b) for a, b, i, is_b in vs])
          for k, f_, e_, vs in groups(base, premium) if k not in ("contract", "arabica_share")]

    def variants(gs, base_inputs):
        """(label, Inputs) of the base and of every one-at-a-time change."""
        return [("**پایه**", base_inputs)] + [(f"{name_fa}: {lab}", i) for _, name_fa, _, vs in gs
                                             for lab, _, i, is_b in vs if not is_b]

    def sens_rows(gs, base_inputs):
        rows = []
        for label, inputs in variants(gs, base_inputs):
            _, r = run(inputs)
            t, dd = r.table, r.decision
            chosen = strat_fa(dd.chosen) if dd.qualifies else \
                f"هیچ (نزدیک‌ترین: {strat_fa(dd.best_probability_strategy)}، {pct(dd.best_probability, 1)})"
            rows.append([label, share_fa(r.min_hedge), chosen, pct(dd.highest_floor, 1)]
                        + [pct(t.loc[s, "mean_cost_pct"], 1) for s in MAIN]
                        + [pct(t.loc[s, "cvar_pct"], 0) for s in MAIN])
        return rows

    sens_head = ["تغییر", "حداقل پوشش", "راهبرد منتخب", "بالاترین کفِ شدنی", "میانگین: امروز", "میانگین: پله‌ای",
                 "میانگین: روز پرداخت", "CVaR95: امروز", "CVaR95: پله‌ای", "CVaR95: روز پرداخت"]
    rows_float, rows_fixed = sens_rows(G, base), sens_rows(Gf, fixed)

    # ---- what the tie rule (D-037) changes: the same tables with the tolerance switched off
    tie_changes, tie_rows = [], 0
    for which, gs, b in (("شناور", G, base), ("ثابت", Gf, fixed)):
        for label, inputs in variants(gs, b):
            _, r = run(inputs)
            literal = dec.decide(r.table, inputs.with_(tie_tolerance=0.0), r.c_plan).chosen
            tie_rows += 1
            if literal != r.decision.chosen:
                tie_changes.append(f"{which}، {label.strip('*')}: از «{strat_fa(literal)}» به «{strat_fa(r.decision.chosen)}»")

    torn_left = [(e_, [(b, run(i)[1].min_hedge, is_b) for _, b, i, is_b in vs]) for _, _, e_, vs in Gf]
    torn_right = [(e_, [(b, run(i)[1].decision.highest_floor, is_b) for _, b, i, is_b in vs])
                  for k, _, e_, vs in G if k != "margin_floor"]

    def spans(gs, value, missing):
        out = []
        for _, name_fa, _, vs in gs:
            xs = [value(run(i)[1]) for _, _, i, _ in vs]
            xs = [missing if x is None else x for x in xs]
            out.append((max(xs) - min(xs), name_fa, min(xs), max(xs)))
        return sorted(out, reverse=True)

    span_fixed = spans(Gf, lambda r: r.min_hedge, 1.0)
    span_float = spans([g for g in G if g[0] != "margin_floor"], lambda r: r.decision.highest_floor, 0.0)

    # ---- pass-through, both modes (D-043)
    pt = {}
    for key, b in (("float", base), ("fixed", fixed)):
        for mode in ("up_only", "symmetric"):
            i = b.with_(pass_through=0.5, pass_through_mode=mode)
            p_, r_ = run(i)
            pt[key, mode] = {"res": r_, "max": max_hedge(r_), "safest": float(r_.curve["p_margin_below_floor"].idxmin()),
                             "rev": float(costs.revenue(p_, i).mean() / (r_.c_plan / (1 - i.planned_margin)))}
    up_float = pt["float", "up_only"]["res"]
    assert up_float.decision.qualifies, "the text below says up-only pass-through makes the floating case feasible"

    # ---- price of insurance
    spread = base.with_(funds_spread=0.10)
    _, Rs = run(spread)
    _, Rz = run(base.with_(drift="zero"))
    _, Rh3 = run(base.with_(fx_holding_cost=0.03))
    curve, curve_f = R.curve, Rf.curve
    ins_rows = []
    for h in (0.0, 0.25, 0.5, 0.75, 1.0):
        ins_rows.append([pct(h, 0), pct(curve.loc[h, "mean_cost_pct"], 2), pct(curve.loc[h, "cvar_pct"], 0),
                         pct(curve.loc[h, "p_margin_below_floor"], 1), pct(curve_f.loc[h, "mean_cost_pct"], 2),
                         pct(curve_f.loc[h, "cvar_pct"], 0), pct(curve_f.loc[h, "p_margin_below_floor"], 1)])
    price = {k: r.curve.loc[1.0, "mean_cost_pct"] - r.curve.loc[0.0, "mean_cost_pct"]
             for k, r in (("base", R), ("fixed", Rf), ("spread", Rs), ("hold3", Rh3), ("zero", Rz))}

    # ---- risk split (payment day)
    late = I.payment_day()
    split = {}
    for key, pth, inputs in (("coffee_frozen", P, fixed), ("usd_frozen", sim.frozen_usd(P), base),
                             ("both", P, base)):
        c = costs.cost_on_paths(pth, late, inputs) / c_plan
        split[key] = {"mean": float(c.mean()), "sd": float(c.std(ddof=1)),
                      "var": float(np.quantile(c, base.confidence)), "cvar": dec.tail_mean(c, base.confidence)}
    now_cost = costs.cost_on_paths(P, I.now(), base)
    late_cost = costs.cost_on_paths(P, late, base)
    excess_share = float((costs.usd_needed(P, base) < base.usd_amount).mean())
    corr_need = float(np.corrcoef(P.usd[:, T], costs.usd_needed(P, base))[0, 1])
    friction = float((now_cost - late_cost).mean() / c_plan)

    # ---- calibration preview (D3b)
    tab = cal.usd_exceedance(wk, N_CALIBRATION, seed=SEED)
    summ = cal.summarize(tab)
    okc = tab[tab["usable"]]
    n_ind = int(summ.loc["all", "independent"])
    bin_lo, bin_hi = cal.binomial_range(n_ind)
    rates = {(m, side): float(summ.loc["all", f"{m}_{side}"]) for m in ("fhs", "plain") for side in ("above", "below")}
    inside = sum(bin_lo <= x <= bin_hi for x in rates.values())
    alt = {dr: cal.summarize(cal.usd_exceedance(wk, N_CALIBRATION, seed=SEED, drift=dr)).loc["all"]
           for dr in ("historical", "zero")}
    band = {m: okc[f"{m}_q95"] - okc[f"{m}_q05"] for m in ("fhs", "plain")}
    fhs_narrower = float((band["fhs"] < band["plain"]).mean())
    today_q = {"fhs": np.quantile(P.usd[:, T] - 1, [0.05, 0.95]), "plain": np.quantile(Pp.usd[:, T] - 1, [0.05, 0.95])}
    years = summ.drop(index="all")
    first_year = int(years.index[0])
    calm = (okc["jalali_year"] == first_year).values
    calm_target = float(np.median([(1 + I.market_rate(wk, t)) ** (T / I.WEEKS_PER_YEAR) - 1
                                   for t in okc.index[calm]]))
    assert okc.loc[calm, "realized"].median() < calm_target
    recent_years = [int(y) for y in years.index[-2:]]
    recent = okc[okc["jalali_year"].isin(recent_years)]

    def hot(col, cut=0.15):
        """Jalali years in which a rate is at least three times the nominal 5%."""
        y = years[years[col] >= cut]
        return "، ".join(f"{fa(k)} ({pct(v, 0)})" for k, v in y[col].items()) or "هیچ سالی"

    # ---- figures
    cost_pct = {s.name: costs.cost_on_paths(P, s, base) / c_plan for s in (I.now(), I.staged(3, T), late)}
    f1 = fig_costs(cost_pct, tb, limit)
    f2 = fig_insurance([("floating contract (base)", BLUE, curve), ("fixed contract", ORANGE, curve_f),
                        ("floating, funds rate = market + 10 pp", AQUA, Rs.curve)], base.confidence, Rf.min_hedge)
    f3 = fig_tornado(torn_left, Rf.min_hedge, torn_right, d.highest_floor, base.margin_floor)
    f4 = fig_split([("coffee frozen (fixed contract): USD risk only", split["coffee_frozen"]["cvar"], ORANGE),
                    ("USD frozen at its expected path: coffee risk only", split["usd_frozen"]["cvar"], BLUE),
                    ("both random (base case)", split["both"]["cvar"], INK2)])
    f5 = fig_calibration(tab, summ)

    # ---- export, tests, backtest start
    ex = exporter.export_paths(as_of, N_EXPORT, seed=SEED, options=base, weekly=wk, daily=daily, imf=imf)
    tests = run_tests()
    d015 = pd.Timestamp(eda.BACKTEST_START)
    weeks_lost = int(((wk.index >= d015) & (wk.index < cal.BACKTEST_START)).sum())
    band_fixed = hedge_band(Rf)

    # ======================================================================
    L: list[str] = []
    add = L.append
    today = dt.date.today()
    add("# گزارش مرحلهٔ D3 — موتور\n")
    add(f"تاریخ ساخت: {jg(today)}. همهٔ عددها، جدول‌ها و شکل‌ها را `scripts/d3_report.py` ساخته است: "
        f"{num(N_PATHS)} مسیر، بذر ثابت `{SEED}`، تاریخ تصمیم = آخرین هفتهٔ پنل، {jg(as_of)}.\n")

    add("## آنچه در D3b عوض شد\n")
    add("این گزارش پس از بازبینی D3 دوباره ساخته شده است. تصمیم‌های بازبینی (D-036 تا D-043) در `DECISIONS.md` "
        "ثبت شده‌اند و در بخش ۷ کنار هر بند آمده‌اند. شش تغییر:\n")
    add(f"1. **هفتهٔ منتهی به تاریخ تصمیم حالا در مخزن و در برآورد نوسان است (D-038).** مخزنِ فیلترشده از "
        f"{fa(Po.info['pool_weeks'])} به {fa(info['pool_weeks'])} هفته رسید و نوسان هفتگی دلار از "
        f"{pct(Po.info['sigma_as_of'], 2)} به {pct(info['sigma_as_of'], 2)}. اثر روی عددهای اصلی کوچک است: احتمال "
        f"حاشیهٔ زیر کف برای «همه امروز» از {pct(Ro.table.loc['now', 'p_margin_below_floor'], 1)} به "
        f"{pct(tb.loc['now', 'p_margin_below_floor'], 1)}، بالاترین کفِ شدنی از {pct(Ro.decision.highest_floor, 1)} به "
        f"{pct(d.highest_floor, 1)}، و حداقل نسبت پوشش با قرارداد ثابت از {share_fa(Rfo.min_hedge)} به "
        f"{share_fa(Rf.min_hedge)}. شبیه‌ساز همچنان فقط گذشته را می‌بیند: هیچ هفته‌ای پس از تاریخ تصمیم.")
    add(f"2. **قاعدهٔ تساوی (D-037).** اختلاف میانگین هزینهٔ کمتر از {pct(base.tie_tolerance, 2)} برنامه تساوی است و "
        "CVaR95 کمتر برنده می‌شود. خروجی اصلی «حداقل نسبت پوشش» است و «راهبرد منتخب» کنارش می‌آید؛ ستون‌های "
        f"جدول‌ها به همین ترتیب شده‌اند. در {fa(len(tie_changes))} ردیف از {fa(tie_rows)} ردیفِ دو جدول حساسیت، "
        "راهبرد منتخب عوض شد: " + "؛ ".join(tie_changes) + ".")
    add(f"3. **حساسیت نرخ تأمین مالی = نرخ بازار + ۱۰ واحد درصد (D-039)**، به‌جای سطح ثابت ٪۴۰. امروز یعنی "
        f"{pct(spread.cost_of_funds, 1)} در سال.")
    add("4. **انتقال به قیمت فروش دو حالت دارد (D-043):** «فقط بالا» (قیمت فروش با جهشِ نامنتظر دلار بالا می‌رود "
        "ولی با اُفت دلار پایین نمی‌آید) و «متقارن». هر جا انتقال بزرگ‌تر از صفر باشد پیش‌فرض «فقط بالا» است. هر "
        f"دو ردیف در جدول‌های بخش ۳ آمده‌اند. یافتهٔ تازه: با انتقال ۰.۵ِ «فقط بالا»، قرارداد شناور هم قید را برآورده "
        f"می‌کند (حداقل نسبت پوشش {share_fa(up_float.min_hedge)}).")
    add("5. **بخش تازه: پیش‌نمایش کالیبراسیون (بخش ۵)**، با شکل ۵. جمع‌بندی به بخش ۶ رفت و «خروجی ماشین‌حساب و "
        "آزمون‌ها» پیوست شد. شمارهٔ بندهای بخش ۷ همان است که بود.")
    add(f"6. **بک‌تست از {jd(cal.BACKTEST_START)} شروع می‌شود (D-041).** پیش‌نمایش کالیبراسیون از همین تاریخ است. "
        "گزارش D2 (بازبینی‌شده) دست نخورده و نمونهٔ «بک‌تست» آن همان تعریف D-015 را دارد.\n")

    add("**این مرحله چه ساخت.** یک موتور آزموده در `src/coffee_risk/engine/` که برای یک تاریخ تصمیم و یک "
        "دسته ورودی: (۱) مسیرهای مشترک ۱۳ هفته‌ای دلار، عربیکا و روبوستا را شبیه‌سازی می‌کند؛ (۲) هزینهٔ ریالی هر "
        "راهبرد خرید ارز را روی هر مسیر حساب می‌کند؛ (۳) معیارهای ریسک را می‌دهد؛ (۴) قاعدهٔ تصمیم D-035 را اجرا "
        "می‌کند؛ (۵) فایل مسیرهای ماشین‌حساب را می‌نویسد. رابط ماشین‌حساب، بک‌تست غلتان (D5) و اجرای تنش (D4) "
        "ساخته نشده‌اند؛ تابع‌های هزینه و تصمیم هر «مسیر»ی را می‌پذیرند (شبیه‌سازی‌شده، یک مسیر تنش یا یک مسیر "
        "واقعی تاریخی)، پس هر دو مرحله همین تابع‌ها را به کار می‌برند.\n")
    add("**همهٔ پیش‌فرض‌ها موقت‌اند.** فقط سه ورودی از داده می‌آید (نرخ دلار، نرخ بازار، پرمیوم). بقیه فرض "
        "کاریِ نویسنده است و در جدول بخش ۱-۱ با «فرض موقت» مشخص شده است.\n")

    add("## نتیجه در چند خط\n")
    add(f"1. **در حالت پایه (قرارداد شناور) حداقل نسبت پوشش وجود ندارد: هیچ راهبردی قید را برآورده نمی‌کند.** قید: "
        f"حاشیهٔ سود با احتمال {pct(base.confidence, 0)} زیر {pct(base.margin_floor, 0)} نیفتد. بهترین راهبرد "
        f"(«{strat_fa(d.best_probability_strategy)}») به {pct(d.best_probability, 1)} می‌رسد. بالاترین کفی که در "
        f"{pct(base.confidence, 0)} شدنی است {pct(d.highest_floor, 1)} است.")
    add("2. **دلیلش قیمت قهوه است، نه دلار.** با قرارداد شناور، حتی وقتی همهٔ دلار امروز خریده شود، مقدار دلارِ "
        "لازم با قیمت قهوه بالا و پایین می‌رود. زمان خرید ارز این ریسک را برنمی‌دارد.")
    add(f"3. **اگر قیمت قهوه امروز ثابت شود، حداقل نسبت پوشش {share_fa(Rf.min_hedge)} است:** خرید دست‌کم همین "
        "سهم از دلار در همین امروز قید را برآورده می‌کند.")
    add(f"4. **انتقالِ «فقط بالا» هم قید را شدنی می‌کند، با حاشیهٔ کم.** با انتقال ۰.۵ و قرارداد شناور حداقل نسبت "
        f"پوشش {share_fa(up_float.min_hedge)} است؛ احتمالِ «همه امروز» "
        f"{pct(1 - up_float.table.loc['now', 'p_margin_below_floor'], 1)} است، فقط کمی بالای {pct(base.confidence, 0)}. "
        "با انتقال متقارن شدنی نیست.")
    add(f"5. **بیمه در حالت پایه تقریباً رایگان است، و این از فرض می‌آید.** رانش خنثی یعنی میانگین هزینهٔ "
        f"«امروز» و «روز پرداخت» برابر است. اختلاف میانگین هزینهٔ این دو {pct(friction, 2)} برنامه است.")
    add(f"6. **پیش‌نمایش کالیبراسیون دو روش نوسان را از هم جدا نمی‌کند.** در {fa(len(okc))} شروع هفتگی، دلارِ واقعی در "
        f"{pct(rates['fhs', 'above'], 1)} موارد از صدک ۹۵ِ روش فیلترشده بالاتر رفت و در {pct(rates['plain', 'above'], 1)} "
        f"موارد از صدک ۹۵ِ بوت‌استرپ ساده. هر دو حدود دو برابر ٪۵ هستند؛ با فقط {fa(n_ind)} پنجرهٔ مستقل این هنوز "
        "در دامنهٔ یک شبیه‌سازِ درست است (بخش ۵).")
    add("7. **@N_OPEN@ تصمیم تازه لازم است** (بندهای آخر بخش ۷).\n")

    # ------------------------------------------------------------------ 1
    add("## ۱. حالت پایه\n")
    add("### ۱-۱. ورودی‌ها\n")
    src_data, src_ph = "از داده", "فرض موقت"
    add(table(["ورودی", "مقدار", "منبع"], [
        ["نرخ دلار در تاریخ تصمیم (s0)", f"{num(base.s0)} ریال", f"{src_data}: فروش Bonbast در {jd(as_of)}"],
        ["نرخ بازار، سالانه (market_rate)", pct(base.market_rate, 1),
         f"{src_data}: بازده ۱۳ هفتهٔ اخیر صندوق، سالانه‌شده"],
        ["سهم پرمیوم فیزیکی از فاکتور (premium_share)", pct(base.premium_share, 1),
         f"{src_data}: ۱ − میانهٔ نسبت آتی به شاخص IMF در ۳۶ ماه (عربیکا {pct(prem['arabica']['premium'], 1)}، "
         f"روبوستا {pct(prem['robusta']['premium'], 1)})، با وزن ترکیب"],
        ["مبلغ فاکتور (usd_amount)", f"{num(base.usd_amount)} دلار", src_ph],
        ["هفتهٔ پرداخت (T)", fa(base.payment_week), src_ph],
        ["تخفیف کانال رسمی (channel_discount)", pct(base.channel_discount, 0), src_ph],
        ["قرارداد قهوه (contract)", "شناور", f"{src_ph}؛ D-036"],
        ["سهم عربیکا (w)", num(base.arabica_share, 1), src_ph],
        ["نرخ تأمین مالی بنگاه، سالانه (funds_rate)", f"برابر نرخ بازار ({pct(base.cost_of_funds, 1)})", src_ph],
        ["هزینهٔ نگهداری ارز، سالانه (fx_holding_cost)", pct(base.fx_holding_cost, 0), src_ph],
        ["اختلاف بازفروش دلار اضافه (sellback_spread)", pct(base.sellback_spread, 0), src_ph],
        ["حاشیهٔ سود برنامه (planned_margin)", pct(base.planned_margin, 0), src_ph],
        ["کف حاشیهٔ سود (margin_floor)", pct(base.margin_floor, 0), src_ph],
        ["اطمینان (confidence)", pct(base.confidence, 0), src_ph],
        ["انتقال به قیمت فروش (pass_through)", f"{num(base.pass_through, 1)}؛ اگر بزرگ‌تر از صفر شود: «فقط بالا»",
         f"{src_ph}؛ D-043"],
        ["نااطمینانی فروش (demand_sd)", pct(base.demand_sd, 0), src_ph],
        ["بودجه (budget)", f"{num(base.budget_factor, 2)} × هزینهٔ برنامه", src_ph],
        ["آستانهٔ تساوی میانگین هزینه (tie_tolerance)", f"{pct(base.tie_tolerance, 2)} برنامه", "D-037"],
        ["رانش دلار (drift)", "خنثی", "D-028"],
        ["نوسان دلار", f"شبیه‌سازی تاریخی فیلترشده، نیمه‌عمر {fa(int(base.half_life))} هفته", "D-026"],
        ["میانگین طول بلوک", f"{fa(int(base.mean_block))} هفته", "D-025"],
    ]))
    add("")
    add(f"* **هزینهٔ برنامه** = مبلغ فاکتور × نرخ امروز × رشد مورد انتظار دلار تا هفتهٔ پرداخت "
        f"({pct(g_T - 1, 1, True)}) = **{bn(c_plan)} میلیارد ریال**. همهٔ هزینه‌ها در این گزارش درصدی از همین عددند "
        f"و به ارزش هفتهٔ پرداخت.")
    add(f"* **درآمد برنامه** = هزینهٔ برنامه ÷ (۱ − حاشیهٔ سود برنامه) = {bn(c_plan / (1 - base.planned_margin))} میلیارد ریال. "
        f"با انتقال صفر و بدون نااطمینانی فروش، درآمد ثابت است؛ پس «حاشیهٔ سود زیر {pct(base.margin_floor, 0)}» "
        f"دقیقاً یعنی «هزینه بالای {pct(limit, 0)} برنامه».")
    add(f"* **پرمیوم** از ماه‌هایی حساب شده که در تاریخ تصمیم منتشر شده بودند: IMF با تأخیر منتشر می‌کند و موتور "
        f"هر ماه را {fa(I.IMF_LAG_MONTHS)} ماه بعد «دانسته» می‌گیرد (D-042). آخرین ماه استفاده‌شده: عربیکا "
        f"{prem['arabica']['last'].strftime('%Y-%m')}، روبوستا {prem['robusta']['last'].strftime('%Y-%m')} "
        f"(ماهی که کمتر از {fa(PREMIUM_MIN_DAYS)} روز دادهٔ آتی دارد کنار می‌رود؛ 2026-07 روبوستا "
        f"{fa(july_robusta_days)} روز دارد).")
    add("* **محدودیت کانال رسمی:** کانال رسمی به شکل «نرخ آزاد × (۱ − تخفیف)» با همان رفتار نرخ آزاد مدل شده "
        "است. نرخی که اداری و پله‌ای حرکت کند مدل نشده است.\n")

    add("### ۱-۲. شبیه‌ساز چه ساخت\n")
    add("چند اصطلاح، هر کدام یک بار:\n")
    add("* **بوت‌استرپ ایستا:** هر مسیر از یک هفتهٔ تصادفیِ تاریخ شروع می‌شود و هر هفته یا با هفتهٔ بعدیِ تاریخ "
        "ادامه می‌دهد یا (با احتمال یک‌سیزدهم) به هفتهٔ تصادفی دیگری می‌پرد. هر سه سری از همان هفته‌ها برداشته "
        "می‌شوند، پس هم‌حرکتی‌شان حفظ می‌شود.")
    add("* **نوسان EWMA:** برآورد نوسانِ هر هفته از هفته‌های پیش از آن، با وزنی که هر ۱۳ هفته نصف می‌شود. نوسانِ "
        "«امروز» از همهٔ هفته‌ها تا خودِ تاریخ تصمیم برآورد می‌شود (D-038).")
    add("* **بازدهٔ استانداردشده:** بازدهٔ دلار در یک هفته تقسیم بر نوسانِ برآوردشده برای همان هفته. شبیه‌ساز "
        "این عددها را نمونه می‌گیرد و در نوسانِ امروز ضرب می‌کند: شکل از تاریخ، اندازه از امروز.")
    add("* **کف نوسان:** اگر برآورد نوسانِ یک هفته از صدک ۵ برآوردهای تا آن روز کمتر باشد، همان صدک به کار "
        "می‌رود تا هفته‌های تقریباً ساکن عدد استانداردشدهٔ بزرگ نسازند.")
    add("* **تنظیم رانش:** سطح‌های هر هفته در یک عدد ضرب می‌شوند تا میانگینِ نسبت قیمت روی همهٔ مسیرها دقیقاً "
        "برابر هدف شود (برای دلار: رشد ریال با نرخ بازار؛ برای هر قهوه: ۱).\n")
    add(table(["مورد", "مقدار"], [
        ["مخزن، دلارِ فیلترشده", f"{fa(info['pool_weeks'])} هفته، از {jd(info['pool_first'])} تا {jd(info['pool_last'])}"],
        ["مخزن، بوت‌استرپ ساده", f"{fa(Pp.info['pool_weeks'])} هفته ({fa(sim.MIN_HISTORY)} هفتهٔ اول برآورد نوسان ندارند)"],
        ["آخرین هفته‌ای که در برآورد نوسان آمده", jg(info["last_week_in_vol"])],
        ["نوسان هفتگی دلار در تاریخ تصمیم", f"{pct(info['sigma_as_of'], 2)} (سالانه {pct(info['sigma_as_of'] * ANN, 0)})"],
        ["میانهٔ همین برآورد در مخزن", f"{pct(info['sigma_pool_median'], 2)} (سالانه {pct(info['sigma_pool_median'] * ANN, 0)})"],
        ["کف نوسان در تاریخ تصمیم", pct(info["sigma_floor_as_of"], 2)],
        ["هفته‌هایی که به کف خوردند", f"{fa(info['floor_hits'])} از {fa(info['pool_weeks'])}، همه بین "
                                     f"{jd(info['floor_hit_first'])} و {jd(info['floor_hit_last'])}"],
        ["میانگین بازده‌های استانداردشده (کم شد)", num(info["z_mean"], 2)],
        ["انحراف معیار بازده‌های استانداردشده", num(info["z_std"], 2)],
        ["بزرگ‌ترین قدرمطلق بازدهٔ استانداردشده", num(info["z_abs_max"], 1)],
        ["انحراف معیار هفتگی دلارِ نمونه‌گیری‌شده", f"{pct(info['sample_usd_std'], 2)} (بوت‌استرپ ساده: {pct(Pp.info['pool_usd_std'], 2)})"],
        ["رشد هدف دلار تا هفتهٔ ۱۳ (خنثی)", pct(g_T - 1, 2, True)],
    ]))
    add("")
    add("آیا مسیرها به تاریخ شبیه‌اند؟ تغییر ۱۳ هفته‌ای، درصد ساده:\n")
    add(table(["سری", "میانگین", "صدک ۵", "صدک ۲۵", "میانه", "صدک ۷۵", "صدک ۹۵"], q_rows))
    add("")
    add(f"* فاصلهٔ صدک ۵ تا ۹۵ دلارِ شبیه‌سازی‌شده {num(sim_spread / hist_spread, 1)} برابر تاریخ است، چون نوسانِ امروز "
        f"{num(info['sigma_as_of'] / info['sigma_pool_median'], 1)} برابر میانهٔ مخزن است. دُم پایین هم پهن شده است: "
        "شبیه‌ساز اُفت بزرگ دلار را محتمل‌تر از تاریخ می‌بیند. اینکه این پهنا درست است یا نه، پرسش بخش ۵ است.")
    add("* میانگین‌ها با تاریخ فرق دارند و باید داشته باشند: رانش را D-028 و D-029 تعیین می‌کنند، نه تاریخ.")
    add("* هفته‌های «کف» همه در سال‌های آرامِ پس از بحران ۱۳۹۱ هستند، وقتی نوسان مدام رکورد پایین تازه می‌زد.\n")

    add("### ۱-۳. جدول راهبردها\n")
    add("* **VaR95:** هزینه‌ای که در ٪۹۵ مسیرها از آن بیشتر نمی‌شود. **CVaR95:** میانگین هزینه در بدترین ٪۵ مسیرها.")
    add("* **خطای شبیه‌سازی:** خطای معیار مونت‌کارلو؛ یعنی عدد با بذر دیگر حدوداً چقدر جابه‌جا می‌شود.")
    add(f"* هزینه‌ها درصدی از هزینهٔ برنامه‌اند. بودجه = {pct(base.budget_factor, 0)} برنامه.\n")
    rows = []
    for name in tb.index:
        r = tb.loc[name]
        rows.append([strat_fa(name), pct(r["mean_cost_pct"], 1), pct(r["median_cost_pct"], 1), pct(r["var_pct"], 0),
                     pct(r["cvar_pct"], 0), pct(r["p_over_budget"], 1), pct(r["margin_q"], 1),
                     pct(r["p_margin_below_floor"], 1), pct(r["se_mean_cost_pct"], 2),
                     pct(r["se_p_margin_below_floor"], 2)])
    add(table(["راهبرد", "میانگین هزینه", "میانه", "VaR95", "CVaR95", "احتمال هزینه بالای بودجه",
               "صدک ۵ حاشیهٔ سود", "احتمال حاشیهٔ زیر کف", "خطای شبیه‌سازیِ میانگین", "خطای شبیه‌سازیِ احتمال"], rows))
    add("")
    add("سه راهبرد اصلی به میلیارد ریال:\n")
    add(table(["راهبرد", "میانگین", "میانه", "VaR95", "CVaR95"],
              [[strat_fa(s), bn(tb.loc[s, "mean_cost"]), bn(tb.loc[s, "median_cost"]), bn(tb.loc[s, "var"]),
                bn(tb.loc[s, "cvar"])] for s in MAIN]))
    add("")
    add(f"![توزیع هزینه]({f1})\n")
    add("*شکل ۱ — توزیع هزینهٔ ریالی سه راهبرد اصلی در حالت پایه، درصدی از هزینهٔ برنامه. خط پر: برنامه. "
        "خط‌چین: هزینه‌ای که حاشیهٔ سود را به کف می‌رساند. نقطه‌چین: CVaR95 همان راهبرد.*\n")

    add("### ۱-۴. قاعدهٔ تصمیم\n")
    add(f"قاعده (D-035 و D-037): راهبردهایی بمانند که احتمال «حاشیهٔ سود ≥ کف» در آن‌ها دست‌کم {pct(base.confidence, 0)} "
        f"است؛ از میان آن‌ها کمترین میانگین هزینه؛ اختلافِ کمتر از {pct(base.tie_tolerance, 2)} برنامه تساوی است و در "
        "تساوی CVaR95 کمتر برنده می‌شود. خروجی اصلی «حداقل نسبت پوشش» است: کوچک‌ترین سهمی از دلار که اگر امروز "
        "خریده شود (و بقیه روز پرداخت) قید برآورده می‌شود. روی شبکهٔ یک‌درصدی جست‌وجو شده است.\n")
    add(table(["مورد", "قرارداد شناور (پایه)", "قرارداد ثابت"], [
        ["**حداقل نسبت پوشش**", "هیچ؛ حتی ٪۱۰۰ کافی نیست", f"**{share_fa(Rf.min_hedge)}**"],
        ["راهبرد منتخب", "هیچ", strat_fa(Rf.decision.chosen)],
        ["بیشترین احتمالِ رسیدن به کف", f"{pct(d.best_probability, 1)} («{strat_fa(d.best_probability_strategy)}»)",
         f"{pct(Rf.decision.best_probability, 1)} («{strat_fa(Rf.decision.best_probability_strategy)}»)"],
        [f"بالاترین کفِ شدنی در اطمینان {pct(base.confidence, 0)}",
         f"{pct(d.highest_floor, 1)} («{strat_fa(d.highest_floor_strategy)}»)",
         f"{pct(Rf.decision.highest_floor, 1)} («{strat_fa(Rf.decision.highest_floor_strategy)}»)"],
    ]))
    add("")
    add(f"خطای شبیه‌سازی: اگر احتمال‌ها دو خطای معیار پایین یا بالا بروند، حداقل پوششِ قرارداد ثابت "
        f"{share_fa(band_fixed[0])} یا {share_fa(band_fixed[1])} می‌شود. در حالت شناور نتیجه عوض نمی‌شود: احتمال «همه "
        f"امروز» {pct(1 - tb.loc['now', 'p_margin_below_floor'], 1)} است با خطای معیار "
        f"{pct(tb.loc['now', 'se_p_margin_below_floor'], 2)}.\n")
    add("**به زبان ساده**\n")
    add(f"* برنامه این است که فاکتور {bn(c_plan)} میلیارد ریال تمام شود. کف {pct(base.margin_floor, 0)} یعنی هزینه نباید "
        f"از {pct(limit, 0)} برنامه ({bn(limit * c_plan)} میلیارد ریال) بگذرد، و این فقط در {pct(1 - base.confidence, 0)} "
        "حالت‌ها مجاز است.")
    add(f"* اگر خرید به روز پرداخت بماند، در {pct(tb.loc['payment_day', 'p_margin_below_floor'], 0)} حالت‌ها هزینه از این "
        f"حد می‌گذرد، و در بدترین ٪۵ حالت‌ها هزینه به‌طور میانگین {pct(tb.loc['payment_day', 'cvar_pct'], 0)} برنامه است.")
    add(f"* اگر همهٔ دلار امروز خریده شود، این احتمال به {pct(tb.loc['now', 'p_margin_below_floor'], 1)} می‌رسد: بهتر، ولی "
        f"هنوز بالای {pct(1 - base.confidence, 0)}. چرا؟ چون فاکتور با قیمت قهوه در روز پرداخت بسته می‌شود. اگر قهوه گران "
        "شود، دلارِ بیشتری لازم است و باید با نرخ همان روز خریده شود.")
    add(f"* خرید سه‌پله‌ای بین این دو است ({pct(tb.loc['staged_3', 'p_margin_below_floor'], 1)}).")
    add(f"* میانگین هزینهٔ همهٔ راهبردها تقریباً یکی است ({pct(tb['mean_cost_pct'].min(), 2)} تا "
        f"{pct(tb['mean_cost_pct'].max(), 2)} برنامه). کمی زیر ٪۱۰۰ است چون در مسیرها دلار و قهوه اندکی خلاف هم "
        f"حرکت می‌کنند (همبستگی نرخ دلار هفتهٔ ۱۳ با مقدار دلارِ لازم: {num(corr_need, 2)}). راهبردها در ریسک "
        "فرق دارند، نه در میانگین.\n")

    # ------------------------------------------------------------------ 2
    add("## ۲. بهای بیمه\n")
    add("برای هر سهم از دلار که امروز خریده شود (و بقیه روز پرداخت): میانگین هزینه و احتمال افتادن حاشیهٔ سود "
        "زیر کف.\n")
    add(f"![بهای بیمه]({f2})\n")
    add(f"*شکل ۲ — بالا: میانگین هزینه (درصد برنامه). پایین: احتمال حاشیهٔ سود زیر {pct(base.margin_floor, 0)}. محور "
        f"افقی: سهم خرید امروز. خط‌چین: حد مجاز ({pct(1 - base.confidence, 0)}). سه خط: حالت پایه (قرارداد شناور)، قرارداد "
        "ثابت، و حالت پایه با نرخ تأمین مالیِ «نرخ بازار + ۱۰ واحد درصد».*\n")
    add(table(["سهم خرید امروز", "شناور: میانگین هزینه", "شناور: CVaR95", "شناور: احتمال زیر کف",
               "ثابت: میانگین هزینه", "ثابت: CVaR95", "ثابت: احتمال زیر کف"], ins_rows))
    add("")
    add(f"* **آنچه بیمه می‌خرد:** با قرارداد شناور، خرید ٪۱۰۰ امروز احتمال زیر کف را از "
        f"{pct(curve.loc[0.0, 'p_margin_below_floor'], 1)} به {pct(curve.loc[1.0, 'p_margin_below_floor'], 1)} می‌رساند و به "
        f"{pct(1 - base.confidence, 0)} نمی‌رسد. با قرارداد ثابت از {pct(curve_f.loc[0.0, 'p_margin_below_floor'], 1)} به صفر "
        f"می‌رسد و از سهم {share_fa(Rf.min_hedge)} به بالا زیر حد مجاز است.")
    add(f"* **آنچه بیمه می‌ارزد:** در حالت پایه تقریباً هیچ. تفاوت میانگین هزینهٔ ٪۱۰۰ با ٪۰ برابر {pct(price['base'], 2, True)} "
        f"برنامه است. این همان اختلاف بازفروش است: در {pct(excess_share, 0)} مسیرها قهوه ارزان می‌شود، دلار اضافه می‌ماند "
        f"و با {pct(base.sellback_spread, 0)} تخفیف فروخته می‌شود.")
    add("* **این «رایگان بودن» نتیجه نیست، فرض است.** رانش خنثی (D-028) طوری ساخته شده که میانگین هزینهٔ زود "
        "خریدن و دیر خریدن برابر شود. بها وقتی پیدا می‌شود که یکی از این‌ها عوض شود:\n")
    add(table(["تغییر نسبت به پایه", "بهای پوشش کامل (میانگین هزینهٔ ٪۱۰۰ منهای ٪۰)"], [
        ["پایه", pct(price["base"], 2, True)],
        [f"نرخ تأمین مالی = نرخ بازار + ۱۰ واحد درصد ({pct(spread.cost_of_funds, 1)})", pct(price["spread"], 2, True)],
        ["هزینهٔ نگهداری ارز ٪۳ در سال", pct(price["hold3"], 2, True)],
        ["رانش صفر (دلار به‌طور میانگین رشد نکند)", pct(price["zero"], 2, True)],
    ]))
    add("")
    add("واحد: درصدی از هزینهٔ برنامهٔ همان ردیف.\n")

    # ------------------------------------------------------------------ 3
    add("## ۳. حساسیت\n")
    add("هر ردیف یک تغییر نسبت به حالت پایه است؛ بقیهٔ ورودی‌ها همان‌اند. همهٔ ردیف‌ها با همان بذر اجرا شده‌اند. "
        "هزینه‌ها درصدی از هزینهٔ برنامهٔ همان ردیف‌اند. «بالاترین کفِ شدنی» بالاترین کف حاشیهٔ سود است که دست‌کم "
        f"یک راهبرد در اطمینان {pct(base.confidence, 0)} برآورده می‌کند. برای ردیف‌های سهم عربیکا، پرمیوم پیش‌فرض از نو "
        "حساب شده است.\n")
    add("**۳-۱. پایه = قرارداد شناور (D-036)**\n")
    add(table(sens_head, rows_float))
    add("")
    plans = {k: run(base.with_(drift=k))[1].c_plan for k in ("neutral", "historical", "zero")}
    feasible = [f"«{name_fa}: {lab}»" for _, name_fa, _, vs in G for lab, _, i, is_b in vs
                if not is_b and run(i)[1].decision.qualifies]
    add(f"* از {fa(len(rows_float) - 1)} تغییر، {fa(len(feasible))} تا قید را شدنی می‌کنند: {'، '.join(feasible)}.")
    add(f"* هزینهٔ برنامه با رانش عوض می‌شود: خنثی {bn(plans['neutral'])}، میانگین تاریخی {bn(plans['historical'])}، "
        f"صفر {bn(plans['zero'])} میلیارد ریال. پس در ردیف «رانش صفر» عدد «میانگین: امروز» بالای ٪۱۰۰ است: دلار "
        "قرار نیست رشد کند ولی پولِ خرید زودهنگام هزینهٔ تأمین مالی دارد.")
    add("* «راهبرد منتخب» از میان ۱۲ راهبرد است (امروز، سه‌پله‌ای، روز پرداخت، و سهم‌های ٪۱۰ تا ٪۹۰). «حداقل "
        "پوشش» روی شبکهٔ یک‌درصدی است. این دو یکی نیستند: حداقل پوشش می‌گوید چه مقدار لازم است؛ راهبرد منتخب "
        f"می‌گوید از میان راهبردهای مجازی که هزینه‌شان در حد {pct(base.tie_tolerance, 2)} برنامه یکی است، کدام کم‌ریسک‌تر "
        "است.\n")
    add("**۳-۲. همان تغییرها، با پایه = قرارداد ثابت**\n")
    add("سهم عربیکا در این جدول نیست: با قرارداد ثابت هزینه به قیمت قهوه بستگی ندارد.\n")
    add(table(sens_head, rows_fixed))
    add("")
    add("* با قرارداد ثابت و رانش خنثی میانگین هزینهٔ همهٔ راهبردها دقیقاً برابر است؛ پس CVaR95 کمتر تصمیم می‌گیرد و "
        "«همه امروز» منتخب می‌شود، در حالی که حداقلِ لازم کمتر است. وقتی زود خریدن هزینه دارد (نرخ تأمین مالیِ "
        "بالاتر، هزینهٔ نگهداری، رانش صفر) راهبرد منتخب به سمت حداقل پوشش برمی‌گردد.\n")
    add("**۳-۳. انتقال به قیمت فروش: دو حالت (D-043)**\n")
    add("انتقال ۰.۵ یعنی نیمی از حرکتِ نامنتظر دلار به قیمت فروش می‌رسد. «متقارن»: در هر دو جهت. «فقط بالا»: "
        "فقط وقتی دلار بالاتر از انتظار تمام شود. «دامنهٔ مجاز» سهم‌هایی از خرید امروز است که قید را برآورده "
        "می‌کنند؛ «امن‌ترین سهم» کمترین احتمالِ حاشیهٔ زیر کف را دارد.\n")
    pt_rows = []
    for key, name in (("float", "شناور"), ("fixed", "ثابت")):
        for mode, mode_fa in (("up_only", "فقط بالا"), ("symmetric", "متقارن")):
            x = pt[key, mode]
            r_ = x["res"]
            rng_ = "هیچ" if r_.min_hedge is None else f"{pct(r_.min_hedge, 0)} تا {pct(x['max'], 0)}"
            pt_rows.append([name, mode_fa, rng_, pct(x["safest"], 0),
                            pct(r_.table.loc["payment_day", "p_margin_below_floor"], 1),
                            pct(r_.table.loc["now", "p_margin_below_floor"], 1), pct(x["rev"] - 1, 1, True)])
    add(table(["قرارداد", "حالت انتقال", "دامنهٔ مجاز", "امن‌ترین سهم", "احتمال زیر کف: همه روز پرداخت",
               "احتمال زیر کف: همه امروز", "میانگین درآمد نسبت به برنامه"], pt_rows))
    add("")
    sym_now = pt["float", "symmetric"]["res"].table.loc["now", "p_margin_below_floor"]
    add(f"* **حالت متقارن خرید کامل را خطرناک نشان می‌دهد:** با قرارداد شناور احتمال زیر کف برای «همه امروز» "
        f"{pct(sym_now, 1)} است، بدتر از حالت بدون انتقال ({pct(tb.loc['now', 'p_margin_below_floor'], 1)})، چون اگر دلار "
        "بیفتد درآمد هم می‌افتد و دلارِ گران‌خریده می‌ماند. این همان اثری است که D-043 می‌گوید شاید واقعی نباشد.")
    add(f"* **حالت «فقط بالا» این اثر را برمی‌دارد:** احتمال «همه امروز» به "
        f"{pct(up_float.table.loc['now', 'p_margin_below_floor'], 1)} می‌رسد و قرارداد شناور از سهم "
        f"{share_fa(up_float.min_hedge)} به بالا قید را برآورده می‌کند. حاشیه کم است: خطای معیار این احتمال "
        f"{pct(up_float.table.loc['now', 'se_p_margin_below_floor'], 2)} است.")
    add(f"* **ولی «فقط بالا» درآمد میانگین را هم بالا می‌برد:** {pct(pt['float', 'up_only']['rev'] - 1, 1, True)} نسبت به "
        "درآمد برنامه، چون جهش‌ها منتقل می‌شوند و اُفت‌ها نه. بخشی از بهبود از همین می‌آید (تصمیم ۷-۸).\n")
    add(f"![حساسیت]({f3})\n")
    add("*شکل ۳ — چپ: حداقل نسبت پوشش با قرارداد ثابت؛ هر میله دامنهٔ یک ورودی است و خط عمودی حالت پایه. راست: "
        "بالاترین کف حاشیهٔ سود که در قرارداد شناور شدنی است؛ خط‌چین کف ٪۱۰ است. نوشتهٔ سرِ هر میله می‌گوید "
        "کدام مقدار آن سر را ساخته است؛ سرِ بی‌نوشته همان حالت پایه است.*\n")
    add("بزرگ‌ترین دامنه‌ها:\n")
    add(table(["رتبه", "قرارداد ثابت: حداقل نسبت پوشش", "دامنه", "قرارداد شناور: بالاترین کفِ شدنی", "دامنه"],
              [[fa(k + 1), span_fixed[k][1], f"{pct(span_fixed[k][2], 0)} تا {pct(span_fixed[k][3], 0)}",
                span_float[k][1], f"{pct(span_float[k][2], 1)} تا {pct(span_float[k][3], 1)}"] for k in range(4)]))
    add("")

    # ------------------------------------------------------------------ 4
    add("## ۴. ریسک از کجا می‌آید\n")
    add("راهبرد: همه روز پرداخت. سه حالت روی همان مسیرها: قهوه ثابت (قرارداد ثابت، فقط ریسک دلار)؛ دلار ثابت روی "
        "مسیر مورد انتظارش (فقط ریسک قهوه)؛ هر دو تصادفی (حالت پایه). هزینه‌ها درصدی از هزینهٔ برنامه.\n")
    add(table(["حالت", "میانگین", "انحراف معیار", "VaR95", "CVaR95"], [
        [lab, pct(split[k]["mean"], 1), pct(split[k]["sd"], 1), pct(split[k]["var"], 0), pct(split[k]["cvar"], 0)]
        for lab, k in (("قهوه ثابت: فقط دلار", "coffee_frozen"), ("دلار ثابت: فقط قهوه", "usd_frozen"),
                       ("هر دو تصادفی", "both"))]))
    add("")
    add(f"![تفکیک ریسک]({f4})\n")
    add("*شکل ۴ — CVaR95 هزینهٔ «همه روز پرداخت» منهای برنامه، در سه حالت.*\n")
    add(f"* ریسک دلار بزرگ‌تر است: به‌تنهایی CVaR95 را به {pct(split['coffee_frozen']['cvar'], 0)} برنامه می‌رساند؛ قهوه "
        f"به‌تنهایی به {pct(split['usd_frozen']['cvar'], 0)}.")
    add(f"* با هم {pct(split['both']['cvar'], 0)} است، خیلی کمتر از جمع دو فاصله: دو ریسک تقریباً مستقل‌اند و "
        "روی هم جمع نمی‌شوند.")
    add(f"* ولی ریسک قهوه به‌تنهایی از حد قید ({pct(limit, 0)}) می‌گذرد. خرید زودهنگام دلار فقط ریسک اول را برمی‌دارد؛ "
        "برای همین در حالت پایه هیچ راهبردی کافی نیست.\n")

    # ------------------------------------------------------------------ 5
    add("## ۵. پیش‌نمایش کالیبراسیون\n")
    add("**فقط برای اطلاع؛ هیچ چیز با این نتیجه تنظیم نشده است.** پرسش: صدک ۹۵ دلارِ شبیه‌سازی‌شده امروز "
        f"{pct(today_q['fhs'][1], 0, True)} است (روش فیلترشده) در برابر {pct(today_q['plain'][1], 0, True)} با بوت‌استرپ ساده، و "
        f"همین انتخاب حداقل نسبت پوشش را از {share_fa(Rf.min_hedge)} به "
        f"{share_fa(run(fixed.with_(usd_vol='plain'))[1].min_hedge)} می‌برد. کدام به واقعیت نزدیک‌تر بوده؟\n")
    add(f"**روش.** برای هر چهارشنبه از {jg(cal.BACKTEST_START)} (D-041) تا {jg(okc.index[-1])} (آخرین شروعی که "
        f"نتیجهٔ ۱۳ هفته‌اش معلوم است): موتور با دادهٔ تا همان روز و {num(N_CALIBRATION)} مسیر شبیه‌سازی می‌کند "
        "(تنظیم‌های پایه: رانش خنثی با نرخ بازارِ همان روز، بلوک ۱۳، نیمه‌عمر ۱۳). بعد تغییر واقعیِ ۱۳ هفته‌ای دلار با "
        "صدک ۵ و ۹۵ شبیه‌سازی مقایسه می‌شود. اگر شبیه‌ساز کالیبره باشد، واقعیت حدود ٪۵ موارد از هر طرف بیرون "
        "می‌افتد.\n")
    add(f"* شروع‌ها: {fa(len(tab))}؛ قابل‌استفاده: {fa(len(okc))} (کنار گذاشته: {fa(len(tab) - len(okc))}).")
    add("* **پنجرهٔ مستقل:** شروع‌های هفتگی ۱۲ هفتهٔ مشترک با همسایه دارند، پس مشاهدهٔ مستقل نیستند. «پنجرهٔ "
        "مستقل» تعداد پنجره‌هایی است که هیچ هفتهٔ مشترکی ندارند. سالِ هر ردیف سالِ شروع پنجره است.\n")
    cal_rows = []
    for key, r in summ.iterrows():
        label = "**همه**" if key == "all" else fa(key)
        cal_rows.append([label, fa(int(r["starts"])), fa(int(r["independent"])), pct(r["fhs_above"], 1),
                         pct(r["fhs_below"], 1), pct(r["plain_above"], 1), pct(r["plain_below"], 1)])
    add(table(["سال شروع", "شروع‌ها", "پنجرهٔ مستقل", "فیلترشده: بالای صدک ۹۵", "فیلترشده: زیر صدک ۵",
               "ساده: بالای صدک ۹۵", "ساده: زیر صدک ۵"], cal_rows))
    add("")
    add(f"![کالیبراسیون]({f5})\n")
    add("*شکل ۵ — خط سیاه: تغییر واقعیِ ۱۳ هفته‌ای دلار برای هر هفتهٔ شروع. نوار رنگی: فاصلهٔ صدک ۵ تا ۹۵ که "
        "موتور در همان هفته شبیه‌سازی کرد. نقطه‌ها: شروع‌هایی که واقعیت بیرون از نوار افتاد. بالا: روش فیلترشده؛ "
        "پایین: بوت‌استرپ ساده.*\n")
    add("**خواندن جدول**\n")
    add(f"* **روی هم، دو روش تقریباً یک‌جور عمل کرده‌اند.** بالای صدک ۹۵: فیلترشده {pct(rates['fhs', 'above'], 1)}، ساده "
        f"{pct(rates['plain', 'above'], 1)}. زیر صدک ۵: فیلترشده {pct(rates['fhs', 'below'], 1)}، ساده "
        f"{pct(rates['plain', 'below'], 1)}. در هر دو، واقعیت {num(rates['fhs', 'above'] / 0.05, 1)} و "
        f"{num(rates['plain', 'above'] / 0.05, 1)} برابرِ ٪۵ از بالای نوار بیرون زده است.")
    add(f"* **نمونه کوچک است.** با {fa(n_ind)} پنجرهٔ مستقل، شبیه‌سازی که واقعاً کالیبره باشد با احتمال ٪۹۵ نرخی بین "
        f"{pct(bin_lo, 1)} و {pct(bin_hi, 1)} نشان می‌دهد (تقریب با توزیع دوجمله‌ای روی پنجره‌های مستقل). {fa(inside)} نرخ از ۴ نرخ بالا داخل این "
        "دامنه‌اند. پس این پیش‌نمایش هیچ‌کدام از دو روش را رد نمی‌کند و بینشان هم داوری نمی‌کند.")
    add(f"* **خطاها خوشه‌ای‌اند.** سال‌هایی که نرخِ «بالای صدک ۹۵» دست‌کم ٪۱۵ بود: فیلترشده: {hot('fhs_above')}؛ ساده: "
        f"{hot('plain_above')}. برای «زیر صدک ۵»: فیلترشده: {hot('fhs_below')}؛ ساده: {hot('plain_below')}.")
    add(f"* **دو روش در زمان فرق دارند، نه در مجموع.** در {pct(fhs_narrower, 0)} شروع‌ها نوار فیلترشده باریک‌تر از نوار "
        "ساده بود؛ امروز برعکس است (جدول زیر). روش فیلترشده در دورهٔ آرام نوار را تنگ می‌کند و در دورهٔ پرنوسان "
        f"باز؛ روش ساده تقریباً ثابت می‌ماند. نمونه: در شروع‌های {fa(first_year)} میانهٔ پهنای نوار فیلترشده "
        f"{pct(band['fhs'][calm].median(), 0)} بود و نوار ساده {pct(band['plain'][calm].median(), 0)}؛ میانهٔ رشد واقعی دلار "
        f"{pct(okc.loc[calm, 'realized'].median(), 1, True)} بود در برابر رشد هدفِ {pct(calm_target, 1, True)} (نرخ بازار)، پس واقعیت از "
        "پایینِ نوارِ باریک بیرون افتاد. شمارش کلی نمی‌گوید این واکنش به‌اندازه است یا نه.")
    add(f"* **دورهٔ پرنوسانِ اخیر.** در {fa(len(recent))} شروعِ سال‌های {fa(recent_years[0])} و {fa(recent_years[1])} واقعیت در "
        f"{pct(recent['plain_above'].mean(), 1)} موارد از صدک ۹۵ِ بوت‌استرپ ساده بالاتر رفت و در "
        f"{pct(recent['fhs_above'].mean(), 1)} موارد از صدک ۹۵ِ روش فیلترشده. این فقط "
        f"{fa(cal.independent_windows(recent.index))} پنجرهٔ مستقل است؛ نشانه است، نه نتیجه.\n")
    add(table(["روش", "میانهٔ صدک ۵ در شروع‌ها", "میانهٔ صدک ۹۵ در شروع‌ها", "میانهٔ پهنای نوار",
               "امروز: صدک ۵", "امروز: صدک ۹۵", "امروز: پهنا"], [
        [name, pct(okc[f"{m}_q05"].median(), 1, True), pct(okc[f"{m}_q95"].median(), 1, True), pct(band[m].median(), 1),
         pct(today_q[m][0], 1, True), pct(today_q[m][1], 1, True), pct(today_q[m][1] - today_q[m][0], 1)]
        for m, name in (("fhs", "فیلترشده"), ("plain", "بوت‌استرپ ساده"))]))
    add("")
    add("«امروز» از اجرای پایه با ۲۰٬۰۰۰ مسیر است.\n")
    add("**اثر رانش بر همین شمارش** (روی هم، همهٔ شروع‌ها):\n")
    add(table(["رانش", "فیلترشده: بالای صدک ۹۵", "فیلترشده: زیر صدک ۵", "ساده: بالای صدک ۹۵", "ساده: زیر صدک ۵"], [
        ["خنثی (پایه)", pct(rates["fhs", "above"], 1), pct(rates["fhs", "below"], 1),
         pct(rates["plain", "above"], 1), pct(rates["plain", "below"], 1)],
        ["میانگین تاریخی (فقط گذشته)", pct(alt["historical"]["fhs_above"], 1), pct(alt["historical"]["fhs_below"], 1),
         pct(alt["historical"]["plain_above"], 1), pct(alt["historical"]["plain_below"], 1)],
        ["صفر", pct(alt["zero"]["fhs_above"], 1), pct(alt["zero"]["fhs_below"], 1),
         pct(alt["zero"]["plain_above"], 1), pct(alt["zero"]["plain_below"], 1)],
    ]))
    add("")
    add("شمارش به رانش هم بستگی دارد، نه فقط به روش نوسان: با رانش صفر مرکز نوار پایین‌تر است و جهش‌ها بیشتر "
        "بیرون می‌افتند. آزمون کامل در D5 می‌آید (تصمیم ۷-۹).\n")

    # ------------------------------------------------------------------ 6
    add("## ۶. جمع‌بندی به زبان ساده\n")
    add("**موتور امروز چه می‌گوید**\n")
    add(f"1. با فرض‌های پایه، زمان خرید ارز به‌تنهایی نمی‌تواند حاشیهٔ سود {pct(base.margin_floor, 0)} را با اطمینان "
        f"{pct(base.confidence, 0)} نگه دارد. بهترین کار («همه امروز») احتمال شکست را از "
        f"{pct(tb.loc['payment_day', 'p_margin_below_floor'], 0)} به {pct(tb.loc['now', 'p_margin_below_floor'], 0)} می‌رساند.")
    add(f"2. اهرم دوم (D-032) لازم است: اگر قیمت قهوه با فروشنده امروز ثابت شود، خرید {share_fa(Rf.min_hedge)} از "
        "دلار در همین امروز قید را برآورده می‌کند.")
    add(f"3. اگر ثابت کردن قیمت قهوه ممکن نباشد، قیدِ شدنی کف {pct(d.highest_floor, 1)} است، نه {pct(base.margin_floor, 0)}؛ "
        f"مگر اینکه بنگاه بتواند جهش دلار را به قیمت فروش منتقل کند: با انتقال ۰.۵ِ «فقط بالا» حداقل پوشش "
        f"{share_fa(up_float.min_hedge)} می‌شود.")
    add("4. زود خریدن در حالت پایه چیزی از میانگین هزینه کم یا زیاد نمی‌کند. این از فرض رانش خنثی می‌آید، "
        "نه از داده. موتور دربارهٔ جهت دلار حرفی نمی‌زند؛ فقط ریسک را اندازه می‌گیرد.\n")
    add("**کدام فرض جواب را بیشتر جابه‌جا می‌کند**\n")
    add(f"1. **نوع قرارداد قهوه.** شناور: هیچ راهبردی مجاز نیست. ثابت: حداقل پوشش {share_fa(Rf.min_hedge)}.")
    add(f"2. با قرارداد ثابت، حداقل پوشش بیش از همه به «{span_fixed[0][1]}» ({pct(span_fixed[0][2], 0)} تا "
        f"{pct(span_fixed[0][3], 0)})، «{span_fixed[1][1]}» ({pct(span_fixed[1][2], 0)} تا {pct(span_fixed[1][3], 0)}) و "
        f"«{span_fixed[2][1]}» ({pct(span_fixed[2][2], 0)} تا {pct(span_fixed[2][3], 0)}) حساس است.")
    add(f"3. با قرارداد شناور، کفِ شدنی بیش از همه به «{span_float[0][1]}»، «{span_float[1][1]}» و "
        f"«{span_float[2][1]}» حساس است.")
    mh = {k: run(fixed.with_(**ch))[1].min_hedge for k, ch in (
        ("b4", dict(mean_block=4.0)), ("b8", dict(mean_block=8.0)), ("hl26", dict(half_life=26.0)),
        ("plain", dict(usd_vol="plain")))}
    add(f"4. طول بلوک و نیمه‌عمر نوسان تقریباً اثری ندارند (حداقل پوشش با بلوک ۴ و ۸ هفته: {share_fa(mh['b4'])} و "
        f"{share_fa(mh['b8'])}؛ با نیمه‌عمر ۲۶: {share_fa(mh['hl26'])}). روش نوسان دارد: با بوت‌استرپ ساده حداقل پوشش "
        f"{share_fa(mh['plain'])} می‌شود، چون نوسان بالای امروز را نمی‌بیند. پیش‌نمایش بخش ۵ هنوز نمی‌گوید کدام "
        "درست‌تر است.\n")

    # --------------------------------------------------------------- appendix
    add("## پیوست: خروجی ماشین‌حساب و آزمون‌ها\n")
    add("**فایل مسیرها (D-014).** `export_paths` بازده‌های لگاریتمی هفتگیِ شبیه‌سازی‌شده را می‌نویسد: بدون تاریخ، "
        "بدون سری قیمت، با رانش صفر و نوسان امروز. ماشین‌حساب رانشِ کاربر را با همان ضربِ هفته‌به‌هفتهٔ موتور "
        "اعمال می‌کند، پس جوابش با موتور یکی است.\n")
    add(table(["مورد", "مقدار"], [
        ["مسیر فایل", f"`{ex['path'].relative_to(paths.ROOT)}` (زیر `data/`، در گیت نیست)"],
        ["اندازه", f"{num(ex['bytes'] / 1e6, 2)} مگابایت؛ فشرده با gzip: {num(ex['gzip_bytes'] / 1e6, 2)} مگابایت"],
        ["عددها", f"{num(ex['numbers'])} = {num(N_EXPORT)} مسیر × {fa(T)} هفته × ۳ سری، عدد صحیح با مقیاس ۰.۰۰۰۰۱ (۵ رقم اعشار)"],
        ["فیلدهای دیگر", "تاریخ تصمیم، s0، نرخ بازار، نوسان EWMA دلار، روش و نیمه‌عمر، طول بلوک، پرمیوم پیش‌فرض هر قهوه"],
        ["میانگین سطح هفتهٔ ۱۳ پس از گرد کردن", "، ".join(f"{k}: {num(v, 5)}" for k, v in ex["mean_level_T"].items())],
    ]))
    add("")
    add("بلوک‌های قهوه در این فایل تکه‌هایی از بازده‌های واقعیِ هفتگی‌اند، بدون تاریخ و بدون سطح قیمت؛ D-014 "
        "همین را مجاز کرده است. بازده‌های دلار استانداردشده و دوباره مقیاس‌شده‌اند.\n")
    add(f"**آزمون‌ها.** `pytest`: {fa(tests['passed'])} آزمون سبز، {fa(tests['failed'])} قرمز؛ {fa(tests['engine_passed'])} تای "
        "آن‌ها برای موتور است (`tests/test_engine.py`). آزمون‌های اسپک D3:\n")
    add("* نوسان صفر: هزینه ÷ (مبلغ × نرخ) برابر ۱.۰۷۰ برای «امروز»، ۱.۰۴۶ برای سه‌پله‌ای و ۱.۰۰۰ برای «روز پرداخت»؛ "
        "هم با مسیر دست‌ساز، هم از راه شبیه‌ساز.")
    add("* قرارداد ثابت: هزینه به مسیر قهوه بستگی ندارد.")
    add("* رانش خنثی، نرخ تأمین مالی برابر نرخ بازار، بدون اصطکاک: میانگین هزینهٔ «امروز» و «روز پرداخت» در "
        "محدودهٔ ۳ خطای شبیه‌سازی برابرند.")
    add("* تنظیم رانش در هر هفته دقیقاً به هدف می‌رسد (هر چهار نوع رانش، هر دو روش نوسان).")
    add("* قاعدهٔ تصمیم روی توزیع دست‌ساز، با حالت «هیچ‌کدام مجاز نیست».")
    add("* بذر یکسان، نتیجهٔ یکسان. شبیه‌ساز با یک پیاده‌سازی کُند و ساده (حلقه‌به‌حلقه) عدد به عدد یکی است.\n")
    add("آزمون‌های D3b:\n")
    add(f"* D-037: اختلاف میانگینِ کمتر از {pct(base.tie_tolerance, 2)} برنامه تساوی است و CVaR95 کمتر برنده می‌شود؛ "
        "اختلافِ برابر یا بیشتر از آن تساوی نیست؛ آستانه سهمی از برنامه است.")
    add("* D-038: هفتهٔ منتهی به تاریخ تصمیم در مخزن و در برآورد نوسان هست؛ اگر بازده‌های پس از تاریخ تصمیم با "
        "عددهای بی‌معنی عوض شوند (یا حذف شوند) خروجی شبیه‌ساز عوض نمی‌شود.")
    add("* D-039: نرخ تأمین مالی = نرخ بازار + فاصله.")
    add("* D-043: در حالت «فقط بالا»، وقتی دلار زیر S₀·g_T تمام شود درآمد همان درآمد برنامه می‌ماند؛ بالای آن با "
        "حالت متقارن یکی است.")
    add("* کالیبراسیون: صدک‌ها همان صدک‌های موتور در همان هفته‌اند؛ آنچه پس از هفتهٔ شروع رخ می‌دهد صدک‌ها را "
        "عوض نمی‌کند؛ شروعِ غیرقابل‌استفاده شمرده می‌شود، حذف نمی‌شود.\n")

    # ------------------------------------------------------------------ 7
    add("## ۷. تصمیم‌ها\n")
    add("بندهای ۷-۱ تا ۷-۷ در بازبینی D3 تصمیم گرفته شدند. بندهای پس از آن تازه‌اند و تصمیم می‌خواهند: برای هر "
        "کدام آنچه دیده شد، گزینه‌ها و پیشنهاد من آمده است.\n")

    add("**۷-۱. حالت پایهٔ یادداشت**")
    add(f"* دیده شد: با قرارداد شناور هیچ راهبردی قید را برآورده نمی‌کند (بهترین {pct(d.best_probability, 1)}). با "
        f"قرارداد ثابت حداقل پوشش {share_fa(Rf.min_hedge)} است.")
    add("* **تصمیم (D-036):** پایهٔ یادداشت «شناور» می‌ماند و «ثابت» کنارش نشان داده می‌شود. یافتهٔ اصلی همین "
        "است: تا قیمت قهوه باز است، زمان خرید ارز به‌تنهایی حاشیهٔ سود را نگه نمی‌دارد.\n")

    add("**۷-۲. تساوی در قاعدهٔ تصمیم**")
    add("* دیده شد: با رانش خنثی میانگین هزینه‌ها به‌طور ساختاری برابرند و اصطکاک‌های ریز، کمتر از خطای "
        "شبیه‌سازی، «کمترین میانگین» را تعیین می‌کردند.")
    add(f"* **تصمیم (D-037):** اختلاف میانگینِ کمتر از {pct(base.tie_tolerance, 2)} برنامه تساوی است و CVaR95 کمتر "
        "برنده می‌شود. خروجی اصلی حداقل نسبت پوشش است؛ راهبرد منتخب کنارش می‌آید. اعمال شد.\n")

    add("**۷-۳. هفتهٔ منتهی به تاریخ تصمیم**")
    add(f"* دیده شد: بازدهٔ این هفته ({pct(float(r_usd.loc[as_of]), 1, True)} برای دلار) در تاریخ تصمیم دانسته است ولی "
        "در D3 بیرون مانده بود.")
    add(f"* **تصمیم (D-038):** این هفته در مخزن و در برآورد نوسان می‌آید. اعمال شد: نوسان "
        f"{pct(info['sigma_as_of'], 2)} (پیش‌تر {pct(Po.info['sigma_as_of'], 2)}).\n")

    add("**۷-۴. حساسیت نرخ تأمین مالی**")
    add(f"* دیده شد: نرخ بازار امروز {pct(base.market_rate, 1)} است، پس سطح ثابت ٪۴۰ تقریباً همان پایه بود.")
    add(f"* **تصمیم (D-039):** حساسیت = نرخ بازار + ۱۰ واحد درصد. اعمال شد: بهای پوشش کامل "
        f"{pct(price['spread'], 2, True)} برنامه (پایه {pct(price['base'], 2, True)}).\n")

    add("**۷-۵. جزئیات برآورد نوسان**")
    add(f"* دیده شد: میانگین در EWMA صفر؛ دست‌کم {fa(sim.MIN_HISTORY)} بازدهٔ قبلی؛ کف = صدک ۵ِ برآوردهای تا آن روز، "
        f"که در {fa(info['floor_hits'])} هفته فعال شد.")
    add("* **تصمیم (D-040):** همین می‌ماند.\n")

    add("**۷-۶. شروع بک‌تست**")
    add(f"* دیده شد: نرخ بازار (بازده ۱۳ هفتهٔ صندوق) در {fa(weeks_lost)} هفتهٔ اولِ پس از {jd(d015)} (D-015) وجود "
        "ندارد.")
    add(f"* **تصمیم (D-041):** بک‌تست از {jg(cal.BACKTEST_START)} شروع می‌شود. پیش‌نمایش بخش ۵ از همین تاریخ است.\n")

    add("**۷-۷. دو تعریف کوچک**")
    add(f"* دیده شد: تأخیر انتشار IMF = {fa(I.IMF_LAG_MONTHS)} ماه؛ رانش «میانگین تاریخی» = میانگین بازدهٔ سادهٔ هفتگی "
        f"دلار در مخزنِ همان تاریخ، مرکب‌شده (امروز {pct(run(base.with_(drift='historical'))[0].growth[T] - 1, 1, True)} "
        f"برای ۱۳ هفته، در برابر {pct(g_T - 1, 1, True)} خنثی).")
    add("* **تصمیم (D-042):** هر دو همین می‌مانند.\n")

    add("**۷-۸. انتقالِ «فقط بالا» و درآمد برنامه** (تازه)")
    add(f"* دیده شد: با انتقال ۰.۵ِ «فقط بالا»، میانگین درآمد {pct(pt['float', 'up_only']['rev'] - 1, 1, True)} بالاتر از "
        "درآمد برنامه است (در حالت متقارن برابر است). پس «حاشیهٔ سود برنامه» دیگر حاشیهٔ میانگین نیست، حاشیه در "
        "حالتی است که دلار طبق انتظار یا پایین‌تر تمام شود. بخشی از شدنی شدنِ قرارداد شناور (حداقل پوشش "
        f"{share_fa(up_float.min_hedge)}) از همین درآمدِ بالاتر می‌آید.")
    add("* گزینه‌ها: (الف) همین بماند و در یادداشت گفته شود؛ (ب) درآمد برنامه طوری پایین بیاید که میانگین درآمد "
        "برابر برنامه بماند، تا «فقط بالا» فقط شکل ریسک را عوض کند، نه سطح درآمد را.")
    add("* پیشنهاد: **(الف)**. بنگاهی که می‌تواند جهش را منتقل کند واقعاً درآمد میانگین بالاتری دارد؛ کم کردن آن "
        "فرض تازه‌ای می‌خواهد. پیش‌فرض موقت: (الف).\n")

    add("**۷-۹. آزمون کامل کالیبراسیون در D5** (تازه)")
    add(f"* دیده شد: شمارش کلی دو روش را جدا نمی‌کند (بالای صدک ۹۵: {pct(rates['fhs', 'above'], 1)} و "
        f"{pct(rates['plain', 'above'], 1)}، با {fa(n_ind)} پنجرهٔ مستقل). تفاوت دو روش در این است که نوار را کِی باز و "
        "بسته می‌کنند، و شمارش کلی این را نمی‌سنجد. نتیجه به رانش هم بستگی دارد.")
    add("* گزینه‌ها: (الف) D5 همین شمارش را تکرار کند؛ (ب) به‌علاوه، شمارش جدا برای هفته‌های پرنوسان و کم‌نوسان "
        "(مثلاً نیمهٔ بالا و پایینِ نوسان EWMAِ همان روز)، چون پرسشِ امروز دربارهٔ یک هفتهٔ پرنوسان است؛ (ج) به‌علاوه، "
        "معیاری که پهنای نوار را هم جریمه کند، تا نوارِ بی‌جهت پهن امتیاز نگیرد.")
    add("* پیشنهاد: **(ب)**، با مرز دسته‌بندی که پیش از دیدن نتیجه ثبت شود. تا آن وقت روش فیلترشده (D-026) "
        "پیش‌فرض می‌ماند و عدد بوت‌استرپ ساده کنارش گزارش می‌شود.\n")

    text = "\n".join(L) + "\n"
    n_open = len(re.findall(r"^\*\*۷-[۰-۹]+\..*\(تازه\)$", text, flags=re.M))
    text = text.replace("@N_OPEN@", fa(n_open))
    (OUT / "README.md").write_text(text, encoding="utf-8")
    return str(OUT / "README.md")


def main() -> None:
    print(build())


if __name__ == "__main__":
    main()
