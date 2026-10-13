#!/usr/bin/env python
"""Build the D5 report: reports/d5_backtest/README.md (Persian) and its figures.

D5 is the backtest: every Wednesday from 1394-03-27 with a known 13-week
outcome is a decision date. Part A prices the simple strategies on what really
happened; part B runs the engine past-only at each date and prices what it
would have chosen; part C checks the calibration, split by volatility regime.
D5b adds an illustration of the stress floor (D-051).
Every number, table and figure in the report is computed here (about a
minute):

    .venv/bin/python scripts/d5_report.py
"""
from __future__ import annotations

import datetime as dt

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from coffee_risk import paths
from coffee_risk.data import fred, panel, weekly
from coffee_risk.engine import backtest as B
from coffee_risk.engine import calibration as cal
from coffee_risk.engine import settings, stress
from coffee_risk.engine.inputs import Inputs
from coffee_risk.reporting import (AQUA, BLUE, GRID, INK, INK2, ORANGE, YELLOW, fa, jd, jg, num, pct,
                                   strategy_fa, style, table)

OUT = paths.REPORTS / "d5_backtest"
T = 13
METHOD_FA = {"fhs": "فیلترشده", "plain": "بوت‌استرپ ساده"}
METHOD_EN = {"fhs": "filtered (FHS)", "plain": "plain bootstrap"}
SUBSET_FA = {"all": "همه", "high": "پرنوسان", "low": "کم‌نوسان"}
SUBSET_EN = {"all": "all starts", "high": "high volatility", "low": "low volatility"}
TARGET_FA = {"usd": "تغییر ۱۳ هفته‌ای دلار", "pd_fixed": "هزینهٔ «همه روز پرداخت»، قرارداد ثابت",
             "pd_floating": "هزینهٔ «همه روز پرداخت»، قرارداد شناور"}
LADDER = ["payment_day"] + [f"partial_{k}" for k in range(10, 100, 10)] + ["now", "staged_3"]


def _save(fig, name: str) -> str:
    fig.savefig(OUT / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return name


def _dates(ax) -> None:
    loc = mdates.AutoDateLocator(minticks=5, maxticks=10)
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))


def band_fa(lo, hi) -> str:
    return f"{pct(lo, 1)} تا {pct(hi, 1)}"


def inside_fa(x, lo, hi) -> str:
    return "داخل" if lo <= x <= hi else "بیرون"


# -------------------------------------------------------------------- figures
def fig_margin(ok: pd.DataFrame, floor: float, plan: float) -> str:
    fig, ax = plt.subplots(figsize=(10.0, 4.6))
    series = [("fixed.now.margin", "all today", BLUE), ("fixed.payment_day.margin", "all on the payment day", ORANGE),
              ("fixed.staged_3.margin", "staged (weeks 0, 4, 9)", AQUA),
              ("fhs.fixed.margin", "model: minimum hedge ratio (FHS)", YELLOW)]
    for col, label, color in series:
        ax.plot(ok.index, ok[col] * 100, color=color, lw=1.4, label=label)
    ax.axhline(floor * 100, color=INK, lw=1.0, ls="--")
    ax.text(ok.index[0], floor * 100 - 2.5, f"floor {floor * 100:.0f}%", fontsize=8.5, color=INK, va="top")
    ax.axhline(plan * 100, color=INK2, lw=0.6)
    ax.set_ylim(-80, 45)
    ax.set_ylabel("realised margin, %")
    ax.set_xlabel("decision date")
    ax.set_title("Fixed contract: the margin each strategy really produced, by decision date", pad=26)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=4, borderaxespad=0.2, fontsize=8.5)
    _dates(ax)
    return _save(fig, "fig1_realised_margin.png")


def fig_breach(rates: dict, band: tuple) -> str:
    """rates: contract -> [(label, rate)]"""
    fig, axes = plt.subplots(2, 1, figsize=(10.0, 6.4), sharey=True)
    for ax, (title, rows) in zip(axes, rates.items()):
        labels = [r[0] for r in rows]
        vals = [r[1] * 100 for r in rows]
        colors = [YELLOW if lab.startswith("model") else (AQUA if lab == "staged" else BLUE) for lab in labels]
        ax.axhspan(band[0] * 100, band[1] * 100, color=GRID, alpha=0.6, lw=0)
        ax.bar(range(len(vals)), vals, color=colors, width=0.62, lw=0)
        for x, v in enumerate(vals):
            ax.text(x, v + 0.5, f"{v:.1f}", ha="center", va="bottom", fontsize=8.5, color=INK)
        ax.axhline(5, color=INK, lw=1.0, ls="--")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=8.5)
        ax.set_ylabel("dates with margin\nbelow the floor, %")
        ax.set_ylim(0, 30)
        ax.set_title(title, fontsize=10, pad=6)
        ax.grid(axis="x", visible=False)
    axes[-1].set_xlabel("share of the USD bought on the decision date (0% = all on the payment day), "
                        "the staged plan, and the model's choice")
    fig.suptitle("How often each strategy really broke the floor  (dashed = 5%; grey = what a true 5% rate "
                 "could show with this many independent windows)", x=0.01, ha="left", fontsize=10.5,
                 fontweight="bold")
    fig.tight_layout()
    return _save(fig, "fig2_breach_rates.png")


def fig_calibration(calib: pd.DataFrame) -> str:
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 4.0))
    x = np.arange(3)
    subsets = ["all", "high", "low"]
    for ax, col, title in zip(axes, ("above", "below", "pinball"),
                              ("above the 95th percentile, % of starts", "below the 5th percentile, % of starts",
                               "pinball loss, points (lower is better)")):
        for j, (m, color) in enumerate((("fhs", BLUE), ("plain", ORANGE))):
            vals = [calib.loc[(s, m), col] * 100 for s in subsets]
            ax.bar(x + (j - 0.5) * 0.36, vals, width=0.34, color=color, lw=0, label=METHOD_EN[m])
            for xi, v in zip(x + (j - 0.5) * 0.36, vals):
                ax.text(xi, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8.5, color=INK)
        if col != "pinball":
            for xi, s in zip(x, subsets):
                lo, hi = calib.loc[(s, "fhs"), ["band_lo", "band_hi"]] * 100
                ax.plot([xi - 0.42, xi + 0.42], [hi, hi], color=INK2, lw=0.9, ls=":")
            ax.axhline(5, color=INK, lw=1.0, ls="--")
        ax.set_xticks(x)
        ax.set_xticklabels([SUBSET_EN[s] for s in subsets], fontsize=8.5)
        ax.set_title(title, fontsize=9.5, fontweight="normal", color=INK, pad=6)
        ax.grid(axis="x", visible=False)
    handles, labels = axes[0].get_legend_handles_labels()
    axes[2].legend(handles, labels, loc="upper right", fontsize=8.5)
    fig.suptitle("Calibration of the 13-week USD change by volatility regime  (dashed = 5%; dotted = top of "
                 "the 95% binomial band)", x=0.01, ha="left", fontsize=10.5, fontweight="bold")
    fig.tight_layout()
    return _save(fig, "fig4_calibration_by_regime.png")


def fig_hstar(ok: pd.DataFrame, level: pd.Series, floor: float) -> str:
    fig, (a, b) = plt.subplots(2, 1, figsize=(10.0, 6.2), sharex=True)
    a.plot(ok.index, ok["fhs.h_star"] * 100, color=BLUE, lw=1.4, label="filtered (FHS)")
    a.plot(ok.index, ok["plain.h_star"] * 100, color=ORANGE, lw=1.2, label="plain bootstrap")
    broke = ok[ok["fhs.fixed.margin"] < floor]
    a.plot(broke.index, broke["fhs.h_star"] * 100, ls="", marker="o", ms=3.6, color=INK,
           label="FHS: the floor broke 13 weeks later")
    a.set_ylim(-3, 100)
    a.set_ylabel("minimum hedge ratio h*, %")
    a.set_title("What the model asked for: minimum share to buy on each decision date (fixed contract)", pad=26)
    a.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, borderaxespad=0.2, fontsize=8.5)
    lv = level.loc[ok.index[0]:]
    b.plot(lv.index, lv.values / 1e6, color=INK2, lw=1.3)
    b.set_yscale("log")
    ticks = [t for t in (0.03, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0) if lv.min() / 1e6 * 0.9 <= t <= lv.max() / 1e6 * 1.1]
    b.set_yticks(ticks)
    b.set_yticklabels([f"{t:g}" for t in ticks])
    b.minorticks_off()
    b.set_ylabel("USD/IRR, million rial (log scale)")
    b.set_xlabel("decision date")
    _dates(b)
    fig.tight_layout()
    return _save(fig, "fig3_hstar.png")


def fig_floor(ok: pd.DataFrame, floor: float) -> str:
    fig, ax = plt.subplots(figsize=(10.0, 4.2))
    ax.plot(ok.index, ok["fhs.h_star"] * 100, color=BLUE, lw=1.3, label="model only: h*")
    ax.plot(ok.index, ok["fhs.floor_hedge"] * 100, color=ORANGE, lw=1.6, label="with the stress floor: max(h*, floor)")
    broke = ok[ok["fhs.floor.margin"] < floor]
    ax.plot(broke.index, broke["fhs.floor_hedge"] * 100, ls="", marker="o", ms=3.6, color=INK,
            label="with the floor: the margin still broke 13 weeks later")
    ax.set_ylim(-3, 100)
    ax.set_ylabel("share bought on the decision date, %")
    ax.set_xlabel("decision date")
    ax.set_title("Stress floor (illustration): never hedge less than what survives the largest rise seen so far",
                 pad=26)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, borderaxespad=0.2, fontsize=8.5)
    _dates(ax)
    return _save(fig, "fig5_stress_floor.png")


# --------------------------------------------------------------------- report
def build() -> str:
    OUT.mkdir(parents=True, exist_ok=True)
    style(plt)
    wk, daily, imf = weekly.load(), panel.load(), fred.load()
    bt = B.run_backtest(wk, daily, imf, n_paths=settings.BACKTEST_PATHS, seed=settings.REPORT_SEED)
    ok = bt[bt["usable"]]
    base = Inputs(s0=1.0, market_rate=0.0, premium_share=0.0)        # only for the fixed placeholders
    floor, plan_margin, conf = base.margin_floor, base.planned_margin, base.confidence
    n_ind = cal.independent_windows(ok.index, T)
    band = cal.binomial_range(n_ind)
    reg = B.regime(bt)
    names = [s.name for s in B.fixed_strategies(T)]
    summ = {c: B.strategy_summary(bt, c, floor, names) for c in B.CONTRACTS}
    cheap = {c: B.cheapest_share(bt, c) for c in B.CONTRACTS}
    calib = {t: B.calibration(bt, t) for t in B.TARGETS}

    def breach(col, part=ok):
        return float((part[col] < floor).mean())

    def model_row(m, contract, part=ok):
        return {"cost": float(part[f"{m}.{contract}.cost"].mean()), "median": float(part[f"{m}.{contract}.cost"].median()),
                "cvar": B.tail_mean(part[f"{m}.{contract}.cost"]), "breach": breach(f"{m}.{contract}.margin", part),
                "worst": float(part[f"{m}.{contract}.margin"].min())}

    model = {(m, c): model_row(m, c) for m in B.METHODS for c in B.CONTRACTS}
    feas = {m: ok[f"{m}.floating.feasible"].astype(bool) for m in B.METHODS}
    high, low = (reg == "high").values, (reg == "low").values

    def year_table(columns):
        return B.by_year(bt, columns, floor, T)

    yr_fixed = year_table({n: f"fixed.{n}.margin" for n in B.MAIN})
    yr_float = year_table({n: f"floating.{n}.margin" for n in B.MAIN})
    yr_model = year_table({"fhs": "fhs.fixed.margin", "plain": "plain.fixed.margin",
                           "late": "fixed.payment_day.margin"})

    def year_stats(contract):
        rows = {}
        for y, part in list(ok.groupby("jalali_year")) + [("all", ok)]:
            rows[y] = {f"{n}.{k}": v for n in B.MAIN for k, v in (
                ("mean", part[f"{contract}.{n}.cost"].mean()), ("median", part[f"{contract}.{n}.cost"].median()),
                ("cvar", B.tail_mean(part[f"{contract}.{n}.cost"])))}
        return pd.DataFrame.from_dict(rows, orient="index")

    ystats = {c: year_stats(c) for c in B.CONTRACTS}
    hot_years = lambda col: "، ".join(f"{fa(int(y))} ({pct(v, 0)})" for y, v in yr_model.drop(index="all")[col].items()
                                     if v >= 0.10) or "هیچ سالی"

    # ---- figures
    f1 = fig_margin(ok, floor, plan_margin)

    def ladder(contract):
        lab = {"payment_day": "0%", "now": "100%", "staged_3": "staged"}
        rows = [(lab[n] if n in lab else n.split("_")[1] + "%", summ[contract].loc[n, "breach"]) for n in LADDER]
        return rows + [("model\nFHS", model["fhs", contract]["breach"]), ("model\nplain", model["plain", contract]["breach"])]

    f2 = fig_breach({"Fixed contract": ladder("fixed"), "Floating contract": ladder("floating")}, band)
    f3 = fig_calibration(calib["usd"])
    f4 = fig_hstar(ok, wk["usd_free"], floor)
    f5 = fig_floor(ok, floor)

    # ---- D5b: the stress floor (D-051), an illustration
    floor_rows = [("فقط مدل (h*)", "fhs.h_star", "fhs.fixed"), ("مدل با کفِ تنش", "fhs.floor_hedge", "fhs.floor"),
                  ("سه‌پله‌ای", None, "fixed.staged_3"), ("همه امروز", None, "fixed.now")]

    def floor_stats(part, hedge_col, prefix):
        return {"hedge": np.nan if hedge_col is None else float(part[hedge_col].mean()),
                "cost": float(part[f"{prefix}.cost"].mean()), "cvar": B.tail_mean(part[f"{prefix}.cost"]),
                "breach": breach(f"{prefix}.margin", part), "worst": float(part[f"{prefix}.margin"].min())}

    fs = {(sub, lab): floor_stats(ok[mask], hc, pre)
          for sub, mask in (("all", np.ones(len(ok), dtype=bool)), ("high", high), ("low", low))
          for lab, hc, pre in floor_rows}
    yr_floor = year_table({"floor": "fhs.floor.margin", "model": "fhs.fixed.margin"})

    # ---- sensitivity of the stress floor: S2 as the scenario (D-053)
    s2 = next(sc for sc in stress.scenarios(wk) if sc.key == "fx_1401_1402")
    s2_rise, s2_end = s2.change("usd"), pd.Timestamp(s2.usd_window[1])
    s2_variants = {"past": B.scenario_floor(bt, wk, s2_rise, available_from=s2_end, horizon=T),
                   "always": B.scenario_floor(bt, wk, s2_rise, horizon=T)}

    def floor_summary(hedge, cost, margin):
        return {"hedge": float(hedge.mean()), "cost": float(cost.mean()), "breach": float((margin < floor).mean())}

    sens = {"model": floor_summary(ok["fhs.h_star"], ok["fhs.fixed.cost"], ok["fhs.fixed.margin"]),
            "worst": floor_summary(ok["fhs.floor_hedge"], ok["fhs.floor.cost"], ok["fhs.floor.margin"]),
            **{k: floor_summary(v["hedge"], v["cost"], v["margin"]) for k, v in s2_variants.items()}}
    s2_before = int((ok.index < s2_end).sum())
    s1_dates = int(np.isclose(ok["stress.max_rise"], ok["stress.max_rise"].max()).sum())
    rise = ok["stress.max_rise"]
    jump = rise.index[(rise.values > rise.values[0] + 1e-12)][0]            # first date with a larger rise in its past
    rise_top = rise.index[(rise.values >= rise.max() - 1e-12)][0]           # first date that has seen the largest rise

    # ======================================================================
    L: list[str] = []
    add = L.append
    add("# گزارش مرحلهٔ D5 — بک‌تست\n")
    add(f"تاریخ ساخت: {jg(dt.date.today())}. همهٔ عددها، جدول‌ها و شکل‌ها را `scripts/d5_report.py` ساخته است: هر "
        f"تاریخ تصمیم {num(settings.BACKTEST_PATHS)} مسیر، بذر ثابت برای هر تاریخ (`{settings.REPORT_SEED}` + شمارهٔ هفته).\n")
    add("## آنچه در D5b عوض شد\n")
    add("این گزارش پس از بازبینی D4 و D5 دوباره ساخته شده است. سه تغییر؛ عددهای بخش‌های ۱ تا ۴ همان‌اند که بودند "
        "(همان بذرها).\n")
    add("1. **بخش تازه «D5b. کفِ تنش»** (پیش از بخش ۵)، با شکل ۵: نمایشِ قاعدهٔ D-051 روی همین بک‌تست. این فکر پس "
        "از دیدن نتیجه‌های D5 آمد؛ پس آن بخش نمایش است، نه شاهدِ بیرون از نمونه.")
    add("2. **بخش ۵ حالا «تصمیم‌ها» است:** تصمیم‌های بازبینی (D-048 تا D-051) کنار هر بند آمده‌اند. تصمیم‌های D4 "
        "(D-046، D-047) در `DECISIONS.md` ثبت شده‌اند.")
    add("3. **افزوده پس از بازبینی D5b: جدول «حساسیت به سناریوی کف»** در بخش D5b، با دو ردیف برای کفِ S2، و "
        "تصمیم D-053 در بخش ۵. بقیهٔ بخش D5b همان است.\n")
    add("**دو پرسش، هر دو روی تاریخ واقعی و فقط با دادهٔ گذشته.** (۱) اگر بنگاه از ۱۳۹۴ هر هفته یکی از راهبردهای "
        "ساده را دنبال کرده بود، حاشیهٔ سود چند بار زیر کف می‌رفت و به چه هزینه‌ای؟ (۲) وعدهٔ مدل درست بود؟ وقتی "
        "گفت «حداکثر ۵٪ ریسک»، ریسکِ واقعی نزدیک ۵٪ بود؟\n")
    add("**قراردادها**\n")
    add(f"* **تاریخ تصمیم:** هر چهارشنبه از {jg(ok.index[0])} (D-041) تا {jg(ok.index[-1])}، آخرین هفته‌ای که نتیجهٔ "
        f"۱۳ هفته‌اش معلوم است. {fa(len(bt))} تاریخ؛ {fa(len(ok))} قابل‌استفاده (کنار گذاشته: {fa(len(bt) - len(ok))}).")
    add(f"* **پنجرهٔ مستقل:** تاریخ‌های هفتگی ۱۲ هفتهٔ مشترک با همسایه دارند. {fa(len(ok))} تاریخ برابر با {fa(n_ind)} "
        "پنجرهٔ بدون هم‌پوشانی است؛ کنار هر نرخ همین عدد آمده است.")
    add(f"* **دامنهٔ دوجمله‌ای:** اگر نرخِ واقعی ۵٪ باشد، با {fa(n_ind)} پنجرهٔ مستقل نرخِ مشاهده‌شده با احتمال ۹۵٪ بین "
        f"{band_fa(*band)} می‌افتد (تقریب). نرخی داخل این دامنه با «۵٪» سازگار است.")
    add("* **ورودی‌های هر تاریخ، همان روز:** نرخ دلار، نرخ بازار (بازده ۱۳ هفتهٔ اخیر صندوق)، پرمیوم با تأخیر "
        f"انتشار IMF. برنامه با رانش خنثی. انتقال صفر، نااطمینانی فروش خاموش، حاشیهٔ برنامه {pct(plan_margin, 0)}، کف "
        f"{pct(floor, 0)}. همه پیش‌فرض‌های موقت D3.")
    add("* **هزینهٔ واقعی:** نرخ‌های واقعی دلار در هفته‌های خرید؛ قهوه از جمع بازده‌های هفتگیِ بدون رول (D-022)؛ "
        "هزینهٔ پولِ خرید زودهنگام = رشد واقعی صندوق از هفتهٔ خرید تا هفتهٔ پرداخت. با همان تابع هزینهٔ موتور حساب "
        "شده است. هزینه‌ها درصدی از هزینهٔ برنامهٔ همان تاریخ‌اند.")
    add(f"* در {fa(int(ok['fund_level_old'].sum()))} تاریخ، یکی از هفته‌های خرید یا پرداخت روی تعطیلی بازار افتاده و "
        "آخرین سطحِ منتشرشدهٔ صندوق (بیش از ۵ روز پیش) به کار رفته است. این تاریخ‌ها حذف نشده‌اند.\n")

    # ---------------------------------------------------------------- summary
    fx, fl = summ["fixed"], summ["floating"]
    mf = model["fhs", "fixed"]
    add("## خلاصه به زبان ساده\n")
    add("**چه می‌شد**\n")
    add(f"1. **با قرارداد ثابت، خرید کاملِ دلار در روز تصمیم هیچ‌وقت کف را نشکست** ({pct(fx.loc['now', 'breach'], 1)} از "
        f"{fa(len(ok))} هفته). صبر تا روز پرداخت در {pct(fx.loc['payment_day', 'breach'], 1)} هفته‌ها کف را شکست، با بدترین "
        f"حاشیهٔ {pct(fx.loc['payment_day', 'worst_margin'], 0)}. خرید سه‌پله‌ای: {pct(fx.loc['staged_3', 'breach'], 1)}.")
    add(f"2. **زود خریدن به‌طور میانگین ارزان‌تر هم بود:** میانگین هزینه {pct(fx.loc['now', 'mean_cost'], 1)} برنامه برای "
        f"«همه امروز» در برابر {pct(fx.loc['payment_day', 'mean_cost'], 1)} برای «همه روز پرداخت». ولی میانه‌ها نزدیک‌اند "
        f"({pct(fx.loc['now', 'median_cost'], 1)} و {pct(fx.loc['payment_day', 'median_cost'], 1)}) و هر کدام تقریباً در نیمی "
        f"از هفته‌ها ارزان‌تر بود ({pct(cheap['fixed']['now'], 0)} و {pct(cheap['fixed']['payment_day'], 0)}): فاصلهٔ میانگین "
        "را چند جهش بزرگ ساخته است.")
    add(f"3. **با قرارداد شناور هیچ راهبردی امن نبود:** «همه امروز» در {pct(fl.loc['now', 'breach'], 1)} هفته‌ها کف را "
        f"شکست و «همه روز پرداخت» در {pct(fl.loc['payment_day', 'breach'], 1)}. این همان یافتهٔ D3 است، این بار روی "
        "تاریخ واقعی.\n")
    add("**می‌شود به مدل اعتماد کرد؟**\n")
    add(f"4. **تا حدی.** اگر بنگاه هر هفته همان «حداقل نسبت پوشش» مدل را می‌خرید (قرارداد ثابت، روش فیلترشده)، کف در "
        f"{pct(mf['breach'], 1)} هفته‌ها می‌شکست. وعده ۵٪ بود. با {fa(n_ind)} پنجرهٔ مستقل این هنوز "
        f"{inside_fa(mf['breach'], *band)}ِ دامنهٔ {band_fa(*band)} است، ولی در سمت بدِ آن. بوت‌استرپ ساده: "
        f"{pct(model['plain', 'fixed']['breach'], 1)}.")
    add(f"5. **در هفته‌های پرنوسان، مثل امروز، روش فیلترشده نزدیک وعده ماند:** نرخ شکست "
        f"{pct(breach('fhs.fixed.margin', ok[high]), 1)} در برابر {pct(breach('plain.fixed.margin', ok[high]), 1)} برای "
        "بوت‌استرپ ساده.\n")
    add("**کجا شکست می‌خورد**\n")
    add(f"6. **در هفته‌های آرامِ پیش از جهش.** در نیمهٔ کم‌نوسان، مدلِ فیلترشده به‌طور میانگین فقط "
        f"{pct(ok.loc[low, 'fhs.h_star'].mean(), 0)} پوشش می‌خواست و کف در {pct(breach('fhs.fixed.margin', ok[low]), 1)} "
        f"هفته‌ها شکست. شکست‌ها در چند سال جمع شده‌اند: {hot_years('fhs')}. مدل آرامش را ادامه‌دار فرض می‌کند و جهش "
        "را نمی‌بیند.")
    add(f"7. **قرارداد شناور:** مدل فقط در {pct(feas['fhs'].mean(), 0)} هفته‌ها راهبرد مجازی پیدا کرد؛ و در همان "
        f"هفته‌ها کف در {pct(breach('fhs.floating.margin', ok[feas['fhs']]), 1)} موارد شکست. وعده حداکثر ۵٪ بود.\n")

    # ------------------------------------------------------------------ 1
    add("## ۱. راهبردهای ساده، بدون مدل\n")
    add("هر ردیف یک راهبرد است که هر هفته بی‌استثنا اجرا شده. **CVaR95** میانگینِ ۵٪ گران‌ترین هفته‌هاست. "
        "«شکست کف» سهم هفته‌هایی است که حاشیهٔ سود زیر کف رفت.\n")
    head = ["راهبرد", "ثابت: میانگین هزینه", "میانه", "CVaR95", "شکست کف", "بدترین حاشیه",
            "شناور: میانگین هزینه", "میانه", "CVaR95", "شکست کف", "بدترین حاشیه"]
    rows = []
    for n in LADDER:
        row = [strategy_fa(n)]
        for c in B.CONTRACTS:
            r = summ[c].loc[n]
            row += [pct(r["mean_cost"], 1), pct(r["median_cost"], 1), pct(r["cvar"], 0), pct(r["breach"], 1),
                    pct(r["worst_margin"], 0)]
        rows.append(row)
    add(table(head, rows))
    add("")
    add(f"تعداد: {fa(len(ok))} تاریخ، {fa(n_ind)} پنجرهٔ مستقل. «همه روز پرداخت» همان سهم ٪۰ است و «همه امروز» همان "
        "سهم ٪۱۰۰.\n")
    add("**کدام ارزان‌تر تمام شد؟** از میان سه راهبرد اصلی، سهم هفته‌هایی که هر کدام ارزان‌ترین بود:\n")
    add(table(["قرارداد"] + [strategy_fa(n) for n in B.MAIN],
              [[("ثابت" if c == "fixed" else "شناور")] + [pct(cheap[c][n], 1) for n in B.MAIN] for c in B.CONTRACTS]))
    add("")
    inner = [n for n in names if n.startswith("partial") and n not in ("partial_0", "partial_100")]
    beat = {c: int((ok[[f"{c}.{n}.cost" for n in inner]].min(axis=1)
                    < ok[[f"{c}.now.cost", f"{c}.payment_day.cost"]].min(axis=1) - 1e-12).sum()) for c in B.CONTRACTS}
    add("سهم‌های میانی (٪۱۰ تا ٪۹۰) در این شمارش نیامده‌اند، چون هزینه‌شان بین «همه امروز» و «همه روز پرداخت» "
        f"است: یک سهم میانی فقط در {fa(beat['fixed'])} تاریخ (قرارداد ثابت) و {fa(beat['floating'])} تاریخ (شناور) از هر "
        "دو ارزان‌تر شد.\n")
    add(f"![حاشیهٔ سود واقعی]({f1})\n")
    add("*شکل ۱ — قرارداد ثابت: حاشیهٔ سودی که هر راهبرد واقعاً ساخت، برای هر تاریخ تصمیم. خط‌چین: کف. زرد: "
        "خریدِ «حداقل نسبت پوشش» مدل (بخش ۲).*\n")
    add(f"![نرخ شکست کف]({f2})\n")
    add(f"*شکل ۲ — سهم تاریخ‌هایی که کف شکست، برای هر سهمِ خرید در روز تصمیم، خرید سه‌پله‌ای و انتخاب مدل. بالا: "
        f"قرارداد ثابت؛ پایین: شناور. خط‌چین: ۵٪. نوار خاکستری: دامنهٔ دوجمله‌ای {band_fa(*band)}.*\n")
    shares = sorted((float(n.split("_")[1]) / 100, fx.loc[n, "breach"]) for n in names if n.startswith("partial"))
    under5 = min(h_ for h_, b_ in shares if all(b2 < 0.05 for h2, b2 in shares if h2 >= h_))
    zero = min(h_ for h_, b_ in shares if all(b2 == 0 for h2, b2 in shares if h2 >= h_))
    add(f"* **قرارداد ثابت:** هر ده واحد پوششِ بیشتر، نرخ شکست را پایین می‌آورد؛ از {pct(under5, 0)} به بالا زیر ۵٪ است "
        f"و از {pct(zero, 0)} به بالا صفر.")
    add(f"* **قرارداد شناور:** حتی پوشش کامل نرخ شکست را زیر {pct(fl.loc['now', 'breach'], 0)} نمی‌آورد. ریسک قهوه "
        "با خرید ارز برداشته نمی‌شود.")
    add(f"* **هزینهٔ صبر:** CVaR95 «همه روز پرداخت» {pct(fx.loc['payment_day', 'cvar'], 0)} برنامه است (ثابت)؛ یعنی در "
        "۵٪ بدترین هفته‌ها فاکتور بیش از یک‌ونیم برابرِ برنامه تمام شد.\n")

    add("### سال به سال\n")
    add("سالِ هر ردیف سالِ تاریخ تصمیم است. نرخ شکست کف:\n")
    rows = []
    for key in yr_fixed.index:
        rows.append(["**همه**" if key == "all" else fa(key), fa(int(yr_fixed.loc[key, "starts"])),
                     fa(int(yr_fixed.loc[key, "independent"]))]
                    + [pct(yr_fixed.loc[key, n], 1) for n in B.MAIN] + [pct(yr_float.loc[key, n], 1) for n in B.MAIN])
    add(table(["سال", "تاریخ‌ها", "پنجرهٔ مستقل"] + [f"ثابت: {strategy_fa(n)}" for n in B.MAIN]
              + [f"شناور: {strategy_fa(n)}" for n in B.MAIN], rows))
    add("")
    for c, name in (("fixed", "ثابت"), ("floating", "شناور")):
        add(f"هزینه (درصد برنامه)، قرارداد {name}: میانگین / میانه / CVaR95\n")
        rows = []
        for key in ystats[c].index:
            r = ystats[c].loc[key]
            rows.append(["**همه**" if key == "all" else fa(int(key))]
                        + [f"{pct(r[f'{n}.mean'], 0)} / {pct(r[f'{n}.median'], 0)} / {pct(r[f'{n}.cvar'], 0)}"
                           for n in B.MAIN])
        add(table(["سال"] + [strategy_fa(n) for n in B.MAIN], rows))
        add("")
    add("در هر خانه سه عدد است، به ترتیب از راست: میانگین، میانه، CVaR95. CVaR95 یک سال میانگینِ یک تا سه گران‌ترین "
        "هفتهٔ همان سال است.\n")

    # ------------------------------------------------------------------ 2
    add("## ۲. پیروی از مدل\n")
    add("در هر تاریخ، موتور فقط با دادهٔ تا همان روز اجرا شد. **قرارداد ثابت:** «حداقل نسبت پوشش» h* همان روز خریده "
        "شد و بقیه روز پرداخت. **قرارداد شناور:** راهبردِ منتخبِ قاعدهٔ تصمیم (D-035، D-037)؛ اگر هیچ راهبردی مجاز "
        "نبود، راهبردی با بیشترین احتمال، با برچسب «ناشدنی». بعد همان انتخاب روی آنچه واقعاً رخ داد قیمت‌گذاری شد.\n")
    add("**قرارداد ثابت**\n")
    rows = []
    for m in B.METHODS:
        r = model[m, "fixed"]
        rows.append([f"مدل، {METHOD_FA[m]}", pct(ok[f"{m}.h_star"].mean(), 0), pct(r["cost"], 1), pct(r["median"], 1),
                     pct(r["cvar"], 0), pct(r["breach"], 1), inside_fa(r["breach"], *band), pct(r["worst"], 0)])
    for n in B.MAIN:
        r = fx.loc[n]
        rows.append([strategy_fa(n), "—", pct(r["mean_cost"], 1), pct(r["median_cost"], 1), pct(r["cvar"], 0),
                     pct(r["breach"], 1), "—", pct(r["worst_margin"], 0)])
    add(table(["راهبرد", "میانگین h*", "میانگین هزینه", "میانه", "CVaR95", "شکست کف",
               f"نسبت به دامنهٔ {band_fa(*band)}", "بدترین حاشیه"], rows))
    add("")
    h = ok["fhs.h_star"]
    broke = ok["fhs.fixed.margin"] < floor
    add(f"* **وعده و واقعیت:** مدل در هر تاریخ وعده داد که با h* احتمالِ ماندن بالای کف دست‌کم {pct(conf, 0)} است "
        f"(میانگین وعدهٔ روش فیلترشده: {pct(ok['fhs.h_promise'].mean(), 1)}). در واقعیت کف در {pct(mf['breach'], 1)} "
        f"تاریخ‌ها شکست ({fa(int(broke.sum()))} تاریخ از {fa(len(ok))}).")
    add(f"* **h* چقدر بود:** میانگین {pct(h.mean(), 0)}، میانه {pct(h.median(), 0)}، از {pct(h.min(), 0)} تا {pct(h.max(), 0)}. "
        f"در {pct((h == 0).mean(), 0)} تاریخ‌ها مدل گفت پوششی لازم نیست. در تاریخ‌هایی که کف شکست میانگین h* "
        f"{pct(h[broke].mean(), 0)} بود.")
    add(f"* **هزینه:** پیروی از مدل (فیلترشده) به‌طور میانگین {pct(mf['cost'], 1)} برنامه تمام شد؛ بین «همه امروز» "
        f"({pct(fx.loc['now', 'mean_cost'], 1)}) و «همه روز پرداخت» ({pct(fx.loc['payment_day', 'mean_cost'], 1)}).\n")
    add("نرخ شکستِ مدل (قرارداد ثابت) به تفکیک رژیم نوسان در تاریخ تصمیم و سال:\n")
    rows = []
    for label, mask in (("همه", np.ones(len(ok), dtype=bool)), ("پرنوسان", high), ("کم‌نوسان", low)):
        part = ok[mask]
        n_i = cal.independent_windows(part.index, T)
        b_ = cal.binomial_range(n_i)
        rows.append([label, fa(len(part)), fa(n_i), band_fa(*b_), pct(part["fhs.h_star"].mean(), 0),
                     pct(breach("fhs.fixed.margin", part), 1), pct(part["plain.h_star"].mean(), 0),
                     pct(breach("plain.fixed.margin", part), 1), pct(breach("fixed.payment_day.margin", part), 1)])
    add(table(["رژیم", "تاریخ‌ها", "پنجرهٔ مستقل", "دامنهٔ دوجمله‌ای", "فیلترشده: میانگین h*", "فیلترشده: شکست کف",
               "ساده: میانگین h*", "ساده: شکست کف", "همه روز پرداخت: شکست کف"], rows))
    add("")
    rows = []
    for key in yr_model.index:
        rows.append(["**همه**" if key == "all" else fa(key), fa(int(yr_model.loc[key, "starts"])),
                     fa(int(yr_model.loc[key, "independent"])), pct(yr_model.loc[key, "fhs"], 1),
                     pct(yr_model.loc[key, "plain"], 1), pct(yr_model.loc[key, "late"], 1)])
    add(table(["سال", "تاریخ‌ها", "پنجرهٔ مستقل", "مدل فیلترشده: شکست کف", "مدل ساده: شکست کف",
               "همه روز پرداخت: شکست کف"], rows))
    add("")
    add(f"![h* در زمان]({f4})\n")
    add("*شکل ۳ — بالا: حداقل نسبت پوششی که مدل در هر تاریخ خواست (آبی: فیلترشده؛ نارنجی: بوت‌استرپ ساده). "
        "نقطه‌های سیاه: تاریخ‌هایی که با پیروی از مدلِ فیلترشده، کف ۱۳ هفته بعد شکست. پایین: نرخ دلار، مقیاس "
        "لگاریتمی.*\n")
    add("* روش فیلترشده با نوسان بالا و پایین می‌رود: در دورهٔ آرام نزدیک صفر، پس از هر جهش بالا. روش ساده کُند "
        "حرکت می‌کند.")
    add("* نقطه‌های سیاه درست پیش از جهش‌ها و در آغازشان جمع شده‌اند، وقتی h* هنوز پایین بود. مدل جهش را پس از "
        "دیدنش به حساب می‌آورد، نه پیش از آن.")
    add(f"* در نیمهٔ کم‌نوسان، مدلِ فیلترشده تقریباً همان «همه روز پرداخت» بود: میانگین h* "
        f"{pct(ok.loc[low, 'fhs.h_star'].mean(), 0)} و شکست کف {pct(breach('fhs.fixed.margin', ok[low]), 1)} در برابر "
        f"{pct(breach('fixed.payment_day.margin', ok[low]), 1)}. تا پیش از {jd(ok.index[(ok['fhs.h_star'] > 0).values][0])} "
        "هر دو روش h* را صفر دادند.\n")

    add("**قرارداد شناور**\n")
    rows = []
    for m in B.METHODS:
        f_ = feas[m]
        r = model[m, "floating"]
        n_f = cal.independent_windows(ok[f_].index, T)
        b_f = cal.binomial_range(n_f)
        rows.append([f"مدل، {METHOD_FA[m]}", pct(f_.mean(), 0), pct(r["cost"], 1), pct(r["cvar"], 0), pct(r["breach"], 1),
                     f"{pct(breach(f'{m}.floating.margin', ok[f_]), 1)} ({fa(int(f_.sum()))} تاریخ، {fa(n_f)} مستقل، "
                     f"دامنه {band_fa(*b_f)})",
                     pct(breach(f"{m}.floating.margin", ok[~f_]), 1)])
    for n in ("now", "payment_day"):
        r = fl.loc[n]
        rows.append([strategy_fa(n), "—", pct(r["mean_cost"], 1), pct(r["cvar"], 0), pct(r["breach"], 1), "—", "—"])
    add(table(["راهبرد", "تاریخ‌های «شدنی»", "میانگین هزینه", "CVaR95", "شکست کف، همهٔ تاریخ‌ها",
               "شکست کف وقتی مدل گفت «شدنی» (وعده: حداکثر ۵٪)", "شکست کف وقتی گفت «ناشدنی»"], rows))
    add("")
    top = ok["fhs.floating.choice"].value_counts()
    add(f"* مدلِ فیلترشده در {pct(1 - feas['fhs'].mean(), 0)} تاریخ‌ها گفت هیچ راهبردی قید را برآورده نمی‌کند. انتخابش "
        f"در {pct(top.iloc[0] / len(ok), 0)} تاریخ‌ها «{strategy_fa(top.index[0])}» بود؛ برای همین نتیجه‌اش به «همه امروز» "
        "نزدیک است.")
    add(f"* وقتی گفت «شدنی»، کف در {pct(breach('fhs.floating.margin', ok[feas['fhs']]), 1)} موارد شکست: دو برابرِ وعده، "
        "هرچند با این تعداد پنجرهٔ مستقل هنوز داخل دامنهٔ دوجمله‌ای است. نشانه‌ای است که مدل ریسک قرارداد شناور "
        "را کم می‌بیند (بخش ۳).\n")

    # ------------------------------------------------------------------ 3
    add("## ۳. کالیبراسیون (D-045)\n")
    add("در هر تاریخ، مقدار واقعی با صدک ۵ و ۹۵ شبیه‌سازیِ همان روز مقایسه شد. شبیه‌سازِ کالیبره از هر طرف حدود ٪۵ "
        "موارد بیرون می‌افتد.\n")
    add(f"* **رژیم نوسان:** «پرنوسان» = نوسان EWMA دلار در تاریخ تصمیم بالاتر از میانهٔ همین نوسان روی همهٔ تاریخ‌ها "
        f"({pct(ok['sigma'].median(), 2)} در هفته). این مرز پیش از دیدن نتیجه ثبت شده بود. امروز "
        f"({pct(ok['sigma'].iloc[-1], 2)} در آخرین تاریخ تصمیم) در نیمهٔ پرنوسان است.")
    add("* **زیان چندکی (pinball):** معیاری که هم بیرون‌افتادن از نوار را جریمه می‌کند و هم پهنای بی‌جهتِ نوار را. "
        "برای صدک ۹۵: اگر واقعیت بالاتر از صدک باشد ۰.۹۵ × فاصله، وگرنه ۰.۰۵ × فاصله (و برعکس برای صدک ۵). کمتر "
        "بهتر است. واحد: برای دلار واحد درصدِ تغییر؛ برای هزینه واحد درصدِ برنامه.")
    add("* **دامنهٔ دوجمله‌ای** هر ردیف با تعداد پنجره‌های مستقلِ همان ردیف حساب شده است.\n")
    for target in B.TARGETS:
        c_ = calib[target]
        add(f"**{TARGET_FA[target]}**\n")
        rows = []
        for subset in ("all", "high", "low"):
            for m in B.METHODS:
                r = c_.loc[(subset, m)]
                rows.append([SUBSET_FA[subset], METHOD_FA[m], fa(int(r["starts"])), fa(int(r["independent"])),
                             band_fa(r["band_lo"], r["band_hi"]),
                             pct(r["above"], 1) + ("" if r["band_lo"] <= r["above"] <= r["band_hi"] else " ✗"),
                             pct(r["below"], 1) + ("" if r["band_lo"] <= r["below"] <= r["band_hi"] else " ✗"),
                             num(r["pinball05"] * 100, 2), num(r["pinball95"] * 100, 2), num(r["pinball"] * 100, 2),
                             pct(r["width"], 0)])
        add(table(["رژیم", "روش", "تاریخ‌ها", "پنجرهٔ مستقل", "دامنهٔ دوجمله‌ای", "بالای صدک ۹۵", "زیر صدک ۵",
                   "زیان چندکی، صدک ۵", "زیان چندکی، صدک ۹۵", "میانگین دو زیان", "میانهٔ پهنای نوار"], rows))
        add("")
    add("علامت ✗ یعنی نرخ بیرون از دامنهٔ دوجمله‌ایِ همان ردیف است. جدول دوم (قرارداد ثابت) همان نرخ‌های جدول دلار را "
        "دارد: هزینهٔ «همه روز پرداخت» با قرارداد ثابت همان تغییر دلار است به واحدی دیگر.\n")
    add(f"![کالیبراسیون به تفکیک رژیم]({f3})\n")
    add("*شکل ۴ — تغییر ۱۳ هفته‌ای دلار: سهم تاریخ‌هایی که واقعیت بالای صدک ۹۵ (چپ) یا زیر صدک ۵ (وسط) افتاد، و "
        "میانگین زیان چندکی (راست)، برای همهٔ تاریخ‌ها و دو رژیم. آبی: فیلترشده؛ نارنجی: بوت‌استرپ ساده. خط‌چین: ۵٪؛ "
        "نقطه‌چین: سقف دامنهٔ دوجمله‌ای.*\n")
    u = calib["usd"]
    fl_c = calib["pd_floating"]
    better = lambda s: "فیلترشده" if u.loc[(s, "fhs"), "pinball"] < u.loc[(s, "plain"), "pinball"] else "بوت‌استرپ ساده"
    add("**خواندن جدول‌ها**\n")
    add(f"* **هفته‌های پرنوسان (وضع امروز):** روش فیلترشده نزدیک هدف است (بالای صدک ۹۵: "
        f"{pct(u.loc[('high', 'fhs'), 'above'], 1)}؛ زیر صدک ۵: {pct(u.loc[('high', 'fhs'), 'below'], 1)}). بوت‌استرپ ساده "
        f"جهش‌ها را دو برابرِ هدف از دست می‌دهد ({pct(u.loc[('high', 'plain'), 'above'], 1)}). زیان چندکی هم به نفع "
        f"{better('high')} است ({num(u.loc[('high', 'fhs'), 'pinball'] * 100, 2)} در برابر "
        f"{num(u.loc[('high', 'plain'), 'pinball'] * 100, 2)})، با اینکه نوارش پهن‌تر است.")
    add(f"* **هفته‌های کم‌نوسان:** برعکس. نوار فیلترشده زیادی باریک است: واقعیت در "
        f"{pct(u.loc[('low', 'fhs'), 'above'], 1)} موارد از بالا و {pct(u.loc[('low', 'fhs'), 'below'], 1)} از پایین بیرون "
        f"افتاد. بوت‌استرپ ساده نزدیک‌تر به هدف است ({pct(u.loc[('low', 'plain'), 'above'], 1)} و "
        f"{pct(u.loc[('low', 'plain'), 'below'], 1)}). زیان چندکی دو روش تقریباً برابر است "
        f"({num(u.loc[('low', 'fhs'), 'pinball'] * 100, 2)} و {num(u.loc[('low', 'plain'), 'pinball'] * 100, 2)}).")
    add(f"* **روی هم:** زیان چندکی {better('all')} کمتر است ({num(u.loc[('all', 'fhs'), 'pinball'] * 100, 2)} در برابر "
        f"{num(u.loc[('all', 'plain'), 'pinball'] * 100, 2)}). نرخ‌های کلی همان‌هایی‌اند که پیش‌نمایش D3b داد.")
    add(f"* **قرارداد شناور:** هر دو روش سمت بالا را دست‌کم می‌گیرند (فیلترشده {pct(fl_c.loc[('all', 'fhs'), 'above'], 1)}، "
        f"ساده {pct(fl_c.loc[('all', 'plain'), 'above'], 1)}) و سمت پایین را دست‌بالا "
        f"({pct(fl_c.loc[('all', 'fhs'), 'below'], 1)} و {pct(fl_c.loc[('all', 'plain'), 'below'], 1)}). یعنی هزینهٔ واقعی "
        "بیشتر از آنچه شبیه‌ساز می‌گفت به سمت گران‌تر کج بوده است. قهوه برای هر دو روش یکسان (بوت‌استرپ ساده) "
        "شبیه‌سازی می‌شود، پس این ضعف مشترک است.")
    add("* **اندازهٔ نمونه:** هر رژیم فقط "
        f"{fa(int(u.loc[('high', 'fhs'), 'independent']))} پنجرهٔ مستقل دارد و دامنهٔ دوجمله‌ای‌اش پهن است "
        f"({band_fa(u.loc[('high', 'fhs'), 'band_lo'], u.loc[('high', 'fhs'), 'band_hi'])}). جهتِ تفاوت‌ها روشن است؛ "
        "اندازهٔ دقیقشان نه.\n")

    # ------------------------------------------------------------------ 4
    add("## ۴. جمع‌بندی برای یادداشت\n")
    add(f"1. **اهرم اول کار می‌کند.** با قرارداد ثابت، هر چه سهم خریدِ روز تصمیم بیشتر، شکست کف کمتر: از "
        f"{pct(fx.loc['payment_day', 'breach'], 1)} (هیچ) تا {pct(fx.loc['now', 'breach'], 1)} (کامل). این روی "
        f"{fa(len(ok))} هفتهٔ واقعی است، نه شبیه‌سازی.")
    add(f"2. **بدون اهرم دوم کافی نیست.** با قرارداد شناور بهترین راهبرد هم در {pct(fl.loc['now', 'breach'], 1)} "
        "هفته‌ها کف را شکست.")
    add(f"3. **عدد مدل را باید «حداقل» خواند، نه «کافی».** پیروی از h* نرخ شکست را از "
        f"{pct(fx.loc['payment_day', 'breach'], 1)} به {pct(mf['breach'], 1)} رساند، نه به ۵٪؛ و در دورهٔ آرام پیش از "
        "جهش کمکی نکرد.")
    add("4. **برای امروز، روش فیلترشده انتخاب بهتری است.** در هفته‌های پرنوسان کالیبره‌تر از بوت‌استرپ ساده بود و "
        "نرخ شکستش نزدیک وعده ماند. ضعفش در هفته‌های آرام است، که وضع امروز نیست.\n")

    # ---------------------------------------------------------------- D5b
    add("## D5b. کفِ تنش (D-051): یک نمایش، نه شاهد\n")
    add("**هشدار.** فکرِ کفِ تنش پس از دیدن نتیجه‌های همین بک‌تست آمد (مدل جهشِ پس از دورهٔ آرام را نمی‌بیند). پس "
        "آزمودنش روی همین داده شاهدِ بیرون از نمونه نیست. عددهای این بخش فقط نشان می‌دهند قاعده چه می‌کند.\n")
    add("**قاعده (بدون پارامتر، فقط گذشته).** در هر تاریخ تصمیم:\n")
    add("1. بزرگ‌ترین رشد ۱۳ هفته‌ای دلار که تا همان روز در داده دیده شده است؛")
    add("2. **کفِ تنش** = سهمی که با قرارداد ثابت و ورودی‌های همان روز، از تکرارِ همان رشد جان به در می‌بَرد "
        "(فرمول تنش معکوس گزارش D4: h ≥ (x − هزینهٔ مجاز) ÷ (x − F))؛")
    add("3. پوشش = بزرگ‌ترِ این دو: h* مدل (روش فیلترشده) و کفِ تنش.\n")
    add(f"تا پیش از {jd(jump)} بزرگ‌ترین رشدِ دیده‌شده {pct(rise.iloc[0], 0, True)} بود و کفِ تنش کوچک "
        f"(میانگین {pct(ok.loc[ok.index < jump, 'stress.floor'].mean(), 0)}). از {jd(rise_top)} که جهش ۱۳۹۷ "
        f"({pct(rise.max(), 0, True)}) کامل در تاریخچه آمد، کف تقریباً {pct(ok.loc[ok.index >= rise_top, 'stress.floor'].mean(), 0)} "
        "شد و همان‌جا ماند.\n")
    rows = []
    for sub, mask in (("all", np.ones(len(ok), dtype=bool)), ("high", high), ("low", low)):
        part = ok[mask]
        n_i = cal.independent_windows(part.index, T)
        for lab, _, _ in floor_rows:
            r = fs[sub, lab]
            rows.append([SUBSET_FA[sub], lab, fa(len(part)), fa(n_i), "—" if np.isnan(r["hedge"]) else pct(r["hedge"], 0),
                         pct(r["cost"], 1), pct(r["cvar"], 0), pct(r["breach"], 1), pct(r["worst"], 0)])
    add(table(["رژیم", "راهبرد", "تاریخ‌ها", "پنجرهٔ مستقل", "میانگین پوشش", "میانگین هزینه", "CVaR95", "شکست کف",
               "بدترین حاشیه"], rows))
    add("")
    add("قرارداد ثابت. هزینه درصدی از برنامه است. «همه امروز» یعنی پوشش ٪۱۰۰.\n")
    add(f"![کفِ تنش]({f5})\n")
    add("*شکل ۵ — سهمی که در هر تاریخ خریده می‌شد: آبی، فقط h* مدل؛ نارنجی، با کفِ تنش. نقطه‌های سیاه: تاریخ‌هایی "
        "که با وجودِ کف، حاشیهٔ سود ۱۳ هفته بعد زیر کف رفت.*\n")
    m_all, f_all = fs["all", "فقط مدل (h*)"], fs["all", "مدل با کفِ تنش"]
    still = "، ".join(f"{fa(int(y))} ({pct(v, 0)})" for y, v in yr_floor.drop(index="all")["floor"].items() if v > 0)
    add(f"* **چه کرد:** نرخ شکست از {pct(m_all['breach'], 1)} به {pct(f_all['breach'], 1)} رسید، با میانگین پوشش "
        f"{pct(f_all['hedge'], 0)} به‌جای {pct(m_all['hedge'], 0)}. بیشترِ بهبود در هفته‌های کم‌نوسان است "
        f"({pct(fs['low', 'فقط مدل (h*)']['breach'], 1)} به {pct(fs['low', 'مدل با کفِ تنش']['breach'], 1)})، همان‌جا که مدل "
        "شکست می‌خورد.")
    add(f"* **هزینه:** میانگین هزینه {pct(f_all['cost'], 1)} برنامه شد (فقط مدل: {pct(m_all['cost'], 1)}). در این "
        "تاریخچه پوششِ بیشتر ارزان‌تر هم تمام شد، چون زود خریدن به‌طور میانگین ارزان‌تر بود (بخش ۱). این را نباید "
        "به آینده تعمیم داد.")
    add(f"* **کجا هنوز شکست:** {still}. کفِ تنش فقط از رشدی محافظت می‌کند که پیش‌تر دیده شده؛ جهش ۱۳۹۷ بزرگ‌تر از "
        "هر چیزی بود که تا آن روز در داده بود.")
    add(f"* **قاعدهٔ زمختی است.** از {jd(rise_top)} به بعد کف هیچ‌وقت کمتر از "
        f"{pct(ok.loc[ok.index >= rise_top, 'stress.floor'].min(), 0)} نبود؛ یعنی «تقریباً همه را امروز بخر». "
        f"«همه امروز» در همین بک‌تست هیچ شکستی نداشت و «سه‌پله‌ای» {pct(fs['all', 'سه‌پله‌ای']['breach'], 1)}. ارزشِ کفِ "
        "تنش در ماشین‌حساب این است که سناریو را مدیریت انتخاب می‌کند (D-051)، نه اینکه از این جدول بهتر باشد.\n")

    add("**حساسیت به سناریوی کف**\n")
    add(f"قاعدهٔ بالا «بدترین جهشِ دیده‌شده تا آن روز» را مبنا می‌گیرد؛ از {jd(rise_top)} این همان جهش ۱۳۹۷ است "
        f"({fa(s1_dates)} تاریخ از {fa(len(ok))}). اگر به‌جای آن جهش ۱۴۰۱–۱۴۰۲ (S2، دلار {pct(s2_rise, 0, True)} در ۱۳ هفته) "
        f"مبنا باشد چه می‌شود؟ همان {fa(len(ok))} تاریخ، قرارداد ثابت، پوشش = بزرگ‌ترِ h* مدل و کف.\n")
    add(table(["کف", "میانگین پوشش", "میانگین هزینه", "شکست کف"], [
        ["بدون کف (فقط مدل)", pct(sens["model"]["hedge"], 0), pct(sens["model"]["cost"], 1), pct(sens["model"]["breach"], 1)],
        ["بدترین جهشِ دیده‌شده تا آن روز (قاعدهٔ این بخش)", pct(sens["worst"]["hedge"], 0), pct(sens["worst"]["cost"], 1),
         pct(sens["worst"]["breach"], 1)],
        ["S2، فقط پس از رخ دادنش (فقط گذشته)", pct(sens["past"]["hedge"], 0), pct(sens["past"]["cost"], 1),
         pct(sens["past"]["breach"], 1)],
        [f"S2 در همهٔ تاریخ‌ها (پیش از {jd(s2_end)} به آینده نگاه می‌کند؛ آزمون منصفانه نیست)",
         pct(sens["always"]["hedge"], 0), pct(sens["always"]["cost"], 1), pct(sens["always"]["breach"], 1)],
    ]))
    add("")
    add(f"* **ردیف «فقط گذشته»:** پنجرهٔ S2 در {jg(s2_end)} تمام شد. پیش از آن ({fa(s2_before)} تاریخ) کف صفر است و "
        f"پوشش همان h* مدل؛ از آن روز کف سهمی است که از تکرار S2 جان به در می‌بَرد "
        f"({pct(s2_variants['past'].loc[s2_variants['past'].index >= s2_end, 'floor'].min(), 0)} تا "
        f"{pct(s2_variants['past'].loc[s2_variants['past'].index >= s2_end, 'floor'].max(), 0)}، بسته به ورودی‌های همان روز).")
    add(f"* **خواندن:** کفِ S2 به‌شکلِ منصفانه‌اش نرخ شکست را به {pct(sens['past']['breach'], 1)} می‌رساند، که هنوز بالای "
        f"هدف ۵٪ است؛ کفِ «بدترینِ دیده‌شده» به {pct(sens['worst']['breach'], 1)}. ردیف آخر ({pct(sens['always']['breach'], 1)}) "
        "بهتر به نظر می‌رسد ولی در بیشترِ تاریخ‌ها از جهشی استفاده می‌کند که هنوز رخ نداده بود.")
    add("* همهٔ این ردیف‌ها نمایش‌اند: قاعده پس از دیدن نتیجهٔ «فقط مدل» انتخاب شد و از پیش ثبت نشده بود.\n")

    # ------------------------------------------------------------------ 5
    add("## ۵. تصمیم‌ها\n")
    add("سه بندِ زیر در بازبینی D5 تصمیم گرفته شدند. تصمیم تازه‌ای لازم نیست.\n")
    add("**۵-۱. فاصلهٔ وعده و واقعیت**")
    add(f"* دیده شد: پیروی از h* (فیلترشده، قرارداد ثابت) در {pct(mf['breach'], 1)} تاریخ‌ها کف را شکست؛ وعده ۵٪ بود. "
        f"داخل دامنهٔ دوجمله‌ای است ({band_fa(*band)})، ولی در نیمهٔ کم‌نوسان {pct(breach('fhs.fixed.margin', ok[low]), 1)} "
        "بود.")
    add(f"* **تصمیم (D-048):** موتور دوباره تنظیم نمی‌شود. یادداشت هر دو عدد را می‌گوید (وعده ۵٪، واقعیت "
        f"{pct(mf['breach'], 1)}) و می‌گوید مدل کجا شکست می‌خورد: هفته‌های آرامِ پیش از جهش.")
    add("* **تصمیم (D-051):** «کفِ تنش» به‌عنوان گزینهٔ سیاستی: پوشش = بزرگ‌ترِ h* مدل و سهمی که از سناریوی "
        "تنشِ منتخبِ مدیریت جان به در می‌بَرد. به ماشین‌حساب می‌رود؛ نمایش آن در بخش D5b است.")
    add("* **تصمیم (D-053):** پیش‌فرض کفِ تنش در ماشین‌حساب همان قاعدهٔ بخش D5b است: بزرگ‌ترین جهش واقعیِ ۱۳ "
        f"هفته‌ای دلار که تا تاریخ تصمیم دیده شده. امروز این جهش ۱۳۹۷ است ({pct(ok['stress.max_rise'].max(), 0, True)}). "
        "S2 در فهرست می‌ماند ولی پیش‌فرض نیست (جدول حساسیت بخش D5b).\n")
    add("**۵-۲. روش نوسان**")
    add(f"* دیده شد: زیان چندکی روی هم به نفع روش فیلترشده است ({num(u.loc[('all', 'fhs'), 'pinball'] * 100, 2)} در برابر "
        f"{num(u.loc[('all', 'plain'), 'pinball'] * 100, 2)}) و در هفته‌های پرنوسان با فاصلهٔ بیشتر. در هفته‌های کم‌نوسان "
        "نوار فیلترشده زیادی باریک است.")
    add("* **تصمیم (D-049):** روش فیلترشده پیش‌فرض می‌ماند.\n")
    add("**۵-۳. ریسک قهوه در شبیه‌ساز**")
    add(f"* دیده شد: با قرارداد شناور، هزینهٔ واقعی در {pct(fl_c.loc[('all', 'fhs'), 'above'], 1)} تاریخ‌ها بالای صدک ۹۵ "
        f"شبیه‌سازی بود؛ و وقتی مدل گفت «شدنی»، کف در {pct(breach('fhs.floating.margin', ok[feas['fhs']]), 1)} موارد "
        "شکست. بخشی از این از دلار می‌آید و بخشی از قهوه، که بوت‌استرپ ساده با رانش صفر است (D-029).")
    add("* **تصمیم (D-050):** کالیبراسیون جداگانهٔ قهوه انجام نمی‌شود. جمله‌های یادداشت دربارهٔ قرارداد شناور از "
        "واقعیت‌های بک‌تست می‌آیند، نه از احتمال‌های مدل؛ ماشین‌حساب برای قرارداد شناور هشدار نشان می‌دهد.\n")

    (OUT / "README.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    return str(OUT / "README.md")


def main() -> None:
    print(build())


if __name__ == "__main__":
    main()
