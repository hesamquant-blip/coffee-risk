#!/usr/bin/env python
"""Build the D2 report: reports/d2_eda/README.md (Persian) and its figures.

D2 measures the facts the simulator (D3) will be built on. It builds no
simulator and no strategy. Every number, table and figure is computed here
from data/processed/panel_weekly.parquet (and the daily panel where needed):

    .venv/bin/python scripts/d2_report.py
"""
from __future__ import annotations

import datetime as dt

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from coffee_risk import eda, paths
from coffee_risk.reporting import (AQUA, BLUE, GRID, INK2, ORANGE, end_label, fa, jd, jg, num,
                                   pct, style, table)
from coffee_risk.data import ico, panel as P, weekly

OUT = paths.REPORTS / "d2_eda"
H = eda.HORIZON
ANN = np.sqrt(eda.WEEKS_PER_YEAR)
FA = {"usd": "دلار", "arabica": "عربیکا", "robusta": "روبوستا", "fund": "صندوق",
      "cost_arabica": "هزینهٔ ریالی، ۱۰۰٪ عربیکا", "cost_robusta": "هزینهٔ ریالی، ۱۰۰٪ روبوستا",
      "cost_mix": "هزینهٔ ریالی، ۵۰/۵۰", "coffee_mix": "قهوهٔ ۵۰/۵۰ (دلاری)"}
EN = {"usd": "USD/IRR (Bonbast)", "arabica": "arabica", "robusta": "robusta"}
COLOR = {"usd": ORANGE, "arabica": BLUE, "robusta": AQUA}
PERIOD_FA = {"2012-10 .. 2017": "۲۰۱۲-۱۰ تا ۲۰۱۷", "2018 .. 2022": "۲۰۱۸ تا ۲۰۲۲", "2023 .. now": "۲۰۲۳ تا امروز"}
EP_FA = {"fx_1397": "جهش ارز ۱۳۹۷", "fx_1401_1402": "جهش ارز ۱۴۰۱–۱۴۰۲", "coffee_2021": "قهوه ۲۰۲۱",
         "coffee_2024": "قهوه ۲۰۲۴", "war": "دورهٔ جنگ"}
EP_EN = {"fx_1397": "FX 1397", "fx_1401_1402": "FX 1401-02", "coffee_2021": "coffee 2021",
         "coffee_2024": "coffee 2024", "war": "war"}
SAMPLE_FA = {"full": "کل", "backtest": "بک‌تست"}


def lp(x, d: int = 1, sign: bool = True) -> str:
    """A log change shown as a percentage of log points."""
    return pct(x, d, sign)


def sp(x, d: int = 0, sign: bool = True) -> str:
    """A log change shown as the simple percentage change it equals."""
    return pct(np.exp(x) - 1, d, sign) if not (x is None or np.isnan(x)) else "—"


def _save(fig, name: str) -> str:
    fig.savefig(OUT / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return name


def _dates(ax) -> None:
    loc = mdates.AutoDateLocator(minticks=5, maxticks=10)
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))


# --------------------------------------------------------------------- figures
def fig_vol(rv: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    for s in eda.RISKY:
        y = rv[s].dropna() * 100
        ax.plot(y.index, y.values, color=COLOR[s], lw=1.5, label=EN[s])
        end_label(ax, y.index[-1], y.iloc[-1], f"{y.iloc[-1]:.0f}")
    ax.set_ylim(bottom=0)
    ax.set_ylabel("annualised volatility, %")
    ax.set_title("Rolling 26-week volatility of weekly log returns", pad=26)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, borderaxespad=0.2)
    _dates(ax)
    return _save(fig, "fig1_rolling_volatility.png")


def fig_carry(c: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(9.5, 4.0))
    y = c["carry"] * 100
    ax.fill_between(y.index, 0, y.values, where=y.values >= 0, color=ORANGE, alpha=0.35, lw=0,
                    label="carry > 0: buying USD at t beat the fund")
    ax.fill_between(y.index, 0, y.values, where=y.values < 0, color=BLUE, alpha=0.35, lw=0,
                    label="carry < 0: the fund beat USD")
    ax.plot(y.index, y.values, color=INK2, lw=0.8)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_ylabel("13-week carry, log %")
    ax.set_xlabel("start week t")
    ax.set_title("13-week carry by start week: USD change minus fund return", pad=26)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, borderaxespad=0.2)
    _dates(ax)
    return _save(fig, "fig2_carry.png")


def fig_hist(ch: pd.DataFrame) -> str:
    fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.4), sharey=True)
    bins = np.linspace(-60, 100, 41)
    for ax, s in zip(axes, eda.RISKY):
        x = ch[s].dropna() * 100
        ax.hist(x.clip(bins[0], bins[-1]), bins=bins, color=COLOR[s], alpha=0.85, lw=0)
        ax.axvline(0, color=INK2, lw=0.8)
        ax.axvline(x.median(), color=INK2, lw=1.2, ls="--")
        ax.set_title(EN[s], fontsize=10)
        ax.set_xlabel("13-week log change, %")
    axes[0].set_ylabel("overlapping windows")
    fig.suptitle("13-week log changes, full sample (dashed line = median)", x=0.01, ha="left",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    return _save(fig, "fig3_13week_changes.png")


def fig_rolling_corr(r: pd.DataFrame, window: int = 104) -> str:
    fig, ax = plt.subplots(figsize=(9.5, 3.8))
    pairs = [("usd", "arabica", ORANGE, "USD - arabica"), ("usd", "robusta", AQUA, "USD - robusta"),
             ("arabica", "robusta", BLUE, "arabica - robusta")]
    for a, b, color, label in pairs:
        y = r[a].rolling(window, min_periods=int(window * 0.8)).corr(r[b]).dropna()
        ax.plot(y.index, y.values, color=color, lw=1.5, label=label)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_ylim(-0.6, 1.0)
    ax.set_ylabel("correlation of weekly returns")
    ax.set_title(f"Rolling {window}-week correlation of weekly log returns", pad=26)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, borderaxespad=0.2)
    _dates(ax)
    return _save(fig, "fig4_rolling_correlation.png")


def fig_acf(r: pd.DataFrame) -> str:
    fig, axes = plt.subplots(2, 3, figsize=(9.5, 4.8), sharex=True, sharey=True)
    lags = np.arange(1, H + 1)
    for j, s in enumerate(eda.RISKY):
        band = eda.acf_band(r[s])
        for i, (series, label) in enumerate(((r[s], "returns"), (r[s].abs(), "absolute returns"))):
            ax = axes[i, j]
            ax.bar(lags, eda.acf(series).values, width=0.55, color=COLOR[s])
            ax.axhspan(-band, band, color=GRID, alpha=0.9, lw=0, zorder=0)
            ax.axhline(0, color=INK2, lw=0.8)
            ax.set_title(f"{EN[s]}: {label}", fontsize=9.5)
            if i == 1:
                ax.set_xlabel("lag, weeks")
    axes[0, 0].set_ylabel("autocorrelation")
    axes[1, 0].set_ylabel("autocorrelation")
    fig.suptitle("Autocorrelation of weekly returns (grey band = approximate 95% range for no memory)",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout()
    return _save(fig, "fig5_autocorrelation.png")


def fig_cost(idx: pd.DataFrame, stale: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    for col, color, label in (("cost_arabica", BLUE, "100% arabica"), ("cost_robusta", AQUA, "100% robusta"),
                              ("cost_mix", ORANGE, "50/50")):
        y = (idx[col] - idx[col].dropna().iloc[0]).where(~stale[col])
        ax.plot(y.index, np.exp(y.values) * 100, color=color, lw=1.5, label=label)
    ax.set_yscale("log")
    ax.set_ylabel("rial cost index, first week = 100 (log scale)")
    ax.set_title("Rial cost of coffee = USD/IRR x coffee price (roll-adjusted)", pad=26)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, borderaxespad=0.2)
    _dates(ax)
    return _save(fig, "fig6_rial_cost_index.png")


def fig_stress(idx: pd.DataFrame, stale: pd.DataFrame, windows: dict) -> str:
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    top = 0
    for col, color, label in (("usd", ORANGE, "USD/IRR (Bonbast)"),
                              ("coffee_mix", BLUE, "50/50 coffee index, USD terms (roll-adjusted)")):
        y = np.exp((idx[col] - idx[col].dropna().iloc[0]).where(~stale[col])) * 100
        ax.plot(y.index, y.values, color=color, lw=1.5, label=label)
        top = max(top, np.nanmax(y.values))
    for k, (key, end) in enumerate(windows.items()):
        start = eda.window_start(end)
        ax.axvspan(start, end, color=INK2, alpha=0.16, lw=0)
        ax.annotate(EP_EN[key], (start, top * (1.0 if k % 2 == 0 else 0.55)), xytext=(2, 0),
                    textcoords="offset points", fontsize=7.5, color=INK2, va="top")
    ax.set_yscale("log")
    ax.set_ylabel("index, first week = 100 (log scale)")
    ax.set_title("USD and the 50/50 coffee index, with the stress windows shaded", pad=26)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, borderaxespad=0.2)
    _dates(ax)
    return _save(fig, "fig7_stress_windows.png")


# ------------------------------------------------------------------ the report
def build() -> str:
    OUT.mkdir(parents=True, exist_ok=True)
    style(plt)
    wk = weekly.load()
    r = eda.weekly_returns(wk)
    rc = eda.rial_cost_returns(r)
    idx, stale, gaps = eda.log_index(wk)
    ch = eda.horizon_changes(idx, stale, gaps)
    bt0 = pd.Timestamp(eda.BACKTEST_START)
    last = wk.index[-1]
    L: list[str] = []
    add = L.append

    # ----------------------------------------------------------- computations
    rv = pd.DataFrame({s: eda.rolling_vol(r[s], 26) for s in eda.RISKY})
    rv_year = rv.groupby(eda.jyears(rv.index)).agg(["mean", "max"])
    cur26 = {s: eda.current_vol(r[s], 26) for s in eda.RISKY}
    cur13 = {s: eda.current_vol(r[s], 13) for s in eda.RISKY}
    full_vol = {s: float(r[s].std() * ANN) for s in eda.RISKY}

    drift_year = r[["usd", "fund"]].groupby(eda.jyears(r.index)).agg(["mean", "count"])
    c_all = eda.carry(idx, stale, gaps)
    c_bt = eda.sample(c_all, "backtest")
    c_year = pd.DataFrame({y: eda.describe_carry(g["carry"]) for y, g in c_bt.groupby(eda.jyears(c_bt.index))}).T
    c_tot = eda.describe_carry(c_bt["carry"])
    fund_stale_bt = int(stale.loc[bt0:, "fund"].sum())

    shapes = {(smp, s): eda.shape(eda.sample(r[s], smp)) for smp in ("full", "backtest") for s in eda.SERIES}
    usd_r = r["usd"].dropna()
    rises, falls = usd_r.nlargest(10), usd_r.nsmallest(10)
    q_cols = ["usd", "arabica", "robusta", "fund", "cost_arabica", "cost_robusta", "cost_mix"]
    quant = {(smp, s): eda.quantiles(eda.sample_windows(ch[s], smp)) for smp in ("full", "backtest") for s in q_cols}

    names = list(eda.SERIES)
    corr = {}
    for label, a, b in [("full", None, None)] + [(p[0], p[1], p[2]) for p in eda.PERIODS]:
        x = r if a is None else r[(r.index >= pd.Timestamp(a)) & ((r.index <= pd.Timestamp(b)) if b else True)]
        corr[label] = (x.corr(), x.corr(method="spearman"), x.notna().all(axis=1).sum(), len(x))
    pairs = [("usd", "arabica"), ("usd", "robusta"), ("arabica", "robusta"), ("usd", "fund")]
    nov = {p: eda.nonoverlapping_corr(ch, *p) for p in pairs}
    tails = {(g, o): eda.tail_comovement(r, g, o) for g, o in
             (("usd", "arabica"), ("usd", "robusta"), ("arabica", "usd"), ("robusta", "usd"))}

    acfs = {s: (eda.acf(r[s]), eda.acf(r[s].abs()), eda.acf_band(r[s])) for s in names}
    blocks = {(smp, s, kind): eda.block_length(eda.sample(r[s], smp) if kind == "ret"
                                               else eda.sample(r[s], smp).abs())
              for smp in ("full", "backtest") for s in names for kind in ("ret", "abs")}
    volpred = {s: eda.vol_predictability(r[s]) for s in eda.RISKY}
    vr = {(smp, s): eda.variance_ratio(eda.sample(r[s], smp)) for smp in ("full", "backtest") for s in eda.RISKY}

    cost_shape = {m: eda.shape(rc[f"cost_{m}"]) for m in eda.MIXES}
    worst = eda.top_nonoverlapping(ch["cost_mix"], 10)

    stress, coffee_extra = {}, {}
    for key, kind, a, b in eda.EPISODES:
        stress[key] = eda.best_window(ch["cost_mix"], a, b)
        if kind == "coffee":
            coffee_extra[key] = {s: eda.best_window(ch[s], a, b) for s in ("arabica", "robusta")}
    war_path = eda.path_change(idx, stale, gaps, pd.Timestamp(eda.WAR_START_WEEK), last)

    pool = eda.pool(r)
    pool_year = pool.groupby(eda.jyears(pool.index)).agg(["sum", "size"])
    war_weeks = pool[pool.index > pd.Timestamp(eda.WAR_START_WEEK)]
    n_pool = int(pool.sum())
    missing_pool = [d for d in pool.index[~pool] if d != wk.index[0]]
    ico_idx = wk.index[wk["robusta_return_source"] == "ico"]
    ico_weeks = len(ico_idx)
    ico_rep = ico.report()
    ico_exc = ico_rep[ico_rep["exception"] != ""]
    ico_test = weekly.robusta_inputs(P.load(), wk.index)["test"]
    # the longest run of consecutive ICO-filled weeks = the big hole in TGJU's robusta data
    runs = (pd.Series(ico_idx).diff() != pd.Timedelta(weeks=1)).cumsum()
    hole = ico_idx[(runs == runs.value_counts().idxmax()).values] if ico_weeks else ico_idx
    hole_ico = float(wk.loc[hole, "robusta_usd_kg_ret"].sum()) if len(hole) else np.nan
    hole_raw = float(wk.loc[hole, "robusta_usd_kg_ret_raw"].sum()) if len(hole) else np.nan
    jg_usd = {smp: eda.jensen_gap(eda.sample_windows(ch["usd"], smp)) for smp in ("full", "backtest")}
    sig = {w: float(r["usd"].dropna().iloc[-w:].std() * np.sqrt(H)) for w in (13, 26)}

    # drift options (decision 3)
    mu = {smp: float(eda.sample(r["usd"], smp).mean()) for smp in ("full", "backtest")}
    fund_now = float(ch["fund"].dropna().iloc[-1])
    fund_avg = float(eda.sample_windows(ch["fund"], "backtest").mean())
    med_hist = float(ch["usd"].median())
    mean_hist = float(ch["usd"].mean())

    # ----------------------------------------------------------------- figures
    f1 = fig_vol(rv)
    f2 = fig_carry(c_bt)
    f3 = fig_hist(ch)
    f4 = fig_rolling_corr(r)
    f5 = fig_acf(r)
    f6 = fig_cost(idx, stale)
    f7 = fig_stress(idx, stale, stress)

    # ================================================================== text
    today = dt.date.today()
    add("# گزارش مرحلهٔ D2 — شناخت داده‌ها")
    add("")
    add(f"تاریخ ساخت: {jg(today)}. همهٔ عددها، جدول‌ها و شکل‌ها را `scripts/d2_report.py` از پنل هفتگی ساخته است. "
        "این مرحله فقط اندازه می‌گیرد: نه شبیه‌ساز، نه راهبرد، نه تنظیم پارامتر.")
    add("")
    add("**قراردادها**")
    add("")
    add(f"* **افق:** {num(H)} هفته، یعنی سه ماه.")
    add(f"* **نمونهٔ «کل»:** همهٔ {num(len(wk))} چهارشنبه از {jg(wk.index[0])} تا {jg(last)}. "
        f"**نمونهٔ «بک‌تست»:** از هفتهٔ {jg(bt0)} (D-015)، {num(len(eda.sample(wk, 'backtest')))} هفته. هر عدد می‌گوید از کدام نمونه است.")
    add("* **دلار** = Bonbast. **قهوه** = آتی ICE، بدون روزهای رول. **صندوق** = شاخص بازده کل صندوق درآمد ثابت «اعتماد».")
    add("* **بازدهٔ لگاریتمی** لگاریتم نسبت دو قیمت است. برای تغییرهای کوچک همان درصد تغییر است؛ برای بزرگ‌ها نه: تغییر لگاریتمی ٪۷۰ یعنی رشد ٪۱۰۱. "
        "در جدول‌ها هر جا «لگاریتمی» نوشته، عدد لگاریتمی است؛ ستون‌های «درصد ساده» تبدیل‌شده‌اند.")
    add(f"* **آمار هفتگی** (نوسان، همبستگی، مخزن) از بازده‌های هفتگی پنل است، با {num(ico_weeks)} هفتهٔ روبوستا که از ICO پر شده (D-020).")
    add(f"* **تغییر {num(H)} هفته‌ای:** برای دلار و صندوق از سطح (پنجره‌ای که اول یا آخرش روی سطح کهنه بیفتد خالی است). "
        f"برای قهوه همه‌جا جمعِ همان بازده‌های هفتگی است: بدون روزهای رول و با هفته‌های پرشده از ICO (D-022). پنجره‌ای که هفته‌ای بدون بازده داشته باشد خالی است.")
    add("")
    add("**هشدار نگاه به آینده.** هر چیزی که بعداً داخل بک‌تست به کار برود (طول بلوک، سطح نوسان، میانگین رانش) اگر روی کل نمونه برآورد شده باشد، در بک‌تست از آینده خبر دارد. "
        "این موارد در متن با علامت ⚠ مشخص شده‌اند.")
    add("")

    # ---- what changed in D2b
    add("## آنچه در D2b عوض شد")
    add("")
    add("این گزارش پس از بازبینی D2 دوباره ساخته شده است. سه تغییر:")
    add("")
    for r_ in ico_exc.itertuples():
        add(f"1. **ماه {fa(f'{int(r_.year)}-{int(r_.month):02d}')} ICO با یک استثنا پذیرفته شد ({r_.exception}).** میانگین {num(r_.rows)} ردیفِ خوانده‌شده {num(r_.mean, 3)} است و PDF {num(r_.printed_average, 2)} چاپ کرده؛ قانون اعتبارسنجی عوض نشده و این ماه هنوز آن را رد می‌کند. "
            f"طبق تصمیم، {num(r_.rows_dropped)} ردیف ({'، '.join(str(d) for d in ico.EXCEPTIONS[(int(r_.year), int(r_.month))]['drop'])}، که عدد روز قبل را تکرار کرده) خالی شد و بقیهٔ ماه استفاده شد. استثنا به همین فایل (با sha256 آن) بسته است.")
    add(f"   * حالا {num(ico_weeks)} بازدهٔ هفتگی روبوستا از ICO می‌آید ({'، '.join(jd(d) for d in ico_idx)}) و هیچ هفته‌ای خالی نمانده است.")
    add(f"   * **مخزن بوت‌استرپ: {num(n_pool)} هفته از {num(len(pool))}** (فقط هفتهٔ اول بیرون است، چون بازده ندارد).")
    add(f"   * آزمون سازگاری ICO با TGJU با ماه تازه دوباره اجرا شد: {num(ico_test['weeks'])} هفته، همبستگی {num(ico_test['corr'], 3)}، نسبت نوسان {num(ico_test['vol_ratio'], 3)}؛ {'قبول' if ico_test['passed'] else 'رد'}.")
    add(f"2. **تغییر چندهفته‌ای قهوه همه‌جا یک‌جور ساخته می‌شود (D-022):** جمع بازده‌های هفتگیِ بدون رول، با هفته‌های ICO. دیگر از «دو سرِ سطح TGJU» استفاده نمی‌شود. "
        f"اثرش روی حفرهٔ بزرگ روبوستا ({jd(hole.min())} تا {jd(hole.max())}): جمع بازده‌های ICO {lp(hole_ico)} است، در برابر {lp(hole_raw)} از دو سرِ سطح TGJU. "
        f"اختلاف ({lp(hole_ico - hole_raw)}) پرش رولِ داخل حفره و تفاوت شاخص فیزیکی با آتی است که حالا بیرون مانده. "
        "این تغییر چندک‌های ۱۳ هفته‌ای روبوستا و ترکیب‌ها، بدترین پنجره‌ها، پنجره‌های تنش و مسیر جنگ را عوض می‌کند (بخش‌های ۳، ۶ و ۷).")
    add("3. **یک اندازه‌گیری تازه در بخش ۲:** فاصلهٔ «میانگین درصد تغییر» با «درصدِ میانگین لگاریتمی» برای دلار، و اینکه چرا رشد لگاریتمی برابر به معنای هزینهٔ مورد انتظار برابر نیست.")
    add("")
    add("تصمیم‌های بازبینی (D-021 تا D-027) در `DECISIONS.md` ثبت شده‌اند و در بخش ۹ کنار هر بند آمده‌اند. الحاقیهٔ D1b همان‌طور که بازبینی شد مانده است؛ وضعیت ICO در آن، پیش از D-021 است.")
    add("")

    # ---- summary
    add("## خلاصه")
    add("")
    u26, u13 = cur26["usd"], cur13["usd"]
    add(f"1. **دلار الان پرنوسان است، قهوه نه.** نوسان ۲۶ هفتهٔ اخیر دلار {pct(u26['vol'], 0)} در سال است، بالاتر از {pct(u26['percentile'], 0)} تاریخچه‌اش. "
        f"عربیکا در صدک {num(cur26['arabica']['percentile'] * 100)} و روبوستا در صدک {num(cur26['robusta']['percentile'] * 100)} است.")
    add(f"2. **زود خریدن دلار به‌طور میانگین برده، ولی نه همیشه.** در نمونهٔ بک‌تست، رشد ۱۳ هفته‌ای دلار در {pct(c_tot['share_positive'], 0)} شروع‌ها از سود صندوق بیشتر بوده؛ "
        f"میانگین اختلاف {lp(c_tot['mean'])} و میانه {lp(c_tot['median'])} است. دامنه بزرگ است: از {lp(c_tot['p5'])} تا {lp(c_tot['p95'])} (صدک ۵ تا ۹۵).")
    add(f"3. **دلار و قهوه با هم حرکت نمی‌کنند.** همبستگی بازدهٔ هفتگی دلار با عربیکا {num(corr['full'][0].loc['usd', 'arabica'], 2)} و با روبوستا {num(corr['full'][0].loc['usd', 'robusta'], 2)} است. "
        f"عربیکا و روبوستا با هم {num(corr['full'][0].loc['arabica', 'robusta'], 2)}.")
    bl_usd = blocks[("full", "usd", "ret")]["circular"]
    bl_abs = blocks[("full", "usd", "abs")]["circular"]
    add(f"4. **تلاطم دلار خوشه‌ای و ماندگار است.** طول بلوک بهینه برای اندازهٔ حرکت‌های دلار حدود {num(bl_abs)} هفته است، بلندتر از افق. "
        f"برای خودِ بازدهٔ دلار برآورد ناپایدار است ({num(bl_usd)} هفته روی کل نمونه، {num(blocks[('backtest', 'usd', 'ret')]['circular'])} هفته روی بک‌تست). قهوه حافظهٔ کوتاهی دارد.")
    w1 = eda.window_row(ch, worst.index[0])
    add(f"5. **بدترین سه‌ماهه برای هزینهٔ ریالی قهوه** (ترکیب ۵۰/۵۰): {jd(w1['start'])} تا {jd(w1['end'])}، با رشد {sp(w1['cost_mix'])} (دلار {sp(w1['usd'])}، قهوه {sp(w1['coffee_mix'])}). "
        f"از آغاز جنگ تا امروز هزینهٔ ریالی {sp(war_path['cost_mix'])} بالا رفته است.")
    add("6. **از پنج تصمیم D3، چهار تا گرفته شده** (بخش ۹): مخزن، طول بلوک، رفتار با نوسان و مجموعهٔ تنش. رانش دلار در اسپک D3 تعیین می‌شود.")
    add("")

    # ---- 1 (A)
    add("## ۱. نوسان در زمان")
    add("")
    add("«نوسان» انحراف معیار بازدهٔ لگاریتمی هفتگی است. برای اینکه با عددهای سالانه قابل مقایسه باشد در جذر ۵۲ ضرب شده («سالانه‌شده»). "
        "«نوسان غلتان ۲۶ هفته‌ای» همین عدد روی ۲۶ هفتهٔ آخر است که هر هفته از نو حساب می‌شود. نمونه: کل.")
    add("")
    rows = []
    for y in rv_year.index:
        row = [fa(y)]
        for s in eda.RISKY:
            row += [pct(rv_year.loc[y, (s, "mean")], 0), pct(rv_year.loc[y, (s, "max")], 0)]
        rows.append(row)
    add(table(["سال", "دلار: میانگین", "دلار: بیشینه", "عربیکا: میانگین", "عربیکا: بیشینه",
               "روبوستا: میانگین", "روبوستا: بیشینه"], rows))
    add("")
    add("هر خانه میانگین یا بیشینهٔ نوسان غلتان ۲۶ هفته‌ای در آن سال جلالی است (سالانه‌شده). سال اول و سال جاری ناقص‌اند.")
    add("")
    add("**نوسان فعلی و جایگاه آن در تاریخچه**")
    add("")
    add("«صدک» می‌گوید چند درصد از تاریخچهٔ همان سنجه پایین‌تر یا برابر عدد فعلی بوده است.")
    add("")
    rows = []
    for s in eda.RISKY:
        a, b = cur13[s], cur26[s]
        rows.append([FA[s], pct(a["vol"], 0), num(a["percentile"] * 100), pct(b["vol"], 0),
                     num(b["percentile"] * 100), pct(b["median"], 0), pct(full_vol[s], 0)])
    add(table(["سری", "۱۳ هفتهٔ اخیر", "صدک", "۲۶ هفتهٔ اخیر", "صدک", "میانهٔ تاریخی نوسان ۲۶ هفته‌ای",
               "نوسان کل نمونه"], rows))
    add("")
    add(f"تا هفتهٔ {jg(last)}. ⚠ صدک‌ها روی کل نمونه حساب شده‌اند.")
    add("")
    add("**آیا این دوره برای هر سری غیرعادی است؟**")
    add("")
    for s in eda.RISKY:
        p26 = cur26[s]["percentile"]
        verdict = ("بله، به‌وضوح بالاست" if p26 >= 0.85 else "کمی بالاتر از معمول" if p26 >= 0.65
                   else "نه، در محدودهٔ معمول است" if p26 >= 0.35 else "نه، پایین‌تر از معمول است")
        add(f"* **{FA[s]}:** {verdict}. نوسان ۲۶ هفته‌ای {pct(cur26[s]['vol'], 0)} است، در برابر میانهٔ تاریخی {pct(cur26[s]['median'], 0)}؛ "
            f"یعنی {num(cur26[s]['vol'] / cur26[s]['median'], 1)} برابر میانه و در صدک {num(p26 * 100)}.")
    add("")
    add(f"![نوسان غلتان]({f1})")
    add("")
    add("*شکل ۱ — نوسان غلتان ۲۶ هفته‌ای، سالانه‌شده، برای دلار، عربیکا و روبوستا.*")
    add("")

    # ---- 2 (B)
    add("## ۲. رانش و کری")
    add("")
    add("«رانش» میانگین بازده است: قیمت به‌طور متوسط هر هفته چقدر بالا می‌رود. این هستهٔ پرسشِ «دلار را الان بخرم یا بعداً» است: "
        "اگر دلار به‌طور متوسط تندتر از سود ریالی بالا برود، زود خریدن می‌ارزد.")
    add("")
    rows = []
    for y in drift_year.index:
        u, f = drift_year.loc[y, ("usd", "mean")], drift_year.loc[y, ("fund", "mean")]
        rows.append([fa(y), num(drift_year.loc[y, ("usd", "count")]), lp(u, 2), lp(u * 52, 0),
                     lp(f, 2) if not np.isnan(f) else "—", lp(f * 52, 0) if not np.isnan(f) else "—",
                     lp((u - f) * 52, 0) if not np.isnan(f) else "—"])
    for smp in ("full", "backtest"):
        x = eda.sample(r, smp)
        u, f = x["usd"].mean(), x["fund"].mean()
        rows.append([f"**{SAMPLE_FA[smp]}**", num(x["usd"].notna().sum()), lp(u, 2), lp(u * 52, 0),
                     lp(f, 2), lp(f * 52, 0), lp((u - f) * 52, 0)])
    add(table(["سال", "هفته", "دلار: میانگین هفتگی", "دلار: سالانه", "صندوق: میانگین هفتگی",
               "صندوق: سالانه", "اختلاف سالانه (دلار − صندوق)"], rows))
    add("")
    add("همه لگاریتمی. «سالانه» = میانگین هفتگی × ۵۲. میانگین صندوق روی هفته‌هایی است که بازدهٔ هفتگی دارد (هفته‌های نوروز طبق D-018 خالی‌اند). ⚠ میانگین‌های «کل» و «بک‌تست» از آینده خبر دارند.")
    add("")
    add(f"**کریِ {num(H)} هفته‌ای**")
    add("")
    add(f"«کری» را این‌طور تعریف می‌کنیم: c(t) = تغییر لگاریتمی دلار از هفتهٔ t تا t+{H}، منهای بازدهٔ لگاریتمی صندوق در همان {num(H)} هفته. "
        f"**c > 0 یعنی کسی که در هفتهٔ t دلار خریده، از کسی که ریالش را تا t+{H} در صندوق نگه داشته و بعد دلار خریده، جلوتر بوده است.** "
        "این یک واقعیتِ داده است، نه بک‌تست راهبرد.")
    add("")
    rows = [[fa(y), num(x["n"]), lp(x["mean"]), lp(x["median"]), lp(x["p5"]), lp(x["p95"]),
             pct(x["share_positive"], 0)] for y, x in c_year.iterrows()]
    rows.append(["**کل بک‌تست**", num(c_tot["n"]), lp(c_tot["mean"]), lp(c_tot["median"]), lp(c_tot["p5"]),
                 lp(c_tot["p95"]), pct(c_tot["share_positive"], 0)])
    add(table(["سال شروع", "تعداد شروع", "میانگین", "میانه", "صدک ۵", "صدک ۹۵", "سهم c > 0"], rows))
    add("")
    add(f"نمونه: بک‌تست؛ همهٔ شروع‌های هفتگی از {jd(c_bt.index.min())} تا {jd(c_bt.index.max())} (آخرین شروعی که {num(H)} هفته بعدش داده دارد). لگاریتمی. "
        f"پنجره‌های پشت‌سرهم {num(H - 1)} هفته هم‌پوشانی دارند، پس مستقل نیستند. "
        f"{num(fund_stale_bt)} چهارشنبه سطح صندوق کهنه دارد؛ پنجره‌هایی که از آن‌ها شروع یا به آن‌ها ختم می‌شوند کنار رفته‌اند.")
    add("")
    best_y, worst_y = c_year["share_positive"].idxmax(), c_year["share_positive"].idxmin()
    add(f"* در کل بک‌تست، زود خریدن در {pct(c_tot['share_positive'], 0)} شروع‌ها برده است: کمی بیشتر از نصف.")
    add(f"* میانگین ({lp(c_tot['mean'])}) خیلی بالاتر از میانه ({lp(c_tot['median'])}) است. یعنی بُردِ زود خریدن از چند جهش بزرگ می‌آید، نه از یک برتری همیشگی.")
    add(f"* سال به سال فرق زیاد است: در {fa(best_y)} سهم c > 0 برابر {pct(c_year.loc[best_y, 'share_positive'], 0)} و در {fa(worst_y)} فقط {pct(c_year.loc[worst_y, 'share_positive'], 0)}.")
    add(f"* صدک ۵ ({lp(c_tot['p5'])}) نشان می‌دهد زود خریدن هم می‌تواند گران تمام شود: دلار افت کرده و سود ریالی هم از دست رفته است.")
    add("")
    add("**رشد لگاریتمی برابر، هزینهٔ مورد انتظار برابر نیست**")
    add("")
    add(f"برای تغییر {num(H)} هفته‌ای دلار دو «میانگین» می‌شود گرفت: میانگینِ درصد تغییرها، و درصدی که از میانگینِ تغییرهای لگاریتمی به دست می‌آید. اولی همیشه بزرگ‌تر است.")
    add("")
    rows = []
    for smp in ("full", "backtest"):
        j = jg_usd[smp]
        rows.append([SAMPLE_FA[smp], num(j["n"]), lp(j["mean_log"]), pct(j["from_mean_log"], 1, True),
                     pct(j["mean_simple"], 1, True), pct(j["gap"], 2, True), pct(j["std_log"], 1),
                     pct(j["half_variance"], 2)])
    add(table(["نمونه", "پنجره", "میانگین تغییر لگاریتمی", "درصدِ همان (e به توان آن، منهای ۱)",
               "میانگین درصد تغییر", "فاصله", f"انحراف معیار تغییر {num(H)} هفته‌ای (σ)", "σ² تقسیم بر ۲"], rows))
    add("")
    rows = [[f"{num(w)} هفتهٔ اخیر", pct(sig[w] * np.sqrt(eda.WEEKS_PER_YEAR / H), 0), pct(sig[w], 1),
             pct(sig[w] ** 2 / 2, 2)] for w in (13, 26)]
    add(table(["نوسان فعلی از", "سالانه‌شده", f"σ برای {num(H)} هفته", "σ² تقسیم بر ۲"], rows))
    add("")
    add(f"جدول دوم همان فاصله را از نوسان *فعلی* می‌سازد: σ = انحراف معیار بازدهٔ هفتگی در هفته‌های اخیر، ضرب در جذر {num(H)}. ردیف ۱۳ هفته همان است که اسپک خواسته؛ ردیف ۲۶ هفته برای مقایسه است. پنجره‌های جدول اول هم‌پوشان‌اند.")
    add("")
    jf = jg_usd["full"]
    add(f"* **چرا؟** هزینهٔ ریالی با e به توانِ تغییر لگاریتمی بالا می‌رود، نه با خودِ آن. یک حرکتِ ٪۲۰ لگاریتمی رو به بالا هزینه را {pct(np.expm1(0.2), 0)} بیشتر می‌کند، ولی همان حرکت رو به پایین فقط {pct(-np.expm1(-0.2), 0)} کمتر. پس وقتی دلار نوسان دارد، میانگینِ هزینه از هزینهٔ «حالتِ میانی» بالاتر می‌افتد.")
    add(f"* **اندازه‌اش:** در تاریخچه {pct(jf['gap'], 1)} برای هر {num(H)} هفته (نمونهٔ کل)، نزدیک به σ²/۲ = {pct(jf['half_variance'], 1)}. با نوسان ۱۳ هفتهٔ اخیر {pct(sig[13] ** 2 / 2, 1)} است.")
    add(f"* **معنایش برای رانش (بند ۹-۳):** اگر در مدل فرض کنیم دلار به‌طور متوسط «به اندازهٔ سود صندوق رشد لگاریتمی می‌کند»، یعنی تقریباً نیمی از حالت‌ها بالاتر و نیمی پایین‌تر از صندوق‌اند (میانه تقریباً برابر). ولی *میانگینِ* هزینهٔ دلار خریدن در آینده هنوز حدود σ²/۲ از هزینهٔ الان خریدن (با احتساب سود صندوق) بیشتر است. "
        "پس «کری صفر» را باید روشن کرد: صفر در میانه (لگاریتمی) یا صفر در میانگین هزینه. این دو به اندازهٔ همین فاصله با هم فرق دارند.")
    add("")
    add(f"![کری]({f2})")
    add("")
    add(f"*شکل ۲ — کریِ {num(H)} هفته‌ای برای هر هفتهٔ شروع در نمونهٔ بک‌تست. نارنجی: دلار خریدن برده؛ آبی: صندوق برده.*")
    add("")

    # ---- 3 (C)
    add("## ۳. شکل توزیع")
    add("")
    add("«چولگی» می‌گوید توزیع به کدام سو کشیده است (مثبت = جهش‌های رو به بالا بزرگ‌ترند). «کشیدگی مازاد» می‌گوید دُم‌ها چقدر از توزیع نرمال سنگین‌ترند (صفر = مثل نرمال؛ عدد بزرگ = حرکت‌های شدید بیشتر از انتظار).")
    add("")
    rows = []
    for smp in ("full", "backtest"):
        for s in names:
            x = shapes[(smp, s)]
            rows.append([SAMPLE_FA[smp], FA[s], num(x["n"]), lp(x["mean"], 2), pct(x["std"]), num(x["skew"], 2),
                         num(x["excess_kurtosis"], 1), lp(x["min"]), lp(x["max"])])
    add(table(["نمونه", "سری", "هفته", "میانگین", "انحراف معیار", "چولگی", "کشیدگی مازاد", "کمینه", "بیشینه"], rows))
    add("")
    add("بازدهٔ لگاریتمی هفتگی.")
    add("")
    add(f"* دلار دُم بسیار سنگین دارد (کشیدگی مازاد {num(shapes[('full', 'usd')]['excess_kurtosis'], 1)}): بیشتر هفته‌ها آرام‌اند و چند هفته حرکت خیلی بزرگ دارند. قهوه به نرمال نزدیک‌تر است.")
    add("")
    add("**۱۰ رشد و ۱۰ افت بزرگ هفتگی دلار** (نمونه: کل، لگاریتمی):")
    add("")
    rows = [[num(i + 1), jg(rises.index[i]), lp(rises.iloc[i]), jg(falls.index[i]), lp(falls.iloc[i])]
            for i in range(10)]
    add(table(["رتبه", "هفتهٔ رشد", "رشد", "هفتهٔ افت", "افت"], rows))
    add("")
    n97 = sum(1 for d in list(rises.index) + list(falls.index) if eda.jyears([d])[0] == 1397)
    add(f"* {num(n97)} مورد از این ۲۰ هفته در ۱۳۹۷ است.")
    add("")
    add(f"**چندک‌های تغییر {num(H)} هفته‌ای** (لگاریتمی؛ پنجره‌های هم‌پوشان)")
    add("")
    rows = []
    for smp in ("full", "backtest"):
        for s in q_cols:
            q = quant[(smp, s)]
            rows.append([SAMPLE_FA[smp], FA[s], num(q["n"]), lp(q["q01"]), lp(q["q05"]), lp(q["q50"]),
                         lp(q["q95"]), lp(q["q99"]), sp(q["q95"])])
    add(table(["نمونه", "سری", "پنجره", "٪۱", "٪۵", "٪۵۰", "٪۹۵", "٪۹۹", "٪۹۵ به درصد ساده"], rows))
    add("")
    add(f"**این پنجره‌ها مستقل نیستند:** هر پنجره با پنجرهٔ بعدی {num(H - 1)} هفتهٔ مشترک دارد. تعداد پنجره‌های واقعاً جدا فقط حدود {num(len(ch['usd'].dropna()) // H)} تاست. پس چندک‌های ٪۱ و ٪۹۹ روی چند رویداد انگشت‌شمار تکیه دارند.")
    add("")
    add(f"![توزیع تغییر ۱۳ هفته‌ای]({f3})")
    add("")
    add(f"*شکل ۳ — توزیع تغییر لگاریتمی {num(H)} هفته‌ای برای دلار، عربیکا و روبوستا (نمونهٔ کل، پنجره‌های هم‌پوشان). خط‌چین میانه است.*")
    add("")

    # ---- 4 (D)
    add("## ۴. وابستگی")
    add("")
    add("همبستگی پیرسون رابطهٔ خطی را می‌سنجد و به چند هفتهٔ شدید حساس است. همبستگی اسپیرمن روی رتبه‌هاست و به آن‌ها حساس نیست. هر دو بین ۱− و ۱+.")
    add("")
    rows = []
    for label, (pe, spc, n_all, n) in corr.items():
        lab = "کل" if label == "full" else PERIOD_FA[label]
        for a, b in (("usd", "arabica"), ("usd", "robusta"), ("arabica", "robusta"), ("usd", "fund"),
                     ("arabica", "fund"), ("robusta", "fund")):
            rows.append([lab, f"{FA[a]} – {FA[b]}", num(pe.loc[a, b], 2), num(spc.loc[a, b], 2)])
    add(table(["دوره", "جفت", "پیرسون", "اسپیرمن"], rows))
    add("")
    add("بازدهٔ هفتگی؛ هر جفت روی هفته‌هایی که هر دو داده دارند. صندوق از ۱۳۹۴ داده دارد.")
    add("")
    add(f"**همبستگی تغییرهای {num(H)} هفته‌ایِ بدون هم‌پوشانی.** بسته به اینکه شمارش از کدام هفته شروع شود {num(H)} حالت هست؛ بازهٔ نتیجه‌ها:")
    add("")
    rows = [[f"{FA[a]} – {FA[b]}", num(v["min"], 2), num(v["median"], 2), num(v["max"], 2),
             f"{num(v['blocks_min'])} تا {num(v['blocks_max'])}"] for (a, b), v in nov.items()]
    add(table(["جفت", "کمینه", "میانه", "بیشینه", "تعداد پنجره در هر حالت"], rows))
    add("")
    add("**هم‌حرکتی در دُم.** در هفته‌هایی که یک سری بالاتر از صدک ۹۵ خودش بوده، سری دیگر چه کرده است؟")
    add("")
    rows = []
    for (g, o), v in tails.items():
        rows.append([f"{FA[g]} بالای صدک ۹۵ ({lp(v['threshold'])})", FA[o], num(v["weeks"]), lp(v["mean"], 2),
                     lp(v["median"], 2), pct(v["share_positive"], 0), lp(v["all_mean"], 2),
                     lp(v["all_median"], 2), pct(v["all_share_positive"], 0)])
    add(table(["شرط", "سری دیگر", "هفته", "میانگین", "میانه", "سهم مثبت", "میانگین در همهٔ هفته‌ها",
               "میانه در همهٔ هفته‌ها", "سهم مثبت در همه"], rows))
    add("")
    add("**آیا جهش دلار با جهش قهوه هم‌زمان می‌شود؟**")
    add("")
    ua, ur = tails[("usd", "arabica")], tails[("usd", "robusta")]
    add(f"* **نه.** در {num(ua['weeks'])} هفته‌ای که دلار بیشترین جهش را داشته، میانهٔ بازدهٔ عربیکا {lp(ua['median'], 2)} و روبوستا {lp(ur['median'], 2)} بوده است؛ "
        f"نزدیک به هفته‌های عادی ({lp(ua['all_median'], 2)} و {lp(ur['all_median'], 2)}).")
    add(f"* همبستگی هفتگی دلار با قهوه در هر سه دوره نزدیک صفر است و همبستگی {num(H)} هفته‌ای هم بین {num(min(nov[('usd', 'arabica')]['min'], nov[('usd', 'robusta')]['min']), 2)} و {num(max(nov[('usd', 'arabica')]['max'], nov[('usd', 'robusta')]['max']), 2)} می‌چرخد.")
    add("* این با انتظار هم می‌خواند: قیمت قهوه در بازار جهانی تعیین می‌شود و دلار آزاد در ایران. در داده، این دو مثل دو منبع ریسک جدا رفتار می‌کنند: در هزینهٔ ریالی روی هم جمع می‌شوند، ولی هم را تشدید نمی‌کنند.")
    add(f"* عربیکا و روبوستا با هم حرکت می‌کنند (هفتگی {num(corr['full'][0].loc['arabica', 'robusta'], 2)}، {num(H)} هفته‌ای میانه {num(nov[('arabica', 'robusta')]['median'], 2)}). پس ترکیب ۵۰/۵۰ فقط کمی از ریسک قهوه کم می‌کند.")
    add("")
    add(f"![همبستگی غلتان]({f4})")
    add("")
    add("*شکل ۴ — همبستگی غلتان ۱۰۴ هفته‌ای بازده‌های هفتگی.*")
    add("")

    # ---- 5 (E)
    add("## ۵. ماندگاری و طول بلوک")
    add("")
    add("«خودهمبستگی» می‌گوید بازدهٔ این هفته چقدر به بازدهٔ k هفته پیش شبیه است. خودهمبستگیِ «قدرمطلق بازده» می‌گوید آیا هفته‌های پرتلاطم پشت‌سرهم می‌آیند (خوشه‌ای بودن نوسان). "
        "اگر سری حافظه نداشته باشد، این عددها باید داخل نوار ۹۵٪ (حدود ±۱.۹۶ تقسیم بر جذر تعداد هفته) بمانند. نمونه: کل.")
    add("")
    rows = []
    for s in names:
        a, b, band = acfs[s]
        rows.append([FA[s], "بازده"] + [num(a[k], 2) for k in (1, 2, 3, 4, 8, 13)]
                    + [num(int((a.abs() > band).sum())), num(band, 2)])
        rows.append([FA[s], "قدرمطلق"] + [num(b[k], 2) for k in (1, 2, 3, 4, 8, 13)]
                    + [num(int((b.abs() > band).sum())), num(band, 2)])
    add(table(["سری", "نوع", "تأخیر ۱", "۲", "۳", "۴", "۸", "۱۳", f"تأخیرهای بیرون نوار (از {num(H)})", "نوار ±"], rows))
    add("")
    add("**طول بلوک بهینه (پولیتیس–وایت).** در «بوت‌استرپ بلوکی» به‌جای هفته‌های تک، تکه‌های پشت‌سرهم از تاریخ را برمی‌داریم تا حافظهٔ سری حفظ شود. «طول بلوک بهینه» را روش پولیتیس و وایت (تابع `optimal_block_length` در بستهٔ `arch`) از روی خودهمبستگی برآورد می‌کند. واحد: هفته.")
    add("")
    rows = []
    for smp in ("full", "backtest"):
        for s in names:
            a, b = blocks[(smp, s, "ret")], blocks[(smp, s, "abs")]
            rows.append([SAMPLE_FA[smp], FA[s], num(a["stationary"], 1), num(a["circular"], 1),
                         num(b["stationary"], 1), num(b["circular"], 1)])
    add(table(["نمونه", "سری", "بازده: ایستا", "بازده: دایره‌ای", "قدرمطلق: ایستا", "قدرمطلق: دایره‌ای"], rows))
    add("")
    add("«ایستا» و «دایره‌ای» دو گونهٔ بوت‌استرپ بلوکی‌اند (بلوک با طول تصادفی یا ثابت). هفته‌های خالی پیش از برآورد حذف شده‌اند. ⚠ برآورد روی کل نمونه از آینده خبر دارد؛ ردیف‌های «بک‌تست» هم روی همهٔ دورهٔ بک‌تست‌اند.")
    add("")
    risky_abs = {s: blocks[("full", s, "abs")]["circular"] for s in eda.RISKY}
    proposed = H
    add(f"**جمع {num(H)} هفته در برابر هفته‌های مستقل.** یک اندازه‌گیری مستقیم‌تر از حافظه: اگر هفته‌ها مستقل بودند، واریانسِ جمعِ {num(H)} هفته {num(H)} برابر واریانس یک هفته می‌شد («نسبت واریانس» = ۱) و کشیدگیِ آن جمع، کشیدگی هفتگی تقسیم بر {num(H)}.")
    add("")
    rows = []
    for smp in ("full", "backtest"):
        for s in eda.RISKY:
            v = vr[(smp, s)]
            rows.append([SAMPLE_FA[smp], FA[s], num(v["variance_ratio"], 2), num(v["kurtosis_sums"], 1),
                         num(v["kurtosis_if_independent"], 1)])
    add(table(["نمونه", "سری", "نسبت واریانس", f"کشیدگی مازاد جمع {num(H)} هفته", "همان، اگر هفته‌ها مستقل بودند"], rows))
    add("")
    add("جمع‌های هم‌پوشان. نسبت بالاتر از ۱ یعنی حرکت‌ها ادامه پیدا می‌کنند؛ کشیدگیِ بیشتر از ستون آخر یعنی هفته‌های آرام و پرتلاطم خوشه‌ای می‌آیند.")
    add("")
    add("**پیشنهاد یک طول بلوک مشترک**")
    add("")
    add("* بوت‌استرپ *مشترک* یعنی هفته‌های دلار، عربیکا و روبوستا با هم برداشته می‌شوند تا هم‌زمانی‌شان حفظ شود. پس یک طول بلوک برای هر سه لازم است و سریِ با حافظهٔ بلندتر تعیین‌کننده است.")
    add(f"* برآورد پولیتیس–وایت برای *بازدهٔ* دلار پایدار نیست: {num(blocks[('full', 'usd', 'ret')]['circular'], 0)} هفته روی کل نمونه و {num(blocks[('backtest', 'usd', 'ret')]['circular'], 0)} هفته روی بک‌تست. به این عدد نمی‌شود تکیه کرد. قهوه در هر دو نمونه {num(max(blocks[(m, c, 'ret')]['circular'] for m in ('full', 'backtest') for c in ('arabica', 'robusta')), 0)} هفته یا کمتر است.")
    add(f"* آنچه پایدار است خوشه‌ای بودن *تلاطم* دلار است: {num(risky_abs['usd'], 0)} هفته روی کل و {num(blocks[('backtest', 'usd', 'abs')]['circular'], 0)} روی بک‌تست؛ و کشیدگی جمع {num(H)} هفته‌ای دلار {num(vr[('full', 'usd')]['kurtosis_sums'], 1)} است، در برابر {num(vr[('full', 'usd')]['kurtosis_if_independent'], 1)} اگر هفته‌ها مستقل بودند. "
        f"نسبت واریانس دلار هم {num(vr[('full', 'usd')]['variance_ratio'], 2)} است.")
    add(f"* نتیجه: بلوک کوتاه (مثلاً ۴ هفته) هفته‌های آرام و بحرانی را قاطی می‌کند و دُم توزیع {num(H)} هفته‌ای دلار را کوچک‌تر از واقع نشان می‌دهد. خوشه‌ها از خودِ افق بلندترند، پس بلوکی به طول افق کافی است تا داخل هر مسیر حفظ شوند؛ بلندتر از آن فایده‌ای ندارد.")
    add(f"* **پیشنهاد: میانگین طول بلوک {num(proposed)} هفته** (برابر افق)، با بوت‌استرپ ایستا (طول تصادفی). "
        "بهایش این است که هر مسیر شبیه‌سازی‌شده تقریباً یک تکهٔ پیوسته از تاریخ است؛ تنوع مسیرها از تعداد نقطه‌های شروع می‌آید، نه از ترکیب تکه‌ها. سطح نوسان را «رفتار با نوسان» تنظیم می‌کند (بند ۹-۴).")
    add("")
    add(f"![خودهمبستگی]({f5})")
    add("")
    add(f"*شکل ۵ — خودهمبستگی بازده (ردیف بالا) و قدرمطلق بازده (ردیف پایین) تا تأخیر {num(H)} هفته. نوار خاکستری محدودهٔ تقریبی ۹۵٪ برای سریِ بی‌حافظه است.*")
    add("")

    # ---- 6 (F)
    add("## ۶. هزینهٔ ریالی قهوه")
    add("")
    add("هزینهٔ ریالی یک کیلو قهوه = قیمت دلاری قهوه × نرخ دلار. پس تغییر لگاریتمی هفتگی آن = بازدهٔ دلار + بازدهٔ قهوه. سه ترکیب: ۱۰۰٪ عربیکا، ۱۰۰٪ روبوستا، و ۵۰/۵۰ که هر هفته به نسبت ۵۰/۵۰ برمی‌گردد "
        "(بازدهٔ قهوه‌اش = لگاریتمِ نصفِ e به توان بازدهٔ عربیکا به‌اضافهٔ نصفِ e به توان بازدهٔ روبوستا). این فقط بخش «آتی» است؛ اضافه‌بهای فیزیکی جدا می‌آید (D-012).")
    add("")
    mix_fa = {"arabica": "۱۰۰٪ عربیکا", "robusta": "۱۰۰٪ روبوستا", "mix": "۵۰/۵۰"}
    rows = []
    for m in eda.MIXES:
        x = cost_shape[m]
        row = [mix_fa[m], pct(x["std"]), pct(x["std"] * ANN, 0)]
        for smp in ("full", "backtest"):
            q = quant[(smp, f"cost_{m}")]
            row += [lp(q["q05"]), lp(q["q50"]), lp(q["q95"]), lp(q["q99"])]
        rows.append(row)
    add(table(["ترکیب", "نوسان هفتگی", "سالانه‌شده", "کل: ٪۵", "کل: ٪۵۰", "کل: ٪۹۵", "کل: ٪۹۹",
               "بک‌تست: ٪۵", "بک‌تست: ٪۵۰", "بک‌تست: ٪۹۵", "بک‌تست: ٪۹۹"], rows))
    add("")
    add(f"چندک‌ها برای تغییر لگاریتمی {num(H)} هفته‌ای، پنجره‌های هم‌پوشان (جدول کامل در بخش ۳). نوسان هفتگی روی نمونهٔ کل.")
    add("")
    vu, vc = shapes[("full", "usd")]["std"], float(rc["coffee_mix"].std())
    add(f"* نوسان هفتگی دلار {pct(vu)} و قهوهٔ ۵۰/۵۰ {pct(vc)} است؛ هزینهٔ ریالی ۵۰/۵۰ {pct(cost_shape['mix']['std'])}. "
        f"چون این دو تقریباً ناهمبسته‌اند، نوسان‌ها به‌صورت «جذر مجموع مربع‌ها» جمع می‌شوند ({pct(float(np.hypot(vu, vc)))})، نه ساده.")
    add("")
    add(f"**۱۰ بدترین پنجرهٔ {num(H)} هفته‌ایِ جدا از هم، ترکیب ۵۰/۵۰** (نمونه: کل)")
    add("")
    rows = []
    for i, end in enumerate(worst.index):
        w = eda.window_row(ch, end)
        share = w["usd"] / w["cost_mix"] if w["cost_mix"] else np.nan
        rows.append([num(i + 1), jg(w["start"]), jg(w["end"]), sp(w["cost_mix"]), lp(w["cost_mix"]),
                     lp(w["usd"]), lp(w["coffee_mix"]), pct(share, 0)])
    add(table(["رتبه", "شروع", "پایان", "رشد هزینهٔ ریالی (ساده)", "همان، لگاریتمی", "سهم دلار (لگاریتمی)",
               "سهم قهوه (لگاریتمی)", "دلار از کل"], rows))
    add("")
    add("«جدا از هم» یعنی هیچ دو پنجره‌ای هفتهٔ مشترک ندارند: اول بدترین پنجره انتخاب می‌شود، بعد بدترینِ باقی‌مانده که با قبلی‌ها هم‌پوشانی ندارد، و همین‌طور. تغییر لگاریتمی هزینه دقیقاً جمع دو سهم است.")
    add("")
    by_usd = sum(1 for end in worst.index if ch.loc[end, "usd"] > ch.loc[end, "coffee_mix"])
    add(f"* در {num(by_usd)} پنجره از ۱۰، سهم دلار از سهم قهوه بزرگ‌تر است.")
    add("")
    add(f"![شاخص هزینهٔ ریالی]({f6})")
    add("")
    add("*شکل ۶ — شاخص هزینهٔ ریالی قهوه برای سه ترکیب، مقیاس لگاریتمی، اولین هفته = ۱۰۰.*")
    add("")

    # ---- 7 (G)
    add("## ۷. پنجره‌های تنش (D-005)")
    add("")
    add(f"برای هر رویداد، در بازهٔ جست‌وجویش پنجرهٔ {num(H)} هفته‌ای با بیشترین رشد هزینهٔ ریالی ۵۰/۵۰ پیدا شده است. پنجره باید کامل داخل بازه باشد. درصدها ساده‌اند (نه لگاریتمی).")
    add("")

    def stress_row(label, end):
        w = eda.window_row(ch, end)
        return [label, jg(w["start"]), jg(w["end"]), sp(w["usd"]), sp(w["arabica"]), sp(w["robusta"]),
                sp(w["cost_arabica"]), sp(w["cost_robusta"]), sp(w["cost_mix"]), sp(w["fund"], 1)]

    head = ["رویداد", "شروع", "پایان", "دلار", "عربیکا", "روبوستا", "هزینه: ۱۰۰٪ عربیکا",
            "هزینه: ۱۰۰٪ روبوستا", "هزینه: ۵۰/۵۰", "صندوق"]
    rows = []
    for key, kind, a, b in eda.EPISODES:
        if stress[key] is not None:
            rows.append(stress_row(EP_FA[key], stress[key]))
    add(table(head, rows))
    add("")
    add("بازه‌های جست‌وجو: " + "؛ ".join(
        f"{EP_FA[key]}: {jd(a)} تا {jd(b) if b else 'آخرین هفته'}" for key, kind, a, b in eda.EPISODES) + ".")
    add("")
    add("**دو رویداد قهوه: پنجره با بیشترین رشد خودِ عربیکا و خودِ روبوستا**")
    add("")
    rows = []
    for key, d in coffee_extra.items():
        for s, end in d.items():
            if end is not None:
                rows.append(stress_row(f"{EP_FA[key]} — بیشترین رشد {FA[s]}", end))
    add(table(head, rows))
    add("")
    add("**دورهٔ جنگ: کل مسیر از آغاز تا امروز**")
    add("")
    wp = war_path
    rows = [[jg(wp["start"]), jg(wp["end"]), num(wp["weeks"]), sp(wp["usd"]), sp(wp["arabica"]), sp(wp["robusta"]),
             sp(wp["cost_arabica"]), sp(wp["cost_robusta"]), sp(wp["cost_mix"]), sp(wp["fund"], 1)]]
    add(table(["شروع", "پایان", "هفته", "دلار", "عربیکا", "روبوستا", "هزینه: ۱۰۰٪ عربیکا",
               "هزینه: ۱۰۰٪ روبوستا", "هزینه: ۵۰/۵۰", "صندوق"], rows))
    add("")
    sw = stress["war"]
    add(f"* شروع، هفتهٔ {jd(eda.WAR_START_WEEK)} است: آخرین چهارشنبه پیش از ۹ اسفند ۱۴۰۴.")
    if sw is not None:
        inside = [d for d in ico_idx if eda.window_start(sw) < d <= sw]
        on_path = [d for d in ico_idx if d > pd.Timestamp(eda.WAR_START_WEEK)]
        add(f"* بدترین پنجرهٔ جنگ {num(len(inside))} هفته با بازدهٔ روبوستا از ICO دارد و کل مسیر جنگ {num(len(on_path))} هفته. تغییر روبوستا در آن‌ها جمع بازده‌های هفتگی است (D-022)، بدون پرش رولِ داخل حفره.")
    f1397 = eda.window_row(ch, stress["fx_1397"])
    add(f"* در جهش ۱۳۹۷ قهوه خودش ارزان شد (عربیکا {sp(f1397['arabica'])}، روبوستا {sp(f1397['robusta'])}) و بخشی از جهش دلار را جبران کرد. در رویدادهای قهوه، دلار هم بالا می‌رفت و دو ریسک روی هم افتادند.")
    add("")
    add(f"![پنجره‌های تنش]({f7})")
    add("")
    add("*شکل ۷ — دلار (نارنجی) و شاخص دلاری قهوهٔ ۵۰/۵۰ (آبی)، هر دو اولین هفته = ۱۰۰، مقیاس لگاریتمی. نوارهای خاکستری پنجره‌های تنش‌اند.*")
    add("")

    # ---- 8 (H)
    add("## ۸. مخزن بوت‌استرپ")
    add("")
    add("«مخزن» مجموعهٔ هفته‌هایی است که شبیه‌ساز از آن‌ها نمونه برمی‌دارد. برای بوت‌استرپ مشترک، هفته‌ای در مخزن است که بازدهٔ هفتگی دلار، عربیکا و روبوستا هر سه را داشته باشد.")
    add("")
    rows = [[fa(y), num(x["size"]), num(x["sum"]), pct(x["sum"] / n_pool, 1)] for y, x in pool_year.iterrows()]
    rows.append(["**جمع**", num(len(pool)), num(n_pool), "٪۱۰۰"])
    add(table(["سال", "هفته", "در مخزن", "سهم از مخزن"], rows))
    add("")
    p97 = int(pool_year.loc[1397, "sum"]) if 1397 in pool_year.index else 0
    add(f"* مخزن: {num(n_pool)} هفته از {num(len(pool))}. بیرون از مخزن: هفتهٔ اول (بازده ندارد)"
        + (f" و {num(len(missing_pool))} هفتهٔ روبوستا که خالی مانده ({'، '.join(jd(d) for d in missing_pool)})." if missing_pool else "."))
    add(f"* سهم ۱۳۹۷: {num(p97)} هفته، {pct(p97 / n_pool, 1)} مخزن. سهم دورهٔ جنگ (پس از {jd(eda.WAR_START_WEEK)}): {num(int(war_weeks.sum()))} هفته، {pct(war_weeks.sum() / n_pool, 1)}.")
    add(f"* {num(ico_weeks)} هفته از مخزن بازدهٔ روبوستا را از ICO دارد (D-020 و D-021).")
    add(f"* مخزنِ نمونهٔ بک‌تست: {num(int(eda.sample(pool, 'backtest').sum()))} هفته.")
    add("")

    # ---- 9
    add("## ۹. این‌ها برای D3 چه معنایی دارد")
    add("")
    add("برای هر تصمیم: آنچه اندازه گرفته شد، گزینه‌ها، پیشنهاد من، و آنچه در بازبینی تصمیم گرفته شد.")
    add("")
    add("**۹-۱. پنجرهٔ مخزن**")
    add(f"* اندازه‌گیری: مخزن کامل {num(n_pool)} هفته است؛ ۱۳۹۷ فقط {pct(p97 / n_pool, 1)} آن است ولی {num(n97)} مورد از ۲۰ حرکت شدید هفتگی دلار را دارد. نوسان دلار پیش از ۱۳۹۷ (۱۳۹۳ تا ۱۳۹۶) حدود {pct(rv_year.loc[1393:1396, ('usd', 'mean')].mean(), 0)} در سال بود و از ۱۳۹۷ به بعد حدود {pct(rv_year.loc[1397:, ('usd', 'mean')].mean(), 0)}.")
    add("* گزینه‌ها: (الف) همهٔ هفته‌ها از مهر ۱۳۹۱؛ (ب) از ۱۳۹۷ به بعد (رژیم پرنوسان فعلی)؛ (ج) پنجرهٔ غلتان، مثلاً ۵ سال آخر. در بک‌تست، هر کدام باید «تا آن روز» باشد: مخزنِ هر تاریخ فقط هفته‌های پیش از آن.")
    add("* پیشنهاد: **(الف)** برای ماشین‌حساب (بیشترین داده، شامل دورهٔ آرام و دورهٔ بحران)، با مخزنِ «تا آن روز» در بک‌تست. دورهٔ آرام ۱۳۹۳ تا ۱۳۹۶ را رفتار با نوسان (۹-۴) تعدیل می‌کند، نه حذف.")
    add("* **تصمیم (D-024):** مخزن = همهٔ هفته‌ها از مهر ۱۳۹۱؛ داخل بک‌تست فقط هفته‌های پیش از هر تاریخ تصمیم.")
    add("")
    add("**۹-۲. طول بلوک**")
    add(f"* اندازه‌گیری: بخش ۵. خوشه‌های تلاطم دلار حدود {num(risky_abs['usd'], 0)} هفته (پایدار در هر دو نمونه)؛ برآورد برای بازدهٔ دلار ناپایدار ({num(blocks[('full', 'usd', 'ret')]['circular'], 0)} در برابر {num(blocks[('backtest', 'usd', 'ret')]['circular'], 0)} هفته)؛ قهوه چند هفته.")
    add(f"* گزینه‌ها: (الف) ۴ هفته (مسیرهای متنوع‌تر، ولی خوشه‌های تلاطم دلار می‌شکند)؛ (ب) میانگین {num(proposed)} هفته، بوت‌استرپ ایستا؛ (ج) طول ثابت {num(H)} هفته = نمونه‌گیری از پنجره‌های تاریخی کامل.")
    add(f"* پیشنهاد: **(ب)**. این عدد از افق می‌آید، نه از برآوردی روی داده؛ پس در بک‌تست از آینده خبر ندارد و برای همهٔ تاریخ‌ها ثابت می‌ماند. ⚠ خودِ اندازه‌گیری‌های بخش ۵ روی کل نمونه‌اند. حساسیت نتیجه به ۴ و ۸ هفته در D3 گزارش شود.")
    add(f"* **تصمیم (D-025):** بوت‌استرپ ایستا با میانگین طول بلوک {num(proposed)} هفته؛ حساسیت در ۴ و ۸ هفته.")
    add("")
    add("**۹-۳. رانش دلار**")
    add("* اندازه‌گیری: آنچه هر گزینه برای تغییر ۱۳ هفته‌ای دلار و برای کری می‌گوید:")
    add("")
    rows = []
    for label, m in (("میانگین تاریخی، نمونهٔ کل ⚠", mu["full"]), ("میانگین تاریخی، نمونهٔ بک‌تست ⚠", mu["backtest"]),
                     ("صفر (بازده‌ها میانگین‌زدایی شوند)", 0.0),
                     ("برابر سود فعلی صندوق (کری صفر)", fund_now / H)):
        rows.append([label, lp(m * 52, 0), lp(m * H), lp(med_hist - mean_hist + m * H), lp(m * H - fund_now),
                     lp(m * H - fund_avg)])
    add(table(["رانش", "سالانه", f"میانگین تغییر {num(H)} هفته‌ای دلار", "میانهٔ تقریبی", "کری در برابر سود فعلی صندوق",
               "کری در برابر میانگین سود صندوق"], rows))
    add("")
    add(f"همه لگاریتمی. سود {num(H)} هفته‌ای صندوق: فعلی {lp(fund_now)}، میانگین بک‌تست {lp(fund_avg)}. «میانهٔ تقریبی» = میانهٔ تاریخی تغییر {num(H)} هفته‌ای دلار ({lp(med_hist)}) که به اندازهٔ تفاوت رانش جابه‌جا شده؛ "
        f"میانه پایین‌تر از میانگین ({lp(mean_hist)}) است چون توزیع به راست چوله است.")
    add("")
    add("* گزینه‌ها: (الف) میانگین تاریخی: می‌گوید زود خریدن به‌طور متوسط می‌برد، ولی این میانگین را چند جهش ساخته‌اند و از آینده خبر دارد؛ (ب) صفر: محافظه‌کارانه برای «زود خریدن»، چون فرض می‌کند دلار رشد نمی‌کند و فقط سود صندوق از دست می‌رود؛ (ج) ورودی کاربر، با یک پیش‌فرض.")
    add("* پیشنهاد: **(ج)**، با پیش‌فرضِ «رانش برابر سود فعلی صندوق» (کری صفر: نه زود خریدن از پیش برنده است، نه دیر خریدن) و نمایش نتیجه برای میانگین تاریخی و صفر در کنارش. این‌طور تصمیم از توزیع ریسک می‌آید، نه از یک فرض پنهان دربارهٔ جهت دلار.")
    add("* **وضعیت:** باز. در اسپک D3 تعیین می‌شود. اندازه‌گیری «رشد لگاریتمی برابر در برابر هزینهٔ مورد انتظار برابر» در بخش ۲ برای همین تصمیم است.")
    add("")
    add("**۹-۴. رفتار با نوسان**")
    add("* اندازه‌گیری:")
    add("")
    rows = []
    for s in eda.RISKY:
        v = volpred[s]
        rows.append([FA[s], pct(cur26[s]["vol"], 0), pct(full_vol[s], 0), num(cur26[s]["vol"] / full_vol[s], 2),
                     num(acfs[s][1][1], 2), num(v["spearman"], 2), num(v["n"])])
    add(table(["سری", "نوسان ۲۶ هفتهٔ اخیر", "نوسان کل نمونه", "نسبت", "خودهمبستگی قدرمطلق، تأخیر ۱",
               "همبستگی رتبه‌ای نوسان ۲۶ هفتهٔ گذشته با ۱۳ هفتهٔ بعد", "تعداد دوره"], rows))
    add("")
    add("ستون آخر روی دوره‌های ۱۳ هفته‌ایِ بدون هم‌پوشانی است: آیا نوسانِ اخیر چیزی دربارهٔ نوسانِ سه‌ماههٔ بعد می‌گوید؟ ⚠ نمونهٔ کل.")
    add("")
    add("* گزینه‌ها: (الف) بوت‌استرپ ساده: نوسان شبیه‌سازی‌شده میانگین تاریخ است، که برای دلار الان کم‌برآورد است؛ "
        "(ب) «شبیه‌سازی تاریخی فیلترشده»: هر بازدهٔ تاریخی بر نوسانِ همان زمانش تقسیم و در نوسان فعلی ضرب می‌شود، تا شکل توزیع از تاریخ و اندازه از امروز بیاید؛ "
        "(ج) فقط پنجرهٔ اخیر (مثلاً ۲ سال): ساده، ولی فقط حدود ۸ سه‌ماههٔ جدا دارد و دُم ۱۳۹۷ را ندارد.")
    add(f"* پیشنهاد: **(ب) برای دلار، (الف) برای قهوه.** دلیل: نوسان دلار ماندگار و قابل پیش‌بینی است (همبستگی {num(volpred['usd']['spearman'], 2)}) و الان {num(cur26['usd']['vol'] / full_vol['usd'], 1)} برابر میانگین تاریخی است؛ "
        f"برای قهوه نوسان فعلی نزدیک معمول است و ماندگاری‌اش ضعیف‌تر. نتیجهٔ بوت‌استرپ ساده هم برای مقایسه گزارش شود. ⚠ برآورد نوسانِ هر زمان باید فقط از دادهٔ پیش از آن باشد.")
    add("* **تصمیم (D-026):** شبیه‌سازی تاریخی فیلترشده برای دلار، با نوسانی که فقط از دادهٔ گذشته برآورد می‌شود؛ بوت‌استرپ ساده برای قهوه؛ نتیجهٔ بوت‌استرپ ساده هم گزارش می‌شود.")
    add("")
    add("**۹-۵. فهرست پنجره‌های تنش**")
    add("* اندازه‌گیری: جدول‌های بخش ۷.")
    add("* گزینه‌ها: (الف) پنج پنجرهٔ جدول اول، همان‌طور که هست؛ (ب) به‌علاوهٔ پنجره‌های «بیشترین رشد عربیکا/روبوستا» برای دو رویداد قهوه؛ (ج) به‌علاوهٔ کل مسیر جنگ به‌عنوان یک سناریوی بلندتر از افق.")
    add("* پیشنهاد: **(الف) + (ج)**. پنجره‌های (ب) با پنجره‌های اصلی هم‌پوشانی زیاد دارند.")
    add("* **تصمیم (D-027):** مجموعهٔ تنش = پنج پنجرهٔ جدول اول بخش ۷ به‌علاوهٔ کل مسیر جنگ.")
    add("")
    return "\n".join(L) + "\n"


def main() -> None:
    text = build()
    (OUT / "README.md").write_text(text, encoding="utf-8")
    print(f"wrote {OUT / 'README.md'} ({len(text.splitlines())} lines) and figures in {OUT}")


if __name__ == "__main__":
    main()
