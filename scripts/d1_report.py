#!/usr/bin/env python
"""Build the D1 report: reports/d1_data/README.md (Persian) and its figures.

Every number, table and figure in the report is computed here from the raw
cache and the panel.

FROZEN since D1b. This script produced the D1 report at the commit tagged
`d1`. D1b changed the panel (Bonbast became the primary dollar, TGJU's chart
endpoint was removed), so the script no longer runs against the current code,
and the D1 report must not be rewritten. To reproduce the D1 report:

    git checkout d1
    .venv/bin/python -m coffee_risk.data build-panel
    .venv/bin/python scripts/d1_report.py

What changed after D1 is reported in reports/d1_data/addendum_d1b.md, built
by scripts/d1b_addendum.py.
"""
from __future__ import annotations

import sys

if __name__ == "__main__":
    sys.exit("scripts/d1_report.py is frozen (see its docstring): "
             "check out the tag `d1` to reproduce the D1 report.")

import csv
import datetime as dt

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from coffee_risk import paths
from coffee_risk.dates import to_jalali
from coffee_risk import units
from coffee_risk.data import (bonbast, checks, common_calendar as cc, fred, fund, http,
                              investing, panel as P, quality, robots, rolls, tgju, tsetmc,
                              yahoo)

OUT = paths.REPORTS / "d1_data"
WAR_START = dt.date(2026, 2, 28)          # D-005: 9 Esfand 1404
ERAS = [(2008, 2012), (2013, 2016), (2017, 2020), (2021, 2026)]
MIN_DAYS_IN_MONTH = 15                    # a monthly mean needs this many daily values
EXPIRY_WINDOW = 13                        # trading days before a switch day
ROBUSTA_BETA = 0.5                        # share of the arabica return removed from robusta

# Known changes in the official FX channel. Month-level dates from general
# knowledge (the last one is from the spec); the author should confirm them.
REGIMES = [
    (dt.date(2018, 4, 1), "Apr 2018: single official rate", "فروردین ۱۳۹۷: اعلام نرخ رسمی واحد"),
    (dt.date(2018, 8, 1), "Aug 2018: NIMA secondary market", "مرداد ۱۳۹۷: بازار ثانویهٔ نیما"),
    (dt.date(2023, 2, 1), "Feb 2023: exchange centre opens", "اسفند ۱۴۰۱: آغاز کار مرکز مبادله"),
    (dt.date(2024, 12, 1), "Dec 2024: agreed-rate market replaces NIMA",
     "آذر ۱۴۰۳: بازار توافقی جای نیما را گرفت"),
]

# ------------------------------------------------------------------ formatting
_FA = str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹")


def fa(x) -> str:
    return str(x).translate(_FA)


def _nan(x) -> bool:
    return x is None or (isinstance(x, (float, np.floating)) and np.isnan(x))


def num(x, d: int = 0) -> str:
    if _nan(x):
        return "—"
    return fa(f"{abs(x):,.{d}f}".replace(",", "٬")) + ("−" if x < 0 else "")


def pct(x, d: int = 2, sign: bool = False) -> str:
    """A fraction as a percentage: 0.0123 -> ٪۱.۲۳ (sign written after it)."""
    if _nan(x):
        return "—"
    tail = "−" if x < 0 else ("+" if sign and x > 0 else "")
    return "٪" + fa(f"{abs(x) * 100:,.{d}f}".replace(",", "٬")) + tail


def _date(d) -> dt.date:
    return d.date() if hasattr(d, "date") else d


def jd(d) -> str:
    return fa(to_jalali(_date(d)))


def jg(d) -> str:
    return f"{jd(d)} ({_date(d)})"


def table(headers: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


# --------------------------------------------------------------------- figures
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"   # validated categorical slots 1-3


def _style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "text.color": INK,
        "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.8, "grid.linestyle": "-", "axes.spines.top": False,
        "axes.spines.right": False, "axes.titlesize": 11, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "font.size": 9, "legend.frameon": False,
        "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
    })


def _end_label(ax, x, y, text) -> None:
    ax.annotate(text, (x, y), xytext=(5, 0), textcoords="offset points",
                va="center", fontsize=8, color=INK2)


def _save(fig, name: str) -> str:
    fig.savefig(OUT / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return name


def fig_usd(panel: pd.DataFrame, pairs: pd.DataFrame) -> str:
    fig, (a, b) = plt.subplots(2, 1, figsize=(9.5, 6.2), sharex=True,
                               gridspec_kw={"height_ratios": [3, 1.4], "hspace": 0.12})
    t, bb = panel["usd_irr"].dropna(), panel["usd_irr_bonbast"].dropna()
    a.plot(bb.index, bb.values, color=ORANGE, lw=1.3, label="Bonbast sell (check)")
    a.plot(t.index, t.values, color=BLUE, lw=1.3, label="TGJU price_dollar_rl (primary)")
    a.set_yscale("log")
    a.set_ylabel("rial per USD (log scale)")
    a.set_title("Free-market USD/IRR from two sources")
    a.legend(loc="upper left")
    _end_label(a, t.index[-1], t.iloc[-1], f"{t.iloc[-1]:,.0f}")
    d = pairs["diff_same"] * 100
    b.fill_between(d.index, 0, d.values, color=BLUE, alpha=0.18, lw=0)
    b.plot(d.index, d.values, color=BLUE, lw=0.7)
    b.axhline(0, color=INK2, lw=0.8)
    b.set_ylabel("TGJU vs Bonbast, %\n(same date)")
    b.margins(x=0.01)
    return _save(fig, "fig1_usd_sources.png")


def fig_coffee(monthly: pd.DataFrame) -> str:
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 6.4), sharex=True,
                             gridspec_kw={"hspace": 0.2})
    m = monthly[monthly.index >= "2008-01-01"]
    for ax, name, title in ((axes[0], "arabica", "Arabica: ICE front month (KC=F) vs IMF Other Mild Arabicas"),
                            (axes[1], "robusta", "Robusta: ICE London front month (TGJU) vs IMF Robustas")):
        d = m[f"{name}_daily"].where(m[f"{name}_days"] >= MIN_DAYS_IN_MONTH)
        ax.plot(m.index, m[f"{name}_imf"], color=ORANGE, lw=1.6, label="IMF monthly indicator (physical)")
        ax.plot(m.index, d, color=BLUE, lw=1.6, label="monthly mean of the daily futures series")
        ax.set_ylabel("USD per kg")
        ax.set_title(title)
        ax.set_ylim(bottom=0)
        last = d.dropna()
        _end_label(ax, last.index[-1], last.iloc[-1], f"{last.iloc[-1]:.2f}")
    axes[0].legend(loc="upper left")
    return _save(fig, "fig2_coffee_vs_imf.png")


def fig_fund(index: pd.DataFrame, label: str) -> str:
    fig, ax = plt.subplots(figsize=(9.5, 3.8))
    x = pd.to_datetime(index["date"])
    ax.plot(x, index["tr_index"], color=BLUE, lw=1.6)
    ax.set_yscale("log")
    ax.set_ylabel("index, first traded day = 100 (log scale)")
    ax.set_title(f"Rial fixed-income total-return index: {label}")
    _end_label(ax, x.iloc[-1], index["tr_index"].iloc[-1], f"{index['tr_index'].iloc[-1]:,.0f}")
    return _save(fig, "fig4_fund_index.png")


def fig_rolls(kc: pd.DataFrame, contract: pd.DataFrame, contract_name: str,
              switches: list[dt.date]) -> str:
    start = dt.date(2026, 4, 1)
    a = kc[kc["date"] >= start].set_index("date")["close"]
    b = contract[contract["date"] >= start].set_index("date")["close"]
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    ax.plot(pd.to_datetime(b.index), b.values, color=ORANGE, lw=1.6,
            label=f"{contract_name} (one contract, never rolls)")
    ax.plot(pd.to_datetime(a.index), a.values, color=BLUE, lw=1.6,
            label="KC=F (continuous front month)")
    top = max(a.max(), b.max())
    for s in switches:
        if s >= start:
            ax.axvline(pd.Timestamp(s), color=INK2, lw=0.8)
            ax.annotate(f"switch day\n{s}", (pd.Timestamp(s), top), xytext=(4, -2),
                        textcoords="offset points", va="top", fontsize=8, color=INK2)
    ax.set_ylabel("US cents per lb")
    ax.set_title("Arabica: the continuous series jumps to the next contract on each switch day")
    ax.legend(loc="lower right")
    return _save(fig, "fig3_arabica_rolls.png")


def _break_gaps(s: pd.Series, days: int = 20) -> pd.Series:
    """Insert an empty point inside every hole longer than `days`, so a line
    is not drawn across a stretch that has no data."""
    s = s.dropna()
    holes = s.index[1:][(s.index[1:] - s.index[:-1]).days > days]
    extra = pd.Series(np.nan, index=[h - pd.Timedelta(days=1) for h in holes])
    return pd.concat([s, extra]).sort_index()


def fig_official(ratios: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    series = [("sana_sell_usd", ORANGE, "SANA sell"), ("nima_sell_usd", BLUE, "NIMA sell"),
              ("ice_transfer_usd_sell", AQUA, "exchange centre, remittance sell")]
    last = None
    for key, color, label in series:
        if key in ratios.columns:
            s = ratios[key].dropna()
            s = s[(s > 0.2) & (s < 1.6)]                 # axis range; x10 unit errors are off-scale
            line = _break_gaps(s)
            ax.plot(line.index, line.values, color=color, lw=1.2, label=label)
            last = s
    if last is not None:                                 # newest official rate, labelled once
        _end_label(ax, last.index[-1], last.iloc[-1], f"{last.iloc[-1]:.2f}")
    ax.axhline(1.0, color=INK2, lw=0.8)
    for i, (d, label, _) in enumerate(REGIMES):
        right = i == len(REGIMES) - 1
        ax.axvline(pd.Timestamp(d), color=INK2, lw=0.8)
        ax.annotate(label, (pd.Timestamp(d), 0.38 if right else 0.24 + 0.07 * (i % 2)),
                    xytext=(-4 if right else 4, 0), textcoords="offset points",
                    ha="right" if right else "left", fontsize=7.5, color=INK2)
    ax.set_ylim(0.2, 1.25)
    ax.set_ylabel("official rate / free-market rate")
    ax.set_title("Official-channel USD rates as a share of the free-market rate", pad=26)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, borderaxespad=0.2)
    return _save(fig, "fig5_official_ratio.png")


# ------------------------------------------------------------------ the report
def _manifest() -> pd.DataFrame:
    return pd.read_csv(paths.MANIFEST, dtype=str).fillna("")


def _coverage_row(name, source, how, s: pd.Series, iran: bool = True) -> list:
    s = s.dropna()
    g = quality.gaps(s, iran=iran)
    kinds = g["kind"].value_counts().to_dict()
    parts = [f"{num(len(g))} شکاف"] if len(g) else ["بدون شکاف"]
    if len(g):
        detail = []
        if kinds.get("nowruz"):
            detail.append(f"{num(kinds['nowruz'])} نوروز")
        if kinds.get("year_end"):
            detail.append(f"{num(kinds['year_end'])} پایان سال میلادی")
        if kinds.get("other"):
            detail.append(f"{num(kinds['other'])} دیگر")
        parts.append("(" + "، ".join(detail) + ")")
    return [name, source, how, jg(s.index.min()), jg(s.index.max()), num(len(s)), " ".join(parts)]


def build() -> str:
    OUT.mkdir(parents=True, exist_ok=True)
    _style()
    p = P.load()
    official = pd.read_parquet(P.OFFICIAL)
    man = _manifest()
    L: list[str] = []
    add = L.append

    # ------------------------------------------------------------ computations
    usd_ct = checks.chart_vs_table(tgju.USD_KEY)
    rob_ct = checks.chart_vs_table(tgju.ROBUSTA_KEY)
    pairs = checks.usd_pairs(p)
    agree = checks.usd_agreement(pairs)
    by_year = checks.usd_agreement_by_year(pairs)
    worst = checks.usd_largest_disagreements(pairs)
    recent = checks.usd_recent_check(p)
    extremes = checks.usd_extremes(p)
    monthly = checks.coffee_vs_imf(p)
    monthly08 = monthly[monthly.index >= "2008-01-01"]
    imf = {n: checks.imf_agreement(monthly08, n, MIN_DAYS_IN_MONTH) for n in ("arabica", "robusta")}
    imf_year = checks.imf_ratio_by_year(monthly08, MIN_DAYS_IN_MONTH)
    gap_abs = pd.DataFrame({n: (monthly08[f"{n}_imf"] - monthly08[f"{n}_daily"])
                            .where(monthly08[f"{n}_days"] >= MIN_DAYS_IN_MONTH)
                            for n in ("arabica", "robusta")})
    gap_year = gap_abs.groupby(gap_abs.index.year).mean()

    kc = yahoo.load()
    contracts = yahoo.loaded_contracts()
    kc_cal = rolls.trading_calendar(list(kc["date"]))
    kc_sw = rolls.switch_days(kc_cal, rolls.ARABICA)
    vol_ev = rolls.volume_evidence(kc, kc_sw)
    vol_ev12 = vol_ev[vol_ev["switch_day"] >= dt.date(2012, 1, 1)]
    # the listed contract with the longest history is the best "no roll" reference
    ref_name = max(contracts, key=lambda k: len(contracts[k])) if contracts else None
    con_ev = rolls.contract_evidence(kc, contracts[ref_name], kc_sw) if ref_name else pd.DataFrame()
    noise = (rolls.expiry_window_noise(kc, contracts[ref_name], kc_sw, EXPIRY_WINDOW)
             if ref_name else {})
    missing_in_kc = sorted(set(contracts[ref_name]["date"]) - set(kc["date"])) if ref_name else []

    rb_obs = p["robusta_usd_t"].where(~p["robusta_weekend"]).dropna()
    rb_series = pd.Series(rb_obs.values, index=[d.date() for d in rb_obs.index])
    ar_obs = p["arabica_clb"].dropna()
    ar_series = pd.Series(ar_obs.values, index=[d.date() for d in ar_obs.index])
    rb_cal = rolls.trading_calendar(list(rb_series.index), ref_dates=list(ar_series.index))
    rb_ev = rolls.days_left_evidence(rb_series, ar_series, rb_cal, rolls.ROBUSTA, ERAS,
                                     beta=ROBUSTA_BETA)

    vol = {
        ("arabica", "all"): checks.volatility(p, "arabica_clb", "arabica_roll"),
        ("arabica", "2012"): checks.volatility(p, "arabica_clb", "arabica_roll", dt.date(2012, 1, 1)),
        ("arabica", "2021"): checks.volatility(p, "arabica_clb", "arabica_roll", dt.date(2021, 1, 1)),
        ("robusta", "all"): checks.volatility(p, "robusta_usd_t", "robusta_roll", None, "robusta_weekend"),
        ("robusta", "2012"): checks.volatility(p, "robusta_usd_t", "robusta_roll", dt.date(2012, 1, 1), "robusta_weekend"),
        ("robusta", "2021"): checks.volatility(p, "robusta_usd_t", "robusta_roll", dt.date(2021, 1, 1), "robusta_weekend"),
    }
    roll_year = pd.DataFrame({
        "arabica": checks.roll_sum_by_year(p, "arabica_clb", "arabica_roll"),
        "robusta": checks.roll_sum_by_year(p, "robusta_usd_t", "robusta_roll", "robusta_weekend")})

    common3 = cc.common_dates(p, P.PRICE_COLS, unobserved=P.UNOBSERVED_FLAGS)
    win3 = cc.window_counts(common3, 91)
    common2 = cc.common_dates(p, ["usd_irr", "arabica_usd_kg"])
    win2 = cc.window_counts(common2, 91)
    common4 = cc.common_dates(p, P.PRICE_COLS + ["fund_tr_index"], unobserved=P.UNOBSERVED_FLAGS)
    win4 = cc.window_counts(common4, 91)
    ret3 = cc.common_calendar_returns(p, P.PRICE_COLS, exclude=P.ROLL_FLAGS,
                                      unobserved=P.UNOBSERVED_FLAGS)

    cands = fund.candidates()
    chosen = fund.choose(cands)
    fund_idx = fund.total_return_index(chosen["ins_code"])
    fund_years = fund.annual_returns(fund_idx)

    off_sum = checks.official_summary(official)
    off_med, off_cnt = checks.official_ratio_by_year(official, p)
    off_ratios = checks.official_ratios(official, p)

    rob = robots.check()
    fails = http.failures()

    # ----------------------------------------------------------------- figures
    f1 = fig_usd(p, pairs)
    f2 = fig_coffee(monthly)
    f3 = fig_fund(fund_idx, f"TSETMC {chosen['ins_code']}")
    f4 = fig_rolls(kc, contracts[ref_name], ref_name, kc_sw) if ref_name else None
    f5 = fig_official(off_ratios)

    # ------------------------------------------------------------- shorthand
    usd = p["usd_irr"].dropna()
    bb = p["usd_irr_bonbast"].dropna()
    rb_gaps = quality.gaps(rb_obs, iran=False).sort_values("calendar_days", ascending=False)
    big = rb_gaps.iloc[0] if len(rb_gaps) else None
    span = (p.index >= common3.min()) & (p.index <= common3.max())
    win_usd = cc.window_counts(p.index[span & p["usd_irr"].notna().values], 91)
    win_ar = cc.window_counts(p.index[span & p["arabica_clb"].notna().values], 91)
    # unconfirmed extreme moves that are one half of a jump-and-return pair
    ex_dates = list(extremes["date"])
    paired = 0
    for _, r in extremes[extremes["verdict"] == "not_confirmed"].iterrows():
        near = extremes[(extremes["date"] != r["date"])
                        & ((extremes["date"] - r["date"]).abs() <= pd.Timedelta(days=10))]
        paired += int((np.sign(near["tgju_return"]) == -np.sign(r["tgju_return"])).any())
    w97 = [d for d in worst.index if to_jalali(d.date())[:7] in ("1397-07", "1397-08")]
    seg97 = pairs.loc[min(w97):max(w97)] if w97 else None
    payers = cands[cands["drops_per_year"] > fund.MAX_DROPS_PER_YEAR]
    nima = off_med["nima_sell_usd"] if "nima_sell_usd" in off_med else pd.Series(dtype=float)
    sana = off_med["sana_sell_usd"] if "sana_sell_usd" in off_med else pd.Series(dtype=float)
    weekend_rows = p.index[p["robusta_weekend"]]
    n_ex = extremes["verdict"].value_counts().to_dict()
    same, nxt, best = agree.loc["same_date"], agree.loc["bonbast_next_day"], agree.loc["closer_of_two"]
    y1397 = by_year.loc[1397] if 1397 in by_year.index else None
    rb_manual = investing.available()
    war3 = win3[win3.index >= pd.Timestamp(WAR_START)]
    a21, r21 = vol[("arabica", "2021")], vol[("robusta", "2021")]
    last3 = con_ev.tail(3) if len(con_ev) else con_ev
    rb21 = rb_ev[rb_ev["era"] == "2021-2026"].set_index("days_left")
    rb_all_other = rb_ev[rb_ev["days_left"] == -1].set_index("era")
    bonbast_stale = quality.stale_runs(p["usd_irr_bonbast"])
    bonbast_extreme = int(quality.extreme_flag(p["usd_irr_bonbast"], quality.USD_EXTREME).sum())
    today = dt.date.today()

    # ================================================================== text
    add("# گزارش مرحلهٔ D1 — داده‌ها")
    add("")
    add(f"تاریخ ساخت گزارش: {jg(today)}. همهٔ عددها، جدول‌ها و شکل‌ها را `scripts/d1_report.py` "
        "از روی دادهٔ خام ساخته است. هیچ عددی دستی نوشته نشده است.")
    add("")

    # ---- 1
    add("## ۱. خلاصه")
    add("")
    if rb_manual:
        rob_line = "فایل دستی Investing.com رسیده و در پنل استفاده شده است."
    elif big is not None:
        rob_line = (f"روبوستا از TGJU خودکار آمد، ولی در ۱۴۰۵ یک حفرهٔ {num(big['calendar_days'])} روزه "
                    f"و {num(len(weekend_rows))} ردیف با تاریخ آخر هفته دارد؛ برای این دوره فایل دستی Investing.com لازم است.")
    else:
        rob_line = "روبوستا از TGJU خودکار آمد و شکاف بیش از ۵ روز ندارد."
    add(f"1. **همهٔ منبع‌ها جواب دادند** و پنل روزانه ساخته شد: دلار آزاد از TGJU ({num(len(usd))} روز از {jd(usd.index.min())})، "
        "Bonbast، عربیکا، روبوستا، شاخص IMF و صندوق درآمد ثابت.")
    add(f"2. **دلار TGJU و Bonbast در بیشتر سال‌ها نزدیک‌اند** (میانهٔ اختلاف {pct(same['median_abs_diff'])}). "
        f"ولی در ۱۳۹۷ جدا می‌شوند (تا {pct(worst['diff_best'].max(), 0)}) و {num(n_ex.get('not_confirmed', 0))} جهش بزرگ روزانهٔ TGJU را Bonbast تأیید نمی‌کند.")
    add(f"3. **قانون «رول» عربیکا روشن و آزموده است** (در {num(int(vol_ev12['volume_jump'].sum()))} از {num(len(vol_ev12))} رول از ۲۰۱۲). "
        "برای روبوستا فقط شاهد آماری از ۲۰۲۱ به بعد داریم.")
    add(f"4. **{rob_line}**")
    add("5. **نتیجه:** داده برای شروع D2 کافی است، به شرط اینکه دربارهٔ بندهای بخش ۹ تصمیم بگیرید. "
        "مهم‌ترین‌ها: رفتار با دلارِ ۱۳۹۷، فایل دستی روبوستا، و فاصلهٔ قیمت آتی با قیمت فیزیکی قهوه.")
    add("")

    # ---- 2
    add("## ۲. پوشش داده‌ها")
    add("")
    add("«شکاف» یعنی بیش از ۵ روز تقویمی پشت‌سرهم بدون داده. تاریخ‌ها جلالی‌اند و میلادی در پرانتز آمده است.")
    add("")
    rb_how = "نقطهٔ نمودار، یک درخواست" + ("؛ + فایل دستی Investing.com" if rb_manual else "")
    rows = [
        _coverage_row("دلار آزاد (اصلی)", "TGJU، کلید `price_dollar_rl`", "نقطهٔ نمودار، یک درخواست", p["usd_irr"]),
        _coverage_row("دلار آزاد (کنترل)", "Bonbast، نرخ فروش", "آرشیو گیت‌هاب، یک فایل", p["usd_irr_bonbast"]),
        _coverage_row("عربیکا", "Yahoo `KC=F`، سنت بر پوند", "کتابخانهٔ yfinance", p["arabica_clb"], iran=False),
        _coverage_row("روبوستا", "TGJU، کلید `commodities-london-coffee`، دلار بر تن", rb_how, rb_obs, iran=False),
        _coverage_row("صندوق درآمد ثابت", f"TSETMC، نماد {chosen['ticker']}", "تاریخچهٔ کامل، یک درخواست", p["fund_close_irr"]),
    ]
    for name, col in (("شاخص ماهانهٔ عربیکا (IMF)", "arabica_imf"), ("شاخص ماهانهٔ روبوستا (IMF)", "robusta_imf")):
        s = monthly[col].dropna()
        rows.append([name, "FRED", "CSV، یک درخواست", jg(s.index.min()), jg(s.index.max()),
                     num(len(s)) + " ماه", "ماهانه"])
    for _, r in off_sum.iterrows():
        rows.append([f"رسمی: `{r['key']}`", "TGJU", "نقطهٔ نمودار، یک درخواست",
                     jg(r["first"]), jg(r["last"]), num(r["rows"]),
                     f"{num(r['gaps_over_30d'])} شکاف بیش از ۳۰ روز"])
    add(table(["سری", "منبع", "روش دریافت", "اولین تاریخ", "آخرین تاریخ", "ردیف", "شکاف‌ها"], rows))
    add("")
    add(f"* پنل: `data/processed/panel_daily.parquet`، {num(len(p))} روز تقویمی از {jg(p.index.min())} تا {jg(p.index.max())}، "
        f"{num(p.shape[1])} ستون. هر خانه فقط وقتی پر است که آن سری در آن روز داده دارد. هیچ مقداری پر یا درون‌یابی نشده است.")
    add(f"* واحدها: دلار به ریال. عربیکا به سنت بر پوند و دلار بر کیلو (ضرب در {fa(units.CENTS_PER_LB_TO_USD_PER_KG)}). "
        f"روبوستا به دلار بر تن و دلار بر کیلو (تقسیم بر {num(units.KG_PER_TONNE)}).")
    add("* سری‌های رسمی جدا در `data/processed/official_usd_daily.parquet` هستند و در مدل استفاده نمی‌شوند.")
    add("")

    # ---- 3
    add("## ۳. نتیجهٔ دسترسی")
    add("")
    add("دو مسیر شبکه آزموده شد: «مستقیم» و «پروکسی». هر تلاش، موفق یا ناموفق، در `data/raw/FETCH_LOG.csv` ثبت شده است. "
f"بین هر دو درخواست دست‌کم {num(http.PAUSE_SECONDS)} ثانیه فاصله است.")
    add("")
    rows = []
    for src, g in man[man["source"] != "robots"].groupby("source"):
        rows.append([f"`{src}`", num(len(g)), "، ".join(sorted(set(g["route"]))),
                     g["fetched_at_utc"].max()[:10]])
    add(table(["منبع", "فایل خام", "مسیر", "آخرین دریافت (UTC)"], rows))
    add("")
    add(f"فهرست کامل فایل‌ها با تعداد ردیف، بازهٔ تاریخ و sha256 در `data/raw/MANIFEST.csv` است ({num(len(man))} ردیف).")
    add("")
    add("**آنچه کار نکرد** (متن دقیق خطا):")
    add("")
    seen, rows = set(), []
    for r in fails:
        if r["source"] == "robots":
            continue
        key = (r["source"], r["file"], r["route"])
        if key in seen:
            continue
        seen.add(key)
        detail = r["detail"][:150].replace("|", "/").replace("`", "'").replace("\\n", " ")
        rows.append([f"`{r['source']}/{r['file']}`", r["route"], f"`{detail}`"])
    add(table(["فایل", "مسیر", "خطا"], rows) if rows else "هیچ خطایی باقی نمانده است.")
    add("")
    add("* قراردادهای منقضی‌شدهٔ عربیکا (مثل `KCU26.NYB`) در Yahoo نیستند. Yahoo فقط قراردادهای زنده را می‌دهد.")
    add("* روبوستا در Yahoo نماد ندارد. Stooq به اسکریپت یک «بررسی مرورگر با جاوااسکریپت» می‌دهد؛ آن را دور نزدیم.")
    add("* TSETMC فقط از مسیر مستقیم جواب می‌دهد (از مسیر پروکسی تایم‌اوت می‌شود).")
    add("")
    add("**robots.txt و شرایط استفاده**")
    add("")
    verdict = {True: "مجاز", False: "**ممنوع**", None: "—"}
    use_fa = {"platform.tgju.org": "TGJU، نقطهٔ نمودار", "api.tgju.org": "TGJU، نقطهٔ جدول تاریخچه",
              "www.tgju.org": "TGJU، صفحهٔ تاریخچه (فقط با چشم)", "raw.githubusercontent.com": "آرشیو Bonbast",
              "query1.finance.yahoo.com": "Yahoo Finance (از راه yfinance)", "fred.stlouisfed.org": "FRED",
              "cdn.tsetmc.com": "TSETMC، تاریخچهٔ روزانه", "service.tsetmc.com": "TSETMC، فهرست نمادها",
              "stooq.com": "Stooq (جایگزین؛ استفاده نشد)", "www.investing.com": "Investing.com (فقط دانلود دستی)"}
    rows = []
    for _, r in rob.iterrows():
        state = "دارد" if r["robots"] == "present" else "ندارد / داده نشد"
        allowed = verdict[None if r["allowed"] is None or pd.isna(r["allowed"]) else bool(r["allowed"])]
        rows.append([f"`{r['host']}`", use_fa.get(r["host"], r["use"]), state, allowed])
    add(table(["میزبان", "کاربرد ما", "robots.txt", "مسیر ما برای ربات عمومی"], rows))
    add("")
    add("* **TGJU:** `platform.tgju.org` (نقطهٔ نمودار) همهٔ مسیرها را برای ربات‌ها بسته است. `api.tgju.org` (نقطهٔ پشت جدول تاریخچه) robots.txt ندارد. "
        "پانوشت سایت می‌گوید: «هر گونه کپی برداری از محتوا ... صرفا با موافقت مکتوب مجاز می باشد». TGJU وب‌سرویس فروشی هم دارد.")
    add("* **Bonbast:** از خود سایت چیزی نگرفتیم. آرشیو گیت‌هاب مجوز MIT دارد.")
    add("* **Yahoo:** robots.txt مسیر API را برای ربات عمومی بسته است. صفحهٔ شرایط استفاده باز نشد (تایم‌اوت) و خوانده نشده است.")
    add("* **FRED / IMF:** استفادهٔ شخصی و غیرتجاری آزاد است. این دو سری حق نشر IMF دارند: برای هر استفادهٔ دیگر باید از IMF اجازه گرفت.")
    add("* **Investing.com:** به اسکریپت خطای ۴۰۳ می‌دهد. فقط دانلود دستی.")
    add("* **نتیجه:** تصمیم D-006 (داده در گیت نمی‌رود) درست است. دربارهٔ نقطهٔ TGJU و ماشین‌حساب عمومی بند ۹ را ببینید.")
    add("")

    # ---- 4
    add("## ۴. دلار آزاد: TGJU در برابر Bonbast")
    add("")
    add("### ۴-۱. دو نقطهٔ دریافت TGJU یکی‌اند")
    add("")
    dup_idx = p.index[p["usd_dup_bar"]]
    dup_early = sum(1 for d in dup_idx if int(to_jalali(d.date())[:4]) <= 1396)
    add(f"* نقطهٔ نمودار {num(usd_ct['chart_bars'])} میله می‌دهد، ولی فقط {num(usd_ct['chart_dates'])} تاریخ یکتا. "
        f"در {num(usd_ct['dup_dates'])} تاریخ دو میله هست ({num(dup_early)} تای آن در ۱۳۹۶ یا پیش‌تر).")
    add(f"* جدول تاریخچهٔ سایت TGJU {num(usd_ct['table_rows'])} ردیف دارد. اگر در تاریخ‌های دوتایی **میلهٔ دوم** را نگه داریم، "
        f"قیمت پایانی در {num(usd_ct['equal_close'])} از {num(usd_ct['common_dates'])} تاریخ و هر چهار قیمت در {num(usd_ct['equal_ohlc'])} تاریخ دقیقاً برابر جدول است.")
    add(f"* میلهٔ اول فقط در {num(usd_ct['dup_first_bar_equals_table'])} تاریخ با جدول برابر است. پس قانون «میلهٔ آخر» درست است. "
        "این تاریخ‌ها در ستون `usd_dup_bar` علامت دارند.")
    chart_raw = tgju.load_chart(tgju.USD_KEY)
    at_midnight = int((chart_raw["ts"] % 86400 == 0).sum())
    add(f"* مُهر زمانی {num(at_midnight)} از {num(len(chart_raw))} میله ساعت ۰۰:۰۰ UTC است، یعنی بامداد همان روز در تهران. "
        f"پس از تبدیل به تاریخ تهران، {num(usd_ct['chart_only'])} تاریخ فقط در نمودار و {num(usd_ct['table_only'])} تاریخ فقط در جدول هست.")
    add("")
    add("### ۴-۲. بررسی روزهای اخیر")
    add("")
    has_page = "page_close" in recent.columns
    rows = [[jg(d), num(r["chart_close"]), num(r["table_close"])]
            + ([num(r["page_close"])] if has_page else [])
            + [num(r["bonbast_same"]), pct(r["diff_same"], 2, True),
               num(r["bonbast_next"]), pct(r["diff_next"], 2, True)]
            for d, r in recent.iterrows()]
    add(table(["تاریخ", "TGJU نمودار", "TGJU نقطهٔ جدول"]
              + (["TGJU صفحهٔ تاریخچه"] if has_page else [])
              + ["Bonbast همان روز", "اختلاف", "Bonbast روز بعد", "اختلاف"], rows))
    add("")
    add(f"واحد: ریال. آرشیو Bonbast تا {jg(bb.index.max())} داده دارد.")
    if has_page:
        same_page = int((recent["chart_close"] == recent["page_close"]).sum())
        add("")
        add(f"«صفحهٔ تاریخچه» همان عددهایی است که در HTML صفحهٔ `www.tgju.org/profile/price_dollar_rl/history` چاپ شده است. "
            f"در {num(same_page)} از {num(len(recent))} تاریخ با نمودار برابر است.")
    add("")
    add("### ۴-۳. توافق دو منبع")
    add("")
    add("«بازده لگاریتمی» یعنی لگاریتم نسبت قیمت امروز به قیمت قبلی. برای تغییرهای کوچک تقریباً همان درصد تغییر است.")
    add("")
    add(f"**نکتهٔ تاریخ:** آرشیو Bonbast هر روز یک عکس از سایت می‌گیرد و همهٔ روزهای تقویم را دارد (حتی جمعه). "
        "در بیشتر سال‌ها عددِ روز *بعدِ* آرشیو به قیمت پایانی TGJU نزدیک‌تر است تا عدد همان روز. پس هر دو هم‌ترازی را گزارش می‌کنیم.")
    add("")
    rows = []
    for label, r in (("همان تاریخ", same), ("Bonbast یک روز بعد", nxt), ("نزدیک‌ترِ این دو", best)):
        rows.append([label, num(r["days"]), pct(r["median_abs_diff"]), pct(r["p95_abs_diff"]),
                     f"{num(r['days_over_2pct'])} ({pct(r['share_over_2pct'], 1)})",
                     num(r["corr_daily"], 2), num(r["corr_5d"], 2)])
    add(table(["هم‌ترازی", "روز مشترک", "میانهٔ قدرمطلق اختلاف", "صدک ۹۵", "روزهای بالای ٪۲",
               "همبستگی بازده روزانه", "همبستگی بازده ۵ روزه"], rows))
    add("")
    add(f"* همبستگی بازده روزانه پایین است، چون تاریخ دو منبع یک روز جابه‌جا می‌شود. همبستگی ۵ روزه (بازه‌های بدون هم‌پوشانی، {num(same['n_5d'])} بازه) این مشکل را ندارد.")
    add("* اختلاف‌ها یکنواخت نیستند. جدول زیر به تفکیک سال جلالی است:")
    add("")
    rows = [[fa(y), num(r["days"]), pct(r["median_abs_same"]), pct(r["median_abs_next"]),
             pct(r["median_signed_same"], 2, True), num(r["days_over_2pct_same"]), num(r["days_over_2pct_best"])]
            for y, r in by_year.iterrows()]
    add(table(["سال", "روز", "میانهٔ اختلاف (همان روز)", "میانهٔ اختلاف (روز بعد)",
               "میانهٔ علامت‌دار (TGJU منهای Bonbast)", "روزهای بالای ٪۲ (همان روز)",
               "روزهای بالای ٪۲ (نزدیک‌ترِ دو هم‌ترازی)"], rows))
    add("")
    if y1397 is not None:
        add(f"* **۱۳۹۷ سال مشکل‌دار است:** {num(y1397['days_over_2pct_best'])} روز از {num(y1397['days'])} روز اختلاف بالای ٪۲ دارد "
            f"و TGJU به‌طور میانه {pct(abs(y1397['median_signed_same']))} پایین‌تر از Bonbast است.")
    others = by_year[(by_year["days_over_2pct_best"] >= 30) & (by_year.index != 1397)]
    if len(others):
        add("* سال‌های دیگر با ۳۰ روز یا بیشتر اختلاف بالای ٪۲: "
            + "، ".join(f"{fa(y)} ({num(r['days_over_2pct_best'])} روز)" for y, r in others.iterrows()) + ".")
    add("")
    add("**۱۰ اختلاف بزرگ** (اختلافی که هیچ‌کدام از دو هم‌ترازی توضیحش نمی‌دهد):")
    add("")
    rows = [[jg(d), num(r["tgju"]), num(r["bonbast_same"]), pct(r["diff_same"], 1, True),
             num(r["bonbast_next"]), pct(r["diff_next"], 1, True)] for d, r in worst.iterrows()]
    add(table(["تاریخ", "TGJU", "Bonbast همان روز", "اختلاف", "Bonbast روز بعد", "اختلاف"], rows))
    add("")
    n_1397 = sum(1 for d in worst.index if to_jalali(d.date()).startswith("1397"))
    add(f"* {num(n_1397)} مورد از ۱۰ مورد در ۱۳۹۷ است."
        + (f" از {jd(seg97.index.min())} تا {jd(seg97.index.max())} میانهٔ TGJU {num(seg97['tgju'].median())} ریال و میانهٔ Bonbast {num(seg97['bonbast_same'].median())} ریال است."
           if seg97 is not None else "")
        + " این دو عدد نمی‌توانند هر دو «قیمت بازار آزاد» باشند. از روی داده نمی‌شود گفت کدام درست است (بند ۹-۲).")
    add("* شکست سطح ×۱۰ یا ×۱۰٬۰۰۰ (اشتباه تومان/ریال یا تغییر واحد پول): "
        + "، ".join(f"{lab} {num(len(quality.level_breaks(p[c])))}" for lab, c in (
            ("TGJU", "usd_irr"), ("Bonbast", "usd_irr_bonbast"), ("عربیکا", "arabica_clb"),
            ("روبوستا", "robusta_usd_t"), ("صندوق", "fund_close_irr"))) + " مورد.")
    add("")
    add(f"![دلار آزاد از دو منبع]({f1})")
    add("")
    add("*شکل ۱ — بالا: دلار آزاد از TGJU و Bonbast، مقیاس لگاریتمی. پایین: درصد اختلاف TGJU با Bonbast در همان تاریخ.*")
    add("")

    # ---- 5
    add("## ۵. قهوه: مقایسه با IMF و مدیریت رول")
    add("")
    add("### ۵-۱. سری روزانه در برابر شاخص ماهانهٔ IMF")
    add("")
    add("سری روزانه قیمت «قرارداد آتی سررسید نزدیک» در بورس ICE است. شاخص IMF قیمت «فیزیکی» است، یعنی قیمت قهوهٔ واقعی تحویلی. "
        f"پس انتظار داریم سطح‌ها فاصله داشته باشند ولی تغییرها هم‌جهت باشند. میانگین ماهانهٔ سری روزانه را (ماه‌های با دست‌کم {num(MIN_DAYS_IN_MONTH)} روز داده) با IMF مقایسه کردیم، از ۲۰۰۸.")
    add("")
    rows = [[lab, num(v["months"]), num(v["corr_monthly_changes"], 3), num(v["ratio_median"], 3),
             f"{num(v['ratio_min'], 3)} تا {num(v['ratio_max'], 3)}"]
            for lab, v in (("عربیکا", imf["arabica"]), ("روبوستا", imf["robusta"]))]
    add(table(["سری", "ماه", "همبستگی تغییر ماهانه (لگاریتمی)", "میانهٔ نسبت آتی به IMF", "بازهٔ نسبت ماهانه"], rows))
    add("")
    rows = []
    for y in imf_year.index:
        r, g = imf_year.loc[y], gap_year.loc[y]
        rows.append([fa(y), num(r["arabica_ratio"], 3), num(g["arabica"], 2),
                     num(r["robusta_ratio"], 3), num(g["robusta"], 2)])
    add(table(["سال میلادی", "عربیکا: نسبت آتی به IMF", "عربیکا: فاصله (دلار بر کیلو)",
               "روبوستا: نسبت آتی به IMF", "روبوستا: فاصله (دلار بر کیلو)"], rows))
    add("")
    add(f"* **تغییرها خیلی خوب هم‌خوان‌اند** (همبستگی {num(imf['arabica']['corr_monthly_changes'], 2)} و {num(imf['robusta']['corr_monthly_changes'], 2)}). پس سری‌های روزانه برای مدل‌کردن *نوسان* مناسب‌اند.")
    add(f"* **روبوستا:** فاصله پایدار است، بین {num(gap_year['robusta'].min(), 2)} و {num(gap_year['robusta'].max(), 2)} دلار بر کیلو. "
        "این هم تأیید می‌کند که کلید `commodities-london-coffee` در TGJU واقعاً روبوستا است.")
    add(f"* **عربیکا — هشدار:** فاصله پایدار نیست، از {num(gap_year['arabica'].min(), 2)} تا {num(gap_year['arabica'].max(), 2)} دلار بر کیلو. "
        "شاخص IMF «عربیکای ملایم» است که بالاتر از قرارداد آتی معامله می‌شود و این اضافه‌بها در زمان عوض می‌شود. "
        "برای *نوسان* مشکلی نیست، ولی برای *سطح* هزینهٔ محموله مهم است (بند ۹-۶).")
    add("")
    add(f"![قهوه در برابر IMF]({f2})")
    add("")
    add("*شکل ۲ — میانگین ماهانهٔ سری روزانهٔ آتی (آبی) و شاخص ماهانهٔ IMF (نارنجی)، دلار بر کیلو.*")
    add("")
    add("### ۵-۲. رول قراردادها")
    add("")
    add("«رول» یعنی این: سری پیوسته همیشه قیمت نزدیک‌ترین قرارداد را نشان می‌دهد. وقتی آن قرارداد تمام می‌شود، سری به قرارداد بعدی می‌پرد. "
        "قیمت دو قرارداد فرق دارد، پس در آن روز یک پرش ساختگی می‌بینیم که برای خریدار تغییر قیمت واقعی نیست.")
    add("")
    add("**عربیکا (`KC=F`) — قانون:** سری تا «آخرین روز معامله»ی قرارداد روی همان قرارداد می‌ماند و از روز کاری بعد قرارداد بعدی را نشان می‌دهد. "
        f"طبق مقررات ICE، آخرین روز معامله {num(rolls.ARABICA.days_left + 1)} روز کاری پیش از آخرین روز کاریِ ماه تحویل است (ماه‌های مارس، مه، ژوئیه، سپتامبر، دسامبر). "
        f"پس «روز پرش» روزی است که دقیقاً {num(rolls.ARABICA.days_left)} روز معاملاتی تا پایان آن ماه مانده است.")
    add("")
    add("شاهد ۱ — مقایسه با یک قرارداد تکی که رول ندارد. Yahoo قراردادهای منقضی را نمی‌دهد، پس از قراردادهای زنده استفاده کردیم:")
    add("")
    if len(last3):
        rows = [[jg(r["switch_day"]), pct(r["continuous_return"], 2, True), pct(r["contract_return"], 2, True),
                 pct(r["gap"], 2, True),
                 "بله" if (r["same_on"] and not r["same_before"]) else "—"]
                for _, r in last3.iterrows()]
        add(table(["روز پرش طبق قانون", "بازده `KC=F`", f"بازده `{ref_name}`", "پرش ساختگی",
                   f"از این روز `KC=F` = `{ref_name}`؟"], rows))
        add("")
        spread = pd.DataFrame({k: rolls.contract_evidence(kc, v, kc_sw).set_index("switch_day")["gap"]
                               for k, v in contracts.items()}).reindex(list(last3["switch_day"]))
        add(f"* در آخرین رول، قیمت `KC=F` دقیقاً از همان روزی که قانون می‌گوید با `{ref_name}` یکی می‌شود و روز قبلش یکی نیست.")
        add(f"* هر {num(len(contracts))} قرارداد زنده همین پرش‌ها را در همین روزها نشان می‌دهند؛ بیشترین اختلاف بین قراردادها {pct((spread.max(axis=1) - spread.min(axis=1)).max())} است. "
            f"جمع این سه پرش {pct(last3['gap'].sum(), 1, True)} است: افتی که در `KC=F` دیده می‌شود ولی هیچ خریداری آن را تجربه نکرده است.")
    add("")
    add("شاهد ۲ — حجم معامله. در ماه تحویل حجم قراردادِ رو به پایان به چند قرارداد در روز می‌رسد. در «آخرین روز معامله» سری یک‌باره حجم قرارداد بعدی را نشان می‌دهد و قیمت یک روز بعد می‌پرد:")
    add("")
    add(f"* برگشت حجم دقیقاً در روز پیش از «روز پرش»: {num(int(vol_ev['volume_jump'].sum()))} از {num(len(vol_ev))} رول از ۲۰۰۰، "
        f"و {num(int(vol_ev12['volume_jump'].sum()))} از {num(len(vol_ev12))} رول از ۲۰۱۲."
        + (f" آخرین مورد ناجور در {fa(vol_ev.loc[~vol_ev['volume_jump'], 'switch_day'].max().year)} است."
           if (~vol_ev["volume_jump"]).any() else ""))
    if noise:
        add(f"* در {num(EXPIRY_WINDOW)} روز پیش از هر پرش، سری هنوز روی قرارداد کم‌معامله است. انحراف معیار اختلاف بازدهٔ روزانهٔ آن با `{ref_name}` در این روزها "
            f"{pct(noise['std_window'])} است، در برابر {pct(noise['std_other'])} در روزهای دیگر ({jd(noise['first'])} تا {jd(noise['last'])}). کمی نویز بیشتر، نه یک مشکل بزرگ.")
    add("")
    add(f"**روبوستا (TGJU) — قانون موقت:** همان منطق. آخرین روز معاملهٔ روبوستا در ICE لندن {num(rolls.ROBUSTA.days_left + 1)} روز کاری پیش از آخرین روز کاریِ ماه تحویل است "
        f"(ژانویه، مارس، مه، ژوئیه، سپتامبر، نوامبر). پس روز پرش روزی است که {num(rolls.ROBUSTA.days_left)} روز معاملاتی تا پایان ماه مانده است.")
    add("")
    add("برای روبوستا نه قرارداد تکی داریم و نه حجم. شاهد فقط آماری است: آن بخش از بازدهٔ روزانهٔ روبوستا را که عربیکا توضیح نمی‌دهد " +
        f"(بازدهٔ روبوستا منهای {num(ROBUSTA_BETA, 1)} برابرِ بازدهٔ عربیکا) بر اساس «چند روز معاملاتی تا پایان ماه تحویل مانده» گروه کردیم:")
    add("")
    rows = []
    for k in range(0, 7):
        row = [fa(k) + (" ← روز پرش" if k == rolls.ROBUSTA.days_left else "")]
        for a, b in ERAS:
            e = rb_ev[(rb_ev["era"] == f"{a}-{b}") & (rb_ev["days_left"] == k)].iloc[0]
            row.append(f"{pct(e['std'])} / {pct(e['mean'], 2, True)}")
        rows.append(row)
    row = ["همهٔ روزها"]
    for a, b in ERAS:
        e = rb_all_other.loc[f"{a}-{b}"]
        row.append(f"{pct(e['std'])} / {pct(e['mean'], 2, True)}")
    rows.append(row)
    add(table(["روز مانده تا پایان ماه"] + [fa(f"{a}–{b}") for a, b in ERAS], rows))
    add("")
    add("هر خانه: انحراف معیار / میانگین.")
    add("")
    k3 = rb21.loc[rolls.ROBUSTA.days_left]
    add(f"* در ۲۰۲۱ تا ۲۰۲۶ روز پرش آشکار است: انحراف معیار {pct(k3['std'])} (در برابر {pct(rb_all_other.loc['2021-2026', 'std'])} در همهٔ روزها) "
        f"و میانگین {pct(k3['mean'], 2, True)}. میانگین منفی یعنی قرارداد نزدیک گران‌تر از قرارداد بعدی بوده است (به این حالت «بک‌واردیشن» می‌گویند).")
    add("* پیش از ۲۰۲۱ هیچ روزی برجسته نیست. یا پرش‌ها کوچک بوده‌اند یا روش منبع فرق داشته؛ از روی داده نمی‌شود فهمید (بند ۹-۳).")
    add("")
    add("**اثر حذف روزهای رول**")
    add("")
    rows = []
    for name, lab in (("arabica", "عربیکا"), ("robusta", "روبوستا")):
        for key, since in (("all", "کل تاریخچه"), ("2012", "از ۲۰۱۲"), ("2021", "از ۲۰۲۱")):
            v = vol[(name, key)]
            rows.append([lab, since, num(v["n_roll"]), pct(v["std_all"], 3), pct(v["std_without_roll"], 3),
                         pct(v["mean_abs_roll_days"]), pct(v["mean_abs_other_days"]),
                         pct(v["roll_sum_per_year"], 1, True)])
    add(table(["سری", "دوره", "روز رول", "نوسان روزانه با رول", "نوسان روزانه بدون رول",
               "میانگین قدرمطلق بازده در روز رول", "در روزهای دیگر", "جمع بازدهٔ روزهای رول در سال"], rows))
    add("")
    add(f"* «نوسان روزانه» انحراف معیار بازدهٔ لگاریتمی روزانه است. حذف رول نوسان را کم تغییر می‌دهد، ولی **جهت** را زیاد: "
        f"از ۲۰۲۱ جمع بازدهٔ روزهای رول عربیکا حدود {pct(a21['roll_sum_per_year'], 1, True)} در سال و روبوستا {pct(r21['roll_sum_per_year'], 1, True)} در سال است. "
        "بخش بزرگی از این، افت ساختگی است.")
    ry = roll_year.tail(4)
    add("* جمع بازدهٔ روزهای رول در سال‌های اخیر: " + "؛ ".join(
        f"{fa(y)}: عربیکا {pct(r['arabica'], 1, True)}، روبوستا {pct(r['robusta'], 1, True)}" for y, r in ry.iterrows()) + ".")
    add(f"* در پنل {num(int(p['arabica_roll'].sum()))} روز رول عربیکا و {num(int(p['robusta_roll'].sum()))} روز رول روبوستا علامت خورده است. "
        "سطح قیمت‌ها دست‌نخورده است؛ فقط بازدهٔ آن روزها در تابع بازده خالی می‌شود.")
    add("")
    if f4:
        add(f"![رول عربیکا]({f4})")
        add("")
        add(f"*شکل ۳ — عربیکا در ۲۰۲۶: سری پیوستهٔ `KC=F` (آبی) و قرارداد `{ref_name}` (نارنجی). خط‌های عمودی روزهای پرش طبق قانون‌اند.*")
        add("")
    add("### ۵-۳. تقویم مشترک")
    add("")
    add("بازار ارز ایران و بورس ICE روزهای کاری متفاوت دارند. «تقویم مشترک» یعنی فقط روزهایی که همهٔ سری‌های انتخاب‌شده داده دارند. "
        "بازده بین دو روز مشترکِ پشت‌سرهم حساب می‌شود، پس هیچ حرکتی گم نمی‌شود و هیچ مقداری پر نمی‌شود.")
    add("")
    rows = [
        ["دلار + عربیکا + روبوستا", num(len(common3)), jd(common3.min()), num(win3.mean(), 1), num(win3.quantile(0.05)), num(win3.min())],
        ["دلار + عربیکا", num(len(common2)), jd(common2.min()), num(win2.mean(), 1), num(win2.quantile(0.05)), num(win2.min())],
        ["دلار + عربیکا + روبوستا + صندوق", num(len(common4)), jd(common4.min()), num(win4.mean(), 1), num(win4.quantile(0.05)), num(win4.min())],
    ]
    add(table(["ستون‌ها", "روز مشترک", "از", "میانگین در پنجرهٔ ۹۱ روزه", "صدک ۵", "کمینه"], rows))
    add("")
    dd = ret3["days"]
    add(f"* **در یک پنجرهٔ ۳ ماهه به‌طور میانگین فقط {num(win3.mean(), 1)} روز مشترک هست.** در همان پنجره دلار به‌تنهایی {num(win_usd.mean(), 1)} روز و عربیکا {num(win_ar.mean(), 1)} روز داده دارد. "
        "دلیل: دلار پنجشنبه و جمعه کم‌داده است و قهوه شنبه و یکشنبه بسته است.")
    add(f"* بازده‌های مشترک هم‌طول نیستند: {num(int((dd == 1).sum()))} بازدهٔ یک‌روزه، {num(int(((dd >= 4) & (dd <= 5)).sum()))} بازدهٔ ۴ یا ۵ روزه (از روی آخر هفته)، "
        f"و بلندترین {num(dd.max())} روز. D2 باید این را در نظر بگیرد (بند ۹-۵).")
    add(f"* در دورهٔ جنگ (از {jd(WAR_START)}): میانگین {num(war3.mean(), 1)} و کمینه {num(war3.min())} روز مشترک در پنجرهٔ ۹۱ روزه.")
    add(f"* به‌خاطر علامت رول، {num(int(ret3['arabica_usd_kg'].isna().sum()))} بازدهٔ عربیکا و {num(int(ret3['robusta_usd_kg'].isna().sum()))} بازدهٔ روبوستا از {num(len(ret3))} بازدهٔ مشترک خالی شده است.")
    add("")

    # ---- 6
    add("## ۶. پرچم‌های کیفیت")
    add("")
    add("### ۶-۱. شکاف‌های بیش از ۵ روز")
    add("")
    # notes are explanations written by hand; the dates and lengths are computed
    notes = {dt.date(2025, 6, 11): "جنگ ۱۲ روزه، خرداد ۱۴۰۴", dt.date(2026, 2, 25): "آغاز جنگ، اسفند ۱۴۰۴",
             dt.date(2001, 9, 10): "تعطیلی بازار نیویورک پس از ۱۱ سپتامبر",
             dt.date(2015, 3, 17): "دو ماه نخست پس از پذیرش، تقریباً بدون معامله"}
    series = (("دلار TGJU", p["usd_irr"], True), ("دلار Bonbast", p["usd_irr_bonbast"], True),
              ("عربیکا", p["arabica_clb"], False), ("روبوستا", rb_obs, False),
              ("صندوق", p["fund_close_irr"], True))
    rows = []
    for label, s, iran in series:
        g = quality.gaps(s, iran=iran)
        hol = g[g["kind"] != "other"]
        rows.append([label, num(len(g)), num(len(hol)),
                     (f"{num(hol['calendar_days'].min())} تا {num(hol['calendar_days'].max())} روز" if len(hol) else "—"),
                     num(len(g) - len(hol))])
    add(table(["سری", "همهٔ شکاف‌ها", "تعطیلات شناخته‌شده", "طول آن‌ها", "شکاف‌های دیگر"], rows))
    add("")
    add("«تعطیلات شناخته‌شده» برای سری‌های ایران نوروز (۱ تا ۱۳ فروردین) و برای قهوه پایان سال میلادی است. شکاف‌های دیگر، تک‌تک:")
    add("")
    rows = []
    for label, s, iran in series:
        g = quality.gaps(s, iran=iran)
        for _, r in g[g["kind"] == "other"].iterrows():
            rows.append([label, jg(r["last_before"]), jg(r["first_after"]), num(r["calendar_days"]),
                         notes.get(r["last_before"], "—")])
    add(table(["سری", "آخرین روز پیش از شکاف", "اولین روز پس از شکاف", "روز تقویمی", "توضیح"], rows))
    add("")
    if big is not None:
        inside = [d for d in rolls.switch_days(rb_cal, rolls.ROBUSTA)
                  if big["last_before"] < d < big["first_after"]]
        add(f"* **بزرگ‌ترین حفرهٔ روبوستا:** از {jg(big['last_before'])} تا {jg(big['first_after'])}، یعنی {num(big['calendar_days'])} روز"
            + ("، در دورهٔ جنگ. " if big["last_before"] >= WAR_START else ". ")
            + (f"روز رولِ {inside[0]} هم داخل همین حفره است؛ علامت رول روی اولین روز پس از حفره نشسته است." if inside else ""))
    usd_war_gaps = quality.gaps(usd[usd.index >= pd.Timestamp(WAR_START)])
    add(f"* دلار TGJU از آغاز جنگ ({jd(WAR_START)}) تا امروز {num(len(usd_war_gaps))} شکاف بیش از ۵ روز دارد.")
    add("")
    add("### ۶-۲. مقدار راکد (۵ مشاهدهٔ پشت‌سرهم یا بیشتر با یک قیمت)")
    add("")
    rows = []
    for label, s, _ in series:
        st = quality.stale_runs(s)
        rows.append([label, num(len(st)), num(int(st["observations"].sum()) if len(st) else 0),
                     num(int(st["observations"].max()) if len(st) else 0)])
    add(table(["سری", "تعداد دوره", "جمع مشاهده‌ها", "بلندترین دوره"], rows))
    add("")
    st = quality.stale_runs(p["usd_irr"])
    for _, r in st.iterrows():
        add(f"* دلار TGJU: از {jg(r['first_date'])} تا {jg(r['last_date'])}، {num(r['observations'])} مشاهده با قیمت {num(r['value'])} ریال.")
    in_nowruz = sum(1 for _, r in bonbast_stale.iterrows()
                    if quality.covers_nowruz(_date(r["first_date"]) - dt.timedelta(days=1),
                                              _date(r["last_date"]) + dt.timedelta(days=1)))
    add(f"* آرشیو Bonbast در روزهای تعطیل همان عدد قبلی را تکرار می‌کند: {num(len(bonbast_stale))} دورهٔ راکد، که {num(in_nowruz)} تای آن روزهای نوروز را در بر دارد.")
    add("")
    add("### ۶-۳. تاریخ تکراری و ردیف‌های مشکوک")
    add("")
    add(f"* دلار TGJU: {num(usd_ct['dup_dates'])} تاریخ با دو میله در نقطهٔ نمودار (بخش ۴-۱). در پنل یک ردیف برای هر تاریخ هست.")
    add(f"* روبوستا TGJU: {num(rob_ct['dup_dates'])} تاریخ با دو میله.")
    dup_other = {"Bonbast": int(bonbast.load()["date"].duplicated().sum()),
                 "عربیکا": int(kc["date"].duplicated().sum()),
                 "صندوق": int(tsetmc.load_daily(chosen["ins_code"])["date"].duplicated().sum())}
    add("* تاریخ تکراری در منبع‌های دیگر: " + "، ".join(f"{k} {num(v)}" for k, v in dup_other.items()) + ".")
    wk = [d for d in weekend_rows]
    first_wk26 = min((d for d in wk if d.year == 2026), default=None)
    n26 = sum(1 for d in wk if d.year == 2026)
    add(f"* **روبوستا {num(len(wk))} ردیف با تاریخ شنبه یا یکشنبه دارد**، که {num(n26)} تای آن در ۲۰۲۶ است"
        + (f" (از {jg(first_wk26)})" if first_wk26 is not None else "") + ". ICE در آخر هفته معامله ندارد. "
        "این ردیف‌ها در ستون `robusta_weekend` علامت دارند و فعلاً در بازده‌ها «مشاهده‌نشده» حساب می‌شوند (بند ۹-۴).")
    add(f"* صندوق: TSETMC {num(int(chosen['zero_volume_rows']))} ردیف بدون معامله (حجم صفر) هم می‌دهد. این ردیف‌ها در پنل نیستند.")
    if missing_in_kc:
        add(f"* عربیکا: `KC=F` {num(len(missing_in_kc))} روز را ندارد که قرارداد `{ref_name}` در آن معامله شده است: "
            + "، ".join(str(d) for d in missing_in_kc) + ".")
    add("")
    add("### ۶-۴. حرکت‌های شدید روزانه")
    add("")
    add(f"**دلار TGJU، قدرمطلق بازده بالای ٪۱۰: {num(len(extremes))} مورد.** هر مورد با Bonbast روی همان دو تاریخ، و با یک روز جابه‌جایی، مقایسه شده است. "
        "«تأیید» یعنی Bonbast هم‌جهت و دست‌کم نصف آن اندازه حرکت کرده است.")
    add("")
    vfa = {"confirmed": "تأیید", "not_confirmed": "**تأیید نشد**", "no_bonbast": "پیش از آغاز Bonbast"}
    rows = [[jg(r["date"]), num(r["prev_close"]), num(r["close"]), pct(r["tgju_return"], 1, True),
             pct(r["bonbast_same"], 1, True), pct(r["bonbast_next"], 1, True), vfa[r["verdict"]]]
            for _, r in extremes.iterrows()]
    add(table(["تاریخ", "قیمت قبلی", "قیمت", "بازده TGJU", "Bonbast همان روزها", "Bonbast یک روز بعد", "نتیجه"], rows))
    add("")
    add(f"* تأیید: {num(n_ex.get('confirmed', 0))}. تأیید نشد: {num(n_ex.get('not_confirmed', 0))}. بدون Bonbast: {num(n_ex.get('no_bonbast', 0))}. "
        "نتیجه در ستون `usd_outlier_check` پنل هم هست.")
    add(f"* {num(paired)} مورد از {num(n_ex.get('not_confirmed', 0))} مورد تأییدنشده جفت‌اند: در کمتر از ۱۰ روز یک حرکت شدید در جهت مخالف هم هست (پرش و برگشت). این شبیه قیمت اشتباه در TGJU است، نه حرکت بازار. "
        f"خودِ Bonbast در کل دوره فقط {num(bonbast_extreme)} حرکت بالای ٪۱۰ دارد.")
    add("")
    for label, col, flag, unit in (("عربیکا", "arabica_clb", "arabica_outlier", "سنت بر پوند"),
                                   ("روبوستا", "robusta_usd_t", "robusta_outlier", "دلار بر تن")):
        s = p[col].where(~p["robusta_weekend"]).dropna() if col == "robusta_usd_t" else p[col].dropna()
        r = np.log(s).diff()
        hit = s.index[p.loc[s.index, flag]]
        add(f"**{label}، قدرمطلق بازده بالای ٪۸: {num(len(hit))} مورد** ({unit}):")
        add("")
        add(table(["تاریخ", "قیمت قبلی", "قیمت", "بازده", "روز رول؟"],
                  [[jg(d), num(s.shift()[d], 2), num(s[d], 2), pct(r[d], 1, True),
                    "بله" if p.loc[d, flag.replace("outlier", "roll")] else "—"] for d in hit]))
        add("")
    z = contracts.get(ref_name) if ref_name else None
    if z is not None:
        zr = np.log(z.set_index("date")["close"]).diff()
        july = [d for d in p.index[p["arabica_outlier"]] if d.year == 2026]
        if july:
            add("* حرکت‌های عربیکا در ۲۰۲۶ واقعی‌اند: قرارداد پرمعاملهٔ " + f"`{ref_name}` در همان روزها "
                + "، ".join(f"{pct(zr.get(d.date(), np.nan), 1, True)}" for d in july) + " حرکت کرده است.")
            add("")

    # ---- 7
    add("## ۷. شاخص هزینهٔ فرصت ریالی و صندوق منتخب")
    add("")
    add("«هزینهٔ فرصت» یعنی سودی که ریال می‌توانست در این فاصله بگیرد اگر زود دلار نمی‌خریدیم. برای اندازه‌گیری آن از یک صندوق درآمد ثابت قابل‌معامله (ETF) استفاده می‌کنیم.")
    add("")
    add(f"* TSETMC {num(len(tsetmc.fixed_income_funds()))} صندوق درآمد ثابت قابل‌معامله فهرست می‌کند (زیرگروه {fa(tsetmc.FIXED_INCOME_SUB_SECTOR)}). تاریخچهٔ همه را گرفتیم ({num(len(cands))} صندوق با دست‌کم یک معامله).")
    add("* قانون انتخاب: بین صندوق‌هایی که هنوز معامله می‌شوند و سود نقدی دوره‌ای نمی‌دهند، آن‌که بیشترین روز معاملاتی را دارد.")
    add("")
    top = cands.sort_values("traded_days", ascending=False).head(10)
    rows = [[r["ticker"], jg(r["first_traded"]), num(r["traded_days"]), num(r["max_gap_days"]),
             num(r["distribution_like_drops"]), num(r["drops_per_year"], 1),
             "**منتخب**" if r["ins_code"] == chosen["ins_code"] else
             ("سود نقدی می‌دهد" if r["drops_per_year"] > fund.MAX_DROPS_PER_YEAR else "—")]
            for _, r in top.iterrows()]
    add(table(["نماد", "اولین روز معامله", "روز معاملاتی", "بلندترین شکاف (روز)",
               "افت بیش از ٪۰.۴", "افت در سال", "وضعیت"], rows))
    add("")
    add(f"* **صندوق منتخب: {chosen['ticker']}** ({chosen['name']}، کد TSETMC `{chosen['ins_code']}`)، از {jg(chosen['first_traded'])}، {num(chosen['traded_days'])} روز معاملاتی.")
    add(f"* **روش بازده کل:** بر پایهٔ قیمت پایانی. این صندوق سود نقدی تقسیم نمی‌کند: قیمتش از {num(chosen['first_close'])} به {num(chosen['last_close'])} ریال رسیده "
        f"و در کل تاریخچه فقط {num(chosen['distribution_like_drops'])} روز افت بیش از ٪۰.۴ دارد. پس قیمت همان بازده کل است.")
    add("* چرا NAV یا قیمت تعدیل‌شده لازم نشد: TSETMC «قیمت دیروز» را پس از تقسیم سود صندوق‌ها اصلاح نمی‌کند. "
        f"در {num(len(payers))} صندوقِ سودپرداز، هر پرداخت یک افت قیمت می‌سازد (میانهٔ افت بین صندوق‌ها: {pct(abs(payers['median_drop'].median()))}). "
        "برای همین صندوقی انتخاب شد که اصلاً سود نقدی نمی‌دهد.")
    add("")
    rows = [[fa(r["jalali_year"]), jd(r["from"]), jd(r["to"]), num(r["observations"]),
             pct(r["total_return"]), pct(r["annualised"])] for _, r in fund_years.iterrows()]
    add(table(["سال جلالی", "از", "تا", "روز معاملاتی", "بازده دوره", "بازده سالانه‌شده"], rows))
    add("")
    add("* بازده هر سال از آخرین قیمت سال قبل تا آخرین قیمت همان سال است و روی روزهای تقویمیِ واقعی سالانه شده است. سال اول و سال جاری ناقص‌اند.")
    add(f"* **محدودیت:** شاخص از {jd(chosen['first_traded'])} شروع می‌شود، ولی دلار از {jd(usd.index.min())}. "
        "TSETMC برای صندوق درآمد ثابتِ قابل‌معامله تاریخچهٔ قدیمی‌تری ندارد (بند ۹-۷).")
    add("")
    add(f"![شاخص صندوق]({f3})")
    add("")
    add(f"*شکل ۴ — شاخص بازده کل صندوق {chosen['ticker']}، مقیاس لگاریتمی، اولین روز = ۱۰۰.*")
    add("")

    # ---- 8
    add("## ۸. کانال رسمی (فقط برای اطلاع)")
    add("")
    add("این بخش در مدل استفاده نمی‌شود (D-002). نسبت هر نرخ رسمی به دلار آزاد TGJU در همان روز حساب شده است. عدد ۰.۸ یعنی نرخ رسمی ٪۲۰ ارزان‌تر از بازار آزاد بوده است.")
    add("")
    rows = [[f"`{r['key']}`", jd(r["first"]), jd(r["last"]), num(r["rows"]), num(r["dup_dates"]),
             num(r["level_breaks_x10"])] for _, r in off_sum.iterrows()]
    add(table(["کلید TGJU", "از", "تا", "روز", "تاریخ تکراری", "شکست سطح ×۱۰"], rows))
    add("")
    keys = [c for c in off_med.columns]
    rows = [[fa(y)] + [num(off_med.loc[y, k], 2) for k in keys] for y in off_med.index]
    add("میانهٔ نسبت به دلار آزاد، به تفکیک سال جلالی (سال‌های با کمتر از ۲۰ روز داده خالی‌اند):")
    add("")
    add(table(["سال"] + [f"`{k}`" for k in keys], rows))
    add("")
    bank = off_sum.set_index("key").loc["bank_usd"] if "bank_usd" in set(off_sum["key"]) else None
    if bank is not None:
        bb_breaks = quality.level_breaks(official["bank_usd"])
        in97 = sum(1 for d in bb_breaks["date"] if to_jalali(d.date()).startswith("1397"))
        add(f"* **`bank_usd` قابل اعتماد نیست:** {num(bank['level_breaks_x10'])} شکست سطح ×۱۰ دارد (جابه‌جایی تومان و ریال؛ {num(in97)} تای آن در ۱۳۹۷). در شکل نیامده است.")
    s_early = sana[[y for y in sana.index if y <= 1400]].dropna()
    if len(s_early):
        add(f"* `sana_sell_usd` تا ۱۴۰۰ نزدیک بازار آزاد است (نسبت سالانه بین {num(s_early.min(), 2)} و {num(s_early.max(), 2)}).")
    n_a = nima[[y for y in nima.index if 1398 <= y <= 1400]].dropna()
    n_b = nima[[y for y in nima.index if 1402 <= y <= 1404]].dropna()
    if len(n_a) and len(n_b):
        def span(x):
            lo, hi = pct(1 - x.max(), 0), pct(1 - x.min(), 0)
            return f"حدود {lo}" if lo == hi else f"بین {lo} و {hi}"
        add(f"* نیما در ۱۳۹۸ تا ۱۴۰۰ {span(n_a)} ارزان‌تر از بازار آزاد بوده، و در ۱۴۰۲ تا ۱۴۰۴ {span(n_b)} (میانهٔ سالانه).")
    if {"nima_sell_usd", "ice_transfer_usd_sell"} <= set(official.columns):
        both = official[["nima_sell_usd", "ice_transfer_usd_sell"]].dropna()
        eq = int((both["nima_sell_usd"] == both["ice_transfer_usd_sell"]).sum())
        add(f"* کلید `nima_sell_usd` در {num(eq)} از {num(len(both))} روز مشترک دقیقاً همان عددِ حوالهٔ مرکز مبادله (`ice_transfer_usd_sell`) است.")
    add("* تغییرهای شناخته‌شدهٔ رژیم (تاریخ‌ها در حد ماه و از دانستهٔ عمومی‌اند؛ لطفاً بازبینی کنید): "
        + "؛ ".join(fa_label for _, _, fa_label in REGIMES) + ".")
    add("")
    add(f"![نسبت نرخ رسمی به آزاد]({f5})")
    add("")
    add("*شکل ۵ — نسبت نرخ‌های کانال رسمی به دلار آزاد. خط‌های عمودی تغییرهای رژیم‌اند.*")
    add("")

    # ---- 9
    add("## ۹. تصمیم‌های لازم")
    add("")
    add("در هر بند، گزینهٔ پیش‌فرضِ موقت همان است که الان در کد است.")
    add("")
    add("**۹-۱. کدام نقطهٔ TGJU؟**")
    add(f"* یافته: `platform.tgju.org` در robots.txt همه‌چیز را برای ربات‌ها بسته است. نقطهٔ پشت جدول تاریخچه (`api.tgju.org`) همان داده را می‌دهد (برابری در {num(usd_ct['equal_ohlc'])} از {num(usd_ct['common_dates'])} تاریخ) و منعی ندارد.")
    add("* گزینه‌ها: (الف) ماندن روی نقطهٔ نمودار، طبق اسپک [پیش‌فرض موقت]؛ (ب) رفتن به نقطهٔ جدول.")
    add("* پیشنهاد: **(ب)**. عددها عوض نمی‌شوند، قانون «میلهٔ آخر» هم دیگر لازم نیست. تا تصمیم شما هر دو فایل خام نگه داشته شده است.")
    add("")
    add("**۹-۲. دلارِ ۱۳۹۷ و جهش‌های تأییدنشدهٔ TGJU**")
    add(f"* یافته: در ۱۳۹۷ دو منبع تا {pct(worst['diff_best'].max(), 0)} فاصله دارند، و {num(n_ex.get('not_confirmed', 0))} حرکت بالای ٪۱۰ در TGJU را Bonbast تأیید نمی‌کند. سناریوی تنش «جهش ارز ۱۳۹۷» (D-005) مستقیم به این بستگی دارد.")
    add("* گزینه‌ها: (الف) TGJU خام، بدون هیچ حذف [پیش‌فرض موقت، طبق D-001]؛ (ب) TGJU، ولی بازدهٔ روزهای `not_confirmed` خالی شود؛ (ج) Bonbast سری اصلی شود؛ (د) TGJU اصلی بماند و D2 همهٔ نتیجه‌ها را یک بار هم با Bonbast حساب کند.")
    add("* پیشنهاد: **(د)**، و در کنارش نتیجهٔ (ب) هم گزارش شود. اگر نتیجه‌ها نزدیک بودند مسئله بسته است؛ اگر نه، آن‌وقت D-001 را بازبینی کنید.")
    add("")
    add("**۹-۳. قانون رول روبوستا**")
    add("* یافته: قانون عربیکا با داده تأیید شد. برای روبوستا شاهد فقط آماری است و فقط از ۲۰۲۱.")
    add(f"* گزینه‌ها: (الف) همان قانون روی کل تاریخچه [پیش‌فرض موقت؛ {num(vol[('robusta', 'all')]['n_roll'])} بازده خالی می‌شود]؛ (ب) فقط از ۲۰۲۱؛ (ج) بدون حذف.")
    add(f"* پیشنهاد: **(الف)**. اگر پیش از ۲۰۲۱ پرشی نبوده، فقط {pct(vol[('robusta', 'all')]['n_roll'] / vol[('robusta', 'all')]['n'], 1)} بازده‌ها را بی‌دلیل کنار گذاشته‌ایم؛ اگر بوده، درست عمل کرده‌ایم. فایل Investing.com هم این را روشن نمی‌کند، چون آن هم سری پیوسته است.")
    add("")
    add("**۹-۴. روبوستا در ۱۴۰۵ — فایل دستی لازم است**")
    if rb_manual:
        add("* فایل دستی رسیده است. شکاف‌ها و ردیف‌های آخر هفتهٔ باقی‌مانده در بخش ۶ آمده‌اند.")
    else:
        holes = "، ".join(f"{num(r['calendar_days'])} روز از {jd(r['last_before'])}" for _, r in rb_gaps.iterrows())
        holes_war = int((rb_gaps["last_before"] >= WAR_START).sum())
        wk_war = sum(1 for d in wk if d.date() >= WAR_START)
        add(f"* یافته: {num(len(rb_gaps))} حفره ({holes}) و {num(n26)} ردیف با تاریخ آخر هفته در ۲۰۲۶. "
            f"{num(holes_war)} حفره و {num(wk_war)} ردیف در دورهٔ جنگ است، که طبق D-005 مهم است.")
    add("* درخواست: فایل روزانهٔ Investing.com (از 2012-01-01 تا امروز) را دانلود و در `data/raw/manual/robusta_investing.csv` ذخیره کنید. خواننده‌اش نوشته و تست شده است. پنل هر جا آن فایل داده داشته باشد از آن استفاده می‌کند و در ستون `robusta_source` علامت می‌زند.")
    add("* ردیف‌های آخر هفته: (الف) مشاهده‌نشده حساب شوند [پیش‌فرض موقت]؛ (ب) ردیف شنبه به‌جای جمعه بنشیند. پیشنهاد: **(الف)** تا فایل دستی برسد.")
    add("")
    add("**۹-۵. بازده‌های تقویم مشترک هم‌طول نیستند (برای D2)**")
    add(f"* یافته: فقط {num(win3.mean(), 1)} روز مشترک در ۳ ماه، آمیخته‌ای از بازده‌های ۱ روزه و ۴ تا ۵ روزه.")
    add("* گزینه‌ها: (الف) بوت‌استرپ روی همین بازده‌ها، با شمردن افق بر حسب روز تقویمی؛ (ب) بازده‌های هفتگی (چهارشنبه تا چهارشنبه)؛ (ج) بوت‌استرپ جدا برای هر تقویم و جفت‌کردن بلوک‌ها با تاریخ.")
    add("* پیشنهاد: در D2 تصمیم بگیرید؛ نظر اولیهٔ من **(ب)** است، چون ساده و هم‌طول است و هیچ حرکتی را گم نمی‌کند.")
    add("")
    add("**۹-۶. قیمت آتی، قیمتی نیست که واردکننده می‌پردازد (برای D2)**")
    add(f"* یافته: قیمت آتی عربیکا {pct(1 - imf_year['arabica_ratio'].max(), 0)} تا {pct(1 - imf_year['arabica_ratio'].min(), 0)} پایین‌تر از شاخص فیزیکی IMF است و این فاصله ناپایدار است. برای روبوستا فاصله پایدار است.")
    add("* گزینه‌ها: (الف) فقط قیمت آتی؛ (ب) یک ورودی «اضافه‌بها روی قیمت آتی» در ماشین‌حساب، با پیش‌فرضی از همین جدول.")
    add("* پیشنهاد: **(ب)**.")
    add("")
    add("**۹-۷. شاخص هزینهٔ فرصت از ۱۳۹۴ شروع می‌شود (برای D2)**")
    add(f"* یافته: صندوق از {jd(chosen['first_traded'])} داده دارد؛ حدود {num((_date(usd.index.max()) - chosen['first_traded']).days / 365.25, 1)} سال.")
    add("* گزینه‌ها: (الف) بک‌تست راهبردها از ۱۳۹۴ [پیشنهاد]؛ (ب) برای پیش از آن یک نرخ ثابت سالانه فرض شود.")
    add("")
    add("**۹-۸. دادهٔ عمومی در ماشین‌حساب (برای مرحله‌های بعد)**")
    add("* یافته: TGJU کپی را به موافقت مکتوب مشروط کرده و سری‌های IMF حق نشر دارند.")
    add("* پیشنهاد: ماشین‌حساب عمومی فقط پارامترهای برآوردشده و شکل‌ها را داشته باشد، نه سری خام. اگر سری خام لازم شد، پیش از انتشار از TGJU اجازه بگیرید.")
    add("")
    return "\n".join(L) + "\n"


def main() -> None:
    text = build()
    (OUT / "README.md").write_text(text, encoding="utf-8")
    print(f"wrote {OUT / 'README.md'} ({len(text.splitlines())} lines) and figures in {OUT}")


if __name__ == "__main__":
    main()
