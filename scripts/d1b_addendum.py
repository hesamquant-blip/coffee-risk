#!/usr/bin/env python
"""Build the D1b addendum: reports/d1_data/addendum_d1b.md (Persian) and its
three figures. The D1 report itself is not touched.

Every number, table and figure is computed here from the raw cache, the daily
panel and the weekly panel.

FROZEN since D2b. This script produced the addendum the author reviewed, at
the commit tagged `d1b`. D2b then accepted the ICO July file by exception
(D-021), which changes what this script would print, and a reviewed report is
not rewritten. What changed is in the note at the top of
reports/d2_eda/README.md. To reproduce the addendum:

    git checkout d1b
    .venv/bin/python -m coffee_risk.data build-panel
    .venv/bin/python scripts/d1b_addendum.py

Section 2 compares the TGJU columns with the ones D1 built from the chart
endpoint. It reads the two D1 panel files from data/processed/d1_snapshot/.
To recreate them: `git checkout d1`, run `build-panel`, copy
data/processed/panel_daily.parquet and official_usd_daily.parquet into that
folder. Without them the section says so.
"""
from __future__ import annotations

import sys

if __name__ == "__main__":
    sys.exit("scripts/d1b_addendum.py is frozen (see its docstring): "
             "check out the tag `d1b` to reproduce the D1b addendum.")

import datetime as dt

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd

from coffee_risk import paths
from coffee_risk.dates import to_jalali
from coffee_risk.reporting import (AQUA, BLUE, INK2, ORANGE, end_label, fa, jd, jg, num, pct,
                                   style, table)
from coffee_risk.data import (checks, http, ico, panel as P, quality, rolls, tgju,
                              weekly)

OUT = paths.REPORTS / "d1_data"
SNAPSHOT = paths.PROCESSED / "d1_snapshot"       # the D1 panel files (tag `d1`), kept for task 1
WAR_START = dt.date(2026, 2, 28)                 # D-005: 9 Esfand 1404
FIG_START = dt.date(2025, 12, 22)                # 1 Dey 1404: shows the Dey freeze before the war
ERAS = [(2008, 2012), (2013, 2016), (2017, 2020), (2021, 2026)]
ROBUSTA_BETA = 0.5                               # share of the arabica return removed from robusta
# windows named in the D1 review (Jalali dates)
WINDOWS = [("1394-07-01", "1394-11-30"), ("1397-01-21", "1397-02-01"),
           ("1397-07-27", "1397-08-27")]
WEEKLY_SERIES = ["usd_free", "usd_tgju", "arabica_usd_kg", "robusta_usd_kg", "fund_tr_index"]
LABEL = {"usd_free": "دلار Bonbast", "usd_tgju": "دلار TGJU", "arabica": "عربیکا",
         "robusta": "روبوستا", "fund": "صندوق", "arabica_usd_kg": "عربیکا",
         "robusta_usd_kg": "روبوستا", "fund_tr_index": "صندوق"}


def _save(fig, name: str) -> str:
    fig.savefig(OUT / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return name


# --------------------------------------------------------------------- figures
def fig_war(p: pd.DataFrame, runs: pd.DataFrame) -> str:
    """Bonbast and TGJU from Dey 1404, with near-stale runs shaded."""
    start = pd.Timestamp(FIG_START)
    b = p.loc[start:, "usd_free"].dropna()
    t = p.loc[start:, "usd_tgju"].dropna()
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    for name, color, label in (("usd_tgju", BLUE, "TGJU near-stale run"),
                               ("usd_free", ORANGE, "Bonbast near-stale run")):
        first = True
        for _, r in runs[(runs["series"] == name) & (runs["last_date"] >= start)].iterrows():
            ax.axvspan(max(r["first_date"], start) - pd.Timedelta(hours=12),
                       r["last_date"] + pd.Timedelta(hours=12), color=color, alpha=0.14, lw=0,
                       label=label if first else None)
            first = False
    ax.plot(b.index, b.values, color=ORANGE, lw=1.6, label="Bonbast sell (primary)")
    ax.plot(t.index, t.values, color=BLUE, lw=1.6, label="TGJU price_dollar_rl (check)")
    ax.axvline(pd.Timestamp(WAR_START), color=INK2, lw=0.8)
    ax.annotate(f"war starts\n{WAR_START}", (pd.Timestamp(WAR_START), b.max()), xytext=(4, 0),
                textcoords="offset points", va="top", fontsize=8, color=INK2)
    ax.set_ylabel("rial per USD")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.set_title("Free-market USD since Dey 1404: Bonbast vs TGJU, near-stale runs shaded", pad=26)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=4, borderaxespad=0.2, fontsize=8)
    end_label(ax, b.index[-1], b.iloc[-1], f"{b.iloc[-1]:,.0f}")
    return _save(fig, "fig_d1b_1_usd_war_period.png")


def _break_holes(s: pd.Series, days: int = 5) -> pd.Series:
    """Insert an empty point inside every hole longer than `days`, so a line
    is not drawn across a stretch that has no data."""
    s = s.dropna()
    holes = s.index[1:][(s.index[1:] - s.index[:-1]).days > days]
    extra = pd.Series(np.nan, index=[h - pd.Timedelta(days=1) for h in holes])
    return pd.concat([s, extra]).sort_index()


def fig_robusta(p: pd.DataFrame) -> str | None:
    """ICO Robustas (physical) and TGJU robusta (futures), both set to 100 on
    their first common date; the holes in TGJU's data are shaded."""
    ico_s = p[weekly.ICO_COL].dropna()
    if ico_s.empty:
        return None
    tg = p.loc[ico_s.index.min():, "robusta_usd_t"].dropna()
    base = ico_s.index.intersection(tg.index)[0]
    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    first = True
    for a, b in zip(tg.index[:-1], tg.index[1:]):
        if (b - a).days > weekly.MAX_AGE_DAYS:
            ax.axvspan(a, b, color=BLUE, alpha=0.13, lw=0,
                       label="hole in TGJU's data" if first else None)
            first = False
    i_line = _break_holes(ico_s / ico_s[base] * 100)
    t_line = _break_holes(tg / tg[base] * 100)
    ax.plot(i_line.index, i_line.values, color=ORANGE, lw=1.6,
            label="ICO Robustas indicator (physical; validated months)")
    ax.plot(t_line.index, t_line.values, color=BLUE, lw=1.6, label="TGJU London robusta (futures)")
    ax.set_ylabel(f"index, {base.date()} = 100")
    ax.set_title("Robusta: ICO indicator vs TGJU futures, both indexed", pad=26)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, borderaxespad=0.2, fontsize=8)
    locator = mdates.AutoDateLocator(minticks=5, maxticks=10)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    return _save(fig, "fig_d1b_2_robusta_sources.png")


def fig_weekly_vol(vol: pd.DataFrame) -> str:
    """Weekly volatility per Jalali year: dollar, arabica, robusta."""
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    x = np.arange(len(vol))
    for col, color, label in (("usd_free_std", ORANGE, "USD (Bonbast)"),
                              ("arabica_usd_kg_std", BLUE, "arabica"),
                              ("robusta_usd_kg_std", AQUA, "robusta")):
        y = vol[col].values * 100
        ax.plot(x, y, color=color, lw=1.8, marker="o", ms=5, mec="#fcfcfb", mew=1.2, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels([str(y) for y in vol.index])
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Jalali year (first and last year are partial)")
    ax.set_ylabel("std of weekly log return, %")
    ax.set_title("Weekly volatility by year (Wednesday to Wednesday)", pad=26)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, borderaxespad=0.2)
    return _save(fig, "fig_d1b_3_weekly_volatility.png")


# ------------------------------------------------------------------ the report
def build() -> str:
    OUT.mkdir(parents=True, exist_ok=True)
    style(plt)
    p = P.load()
    wk = weekly.load()
    official = pd.read_parquet(P.OFFICIAL)
    L: list[str] = []
    add = L.append

    # ------------------------------------------------------------ computations
    snap = checks.snapshot_equality(p, official, SNAPSHOT) if (SNAPSHOT / "panel_daily.parquet").exists() else None
    chart_files = sorted(f.name for f in http.raw_path(tgju.SOURCE, "x").parent.glob("chart_*.json"))
    usd, tg_usd = p["usd_free"].dropna(), p["usd_tgju"].dropna()
    by_year = checks.usd_by_year(p)
    runs = checks.near_stale_table(p, P.CHECKED)
    wins = checks.window_medians(p, WINDOWS)
    quarter = checks.quarter_changes(wk)
    big = checks.weekly_big_moves(wk)
    wcorr = checks.weekly_usd_correlation(p).set_index("bonbast")
    extremes = checks.usd_extremes(p)
    n_ex = extremes["verdict"].value_counts().to_dict()

    rb, rb_tgju = P.robusta_series()
    ico_rep = ico.report()
    rbi = weekly.robusta_inputs(p, wk.index)
    test = rbi["test"]
    rb_missing = wk.index[weekly.missing_weeks(p["robusta_usd_kg"], wk.index)]
    rb_gaps = quality.gaps(p["robusta_usd_t"], iran=False).sort_values("calendar_days", ascending=False)
    weekend = p.index[p["robusta_weekend"]]
    ar = p["arabica_clb"].dropna()
    rb_primary = p["robusta_usd_t"].dropna()
    we26 = [d for d in weekend if d.year == 2026]
    periods = [(str(y), dt.date(y, 1, 1), dt.date(y, 12, 31)) for y in range(2021, 2026)]
    if we26:
        periods += [("2026a", dt.date(2026, 1, 1), (min(we26) - pd.Timedelta(days=1)).date()),
                    ("2026b", min(we26).date(), max(we26).date()),
                    ("2026c", (max(we26) + pd.Timedelta(days=1)).date(), p.index.max().date())]
    lag_ar = checks.lag_by_period(rb_primary, ar, periods)
    # weekend-dated rows against the weekday rows around them
    t_all = p["robusta_tgju_usd_t"].dropna()
    t_wd = t_all[~p["robusta_weekend"].reindex(t_all.index)]
    we_rows = []
    for d in weekend:
        before, after = t_wd[:d], t_wd[d:]
        if len(before) and len(after):
            we_rows.append((abs(t_all[d] / before.iloc[-1] - 1), abs(t_all[d] / after.iloc[0] - 1)))
    we_cmp = pd.DataFrame(we_rows, columns=["vs_prev", "vs_next"])
    rb_series = pd.Series(rb_primary.values, index=[d.date() for d in rb_primary.index])
    ar_series = pd.Series(ar.values, index=[d.date() for d in ar.index])
    rb_cal = rolls.trading_calendar(list(rb_series.index), ref_dates=list(ar_series.index))
    rb_ev = rolls.days_left_evidence(rb_series, ar_series, rb_cal, rolls.ROBUSTA, ERAS,
                                     beta=ROBUSTA_BETA)

    flags = checks.weekly_flags_by_year(wk, WEEKLY_SERIES)
    vol = checks.weekly_volatility_by_year(wk, WEEKLY_SERIES)
    roll_fx = checks.weekly_roll_effect(wk, ["arabica_usd_kg", "robusta_usd_kg"]).set_index("series")
    fund_old = wk.index[wk["fund_tr_index_old"] & wk["fund_tr_index"].notna()]

    # ----------------------------------------------------------------- figures
    f1 = fig_war(p, runs)
    f2 = fig_robusta(p)
    f3 = fig_weekly_vol(vol)

    # ================================================================== text
    today = dt.date.today()
    add("# الحاقیهٔ D1b — دلار Bonbast، پنل هفتگی، حفره‌های روبوستا")
    add("")
    add(f"تاریخ ساخت: {jg(today)}. این الحاقیه گزارش D1 را بازنویسی نمی‌کند؛ فقط آنچه پس از بازبینی D1 عوض شد را گزارش می‌کند. "
        "همهٔ عددها، جدول‌ها و شکل‌ها را `scripts/d1b_addendum.py` ساخته است.")
    add("")
    add("تصمیم‌های بازبینی (D-001 بازنگری‌شده و D-007 تا D-020) در `DECISIONS.md` ثبت شده‌اند. "
        "گزارش D1 با کدِ برچسب `d1` در گیت بازتولید می‌شود؛ اسکریپت آن دیگر روی کد فعلی اجرا نمی‌شود.")
    add("")

    # ---- 1
    add("## ۱. خلاصه")
    add("")
    if snap is not None:
        s_usd = snap.set_index("key").loc[tgju.USD_KEY]
        add(f"1. **TGJU حالا فقط از نقطهٔ جدول تاریخچه می‌آید** و ستون D1 را دقیقاً بازمی‌سازد "
            f"({num(s_usd['equal'])} از {num(s_usd['common'])} تاریخ برابر). نقطهٔ نمودار و قانون «میلهٔ آخر» از کد حذف شد.")
    else:
        add("1. **TGJU حالا فقط از نقطهٔ جدول تاریخچه می‌آید.** فایل‌های پنل D1 برای مقایسه در دسترس نبود.")
    n_tg = int((runs["series"] == "usd_tgju").sum())
    n_bb = runs[runs["series"] == "usd_free"]
    add(f"2. **دلار اصلی Bonbast است** (`usd_free`) و TGJU ستون کنترل (`usd_tgju`). آشکارساز «تقریباً راکد» "
        f"{num(n_tg)} دوره در TGJU پیدا می‌کند که بررسی دقیق‌برابر فقط {num(int(quality.stale_runs(p['usd_tgju']).shape[0]))} تای آن را می‌دید. "
        f"از {num(len(n_bb))} دورهٔ Bonbast، {num(int(n_bb['exact_repeat'].sum()))} تا تکرار دقیق عدد در تعطیلات است.")
    add(f"3. **پنل هفتگی ساخته شد:** {num(len(wk))} چهارشنبه از {jd(wk.index.min())} تا {jd(wk.index.max())}. "
        f"بازده‌های هفتگی دلار جمع‌پذیرند و بازدهٔ هفتگی قهوه روزهای رول را کنار می‌گذارد.")
    n_fill = int((wk["robusta_return_source"] == "ico").sum())
    still = [d for d in rb_missing if pd.isna(wk.loc[d, "robusta_usd_kg_ret"])]
    bad_months = ico_rep[ico_rep["loaded"] & (ico_rep["ok"] != True)]             # noqa: E712
    add(f"4. **حفره‌های روبوستا با شاخص ICO پر شد (D-020):** آزمون سازگاری {'قبول شد' if test['passed'] else '**رد شد**'} "
        f"({num(test['weeks'])} هفته، همبستگی {num(test['corr'], 2)}، نسبت نوسان {num(test['vol_ratio'], 2)}). "
        f"{num(n_fill)} بازدهٔ هفتگی از ICO آمد و {num(len(still))} هفته خالی ماند"
        + (f"، چون PDF ماه {'، '.join(fa(f'{int(r.year)}-{int(r.month):02d}') for r in bad_months.itertuples())} اعتبارسنجی را رد شد." if len(bad_months) and still else "."))
    add("5. **از شما لازم است:** تصمیم دربارهٔ ماه ردشدهٔ ICO و تأیید برداشت من از D-018 (بخش ۷).")
    add("")

    # ---- 2
    add("## ۲. نقطهٔ دریافت TGJU (D-007)")
    add("")
    add("همهٔ کلیدهای TGJU حالا از `api.tgju.org` (نقطهٔ پشت جدول تاریخچه) خوانده می‌شوند: دلار آزاد، روبوستا و پنج سری کانال رسمی. "
        "جدول برای هر تاریخ فقط یک ردیف دارد، پس قانون «میلهٔ آخر» لازم نیست. اگر روزی تاریخی دو بار بیاید، کد می‌ایستد و بی‌صدا انتخاب نمی‌کند.")
    add("")
    if snap is not None:
        add("مقایسهٔ ستون تازه (از جدول) با ستون D1 (از نمودار)، تاریخ به تاریخ:")
        add("")
        rows = []
        for _, r in snap.iterrows():
            rows.append([f"`{r['key']}`", num(r["d1_rows"]), num(r["table_rows"]), num(r["equal"]),
                         num(r["common"] - r["equal"]), num(r["only_d1"]), num(r["only_table"])])
        add(table(["کلید", "ردیف در D1", "ردیف در جدول", "برابر", "نابرابر",
                   "فقط در D1", "فقط در جدول"], rows))
        add("")
        diff = snap[snap["common"] != snap["equal"]]
        ok = snap[snap["key"].isin([tgju.USD_KEY, tgju.ROBUSTA_KEY])]
        if (ok["common"] == ok["equal"]).all() and (ok["only_d1"] + ok["only_table"]).sum() == 0:
            add("* **دلار و روبوستا دقیقاً یکی‌اند.** هیچ عددی در پنل با این تغییر عوض نشد.")
        for _, r in diff.iterrows():
            add(f"* `{r['key']}` در {num(len(r['differing_dates']))} تاریخ فرق دارد: "
                + "، ".join(str(d) for d in r["differing_dates"])
                + ". در این تاریخ‌ها جدول میله‌ای غیر از «میلهٔ آخرِ» نمودار را نشان می‌دهد. این سری فقط برای اطلاع است.")
    add(f"* فایل‌های خام نقطهٔ نمودار ({num(len(chart_files))} فایل `chart_*.json`) به‌عنوان سابقه در `data/raw/tgju/` مانده‌اند. هیچ کدی آن‌ها را نمی‌خواند.")
    add("")

    # ---- 3
    add("## ۳. دلار اصلی = Bonbast (D-001 بازنگری‌شده)")
    add("")
    add("### ۳-۱. ستون‌های پنل")
    add("")
    rows = [["`usd_free`", "Bonbast، نرخ فروش، تومان × ۱۰، تاریخ همان‌طور که منتشر شده", "اصلی",
             jg(usd.index.min()), jg(usd.index.max()), num(len(usd))],
            ["`usd_tgju`", "TGJU، کلید `price_dollar_rl`، قیمت پایانی", "کنترل",
             jg(tg_usd.index.min()), jg(tg_usd.index.max()), num(len(tg_usd))]]
    add(table(["ستون", "منبع", "نقش", "اولین تاریخ", "آخرین تاریخ", "ردیف"], rows))
    add("")
    add("* نام ستون‌ها نسبت به D1 عوض شده است: `usd_irr_bonbast` ← `usd_free` و `usd_irr` ← `usd_tgju`.")
    add(f"* Bonbast برای هر روز تقویمی یک عدد دارد، حتی جمعه و تعطیلات. پیش از {jd(usd.index.min())} دلار اصلی داده ندارد.")
    add("")

    add("### ۳-۲. بازسازی یافتهٔ بازبینی")
    add("")
    add("یافتهٔ شما از روی فایل‌های خام این بود که TGJU در چند دوره نرخ بازار آزاد خیابانی نیست. همان عددها از روی پنل:")
    add("")
    rows = [[f"{fa(r['from'])} تا {fa(r['to'])}", num(r["common_days"]),
             f"{num(r['tgju_min'])} تا {num(r['tgju_max'])}", num(r["tgju_median"]),
             f"{num(r['bonbast_min'])} تا {num(r['bonbast_max'])}", num(r["bonbast_median"]),
             pct(r["median_gap"], 1, True), pct(r["tgju_median_abs_step"], 2)]
            for _, r in wins.iterrows()]
    add(table(["بازه", "روز مشترک", "بازهٔ TGJU", "میانهٔ TGJU", "بازهٔ Bonbast", "میانهٔ Bonbast",
               "میانهٔ فاصله (TGJU نسبت به Bonbast)", "میانهٔ گام روزانهٔ TGJU"], rows))
    add("")
    add(f"* بزرگ‌ترین رشد ۱۳ هفته‌ای که در ۱۳۹۷ تمام می‌شود: Bonbast {pct(quarter['max_rise_1397_bonbast'], 0, True)}، TGJU {pct(quarter['max_rise_1397_tgju'], 0, True)} "
        f"(هر دو در هفتهٔ {jd(quarter['max_rise_1397_bonbast_week'])})." if quarter["max_rise_1397_bonbast_week"] == quarter["max_rise_1397_tgju_week"] else
        f"* بزرگ‌ترین رشد ۱۳ هفته‌ای که در ۱۳۹۷ تمام می‌شود: Bonbast {pct(quarter['max_rise_1397_bonbast'], 0, True)} (هفتهٔ {jd(quarter['max_rise_1397_bonbast_week'])})، "
        f"TGJU {pct(quarter['max_rise_1397_tgju'], 0, True)} (هفتهٔ {jd(quarter['max_rise_1397_tgju_week'])}).")
    add(f"* همبستگی تغییرهای ۱۳ هفته‌ایِ بدون هم‌پوشانی، بسته به اینکه شمارش از کدام هفته شروع شود ({num(13)} حالت، هر کدام حدود {num(quarter['blocks'])} بازه): "
        f"با ۱۳۹۷ بین {num(quarter['corr_all_min'], 2)} و {num(quarter['corr_all_max'], 2)}؛ "
        f"بدون بازه‌هایی که به ۱۳۹۷ می‌خورند بین {num(quarter['corr_ex1397_min'], 2)} و {num(quarter['corr_ex1397_max'], 2)}.")
    if quarter["corr_ex1397_min"] < 0.98:
        add(f"* این عددها با یافتهٔ شما می‌خواند، با یک تفاوت: همبستگی ۱۳ هفته‌ای بیرون از ۱۳۹۷ در بعضی نقطه‌های شروع تا {num(quarter['corr_ex1397_min'], 2)} پایین می‌آید، نه ۰.۹۸.")
    else:
        add("* این عددها با یافتهٔ شما می‌خواند.")
    add("")

    add("### ۳-۳. آشکارساز «تقریباً راکد»")
    add("")
    add(f"«تقریباً راکد» یعنی {num(quality.STALE_RUN)} مشاهدهٔ پشت‌سرهم یا بیشتر که هر کدام کمتر از {pct(quality.NEAR_STALE_STEP, 2)} با مشاهدهٔ قبلی فرق دارد (تغییر لگاریتمی). "
        "بررسی قدیمیِ «عدد دقیقاً تکراری» هم سر جایش است؛ هر دو در پنل ستون دارند (`*_near_stale` و `*_stale`).")
    add("")
    rows = []
    for name, col in P.CHECKED.items():
        r = runs[runs["series"] == name]
        ex = quality.stale_runs(p[col])
        rows.append([LABEL[name], num(len(r)), num(int(r["observations"].sum()) if len(r) else 0),
                     num(int(r["observations"].max()) if len(r) else 0),
                     num(len(ex)), num(int(ex["observations"].sum()) if len(ex) else 0)])
    add(table(["سری", "دورهٔ تقریباً راکد", "مشاهده در این دوره‌ها", "بلندترین دوره",
               "دورهٔ دقیقاً تکراری", "مشاهده در آن‌ها"], rows))
    add("")
    add("**دوره‌های TGJU:**")
    add("")
    tr = runs[runs["series"] == "usd_tgju"]
    b_all = p["usd_free"]
    rows = []
    for _, r in tr.iterrows():
        seg = b_all.loc[r["first_date"]:r["last_date"]].dropna()
        rows.append([jg(r["first_date"]), jg(r["last_date"]), num(r["observations"]),
                     num(r["first_value"]), num(r["last_value"]), pct(r["max_abs_step"], 3),
                     (f"{num(seg.iloc[0])} ← {num(seg.iloc[-1])}" if len(seg) else "—"),
                     (pct(seg.iloc[-1] / seg.iloc[0] - 1, 1, True) if len(seg) else "—")])
    add(table(["از", "تا", "مشاهده", "اولین عدد", "آخرین عدد", "بزرگ‌ترین گام",
               "Bonbast در همان بازه (اول ← آخر)", "تغییر Bonbast"], rows))
    add("")
    war_runs = tr[tr["last_date"] >= pd.Timestamp(FIG_START)]
    add(f"* {num(int((tr['first_date'].map(lambda d: to_jalali(d.date())[:4]) == '1394').sum()))} دوره در ۱۳۹۴ است و {num(len(war_runs))} دوره از دی ۱۴۰۴ به بعد.")
    first_war = tr[(tr["first_date"] <= pd.Timestamp(WAR_START) + pd.Timedelta(days=3)) & (tr["last_date"] >= pd.Timestamp(WAR_START))]
    if len(first_war):
        r = first_war.iloc[0]
        seg = b_all.loc[r["first_date"]:r["last_date"]].dropna()
        nxt = tg_usd[tg_usd.index > r["last_date"]]
        add(f"* **دو هفتهٔ اول جنگ:** TGJU از {jd(r['first_date'])} تا {jd(r['last_date'])} روی حدود {num(r['first_value'])} ریال مانده است، "
            f"در حالی که Bonbast از {num(seg.iloc[0])} به {num(seg.iloc[-1])} رسیده ({pct(seg.iloc[-1] / seg.iloc[0] - 1, 1, True)}). "
            + (f"روز بعد ({jd(nxt.index[0])}) TGJU یک‌باره {pct(np.log(nxt.iloc[0] / r['last_value']), 1, True)} حرکت می‌کند: "
               "همان «جهش تأییدنشده»ی گزارش D1، که در واقع جبرانِ دورهٔ راکد است." if len(nxt) else ""))
    add("")
    add("**دوره‌های Bonbast:**")
    add("")
    br = runs[runs["series"] == "usd_free"]
    rows = []
    for _, r in br.iterrows():
        nz = quality.covers_nowruz(r["first_date"].date() - dt.timedelta(days=1),
                                   r["last_date"].date() + dt.timedelta(days=1))
        rows.append([jg(r["first_date"]), jg(r["last_date"]), num(r["observations"]), num(r["first_value"]),
                     "بله" if r["exact_repeat"] else pct(r["max_abs_step"], 3),
                     "نوروز" if nz else ("جنگ" if r["first_date"].date() >= WAR_START else "—")])
    add(table(["از", "تا", "روز", "عدد", "تکرار دقیق؟ (وگرنه بزرگ‌ترین گام)", "توضیح"], rows))
    add("")
    br_war = br[br["first_date"] >= pd.Timestamp(WAR_START)]
    br_nz = sum(1 for _, r in br.iterrows() if quality.covers_nowruz(
        r["first_date"].date() - dt.timedelta(days=1), r["last_date"].date() + dt.timedelta(days=1)))
    add(f"* {num(br_nz)} دوره از {num(len(br))} دوره روزهای نوروز را در بر دارد و {num(len(br_war))} دوره پس از آغاز جنگ است. "
        "این همان محدودیت شناخته‌شدهٔ Bonbast است: در تعطیلی یا قطعی، آرشیو عدد قبلی را تکرار می‌کند.")
    fr = runs[runs["series"] == "fund"]
    coffee_runs = int(runs["series"].isin(["arabica", "robusta"]).sum())
    add(f"* صندوق {num(len(fr))} دوره دارد که {num(int(fr['exact_repeat'].sum()))} تای آن تکرار دقیق عدد است: در این دوره‌ها قیمت روزی کمتر از {pct(quality.NEAR_STALE_STEP, 2)} بالا رفته است. "
        f"دو سری قهوه روی هم {num(coffee_runs)} دوره دارند.")
    if len(fr):
        add("* دوره‌های صندوق: " + "؛ ".join(f"{jd(r['first_date'])} تا {jd(r['last_date'])} ({num(r['observations'])} مشاهده)" for _, r in fr.iterrows()) + ".")
    add("")
    add(f"![دلار در دورهٔ جنگ]({f1})")
    add("")
    add(f"*شکل ۱ — دلار آزاد از {jd(FIG_START)}: Bonbast (نارنجی، اصلی) و TGJU (آبی، کنترل). نوارهای کم‌رنگ دوره‌های تقریباً راکدِ هر منبع‌اند، به همان رنگ.*")
    add("")

    add("### ۳-۴. TGJU در برابر Bonbast، به تفکیک سال")
    add("")
    add("مرجع Bonbast است. فاصله = TGJU تقسیم بر Bonbast در همان تاریخ، منهای یک.")
    add("")
    rows = [[fa(y), num(r["bonbast_days"]), num(r["tgju_days"]), pct(r["median_abs_gap"]),
             pct(r["median_gap"], 2, True), num(r["days_over"]), num(r["days_over_best"]),
             num(r["usd_free_near_stale_days"]), num(r["usd_tgju_near_stale_days"])]
            for y, r in by_year.iterrows()]
    add(table(["سال", "روز Bonbast", "روز مشترک با TGJU", "میانهٔ قدرمطلق فاصله", "میانهٔ فاصله",
               f"روزهای فاصله بالای {pct(checks.GAP, 0)}", "همان، با یک روز جابه‌جایی مجاز",
               "روز تقریباً راکد Bonbast", "روز تقریباً راکد TGJU"], rows))
    add("")
    worst = by_year.sort_values("days_over_best", ascending=False).head(2)
    add("* بدترین سال‌ها: " + " و ".join(f"{fa(y)} ({num(r['days_over_best'])} روز، میانهٔ فاصله {pct(r['median_gap'], 1, True)})"
                                        for y, r in worst.iterrows()) + ". در هر دو TGJU پایین‌تر از Bonbast است.")
    add(f"* بررسی روزانهٔ D1 هم سر جایش است (ستون `usd_outlier_check`، فقط برای اطلاع، D-008): از {num(len(extremes))} حرکت روزانهٔ بالای {pct(quality.USD_EXTREME, 0)} در TGJU، "
        f"{num(n_ex.get('confirmed', 0))} تأیید، {num(n_ex.get('not_confirmed', 0))} تأییدنشده و {num(n_ex.get('no_bonbast', 0))} پیش از آغاز Bonbast. "
        f"خود Bonbast {num(int(p['usd_free_outlier'].sum()))} حرکت روزانهٔ بالای {pct(quality.USD_EXTREME, 0)} دارد.")
    add("")

    add("### ۳-۵. حرکت‌های هفتگی بزرگ Bonbast و بررسی با TGJU")
    add("")
    add(f"هفته‌هایی که بازدهٔ لگاریتمی چهارشنبه تا چهارشنبهٔ Bonbast از {pct(checks.WEEKLY_BIG, 0)} بیشتر است: {num(len(big))} هفته از {num(int(wk['usd_free_ret'].notna().sum()))} هفته. "
        "«تأیید» یعنی TGJU در همان هفته هم‌جهت و دست‌کم نصف آن اندازه حرکت کرده است.")
    add("")
    vfa = {"confirmed": "تأیید", "not_confirmed": "**تأیید نشد**", "no_tgju": "TGJU ندارد"}
    rows = []
    for d, r in big.iterrows():
        note = []
        if r["tgju_old"]:
            note.append("TGJU بیش از ۵ روز بدون داده")
        if r["tgju_near_stale"]:
            note.append("TGJU تقریباً راکد")
        if r["bonbast_near_stale"]:
            note.append("Bonbast تقریباً راکد")
        rows.append([jg(d), num(r["level_before"]), num(r["level"]), pct(r["bonbast_ret"], 1, True),
                     pct(r["tgju_ret"], 1, True), vfa[r["verdict"]], "؛ ".join(note) or "—"])
    add(table(["چهارشنبه", "Bonbast هفتهٔ قبل", "Bonbast", "بازده Bonbast", "بازده TGJU", "نتیجه", "یادداشت"], rows))
    add("")
    nb = big["verdict"].value_counts().to_dict()
    in97 = sum(1 for d in big.index if to_jalali(d.date()).startswith("1397"))
    add(f"* تأیید: {num(nb.get('confirmed', 0))}. تأیید نشد: {num(nb.get('not_confirmed', 0))}. {num(in97)} هفته از {num(len(big))} هفته در ۱۳۹۷ است.")
    unc = big[big["verdict"] == "not_confirmed"]
    if len(unc):
        add("* هفته‌های تأییدنشده: " + "، ".join(jd(d) for d in unc.index) + ". "
            "این‌ها هفته‌هایی هستند که TGJU یا بسته بوده یا نرخ مدیریت‌شده نشان داده است؛ یعنی همان دوره‌هایی که دلیل تغییر D-001 بودند.")
    add("")

    add("### ۳-۶. همبستگی بازدهٔ هفتگی با TGJU")
    add("")
    add("فقط برای اطلاع. Bonbast یک بار همان‌طور که منتشر شده و یک بار با یک روز جابه‌جایی به عقب (عددِ روز t+1 روی روز t) هفتگی شده است. **در پنل هیچ جابه‌جایی انجام نشده است.**")
    add("")
    lab = {"as_published": "همان‌طور که منتشر شده", "shifted_minus_1_day": "یک روز به عقب"}
    rows = [[lab[k], num(r["weeks"]), num(r["corr"], 3), num(r["weeks_without_1397"]),
             num(r["corr_without_1397"], 3), pct(r["std_of_difference"])] for k, r in wcorr.iterrows()]
    add(table(["Bonbast", "هفته", "همبستگی", "هفته بدون ۱۳۹۷", "همبستگی بدون ۱۳۹۷",
               "انحراف معیار اختلاف دو بازده"], rows))
    add("")
    a, b = wcorr.loc["as_published"], wcorr.loc["shifted_minus_1_day"]
    add(f"* جابه‌جایی یک‌روزه همبستگی را از {num(a['corr'], 2)} به {num(b['corr'], 2)} می‌برد. یعنی بخشی از اختلاف هفتگی دو منبع فقط زمان‌بندی عکس روزانهٔ Bonbast است؛ "
        "بقیه اختلاف واقعی دو منبع است.")
    add(f"* همبستگی هفتگی با اینکه بالا نیست، در افق ۱۳ هفته‌ای به {num(quarter['corr_all_min'], 2)} تا {num(quarter['corr_all_max'], 2)} می‌رسد (بخش ۳-۲). برای افق سه‌ماههٔ مدل، زمان‌بندی یک‌روزه اهمیت کمی دارد.")
    add("")

    # ---- 4
    add("## ۴. روبوستا: پر کردن حفره‌ها با شاخص ICO (D-020)")
    add("")
    add("مسیر Investing.com کنار گذاشته شد (دانلودش بازهٔ تاریخ انتخاب‌شده را نادیده می‌گیرد). فایل یک‌ماههٔ قبلی دیگر خوانده نمی‌شود و خواننده‌اش از کد حذف شد. "
        "به‌جای آن از PDFهای ماهانهٔ «قیمت‌های شاخص» سازمان بین‌المللی قهوه (ICO) استفاده می‌کنیم: یک ردیف برای هر روز معاملاتی، به سنت بر پوند.")
    add("")
    add("**نکتهٔ مهم:** ستون Robustas در ICO قیمت «فیزیکی» است، نه قیمت آتی ICE. پس فقط برای *بازده* استفاده می‌شود و هیچ‌وقت برای *سطح*. "
        "ستون سطح روبوستا همان TGJU می‌ماند و حفره‌هایش پر نمی‌شود.")
    add("")
    add("### ۴-۱. دانلود و اعتبارسنجی")
    add("")
    add(f"بازهٔ خواسته‌شده: {fa(f'{ico.FIRST_MONTH[0]}-{ico.FIRST_MONTH[1]:02d}')} تا {fa(f'{ico.LAST_MONTH[0]}-{ico.LAST_MONTH[1]:02d}')} ({num(len(ico_rep))} ماه). "
        f"هر ماه با {num(len(ico.candidate_urls(2026, 1)))} نام آزموده شد (دو الگوی نام، در `ico.org` و `icocoffee.org`). "
        f"شرط اعتبار: میانگین، بیشینه و کمینهٔ ردیف‌های خوانده‌شده باید با ردیف‌های Average، High و Low چاپ‌شده در خود PDF تا {num(ico.TOLERANCE, 2)} یکی باشد.")
    add("")
    rows = []
    for _, r in ico_rep.iterrows():
        month = fa(f"{int(r['year'])}-{int(r['month']):02d}")
        if not r["loaded"]:
            rows.append([month, "**بارگیری نشد**", "—", "—", "—", "—", "استفاده نمی‌شود"])
        else:
            rows.append([month, "بله", num(r["rows"]),
                         f"{num(r['mean'], 3)} / {num(r['printed_average'], 2)}",
                         f"{num(r['max'], 2)} / {num(r['printed_high'], 2)}",
                         f"{num(r['min'], 2)} / {num(r['printed_low'], 2)}",
                         "معتبر" if r["ok"] is True else "**ردشده؛ استفاده نمی‌شود**"])
    add(table(["ماه", "بارگیری", "ردیف", "میانگین: خوانده / چاپ‌شده", "بیشینه: خوانده / چاپ‌شده",
               "کمینه: خوانده / چاپ‌شده", "نتیجه"], rows))
    add("")
    n_loaded, n_ok = int(ico_rep["loaded"].sum()), int((ico_rep["ok"] == True).sum())          # noqa: E712
    not_loaded = ico_rep[~ico_rep["loaded"]]
    add(f"* بارگیری‌شده: {num(n_loaded)} ماه از {num(len(ico_rep))}. معتبر: {num(n_ok)} ماه.")
    if len(not_loaded):
        add(f"* **بارگیری نشد ({num(len(not_loaded))} ماه):** "
            + "، ".join(fa(f"{int(r.year)}-{int(r.month):02d}") for r in not_loaded.itertuples())
            + ". هیچ‌کدام از نام‌ها جواب نداد (خطای ۴۰۴). صفحهٔ «اطلاعات عمومی بازار» ICO فقط ماه‌های اخیر را پیوند می‌دهد.")
    for r in bad_months.itertuples():
        body = http.raw_path(ico.SOURCE, r.file).read_bytes()
        df_m, _ = ico.parse(body)
        diff = r.printed_average - r.mean
        repeats = df_m[df_m["robustas"] == df_m["robustas"].shift()]
        add(f"* **ماه {fa(f'{int(r.year)}-{int(r.month):02d}')} رد شد:** {num(r.rows)} ردیف خوانده شد و بیشینه و کمینه درست‌اند، ولی میانگین ردیف‌ها {num(r.mean, 3)} است و PDF {num(r.printed_average, 2)} چاپ کرده "
            f"(اختلاف {num(diff, 3)}). این اندازه برابر است با اینکه یک ردیف {num(diff * r.rows, 2)} سنت جابه‌جا باشد. "
            + (f"تنها ردیفی که عیناً عدد روز قبل را تکرار کرده {'، '.join(str(d) for d in repeats['date'])} است. " if len(repeats) else "")
            + "یعنی خواندن درست است و ناسازگاری در خود PDF است. طبق قانون، این ماه استفاده نشد (بخش ۷-۱).")
    add("* فایل‌ها بدون تغییر در `data/raw/ico/` ذخیره و در MANIFEST ثبت شده‌اند. robots.txt سایت ICO این مسیر را منع نکرده است.")
    add("")

    add("### ۴-۲. آزمون سازگاری (پیش از دیدن نتیجه ثابت شده بود)")
    add("")
    add(f"قانون: روی هفته‌های {weekly.ICO_TEST_FROM} تا {weekly.ICO_TEST_TO} که هر دو سری قیمت تازه دارند و روز رول TGJU در هفته نیست، بازدهٔ لگاریتمی هفتگی ICO Robustas با روبوستای TGJU مقایسه می‌شود. "
        f"قبولی = همبستگی دست‌کم {num(weekly.ICO_MIN_CORR, 2)} و نسبت نوسان هفتگی (ICO به TGJU) بین {num(weekly.ICO_VOL_RATIO[0], 2)} و {num(weekly.ICO_VOL_RATIO[1], 2)}.")
    add("")
    rows = [["هفته‌های آزمون", num(test["weeks"]), "—"],
            ["همبستگی بازدهٔ هفتگی", num(test["corr"], 3), "بله" if test["corr"] >= weekly.ICO_MIN_CORR else "**نه**"],
            ["نوسان هفتگی ICO", pct(test["vol_ico"]), "—"],
            ["نوسان هفتگی TGJU", pct(test["vol_tgju"]), "—"],
            ["نسبت نوسان (ICO / TGJU)", num(test["vol_ratio"], 3),
             "بله" if weekly.ICO_VOL_RATIO[0] <= test["vol_ratio"] <= weekly.ICO_VOL_RATIO[1] else "**نه**"]]
    add(table(["سنجه", "مقدار", "در بازهٔ قبولی؟"], rows))
    add("")
    add(f"* **نتیجه: {'قبول' if test['passed'] else 'رد'}.**")
    if test["weeks"]:
        add(f"* هفته‌های آزمون از {jd(test['first_week'])} تا {jd(test['last_week'])} است، نه از ژانویهٔ ۲۰۲۵: PDFهای پیش از آن بارگیری نشد. "
            "هفته‌های ماه ردشده هم در آزمون نیستند. پس آزمون روی هفته‌های کمتری از آنچه در نظر بود انجام شده است.")
    add("")

    add("### ۴-۳. هفته‌های پرشده")
    add("")
    add("هفته‌هایی که بازدهٔ روبوستای TGJU طبق D-018 خالی است (بخش ۵)، و آنچه جایش نشست:")
    add("")
    lo = weekly.last_observation(p["robusta_usd_kg"], wk.index)
    stale = weekly.stale_weeks(lo["obs_date"])
    rows = []
    for d in rb_missing:
        why = "سطح کهنه (داخل حفره)" if stale[d] else "اولین مشاهده پس از حفره"
        src = wk.loc[d, "robusta_return_source"]
        rows.append([jg(d), why, jd(wk.loc[d, "robusta_usd_kg_obs_date"]),
                     pct(wk.loc[d, "robusta_usd_kg_ret_raw"], 2, True), pct(rbi["ico_ret"][d], 2, True),
                     "ICO" if src == "ico" else "**خالی ماند**"])
    add(table(["چهارشنبه", "چرا خالی بود", "آخرین مشاهدهٔ TGJU", "تغییر خام سطح TGJU",
               "بازدهٔ هفتگی ICO", "بازدهٔ نهایی"], rows))
    add("")
    add(f"* {num(n_fill)} هفته از ICO پر شد (`robusta_return_source = \"ico\"`). در این هفته‌ها چیزی بابت رول کنار گذاشته نمی‌شود، چون ICO رول ندارد.")
    if still:
        add(f"* **{num(len(still))} هفته خالی ماند** ({'، '.join(jd(d) for d in still)}): بازدهٔ ICO این هفته‌ها به سطح‌های ماه ردشده نیاز دارد.")
    add("* «تغییر خام سطح TGJU» در هفته‌های داخل حفره صفر است (سطح تکان نخورده) و در اولین هفتهٔ پس از حفره کل حرکتِ حفره را یک‌جا دارد، به‌اضافهٔ پرش رولِ داخل حفره. برای همین این بازده‌ها استفاده نمی‌شوند.")
    if len(rb_gaps):
        add("* حفره‌های سطح روبوستا سر جایشان هستند: "
            + "؛ ".join(f"{jd(r['last_before'])} تا {jd(r['first_after'])} ({num(r['calendar_days'])} روز)" for _, r in rb_gaps.iterrows()) + ".")
    add(f"* ردیف‌های TGJU با تاریخ شنبه یا یکشنبه ({num(len(weekend))} ردیف) در سری روبوستا نیستند (D-010). در ستون `robusta_tgju_usd_t` مانده‌اند و `robusta_weekend` آن‌ها را علامت می‌زند.")
    add("")
    if f2:
        add(f"![روبوستا: ICO و TGJU]({f2})")
        add("")
        add("*شکل ۲ — شاخص فیزیکی ICO Robustas (نارنجی، فقط ماه‌های معتبر) و آتی روبوستای TGJU (آبی)، هر دو در اولین روز مشترک برابر ۱۰۰. نوارهای آبی کم‌رنگ حفره‌های دادهٔ TGJU‌اند.*")
        add("")

    add("### ۴-۴. آیا تاریخ‌های TGJU در ۲۰۲۶ یک روز جابه‌جا شده‌اند؟")
    add("")
    add("فایل Investing بازهٔ مشکوک (بهمن ۱۴۰۴ تا مرداد ۱۴۰۵) را ندارد. پس با عربیکا آزمودیم، که تاریخ‌هایش از منبع دیگری (Yahoo) است: "
        "همبستگی بازدهٔ روزانهٔ روبوستا با عربیکا در تأخیرهای −۱، ۰ و +۱.")
    add("")
    plab = {"2026a": "۲۰۲۶، پیش از اولین ردیف آخر هفته", "2026b": "۲۰۲۶، دورهٔ ردیف‌های آخر هفته",
            "2026c": "۲۰۲۶، پس از آخرین ردیف آخر هفته"}
    rows = [[plab.get(r["period"], fa(r["period"])), f"{jd(r['from'])} تا {jd(r['to'])}", num(r["n"]),
             num(r["lag-1"], 2), f"**{num(r['lag+0'], 2)}**", num(r["lag+1"], 2)] for _, r in lag_ar.iterrows()]
    add(table(["دوره", "بازه", "بازدهٔ روزانه", "تأخیر −۱", "تأخیر ۰", "تأخیر +۱"], rows))
    add("")
    peak0 = int((lag_ar[["lag-1", "lag+0", "lag+1"]].idxmax(axis=1) == "lag+0").sum())
    add(f"* **نه، جابه‌جا نشده‌اند.** در {num(peak0)} دوره از {num(len(lag_ar))} دوره اوج همبستگی در تأخیر ۰ است، از جمله در دورهٔ ردیف‌های آخر هفته. "
        "ردیف‌های روز کاری TGJU تاریخ درست دارند؛ ردیف‌های آخر هفته ردیف اضافه‌اند، نه جابه‌جایی کل سری.")
    if len(we_cmp):
        add(f"* خود ردیف‌های آخر هفته نه تکرار جمعه‌اند و نه پیش‌نمایش دوشنبه: میانهٔ فاصله‌شان از روز کاری قبل {pct(we_cmp['vs_prev'].median())} و از روز کاری بعد {pct(we_cmp['vs_next'].median())} است "
            f"({num(len(we_cmp))} ردیف). منشأشان روشن نیست؛ کنار گذاشتنشان (D-010) درست است.")
    add("")

    add("### ۴-۵. شاهد رول روی سری اصلی")
    add("")
    add(f"قانون رول روبوستا روی کل تاریخچه اعمال می‌شود (D-009): روز پرش روزی است که {num(rolls.ROBUSTA.days_left)} روز معاملاتی تا پایان ماه تحویل مانده است. "
        f"شاهد آماری D1 روی سری اصلی فعلی دوباره حساب شد (بازدهٔ روبوستا منهای {num(ROBUSTA_BETA, 1)} برابر بازدهٔ عربیکا؛ هر خانه: انحراف معیار / میانگین):")
    add("")
    rows = []
    for k in range(0, 7):
        row = [fa(k) + (" ← روز پرش" if k == rolls.ROBUSTA.days_left else "")]
        for a_, b_ in ERAS:
            e = rb_ev[(rb_ev["era"] == f"{a_}-{b_}") & (rb_ev["days_left"] == k)].iloc[0]
            row.append(f"{pct(e['std'])} / {pct(e['mean'], 2, True)}")
        rows.append(row)
    row = ["همهٔ روزها"]
    for a_, b_ in ERAS:
        e = rb_ev[(rb_ev["era"] == f"{a_}-{b_}") & (rb_ev["days_left"] == -1)].iloc[0]
        row.append(f"{pct(e['std'])} / {pct(e['mean'], 2, True)}")
    rows.append(row)
    add(table(["روز مانده تا پایان ماه"] + [fa(f"{a_}–{b_}") for a_, b_ in ERAS], rows))
    add("")
    add("* سری روبوستا همان سری D1 است (TGJU، روزهای کاری)، پس شاهد همان است: فقط در ۲۰۲۱ تا ۲۰۲۶ روز پرش برجسته است. "
        f"در پنل {num(int(p['robusta_roll'].sum()))} روز رول روبوستا و {num(int(p['arabica_roll'].sum()))} روز رول عربیکا علامت دارد.")
    add("")

    # ---- 5
    add("## ۵. پنل هفتگی (D-011)")
    add("")
    add("فایل: `data/processed/panel_weekly.parquet`. هر ردیف یک چهارشنبه است. برای هر سری:")
    add("")
    add("* **سطح** = آخرین مشاهده در آن چهارشنبه یا پیش از آن. اگر چهارشنبه تعطیل باشد، آخرین مشاهدهٔ قبلی می‌آید؛ چیزی پر نمی‌شود. تاریخ همان مشاهده هم ذخیره می‌شود (`*_obs_date`).")
    add("* **بازدهٔ هفتگی دلار و صندوق** = لگاریتم نسبت سطح این چهارشنبه به چهارشنبهٔ قبل.")
    add("* **بازدهٔ هفتگی قهوه** = جمع بازده‌های لگاریتمی روزانهٔ داخل هفته، بدون روزهای رول (`*_ret`). تغییر خامِ سطح هم برای مقایسه هست (`*_ret_raw`).")
    add(f"* **پرچم `*_old`:** آخرین مشاهده بیش از {num(weekly.MAX_AGE_DAYS)} روز تقویمی کهنه است. **پرچم `*_near_stale`:** آن مشاهده داخل یک دورهٔ تقریباً راکد است.")
    add(f"* **بازدهٔ خالی (D-018):** اگر سطحِ آخر هفته کهنه باشد، یا اولین مشاهده پس از حفره‌ای بیش از {num(weekly.MAX_AGE_DAYS)} روزه در آن هفته باشد، بازدهٔ هفتگی خالی می‌ماند، چون حرکتِ یک هفته نیست. `*_ret_raw` خالی نمی‌شود.")
    add("* **روبوستا:** بازده‌های خالی، اگر ICO داشته باشد، از ICO پر می‌شوند و `robusta_return_source` منبع هر بازده را می‌گوید (بخش ۴).")
    add("")
    add(f"* تعداد هفته‌ها: **{num(len(wk))}**، از {jg(wk.index.min())} تا {jg(wk.index.max())}. اولین چهارشنبه، اولین چهارشنبه در یا پس از اولین روز Bonbast ({jd(usd.index.min())}) است.")
    x = wk["usd_free"].dropna()
    tele = abs(wk["usd_free_ret"].sum() - np.log(x.iloc[-1] / x.iloc[0]))
    add(f"* جمع‌پذیری: برای Bonbast، که حفره ندارد، جمع بازده‌های هفتگی برابر لگاریتم نسبت آخرین سطح به اولین سطح است (خطا: {fa(f'{tele:.1e}')}). برای سری‌های حفره‌دار این برابری با D-018 دیگر برقرار نیست؛ تغییر چندهفته‌ای آن‌ها باید از سطح حساب شود.")
    add(f"* ستون‌ها: {num(wk.shape[1])}. `usd_tgju` هم به‌عنوان کنترل در پنل هفتگی هست.")
    add("")
    add("**هفته‌های پرچم‌دار، به تفکیک سال** (هر خانه: کهنه / تقریباً راکد / بازدهٔ خالی):")
    add("")
    rows = []
    for y, r in flags.iterrows():
        row = [fa(y), num(r["weeks"])]
        for s in WEEKLY_SERIES:
            row.append("—" if r[f"{s}_weeks"] == 0 else
                       f"{num(r[f'{s}_old'])} / {num(r[f'{s}_near_stale'])} / {num(r[f'{s}_no_ret'])}")
        rows.append(row)
    tot = ["جمع", num(flags["weeks"].sum())] + [
        f"{num(flags[f'{s}_old'].sum())} / {num(flags[f'{s}_near_stale'].sum())} / {num(flags[f'{s}_no_ret'].sum())}"
        for s in WEEKLY_SERIES]
    rows.append(tot)
    add(table(["سال", "هفته", "دلار Bonbast", "دلار TGJU", "عربیکا", "روبوستا", "صندوق"], rows))
    add("")
    add("* Bonbast هیچ‌وقت «کهنه» نمی‌شود، چون برای هر روز عدد دارد؛ کهنگی واقعی‌اش را پرچم تقریباً راکد نشان می‌دهد.")
    rb_old = wk.index[wk["robusta_usd_kg_old"] & wk["robusta_usd_kg"].notna()]
    if len(rb_old):
        add(f"* روبوستا: {num(len(rb_old))} هفتهٔ کهنه، همه در حفرهٔ ۱۴۰۵ ({'، '.join(jd(d) for d in rb_old)}). بازدهٔ خالیِ باقی‌مانده پس از پر کردن با ICO در ستون سوم آمده است.")
    if len(fund_old):
        first_block = [d for d in fund_old if to_jalali(d.date()) < "1394-03"]
        if first_block:
            nxt = wk.index[wk.index > first_block[-1]][0]
            add(f"* صندوق: {num(len(first_block))} هفتهٔ اول ({jd(first_block[0])} تا {jd(first_block[-1])}) کهنه است؛ صندوق پس از پذیرش تقریباً معامله نشده بود. "
                f"بک‌تست از هفتهٔ {jd(nxt)} شروع می‌شود (D-015). بازده‌های خالی دیگرِ صندوق و TGJU هفته‌های نوروز و تعطیلی‌های جنگ‌اند.")
    add("")
    add("**نوسان هفتگی، به تفکیک سال جلالی** (انحراف معیار بازدهٔ لگاریتمی هفتگی؛ برای قهوه بدون روزهای رول):")
    add("")
    rows = [[fa(y), num(r["usd_free_n"])] + [pct(r[f"{s}_std"]) for s in WEEKLY_SERIES] for y, r in vol.iterrows()]
    allv = [pct(wk[f"{s}_ret"].std()) for s in WEEKLY_SERIES]
    rows.append(["کل دوره", num(int(wk["usd_free_ret"].notna().sum()))] + allv)
    add(table(["سال", "هفته", "دلار Bonbast", "دلار TGJU", "عربیکا", "روبوستا", "صندوق"], rows))
    add("")
    top = vol["usd_free_std"].sort_values(ascending=False).head(3)
    add("* پرنوسان‌ترین سال‌های دلار: " + "، ".join(f"{fa(y)} ({pct(v, 1)})" for y, v in top.items()) + ". سال اول و سال جاری ناقص‌اند.")
    add("")
    rows = [[LABEL[s], num(r["weeks"]), num(r["weeks_with_roll"]), pct(r["std_ret"]), pct(r["std_ret_raw"]),
             pct(r["sum_removed"], 1, True), pct(r["mean_abs_removed"])] for s, r in roll_fx.iterrows()]
    add("**اثر کنار گذاشتن روزهای رول در بازدهٔ هفتگی قهوه:**")
    add("")
    add(table(["سری", "هفته", "هفتهٔ دارای روز رول", "نوسان هفتگی بدون رول", "نوسان هفتگی خام",
               "جمع بازدهٔ کنارگذاشته", "میانگین قدرمطلق آن"], rows))
    add("")
    add("* «جمع بازدهٔ کنارگذاشته» حرکتی است که در سری پیوسته دیده می‌شود ولی از بازدهٔ هفتگی مدل بیرون مانده است: پرش رول به‌اضافهٔ حرکت واقعی همان روز.")
    add("")
    add(f"![نوسان هفتگی]({f3})")
    add("")
    add("*شکل ۳ — انحراف معیار بازدهٔ هفتگی به تفکیک سال جلالی: دلار Bonbast، عربیکا و روبوستا. صندوق در جدول هست ولی در شکل نیست (مقیاسش بسیار کوچک‌تر است).*")
    add("")

    # ---- 6
    add("## ۶. کانال رسمی: تاریخ‌های رژیم (به‌روزشده)")
    add("")
    add("تاریخ‌های بخش ۸ گزارش D1 تأیید شدند و دو مورد اضافه شد. فهرست کامل (در حد ماه):")
    add("")
    for _, _, label in checks.OFFICIAL_REGIMES:
        add(f"* {label}")
    add("")
    add("شکل نسبت نرخ رسمی به آزاد در گزارش D1 با TGJU ساخته شده بود و بازسازی نشده است؛ این بخش در مدل استفاده نمی‌شود (D-002).")
    add("")

    # ---- 7
    add("## ۷. نکته‌ها و تصمیم‌های باز")
    add("")
    add("**۷-۱. ماه ردشدهٔ ICO**")
    if len(bad_months):
        for r in bad_months.itertuples():
            add(f"* یافته: PDF ماه {fa(f'{int(r.year)}-{int(r.month):02d}')} شرط میانگین را رد کرد (بخش ۴-۱). در نتیجه {num(len(still))} هفته ({'، '.join(jd(d) for d in still)}) خالی مانده است.")
        add("* گزینه‌ها: (الف) قانون بماند و این هفته‌ها خالی بمانند [وضع فعلی]؛ (ب) این ماه پذیرفته شود، چون ردیف‌ها درست خوانده شده‌اند و ناسازگاری به یک ردیف برمی‌گردد که در هیچ‌کدام از چهارشنبه‌های لازم نیست؛ (ج) از ICO بخواهید PDF را اصلاح کند.")
        add("* پیشنهاد: **(ب)**، ولی چون قانون را پیش از دیدن داده ثابت کرده بودید، بدون تصمیم شما اجرا نکردم.")
    else:
        add("* همهٔ ماه‌های بارگیری‌شده معتبرند.")
    add("")
    add("**۷-۲. برداشت من از D-018**")
    add("* متن اصلی D-018 به دستم نرسید؛ فقط بازگفتهٔ آن در اسپک پر کردن حفره بود: «اولین بازدهٔ هفتگی پس از هر حفرهٔ بیش از ۵ روز تقویمی خالی است».")
    add(f"* آنچه اجرا شد: بازدهٔ هفتگی خالی است اگر (۱) سطحِ آخر هفته بیش از {num(weekly.MAX_AGE_DAYS)} روز کهنه باشد (هفتهٔ داخل حفره، که بازده‌اش صفرِ ساختگی می‌شد)، یا (۲) اولین مشاهده پس از یک حفرهٔ بیش از {num(weekly.MAX_AGE_DAYS)} روزه در آن هفته باشد.")
    miss = {s: int((wk[s].notna() & wk[s].shift().notna() & wk[f"{s}_ret"].isna()).sum()) for s in WEEKLY_SERIES}
    add("* این قانون روی همهٔ سری‌ها اجرا شده است. بازده‌های خالی: "
        + "، ".join(f"{LABEL[s]} {num(n)}" for s, n in miss.items()) + ". برای صندوق و TGJU این‌ها بیشتر هفته‌های نوروزند.")
    add("* اگر منظورتان فقط روبوستا یا فقط بند (۲) بود، بگویید؛ تغییرش یک خط است.")
    add("")
    add("**۷-۳. ماه‌های ۲۰۲۵ که بارگیری نشد**")
    add(f"* {num(len(not_loaded))} ماه از بازهٔ خواسته‌شده در سایت ICO زیر هیچ‌کدام از دو الگوی نام نبود. اگر نشانی دیگری برایشان دارید، آزمون سازگاری با هفته‌های بیشتری تکرار می‌شود.")
    add("")
    add("**۷-۴. تصمیم‌های بسته‌شده در این دور**")
    add("* آغاز بک‌تست: هفتهٔ ۱۳۹۴-۰۲-۳۰ (D-015). آستانه‌های Investing (D-016) و ردیف روز دانلود (D-017) ثبت شدند، ولی با کنار رفتن مسیر Investing دیگر به کار نمی‌روند.")
    add("* ستون‌های افزوده: پنل روزانه `robusta_ico_clb` (شاخص ICO، فقط برای بازده) و `robusta_tgju_usd_t`؛ پنل هفتگی `robusta_return_source`، `usd_tgju` و پرچم‌های `*_near_stale`.")
    add("")
    return "\n".join(L) + "\n"


def main() -> None:
    text = build()
    (OUT / "addendum_d1b.md").write_text(text, encoding="utf-8")
    print(f"wrote {OUT / 'addendum_d1b.md'} ({len(text.splitlines())} lines) and figures in {OUT}")


if __name__ == "__main__":
    main()
