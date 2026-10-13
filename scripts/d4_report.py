#!/usr/bin/env python
"""Build the D4 report: reports/d4_stress/README.md (Persian) and its figures.

D4 replays real crisis paths from today's position: the weekly returns of a
historical window are applied to today's rate and today's coffee prices, and
every strategy is priced on that single path with the engine's own cost
functions. It ends with a reverse stress test. Every number, table and figure
in the report is computed here:

    .venv/bin/python scripts/d4_report.py
"""
from __future__ import annotations

import datetime as dt

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from coffee_risk import paths
from coffee_risk.data import fred, panel, weekly
from coffee_risk.engine import calibration as cal
from coffee_risk.engine import cost as costs
from coffee_risk.engine import decision as dec
from coffee_risk.engine import inputs as I
from coffee_risk.engine import settings, stress
from coffee_risk.engine import simulator as sim
from coffee_risk.engine.inputs import Inputs
from coffee_risk.reporting import (BLUE, INK, INK2, NEUTRAL, ORANGE, RED, billions as bn, fa, jd, jg, num,
                                   pct, strategy_fa, style, table)

OUT = paths.REPORTS / "d4_stress"
SC_FA = {"fx_1397": "S1 جهش ارز ۱۳۹۷", "fx_1401_1402": "S2 جهش ارز ۱۴۰۱–۱۴۰۲", "coffee_2021": "S3 قهوه ۲۰۲۱",
         "coffee_2024": "S4 قهوه ۲۰۲۴", "war": "S5 جنگ، بدترین ۱۳ هفته", "war_full": "S6 جنگ، کل مسیر",
         "perfect_storm": "S7 توفان کامل (فرضی)"}
SC_EN = {"fx_1397": "S1 FX 1397", "fx_1401_1402": "S2 FX 1401-02", "coffee_2021": "S3 coffee 2021",
         "coffee_2024": "S4 coffee 2024", "war": "S5 war, worst 13 weeks", "war_full": "S6 war, full path",
         "perfect_storm": "S7 perfect storm (hypothetical)"}
REVERSE_H = (0.0, 0.25, 0.50, None, 0.75, 1.0)          # None = the base-case minimum hedge ratio


def _save(fig, name: str) -> str:
    fig.savefig(OUT / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return name


def mark(margin: float, breaks: bool, d: int = 1) -> str:
    return pct(margin, d) + (" ✗" if breaks else "")


# -------------------------------------------------------------------- figures
def fig_paths(scs: list, arabica_share: float) -> str:
    fig, axes = plt.subplots(2, 4, figsize=(11.5, 5.4), sharey=True)
    flat = axes.ravel()
    for ax, sc in zip(flat, scs):
        r = sc.returns
        k = np.arange(sc.horizon + 1)
        usd = np.r_[1.0, np.exp(r["usd"].cumsum().values)] * 100
        mix = (arabica_share * np.r_[1.0, np.exp(r["arabica"].cumsum().values)]
               + (1 - arabica_share) * np.r_[1.0, np.exp(r["robusta"].cumsum().values)]) * 100
        ax.plot(k, usd, color=ORANGE, lw=1.8, label="USD/IRR")
        ax.plot(k, mix, color=BLUE, lw=1.8, label="coffee 50/50, in USD")
        ax.axhline(100, color=INK2, lw=0.6)
        ax.set_title(SC_EN[sc.key], fontsize=9.5, fontweight="normal", color=INK, pad=4)
        near = abs(usd[-1] - mix[-1]) < 9                      # keep the two end labels apart
        hi_is_usd = usd[-1] >= mix[-1]
        ax.text(k[-1], usd[-1], f" {usd[-1]:.0f}", fontsize=8, color=INK2,
                va=("bottom" if hi_is_usd else "top") if near else "center")
        ax.text(k[-1], mix[-1], f" {mix[-1]:.0f}", fontsize=8, color=INK2,
                va=("top" if hi_is_usd else "bottom") if near else "center")
        ax.set_xlim(0, sc.horizon * 1.18)
        ax.set_xlabel("week")
    flat[-1].axis("off")
    handles, labels = flat[0].get_legend_handles_labels()
    flat[-1].legend(handles, labels, loc="center", fontsize=9)
    axes[0, 0].set_ylabel("today = 100")
    axes[1, 0].set_ylabel("today = 100")
    fig.suptitle("The scenario paths, replayed from today's levels", x=0.01, ha="left", fontsize=11,
                 fontweight="bold")
    fig.tight_layout()
    return _save(fig, "fig1_scenario_paths.png")


def fig_heat(grids: dict, cols: list, col_labels: list, scs: list, floor: float) -> str:
    """grids: contract -> DataFrame (scenario x strategy) of margins."""
    cmap = LinearSegmentedColormap.from_list("floor", [RED, NEUTRAL, BLUE])
    norm = TwoSlopeNorm(vmin=floor - 0.5, vcenter=floor, vmax=floor + 0.5)
    fig, axes = plt.subplots(2, 1, figsize=(11.0, 7.4))
    for ax, (title, grid) in zip(axes, grids.items()):
        z = grid[cols].values
        ax.imshow(np.clip(z, floor - 0.5, floor + 0.5), cmap=cmap, norm=norm, aspect="auto")
        for y in range(z.shape[0]):
            for x in range(z.shape[1]):
                ax.text(x, y, str(int(round(z[y, x] * 100))), ha="center", va="center", fontsize=8.5, color=INK,
                        fontweight="bold" if z[y, x] < floor else "normal")
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels(col_labels, fontsize=8.5)
        ax.set_yticks(range(len(scs)))
        ax.set_yticklabels([SC_EN[s.key] for s in scs], fontsize=9)
        ax.set_xticks(np.arange(-0.5, len(cols)), minor=True)
        ax.set_yticks(np.arange(-0.5, len(scs)), minor=True)
        ax.grid(which="minor", color="#fcfcfb", lw=2)
        ax.grid(which="major", visible=False)
        ax.tick_params(which="both", length=0)
        ax.set_title(title, fontsize=10, pad=6)
    axes[-1].set_xlabel("strategy: staged, or the share of the USD bought today (the rest on the payment day)")
    fig.suptitle(f"Margin in each scenario, %  (bold and red = below the {floor * 100:.0f}% floor; "
                 "colour stops at 50 points from the floor)", x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout()
    return _save(fig, "fig2_margin_heatmap.png")


def fig_reverse(curve: pd.DataFrame, points: pd.DataFrame, lines: list) -> str:
    fig, (a, b) = plt.subplots(2, 1, figsize=(9.5, 6.4), sharex=True)
    h = curve.index.values * 100
    a.plot(h, curve["rise"] * 100, color=ORANGE, lw=1.8)
    a.plot(points.index.values * 100, points["rise"] * 100, ls="", marker="o", ms=6, color=ORANGE)
    for h_, r in points["rise"].items():
        a.annotate(f"+{r * 100:.0f}%", (h_ * 100, r * 100), xytext=(0, 8), textcoords="offset points",
                   ha="center", fontsize=8.5, color=INK)
    for label, level in lines:
        a.axhline(level * 100, color=INK2, lw=0.7, ls=":")
        a.text(99, level * 100 + 2, f"{label}: +{level * 100:.0f}%", fontsize=8, color=INK2, va="bottom",
               ha="right")
    a.set_ylim(0, 260)
    a.set_ylabel("USD rise that breaks the floor, % vs today")
    a.set_title("Reverse stress: how far USD must rise in 13 weeks before the margin falls below the floor",
                pad=10)
    b.plot(h, curve["share"] * 100, color=BLUE, lw=1.8)
    b.plot(points.index.values * 100, points["share"] * 100, ls="", marker="o", ms=6, color=BLUE)
    for h_, s in points["share"].items():
        b.annotate(f"{s * 100:.1f}%", (h_ * 100, s * 100), xytext=(0, 8), textcoords="offset points",
                   ha="center", fontsize=8.5, color=INK)
    b.set_ylim(bottom=0)
    b.set_ylabel("share of historical 13-week\nUSD changes at least that large, %")
    b.set_xlabel("share of the USD bought today, %  (fixed contract)")
    fig.tight_layout()
    return _save(fig, "fig3_reverse_stress.png")


# --------------------------------------------------------------------- report
def build() -> str:
    OUT.mkdir(parents=True, exist_ok=True)
    style(plt)
    wk, daily, imf = weekly.load(), panel.load(), fred.load()
    as_of = wk.index[-1]
    base = Inputs.from_data(as_of, wk, daily, imf)
    fixed = base.with_(contract="fixed")
    up = base.with_(pass_through=0.5)                               # up only is the default mode (D-043)
    assert up.pass_through_mode == "up_only"

    # the base-case minimum hedge ratio (fixed contract), as in the D3 report
    base_run = dec.run(sim.simulate(as_of, settings.REPORT_PATHS, 13, settings.REPORT_SEED, fixed, wk), fixed)
    h_star = base_run.min_hedge
    assert h_star is not None

    scs = stress.scenarios(wk)
    by_key = {s.key: s for s in scs}
    storm = by_key["perfect_storm"]
    assert np.array_equal(storm.returns["usd"].values, by_key["fx_1397"].returns["usd"].values)
    assert np.array_equal(storm.returns["arabica"].values, by_key["coffee_2024"].returns["arabica"].values)

    def strategies(T):
        return ([I.now(), I.staged(3, T), I.payment_day()] + I.partial_family() + [I.partial(h_star)])

    main = ["now", "staged_3", "payment_day", I.partial(h_star).name]
    family = [s.name for s in I.partial_family()]
    res, plan = {}, {}
    for sc in scs:
        for label, b in (("floating", base), ("fixed", fixed), ("up_only", up)):
            i = b.with_(payment_week=sc.horizon)
            p = stress.replay(sc.returns, i)
            res[sc.key, label] = stress.outcomes(p, i, strategies(sc.horizon))
            plan[sc.key] = costs.planned_cost(p, i)

    def first_safe(o: pd.DataFrame):
        ok = o.loc[family][~o.loc[family, "breaks"]]
        return None if ok.empty else float(ok["share_at_week0"].min())

    def breaks_count(label, name):
        return int(sum(res[s.key, label].loc[name, "breaks"] for s in scs))

    # ---- reverse stress
    clean = fixed.with_(sellback_spread=0.0, fx_holding_cost=0.0, channel_discount=0.0)
    g_T = (1 + clean.market_rate) ** (clean.payment_week / I.WEEKS_PER_YEAR)
    hist = stress.historical_changes(wk, cal.BACKTEST_START)
    n_ind = cal.independent_windows(hist.index)

    def reverse_row(h):
        x = stress.breaking_usd_level(clean, h)
        return {"level": x, "rise": x - 1.0, "above_plan": x / g_T - 1.0,
                "share": float((hist >= x - 1.0).mean()) if np.isfinite(x) else 0.0,
                "count": int((hist >= x - 1.0).sum()) if np.isfinite(x) else 0}

    hs = [h_star if h is None else h for h in REVERSE_H]
    rev = pd.DataFrame({h: reverse_row(h) for h in hs}).T
    curve = pd.DataFrame({h: reverse_row(h) for h in np.round(np.arange(0.0, 0.96, 0.01), 2)}).T
    worst_start = hist.idxmax()
    worst_end = worst_start + pd.Timedelta(weeks=13)

    # ---- figures
    f1 = fig_paths(scs, base.arabica_share)
    cols = ["staged_3"] + family[:6] + [I.partial(h_star).name] + family[6:]
    col_labels = ["staged"] + [f"{int(round(float(c.split('_')[1])))}%" for c in cols[1:]]
    grids = {"Floating contract (base case)": pd.DataFrame({s.key: res[s.key, "floating"]["margin"] for s in scs}).T,
             "Fixed contract": pd.DataFrame({s.key: res[s.key, "fixed"]["margin"] for s in scs}).T}
    f2 = fig_heat(grids, cols, col_labels, scs, base.margin_floor)
    lines = [(SC_EN[k].split(" ", 1)[0], by_key[k].change("usd")) for k in ("fx_1397", "fx_1401_1402", "war")]
    f3 = fig_reverse(curve, rev[np.isfinite(rev["level"])], lines)

    # ======================================================================
    L: list[str] = []
    add = L.append
    add("# گزارش مرحلهٔ D4 — سناریوهای تنش\n")
    add(f"تاریخ ساخت: {jg(dt.date.today())}. همهٔ عددها، جدول‌ها و شکل‌ها را `scripts/d4_report.py` ساخته است. موقعیت "
        f"امروز = حالت پایهٔ D3 در آخرین هفتهٔ پنل، {jg(as_of)}.\n")
    add("**این مرحله چه می‌کند.** چیزی شبیه‌سازی نمی‌شود. هر سناریو یک مسیر واقعی است که «از امروز» دوباره "
        "بازی می‌شود: بازده‌های هفتگیِ همان پنجرهٔ تاریخی (دلار؛ قهوه بدون روزهای رول، D-022) روی نرخ دلارِ امروز و "
        "قیمت قهوهٔ امروز سوار می‌شوند. نرخ تأمین مالی، پرمیوم و برنامه مالِ امروزند. هزینهٔ هر راهبرد با همان "
        "تابع‌های موتور D3 حساب می‌شود. در پایان یک تنش معکوس می‌آید: دلار چقدر باید بالا برود تا کف بشکند.\n")
    add("**قراردادها**\n")
    add(f"* **برنامه:** {num(base.usd_amount)} دلار با نرخ امروز ({num(base.s0)} ریال) و رشد خنثی تا هفتهٔ پرداخت "
        f"(نرخ بازار {pct(base.market_rate, 1)} در سال). برای ۱۳ هفته: {bn(plan['fx_1397'])} میلیارد ریال.")
    add(f"* **حاشیهٔ سود** = ۱ − هزینه ÷ درآمد برنامه. برنامه {pct(base.planned_margin, 0)}، کف {pct(base.margin_floor, 0)}. "
        "علامت ✗ یعنی کف شکسته است. حاشیهٔ منفی یعنی زیان.")
    add(f"* **٪{fa(int(round(h_star * 100)))} امروز** همان «حداقل نسبت پوشش» حالت پایه با قرارداد ثابت است (گزارش D3).")
    add("* **قرارداد شناور** (پایه، D-036): فاکتور با قیمت قهوهٔ روز پرداخت بسته می‌شود. **قرارداد ثابت:** قیمت قهوه "
        "امروز بسته شده و مسیر قهوه اثری ندارد.")
    add("* همهٔ ورودی‌ها همان پیش‌فرض‌های موقت D3 هستند.\n")

    # ---------------------------------------------------------------- summary
    fl_now, fx_now = breaks_count("floating", "now"), breaks_count("fixed", "now")
    fl_late, fx_late = breaks_count("floating", "payment_day"), breaks_count("fixed", "payment_day")
    fx_star = breaks_count("fixed", I.partial(h_star).name)
    star_fail = [SC_FA[s.key].split(" ", 1)[0] for s in scs if res[s.key, "fixed"].loc[I.partial(h_star).name, "breaks"]]
    add("## نتیجه در چند خط\n")
    def of_all(n):
        return f"همهٔ {fa(len(scs))} سناریو" if n == len(scs) else f"{fa(n)} سناریو از {fa(len(scs))} سناریو"
    add(f"1. **صبر تا روز پرداخت در {of_all(fl_late)} کف را می‌شکند** (قرارداد شناور)؛ با قرارداد ثابت در "
        f"{of_all(fx_late)}.")
    add(f"2. **قرارداد ثابت + خرید کاملِ امروز در هیچ سناریویی نمی‌شکند** ({fa(fx_now)} از {fa(len(scs))}): هزینه از "
        "پیش بسته شده است. این تنها ترکیبی است که از همهٔ هفت مسیر سالم بیرون می‌آید.")
    add(f"3. **قرارداد شناور + خرید کاملِ امروز در {fa(fl_now)} سناریو می‌شکند**: هر جا قهوه گران شده، دلارِ اضافه باید "
        "با نرخ جهش‌کرده خریده شود.")
    add(f"4. **حداقل پوشش ٪{fa(int(round(h_star * 100)))} برای تکرار ۱۳۹۷ کافی نیست.** با قرارداد ثابت در {fa(fx_star)} سناریو "
        f"می‌شکند ({'، '.join(star_fail)}). این عدد برای «۹۵٪ حالت‌ها» ساخته شده، نه برای بدترین مسیر تاریخ.")
    add(f"5. **تنش معکوس:** با پوشش ٪{fa(int(round(h_star * 100)))} کف وقتی می‌شکند که دلار در ۱۳ هفته بیش از "
        f"{pct(rev.loc[h_star, 'rise'], 0)} بالا برود. در {fa(len(hist))} پنجرهٔ تاریخی (از {jd(cal.BACKTEST_START)}) این در "
        f"{pct(rev.loc[h_star, 'share'], 1)} موارد رخ داده است؛ بدون پوشش ({pct(rev.loc[0.0, 'rise'], 0)}) در "
        f"{pct(rev.loc[0.0, 'share'], 1)} موارد.\n")

    # ------------------------------------------------------------------ 1
    add("## ۱. سناریوها\n")
    add("پنج پنجرهٔ ۱۳ هفته‌ای همان پنجره‌های بخش ۷ گزارش D2 هستند (D-027): در هر رویداد، ۱۳ هفته‌ای که هزینهٔ "
        "ریالی قهوهٔ ۵۰/۵۰ بیشترین رشد را داشت. درصدها ساده‌اند.\n")
    rows = []
    for sc in scs:
        usd_w = f"{jd(sc.usd_window[0])} تا {jd(sc.usd_window[1])}"
        cof_w = f"{jd(sc.coffee_window[0])} تا {jd(sc.coffee_window[1])}"
        window = usd_w if not sc.hypothetical else f"دلار: {usd_w}؛ قهوه: {cof_w}"
        mix = sc.mix_change(base.arabica_share)
        rows.append([SC_FA[sc.key], window, fa(sc.horizon), pct(sc.change("usd"), 0, True),
                     pct(sc.change("arabica"), 0, True), pct(sc.change("robusta"), 0, True), pct(mix, 0, True),
                     pct((1 + sc.change("usd")) * (1 + mix) - 1, 0, True), bn(plan[sc.key])])
    add(table(["سناریو", "پنجره", "هفته", "دلار", "عربیکا", "روبوستا", "قهوهٔ ۵۰/۵۰ (دلاری)",
               "هزینهٔ ریالی ۵۰/۵۰", "برنامه (میلیارد ریال)"], rows))
    add("")
    s6 = by_key["war_full"]
    add(f"* **S6** کل مسیر جنگ است، {fa(s6.horizon)} هفته: هفتهٔ پرداخت {fa(s6.horizon)}، خرید سه‌پله‌ای در هفته‌های "
        f"{'، '.join(fa(t) for t, _ in I.staged(3, s6.horizon).tranches)}، و برنامه با رشد خنثی تا هفتهٔ {fa(s6.horizon)} "
        "(برای همین برنامه‌اش بزرگ‌تر است).")
    add("* **S7 فرضی است:** مسیر دلارِ S1 با مسیر قهوهٔ S4. این دو هرگز با هم رخ نداده‌اند، و D2 دلار و قهوه را "
        "تقریباً مستقل یافت. برای دیدنِ بدترین ترکیب است، نه برای برآورد احتمال.")
    add("* نرخ تأمین مالی در همهٔ سناریوها نرخ بازارِ امروز است، نه سود صندوق در همان دوره.\n")
    add(f"![مسیر سناریوها]({f1})\n")
    add("*شکل ۱ — دلار (نارنجی) و سبد قهوهٔ ۵۰/۵۰ به دلار (آبی) در هر سناریو؛ امروز = ۱۰۰. عدد کنار هر خط سطح "
        "پایانی است.*\n")

    # -------------------------------------------------------------- 2 and 3
    def contract_section(number, title, label, intro):
        add(f"## {number}. {title}\n")
        add(intro + "\n")
        add("**هزینه (درصد برنامه) و حاشیهٔ سود، چهار راهبرد اصلی**\n")
        rows = []
        for sc in scs:
            o = res[sc.key, label]
            row = [SC_FA[sc.key]]
            for name in main:
                row += [pct(o.loc[name, "cost_pct"], 0), mark(o.loc[name, "margin"], o.loc[name, "breaks"])]
            rows.append(row)
        head = ["سناریو"]
        for name in main:
            head += [f"{strategy_fa(name)}: هزینه", "حاشیه"]
        add(table(head, rows))
        add("")
        add("**همان هزینه‌ها به میلیارد ریال**\n")
        add(table(["سناریو", "برنامه"] + [strategy_fa(n) for n in main],
                  [[SC_FA[sc.key], bn(plan[sc.key])] + [bn(res[sc.key, label].loc[n, "cost"]) for n in main]
                   for sc in scs]))
        add("")
        add("**حاشیهٔ سود برای هر سهمِ خرید امروز** (بقیه روز پرداخت)\n")
        rows = []
        for sc in scs:
            o = res[sc.key, label]
            safe = first_safe(o)
            rows.append([SC_FA[sc.key]] + [mark(o.loc[n, "margin"], o.loc[n, "breaks"], 0) for n in family]
                        + ["هیچ" if safe is None else pct(safe, 0)])
        add(table(["سناریو"] + [pct(k / 10, 0) for k in range(11)] + ["کمترین سهمی که کف را نگه می‌دارد"], rows))
        add("")

    add_note = ("ستون آخرِ جدول سوم از همان جدول خوانده شده است (شبکهٔ ده‌درصدی): کوچک‌ترین سهمِ خرید امروز که در "
                "آن سناریو حاشیه را بالای کف نگه می‌داشت.")
    contract_section("۲", "قرارداد شناور (حالت پایه)", "floating",
                     "فاکتور با قیمت قهوهٔ روز پرداخت بسته می‌شود؛ هم دلار و هم قهوه اثر دارند. " + add_note)
    s1, s4 = res["fx_1397", "floating"], res["coffee_2024", "floating"]
    add(f"* **S1 (۱۳۹۷):** خرید کاملِ امروز حاشیه را به {pct(s1.loc['now', 'margin'], 0)} می‌رساند، بالاتر از برنامه. "
        f"دلیل: قهوه {pct(by_key['fx_1397'].mix_change(base.arabica_share), 0, True)} ارزان شد، پس دلارِ کمتری لازم بود و "
        "دلارِ اضافه با نرخ جهش‌کرده فروخته شد. این سودِ بادآورده به فرضِ «بازفروش با نرخ بازار» بسته است "
        "(تصمیم ۷-۱).")
    add(f"* **S4 (قهوه ۲۰۲۴):** حتی خرید کاملِ امروز زیان می‌دهد ({pct(s4.loc['now', 'margin'], 0)}): قهوه "
        f"{pct(by_key['coffee_2024'].mix_change(base.arabica_share), 0, True)} گران شد و دلارِ اضافه با نرخ "
        f"{pct(by_key['coffee_2024'].change('usd'), 0, True)} بالاتر خریده شد. زمان خرید ارز جلوی این را نمی‌گیرد.")
    add(f"* در {fa(sum(first_safe(res[s.key, 'floating']) is None for s in scs))} سناریو از {fa(len(scs))} سناریو هیچ "
        "سهمی کف را نگه نمی‌دارد.\n")

    contract_section("۳", "قرارداد ثابت", "fixed",
                     "قیمت قهوه امروز بسته شده است؛ فقط مسیر دلار اثر دارد. برای همین S7 با S1 یکی است. " + add_note)
    add("* خرید کاملِ امروز در همهٔ سناریوها دقیقاً حاشیهٔ برنامه را می‌دهد: نرخ تأمین مالی برابر نرخ بازار است و "
        "هیچ چیزِ دیگری باز نمانده.")
    add(f"* با پوشش ٪{fa(int(round(h_star * 100)))}: در S1 حاشیه {pct(res['fx_1397', 'fixed'].loc[I.partial(h_star).name, 'margin'], 0)} "
        f"می‌شود؛ در بقیهٔ مسیرهای واقعی بالای کف می‌ماند (کمترین: "
        f"{pct(min(res[s.key, 'fixed'].loc[I.partial(h_star).name, 'margin'] for s in scs if s.key not in ('fx_1397', 'perfect_storm')), 0)}).")
    assert all(not res[s.key, "fixed"].loc[I.partial(h_star).name, "breaks"] for s in scs
               if s.key not in ("fx_1397", "perfect_storm"))
    assert s1.loc["now", "margin"] > base.planned_margin and fx_now == 0
    s6f = res["war_full", "fixed"]
    add(f"* **S6:** در مسیر ۳۲ هفته‌ای جنگ، خرید سه‌پله‌ای ({pct(s6f.loc['staged_3', 'margin'], 0)}) تقریباً به خوبیِ "
        f"خرید کاملِ امروز بود و صبر تا پایان ({pct(s6f.loc['payment_day', 'margin'], 0)}) کف را شکست.\n")
    add(f"![نقشهٔ حاشیهٔ سود]({f2})\n")
    add(f"*شکل ۲ — حاشیهٔ سود (درصد) در هر سناریو و هر راهبرد. بالا: قرارداد شناور؛ پایین: قرارداد ثابت. آبی: بالای "
        f"کف {pct(base.margin_floor, 0)}؛ قرمز و پررنگ: زیر کف؛ خاکستری: نزدیک کف. رنگ در ۵۰ واحد فاصله از کف ثابت "
        "می‌ماند؛ عددِ داخل هر خانه مقدار واقعی است.*\n")

    # ------------------------------------------------------------------ 4
    add("## ۴. قرارداد شناور با انتقال ۰.۵ِ «فقط بالا»\n")
    add("همان حالت پایه، با این تفاوت که نیمی از جهشِ نامنتظر دلار به قیمت فروش می‌رسد و اُفت دلار منتقل نمی‌شود "
        "(D-043، D-044). درآمد دیگر ثابت نیست؛ هزینه‌ها همان هزینه‌های بخش ۲ هستند.\n")
    rows = []
    for sc in scs:
        o, o0 = res[sc.key, "up_only"], res[sc.key, "floating"]
        rows.append([SC_FA[sc.key]] + [mark(o.loc[n, "margin"], o.loc[n, "breaks"]) for n in main]
                    + [pct(o0.loc["payment_day", "margin"], 1)])
    add(table(["سناریو"] + [f"{strategy_fa(n)}: حاشیه" for n in main] + ["روز پرداخت، بدون انتقال"], rows))
    add("")
    add("**حاشیهٔ سود برای هر سهمِ خرید امروز**\n")
    rows = []
    for sc in scs:
        o = res[sc.key, "up_only"]
        safe = first_safe(o)
        rows.append([SC_FA[sc.key]] + [mark(o.loc[n, "margin"], o.loc[n, "breaks"], 0) for n in family]
                    + ["هیچ" if safe is None else pct(safe, 0)])
    add(table(["سناریو"] + [pct(k / 10, 0) for k in range(11)] + ["کمترین سهمی که کف را نگه می‌دارد"], rows))
    add("")
    up_late = breaks_count("up_only", "payment_day")
    up_now = breaks_count("up_only", "now")
    add(f"* انتقال، ضربهٔ جهش دلار را نرم می‌کند ولی برنمی‌دارد: «همه روز پرداخت» هنوز در {fa(up_late)} سناریو "
        f"می‌شکند (بدون انتقال: {fa(fl_late)}). «همه امروز» در {fa(up_now)} سناریو می‌شکند (بدون انتقال: {fa(fl_now)}).")
    add("* انتقال به ریسک قهوه کمکی نمی‌کند: درآمد با دلار بالا می‌رود، نه با قیمت جهانی قهوه.\n")

    # ------------------------------------------------------------------ 5
    add("## ۵. تنش معکوس\n")
    add("پرسش وارونه: با قرارداد ثابت، قهوهٔ بی‌تغییر و بدون اصطکاک، دلار تا هفتهٔ پرداخت چقدر باید بالا برود تا "
        "حاشیهٔ سود زیر کف بیفتد؟\n")
    add("**فرمول.** سهم h امروز خریده می‌شود و بقیه روز پرداخت. به واحدِ «مبلغ × نرخ امروز»: هزینه = h·F + (۱ − h)·x "
        "که x نسبتِ نرخ روز پرداخت به نرخ امروز است و F رشد پولِ خرید زودهنگام تا روز پرداخت. هزینهٔ مجاز = "
        "(۱ − کف) ÷ (۱ − حاشیهٔ برنامه) × g، که g رشد مورد انتظار دلار است. پس کف وقتی می‌شکند که\n")
    add("> x > (هزینهٔ مجاز − h·F) ÷ (۱ − h)\n")
    allowed = (1 - clean.margin_floor) / (1 - clean.planned_margin)
    add(f"با ورودی‌های پایه (حاشیهٔ برنامه {pct(clean.planned_margin, 0)}، کف {pct(clean.margin_floor, 0)}، نرخ تأمین مالی برابر "
        f"نرخ بازار، پس F = g = {num(g_T, 4)}): دلار باید {num(allowed - 1, 1)} ÷ (۱ − h) بالاتر از سطح مورد انتظارش تمام "
        "شود.\n")
    rows = []
    for h in hs:
        r = rev.loc[h]
        if np.isfinite(r["level"]):
            rows.append([pct(h, 0), pct(r["above_plan"], 0, True), pct(r["rise"], 0, True),
                         f"{num(base.s0 * r['level'])} ریال", pct(r["share"], 1), fa(int(r["count"]))])
        else:
            rows.append([pct(h, 0), "هیچ‌وقت", "هیچ‌وقت", "—", pct(0.0, 1), fa(0)])
    add(table(["سهم خرید امروز", "رشدِ شکننده، بالاتر از انتظار", "رشدِ شکننده نسبت به امروز", "نرخ دلارِ شکننده",
               "سهم پنجره‌های تاریخی با رشدِ دست‌کم همین‌قدر", "تعداد پنجره"], rows))
    add("")
    add(f"* **نمونهٔ تاریخی:** همهٔ پنجره‌های ۱۳ هفته‌ایِ دلار با شروع هفتگی از {jg(cal.BACKTEST_START)} (D-041) که "
        f"پایانشان معلوم است: {fa(len(hist))} پنجرهٔ هم‌پوشان، برابر با {fa(n_ind)} پنجرهٔ مستقل. سهم‌ها را باید با همین "
        "عددِ کوچک خواند.")
    add(f"* **بزرگ‌ترین رشد ۱۳ هفته‌ای:** {pct(hist.max(), 0, True)}، از {jg(worst_start)} تا {jg(worst_end)}. فقط "
        f"پوشش بالای {pct(1 - (allowed - 1) * g_T / (1 + hist.max() - g_T), 0)} از آن جان به در می‌بُرد.")
    add(f"* با خرید کاملِ امروز هیچ سطحی از دلار کف را نمی‌شکند: هزینه بسته شده است.\n")
    add(f"![تنش معکوس]({f3})\n")
    add("*شکل ۳ — بالا: رشدی از دلار (نسبت به امروز، در ۱۳ هفته) که کف را می‌شکند، برای هر سهمِ خرید امروز. "
        "خط‌چین‌ها رشد دلار در S1، S2 و S5 هستند: هر جا منحنی زیر خط‌چین باشد، آن سناریو کف را می‌شکند. پایین: سهم "
        "پنجره‌های تاریخی که دلار دست‌کم همین‌قدر بالا رفت.*\n")

    # ------------------------------------------------------------------ 6
    add("## ۶. به زبان ساده\n")
    add("**در چند سناریو از هفت سناریو کف می‌شکند**\n")
    add(table(["راهبرد", "قرارداد شناور", "شناور با انتقال ۰.۵ِ فقط بالا", "قرارداد ثابت"],
              [[strategy_fa(n), fa(breaks_count("floating", n)), fa(breaks_count("up_only", n)),
                fa(breaks_count("fixed", n))] for n in main]))
    add("")
    add(f"1. **صبر کردن بدترین انتخاب است.** خرید در روز پرداخت با قرارداد شناور در {of_all(fl_late)} و با قرارداد "
        f"ثابت در {of_all(fx_late)} کف را می‌شکند. تعجبی ندارد: این پنجره‌ها برای بیشترین رشد هزینه انتخاب شده‌اند.")
    add("2. **دو اهرم با هم کار می‌کنند، نه جدا.** خرید کاملِ امروز جلوی دلار را می‌گیرد ولی جلوی قهوه را نه؛ "
        "قرارداد ثابت جلوی قهوه را می‌گیرد ولی جلوی دلار را نه. فقط هر دو با هم از همهٔ سناریوها سالم بیرون می‌آیند.")
    add(f"3. **«حداقل پوشش» بیمهٔ بحران نیست.** ٪{fa(int(round(h_star * 100)))} یعنی در ۹۵٪ مسیرهای شبیه‌سازی‌شده کف "
        f"می‌ماند. برای جهشی مثل ۱۳۹۷ ({pct(by_key['fx_1397'].change('usd'), 0, True)} در ۱۳ هفته) پوششی نزدیک به کامل لازم است.")
    add("4. **سناریوها احتمال ندارند.** این‌ها بدترین پنجره‌های ۱۴ سال‌اند، از امروز بازی‌شده. می‌گویند اگر تاریخ "
        "تکرار شود چه می‌شود، نه اینکه چقدر محتمل است؛ احتمال را بخش ۵ و بک‌تست D5 می‌دهند.\n")

    # ------------------------------------------------------------------ 7
    add("## ۷. تصمیم‌های لازم\n")
    add("**۷-۱. بازفروش دلارِ اضافه در بحران**")
    add(f"* دیده شد: در S1 با قرارداد شناور، خرید کاملِ امروز حاشیهٔ {pct(s1.loc['now', 'margin'], 0)} می‌دهد (برنامه "
        f"{pct(base.planned_margin, 0)}). بیشترِ این از فروش دلارِ اضافه با نرخ بازار آزاد (منهای "
        f"{pct(base.sellback_spread, 0)}) می‌آید، وقتی قهوه ارزان شده و دلار {pct(by_key['fx_1397'].change('usd'), 0, True)} "
        "بالا رفته است. موتور این را سودِ واقعی حساب می‌کند.")
    add("* گزینه‌ها: (الف) همین بماند (قاعدهٔ D3)؛ (ب) دلارِ اضافه حداکثر به قیمت خریدش ارزش‌گذاری شود، تا "
        "راهبردها از سفته‌بازیِ ناخواسته سود نبرند؛ (ج) اختلاف بازفروش در بحران بزرگ‌تر گرفته شود.")
    add("* پیشنهاد: **(الف)**، با یک جمله در یادداشت: حاشیهٔ بالاتر از برنامه در S1 سودِ نگه‌داشتن دلار است، نه "
        "هدف راهبرد. پیش‌فرض موقت: (الف).\n")
    add("**۷-۲. نرخ تأمین مالی در سناریوها**")
    add(f"* دیده شد: طبق اسپک، همهٔ سناریوها با نرخ بازارِ امروز ({pct(base.market_rate, 1)}) قیمت‌گذاری شده‌اند، نه با "
        "سود صندوق در همان دوره. برای همین «قرارداد ثابت + همه امروز» در همهٔ سناریوها دقیقاً حاشیهٔ برنامه را "
        "می‌دهد.")
    add("* گزینه‌ها: (الف) همین (سناریو فقط قیمت‌ها را جابه‌جا می‌کند)؛ (ب) سود صندوق در همان پنجره هم بازی شود.")
    add("* پیشنهاد: **(الف)**. بک‌تست D5 سود واقعی صندوق را به کار می‌برد و این پرسش را آنجا جواب می‌دهد.\n")

    (OUT / "README.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    return str(OUT / "README.md")


def main() -> None:
    print(build())


if __name__ == "__main__":
    main()
