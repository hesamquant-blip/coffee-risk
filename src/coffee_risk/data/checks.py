"""Checks behind the data reports: source agreement, robusta sources, weekly
panel. Every function reads the raw cache or a panel and returns a table.
Nothing here changes data.

Since D1b the free-market dollar is Bonbast (`usd_free`) and TGJU is the
check (`usd_tgju`).
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

from coffee_risk.dates import from_jalali, jalali_year
from coffee_risk.units import cents_lb_to_usd_kg
from coffee_risk.data import fred, quality, tgju, weekly

GAP = 0.02                      # a TGJU-Bonbast gap above this is "large"
WEEKLY_BIG = 0.10               # a weekly |log return| above this is listed

# Known changes in the official FX channel, month-level (author-confirmed,
# D1 review 2026-10-12). (month start, English label, Persian label)
OFFICIAL_REGIMES = [
    (dt.date(2018, 4, 1), "Apr 2018: single official rate", "فروردین ۱۳۹۷: اعلام نرخ رسمی واحد"),
    (dt.date(2018, 8, 1), "Aug 2018: NIMA secondary market", "مرداد ۱۳۹۷: بازار ثانویهٔ نیما"),
    (dt.date(2023, 2, 1), "Feb 2023: exchange centre opens", "اسفند ۱۴۰۱: آغاز کار مرکز مبادله"),
    (dt.date(2024, 12, 1), "Dec 2024: agreed-rate market replaces NIMA",
     "آذر ۱۴۰۳: بازار توافقی جای نیما را گرفت"),
    (dt.date(2026, 1, 1), "Jan 2026: preferential rate abolished in the budget",
     "دی ۱۴۰۴: حذف ارز ترجیحی در بودجه"),
    (dt.date(2026, 5, 1), "May 2026: preferential rate restored for a few essentials (not coffee)",
     "اردیبهشت ۱۴۰۵: بازگشت ارز ترجیحی برای چند کالای اساسی (قهوه جزو آن‌ها نیست)"),
]


def _jyears(index) -> list[int]:
    return [jalali_year(d.date()) for d in index]


def _shift_dates(s: pd.Series, days: int) -> pd.Series:
    out = s.copy()
    out.index = out.index + pd.Timedelta(days=days)
    return out


# ------------------------------------------------- D1 snapshot vs table endpoint
def snapshot_equality(panel: pd.DataFrame, official: pd.DataFrame,
                      snapshot_dir: Path) -> pd.DataFrame:
    """Does each TGJU column built from the table endpoint equal the column
    D1 built from the chart endpoint? `snapshot_dir` holds the D1 panel files
    (commit tagged `d1`). One row per series."""
    old_p = pd.read_parquet(snapshot_dir / "panel_daily.parquet")
    old_o = pd.read_parquet(snapshot_dir / "official_usd_daily.parquet")
    pairs = [("price_dollar_rl", old_p["usd_irr"], panel["usd_tgju"]),
             ("commodities-london-coffee", old_p["robusta_usd_t"], panel["robusta_tgju_usd_t"])]
    pairs += [(k, old_o[k], official[k]) for k in tgju.OFFICIAL_KEYS
              if k in old_o.columns and k in official.columns]
    rows = []
    for key, old, new in pairs:
        old, new = old.dropna(), new.dropna()
        both = old.index.intersection(new.index)
        differ = both[old[both].values != new[both].values]
        rows.append({"key": key, "d1_rows": len(old), "table_rows": len(new),
                     "common": len(both), "equal": len(both) - len(differ),
                     "only_d1": len(old.index.difference(new.index)),
                     "only_table": len(new.index.difference(old.index)),
                     "differing_dates": [d.date() for d in differ]})
    return pd.DataFrame(rows)


# ------------------------------------------------------------ TGJU vs Bonbast
def usd_pairs(panel: pd.DataFrame) -> pd.DataFrame:
    """Dates on which both sources have a value. gap = TGJU / Bonbast - 1,
    with Bonbast on the same date (`gap`) and on the next calendar day
    (`gap_next`); `gap_best` is the smaller of the two in size."""
    b = panel["usd_free"].dropna()
    t = panel["usd_tgju"].dropna()
    df = pd.DataFrame({"bonbast": b.reindex(t.index), "tgju": t,
                       "bonbast_next": _shift_dates(b, -1).reindex(t.index)})
    df = df.dropna(subset=["bonbast"])
    df["gap"] = df["tgju"] / df["bonbast"] - 1
    df["gap_next"] = df["tgju"] / df["bonbast_next"] - 1
    df["gap_best"] = df[["gap", "gap_next"]].abs().min(axis=1)
    return df


def usd_by_year(panel: pd.DataFrame) -> pd.DataFrame:
    """TGJU against Bonbast (the reference), per Jalali year."""
    pairs = usd_pairs(panel)
    g = pairs.assign(jy=_jyears(pairs.index)).groupby("jy")
    b = panel["usd_free"].dropna()
    out = pd.DataFrame({
        "bonbast_days": b.groupby(_jyears(b.index)).size(),
        "tgju_days": g.size(),
        "median_abs_gap": g["gap"].apply(lambda x: x.abs().median()),
        "median_gap": g["gap"].median(),
        "days_over": g["gap"].apply(lambda x: int((x.abs() > GAP).sum())),
        "days_over_best": g["gap_best"].apply(lambda x: int((x > GAP).sum())),
    })
    for name in ("usd_free", "usd_tgju"):
        f = panel.loc[panel[name].notna(), f"{name}_near_stale"]
        out[f"{name}_near_stale_days"] = f.groupby(_jyears(f.index)).sum()
    return out.dropna(subset=["tgju_days"]).fillna(0)


def usd_extremes(panel: pd.DataFrame) -> pd.DataFrame:
    """Every extreme daily TGJU move, with what Bonbast shows over the same
    step. confirmed = Bonbast moved the same way by at least half as much, on
    the same dates or one day later. Information only (D-008)."""
    t = panel["usd_tgju"].dropna()
    r = np.log(t).diff()
    b = panel["usd_free"].dropna()
    b_next = _shift_dates(b, -1)
    rows = []
    for d in t.index[panel.loc[t.index, "usd_tgju_outlier"]]:
        prev = t.index[t.index.get_loc(d) - 1]
        same = np.log(b.get(d, np.nan) / b.get(prev, np.nan))
        nxt = np.log(b_next.get(d, np.nan) / b_next.get(prev, np.nan))
        cands = [x for x in (same, nxt) if not np.isnan(x)]
        if not cands:
            verdict = "no_bonbast"
        elif any(np.sign(x) == np.sign(r[d]) and abs(x) >= 0.5 * abs(r[d]) for x in cands):
            verdict = "confirmed"
        else:
            verdict = "not_confirmed"
        rows.append({"date": d, "prev_date": prev, "close": t[d], "prev_close": t[prev],
                     "tgju_return": r[d], "bonbast_same": same, "bonbast_next": nxt,
                     "verdict": verdict})
    return pd.DataFrame(rows)


def near_stale_table(panel: pd.DataFrame, columns: dict[str, str]) -> pd.DataFrame:
    """Near-stale runs of every checked series, one row per run. `columns`
    maps a label to its level column. `exact_repeat` marks the runs in which
    the value does not change at all (what the exact-repeat check finds)."""
    rows = []
    for name, col in columns.items():
        for _, r in quality.near_stale_runs(panel[col]).iterrows():
            rows.append({"series": name, **r.to_dict(),
                         "calendar_days": (r["last_date"] - r["first_date"]).days + 1,
                         "exact_repeat": bool(r["max_abs_step"] == 0)})
    return pd.DataFrame(rows, columns=["series", "first_date", "last_date", "observations",
                                       "first_value", "last_value", "max_abs_step",
                                       "calendar_days", "exact_repeat"])


def window_medians(panel: pd.DataFrame, windows: list[tuple[str, str]]) -> pd.DataFrame:
    """TGJU and Bonbast inside Jalali-dated windows ('1397-01-21', ...):
    medians of the two levels and of the gap on the dates both have."""
    pairs = usd_pairs(panel)
    rows = []
    for a, b in windows:
        da, db = (from_jalali(*map(int, x.split("-"))) for x in (a, b))
        w = pairs.loc[pd.Timestamp(da):pd.Timestamp(db)]
        t = panel["usd_tgju"].loc[pd.Timestamp(da):pd.Timestamp(db)].dropna()
        steps = np.log(t).diff().abs().dropna()
        rows.append({"from": a, "to": b, "common_days": len(w),
                     "tgju_median": w["tgju"].median(), "bonbast_median": w["bonbast"].median(),
                     "tgju_min": w["tgju"].min(), "tgju_max": w["tgju"].max(),
                     "bonbast_min": w["bonbast"].min(), "bonbast_max": w["bonbast"].max(),
                     "median_gap": w["gap"].median(),
                     "tgju_median_abs_step": steps.median() if len(steps) else np.nan})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- weekly USD
def weekly_big_moves(wk: pd.DataFrame, threshold: float = WEEKLY_BIG) -> pd.DataFrame:
    """Weeks in which Bonbast moved by more than `threshold` (log), with
    TGJU's move in the same week. confirmed = TGJU moved the same way by at
    least half as much."""
    hit = wk[wk["usd_free_ret"].abs() > threshold]
    out = pd.DataFrame({
        "level_before": wk["usd_free"].shift(1).reindex(hit.index), "level": hit["usd_free"],
        "bonbast_ret": hit["usd_free_ret"], "tgju_ret": hit["usd_tgju_ret"],
        "bonbast_near_stale": hit["usd_free_near_stale"],
        "tgju_old": hit["usd_tgju_old"], "tgju_near_stale": hit["usd_tgju_near_stale"]})
    same_way = np.sign(out["tgju_ret"]) == np.sign(out["bonbast_ret"])
    out["verdict"] = np.where(out["tgju_ret"].isna(), "no_tgju", np.where(
        same_way & (out["tgju_ret"].abs() >= 0.5 * out["bonbast_ret"].abs()),
        "confirmed", "not_confirmed"))
    return out


def weekly_usd_correlation(panel: pd.DataFrame) -> pd.DataFrame:
    """Correlation of weekly log returns, TGJU vs Bonbast, with Bonbast as
    published and with its dates moved one day back (the value published for
    day t+1 placed on day t). Information only: nothing is shifted in the
    panel (D-011)."""
    weds = weekly.wednesdays(panel["usd_free"].dropna().index.min().date(),
                             panel.index.max().date())
    t = weekly.ratio_return(weekly.last_observation(panel["usd_tgju"], weds)["level"])
    rows = []
    for label, shift in (("as_published", 0), ("shifted_minus_1_day", -1)):
        b = _shift_dates(panel["usd_free"].dropna(), shift)
        rb = weekly.ratio_return(weekly.last_observation(b, weds)["level"])
        both = pd.concat([rb.rename("bonbast"), t.rename("tgju")], axis=1).dropna()
        jy = pd.Series(_jyears(both.index), index=both.index)
        ex97 = both[jy != 1397]
        rows.append({"bonbast": label, "weeks": len(both),
                     "corr": both.corr().iloc[0, 1],
                     "weeks_without_1397": len(ex97), "corr_without_1397": ex97.corr().iloc[0, 1],
                     "std_of_difference": (both["bonbast"] - both["tgju"]).std()})
    return pd.DataFrame(rows)


def quarter_changes(wk: pd.DataFrame, weeks: int = 13) -> dict:
    """13-week log changes of the two dollar series: the largest rise ending
    in Jalali 1397, and the correlation of non-overlapping 13-week changes
    over every possible starting week, with and without 1397."""
    lv = np.log(wk[["usd_free", "usd_tgju"]])
    ch = (lv - lv.shift(weeks)).dropna()
    jy = pd.Series(_jyears(ch.index), index=ch.index)
    in97 = ch[jy == 1397]
    start97 = pd.Series(_jyears(ch.index - pd.Timedelta(weeks=weeks)), index=ch.index)
    corr_all, corr_ex = [], []
    for off in range(weeks):
        block = ch.iloc[off::weeks]
        corr_all.append(block.corr().iloc[0, 1])
        touches = (jy.reindex(block.index) == 1397) | (start97.reindex(block.index) == 1397)
        corr_ex.append(block[~touches].corr().iloc[0, 1])
    return {"max_rise_1397_bonbast": float(np.exp(in97["usd_free"].max()) - 1),
            "max_rise_1397_tgju": float(np.exp(in97["usd_tgju"].max()) - 1),
            "max_rise_1397_bonbast_week": in97["usd_free"].idxmax(),
            "max_rise_1397_tgju_week": in97["usd_tgju"].idxmax(),
            "corr_all_min": float(min(corr_all)), "corr_all_max": float(max(corr_all)),
            "corr_ex1397_min": float(min(corr_ex)), "corr_ex1397_max": float(max(corr_ex)),
            "blocks": int(len(ch.iloc[0::weeks]))}


# -------------------------------------------------------------------- robusta
def lagged_correlation(a: pd.Series, b: pd.Series, lags=(-1, 0, 1)) -> dict:
    """Correlation of daily log returns on the dates both series have:
    corr(a[t], b[t + lag]). If `a` stamps each day's price one day late, the
    peak moves from lag 0 to lag -1."""
    both = pd.concat([a.rename("a"), b.rename("b")], axis=1, join="inner").dropna()
    r = np.log(both).diff().dropna()
    out = {"n": len(r)}
    for lag in lags:
        out[f"lag{lag:+d}"] = float(r["a"].corr(r["b"].shift(-lag))) if len(r) > 3 else np.nan
    return out


def lag_by_period(a: pd.Series, b: pd.Series,
                  periods: list[tuple[str, dt.date, dt.date]]) -> pd.DataFrame:
    rows = []
    for label, start, end in periods:
        sl = slice(pd.Timestamp(start), pd.Timestamp(end))
        rows.append({"period": label, "from": start, "to": end,
                     **lagged_correlation(a.loc[sl], b.loc[sl])})
    return pd.DataFrame(rows)


# --------------------------------------------------------------- weekly panel
def weekly_flags_by_year(wk: pd.DataFrame, series: list[str]) -> pd.DataFrame:
    """Per Jalali year: weeks, and for each series the weeks flagged because
    the last observation is too old (`*_old`) or sits in a near-stale run, and
    the weeks whose return is empty although the series has started
    (`*_no_ret`, D-018 after any fill)."""
    jy = _jyears(wk.index)
    out = pd.DataFrame({"weeks": wk.groupby(jy).size()})
    for s in series:
        has = wk[s].notna()
        out[f"{s}_weeks"] = has.groupby(jy).sum()
        out[f"{s}_old"] = (wk[f"{s}_old"] & has).groupby(jy).sum()
        out[f"{s}_near_stale"] = (wk[f"{s}_near_stale"] & has).groupby(jy).sum()
        out[f"{s}_no_ret"] = (has & wk[s].shift().notna() & wk[f"{s}_ret"].isna()).groupby(jy).sum()
    return out


def weekly_volatility_by_year(wk: pd.DataFrame, series: list[str]) -> pd.DataFrame:
    """Std of the weekly log return per series per Jalali year, and the count."""
    jy = _jyears(wk.index)
    out = {}
    for s in series:
        g = wk[f"{s}_ret"].groupby(jy)
        out[f"{s}_std"] = g.std()
        out[f"{s}_n"] = g.count()
    return pd.DataFrame(out)


def weekly_roll_effect(wk: pd.DataFrame, series: list[str]) -> pd.DataFrame:
    """Coffee: weeks whose return changes when roll days are left out."""
    rows = []
    for s in series:
        d = (wk[f"{s}_ret"] - wk[f"{s}_ret_raw"]).dropna()
        hit = d[d.abs() > 1e-12]
        rows.append({"series": s, "weeks": len(d), "weeks_with_roll": len(hit),
                     "std_ret": wk[f"{s}_ret"].std(), "std_ret_raw": wk[f"{s}_ret_raw"].std(),
                     "sum_removed": float(-hit.sum()), "mean_abs_removed": float(hit.abs().mean())})
    return pd.DataFrame(rows)


# ------------------------------------------------------------- coffee vs IMF
def coffee_vs_imf(panel: pd.DataFrame) -> pd.DataFrame:
    """Monthly mean of the daily series (USD/kg) next to the IMF monthly
    indicator (USD/kg). `*_days` = daily observations in the month."""
    imf = fred.load()
    imf.index = pd.to_datetime(imf.pop("month"))
    out = pd.DataFrame(index=imf.index)
    out["arabica_imf"] = cents_lb_to_usd_kg(imf["arabica_imf_clb"])
    out["robusta_imf"] = cents_lb_to_usd_kg(imf["robusta_imf_clb"])
    for name in ("arabica", "robusta"):
        s = panel[f"{name}_usd_kg"]
        m = s.groupby(s.index.to_period("M")).agg(["mean", "count"])
        m.index = m.index.to_timestamp()
        out[f"{name}_daily"] = m["mean"].reindex(out.index)
        out[f"{name}_days"] = m["count"].reindex(out.index)
    return out
