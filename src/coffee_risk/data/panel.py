"""Build data/processed/panel_daily.parquet from the raw cache.

One row per calendar day from the first to the last observation of any series.
A cell is empty unless that series has an observation on that day: nothing is
filled or interpolated. Checks only add flag columns.

Columns (layout since D1b)
  jdate                    Jalali date, 'YYYY-MM-DD'
  usd_free                 free-market USD, rial: Bonbast sell rate, dates as
                           published (primary, D-001 revised)
  usd_tgju                 free-market USD, rial: TGJU close (check)
  arabica_clb              ICE arabica front month, US cents/lb (native)
  arabica_usd_kg           same, USD/kg
  robusta_usd_t            ICE London robusta front month, USD/tonne (native):
                           TGJU's London-coffee key, weekday rows
  robusta_usd_kg           same, USD/kg
  robusta_tgju_usd_t       every row of TGJU's London-coffee key, as served
  robusta_ico_clb          ICO Robustas indicator, US cents/lb. A PHYSICAL
                           price, not futures: for returns only, never for
                           levels (D-020). Only months that pass validation.
  fund_close_irr           closing price of the chosen fixed-income fund, rial
  fund_tr_index            its total-return index, 100 on the first traded day
  *_source                 where the value came from
  robusta_weekend          TGJU robusta row dated on a Saturday or Sunday; such
                           rows stay in robusta_tgju_usd_t but are not
                           observations of the primary series (D-010)
  *_roll                   first day the continuous series shows the next contract
  *_stale                  same value on 5 or more consecutive observations
  *_near_stale             5 or more consecutive observations, each within
                           0.05% of the one before
  *_outlier                |log return| vs the previous observation above the
                           threshold (10% USD, 8% coffee)
  usd_outlier_check        for each TGJU outlier: confirmed / not_confirmed by
                           Bonbast, or no_bonbast (information only, D-008)
"""
from __future__ import annotations

import pandas as pd

from coffee_risk import paths
from coffee_risk.dates import to_jalali
from coffee_risk.units import cents_lb_to_usd_kg, usd_tonne_to_usd_kg
from coffee_risk.data import bonbast, fund, ico, quality, rolls, tgju, weekly, yahoo

OFFICIAL = paths.PROCESSED / "official_usd_daily.parquet"

# Columns D2 will model, and the flags the return functions need.
PRICE_COLS = ["usd_free", "arabica_usd_kg", "robusta_usd_kg"]
ROLL_FLAGS = {"arabica_usd_kg": "arabica_roll", "robusta_usd_kg": "robusta_roll"}

# series checked for stale / near-stale values: label -> level column
CHECKED = {"usd_free": "usd_free", "usd_tgju": "usd_tgju", "arabica": "arabica_clb",
           "robusta": "robusta_usd_t", "fund": "fund_close_irr"}
OUTLIER = {"usd_free": quality.USD_EXTREME, "usd_tgju": quality.USD_EXTREME,
           "arabica": quality.COFFEE_EXTREME, "robusta": quality.COFFEE_EXTREME}


def tgju_robusta() -> pd.DataFrame:
    """TGJU's London-coffee key: date, close, weekend (dated Sat/Sun)."""
    t = tgju.series(tgju.ROBUSTA_KEY)
    t["weekend"] = [d.weekday() >= 5 for d in t["date"]]
    return t


def robusta_series() -> tuple[pd.DataFrame, pd.DataFrame]:
    """(robusta series, TGJU rows as served).

    The series is TGJU's weekday rows: date, close, source. Weekend-dated TGJU
    rows are never part of it (D-010); they stay in the second frame, marked.
    Its holes are not filled here: the weekly panel fills the weekly RETURNS
    of those weeks from the ICO indicator (D-020), levels stay as they are.
    """
    tg = tgju_robusta()
    series = tg.loc[~tg["weekend"], ["date", "close"]].assign(source="tgju")
    return series.reset_index(drop=True), tg


def _col(df: pd.DataFrame, value: str, index: pd.DatetimeIndex) -> pd.Series:
    s = pd.Series(df[value].values, index=pd.to_datetime(df["date"]))
    return s.reindex(index)


def _flag(dates, index: pd.DatetimeIndex) -> pd.Series:
    out = pd.Series(False, index=index)
    out.loc[pd.to_datetime(list(dates))] = True
    return out


def _source(level: pd.Series, name: str) -> pd.Series:
    return level.notna().map({True: name, False: None})


def build(save: bool = True) -> pd.DataFrame:
    bb = bonbast.load()
    tg_usd = tgju.series(tgju.USD_KEY)
    kc = yahoo.load()
    rb, rb_tgju = robusta_series()
    ico_rb = ico.load() if ico.available() else pd.DataFrame(columns=["date", "robustas"])
    chosen = fund.choose(fund.candidates())
    fd = fund.total_return_index(chosen["ins_code"])

    frames = (bb, tg_usd, kc, rb, rb_tgju, fd)
    idx = pd.date_range(min(df["date"].min() for df in frames),
                        max(df["date"].max() for df in frames), freq="D", name="date")
    p = pd.DataFrame(index=idx)
    p["jdate"] = [to_jalali(d.date()) for d in idx]

    # --- levels -----------------------------------------------------------
    p["usd_free"] = _col(bb, "sell", idx)
    p["usd_free_source"] = _source(p["usd_free"], "bonbast")
    p["usd_tgju"] = _col(tg_usd, "close", idx)
    p["arabica_clb"] = _col(kc, "close", idx)
    p["arabica_usd_kg"] = cents_lb_to_usd_kg(p["arabica_clb"])
    p["arabica_source"] = _source(p["arabica_clb"], yahoo.continuous_source())
    p["robusta_usd_t"] = _col(rb, "close", idx)
    p["robusta_usd_kg"] = usd_tonne_to_usd_kg(p["robusta_usd_t"])
    p["robusta_source"] = _col(rb, "source", idx)
    p["robusta_tgju_usd_t"] = _col(rb_tgju, "close", idx)
    p["robusta_ico_clb"] = _col(ico_rb, "robustas", idx)
    p["fund_close_irr"] = _col(fd, "close", idx)
    p["fund_tr_index"] = _col(fd, "tr_index", idx)
    p["fund_source"] = _source(p["fund_close_irr"], f"tsetmc:{chosen['ins_code']}")

    # --- flags ------------------------------------------------------------
    p["robusta_weekend"] = _flag(rb_tgju.loc[rb_tgju["weekend"], "date"], idx)
    kc_dates, rb_dates = list(kc["date"]), list(rb["date"])
    kc_cal = rolls.trading_calendar(kc_dates)
    rb_cal = rolls.trading_calendar(rb_dates, ref_dates=kc_dates)
    p["arabica_roll"] = _flag(
        [d for d, f in rolls.roll_flags(kc_dates, rolls.switch_days(kc_cal, rolls.ARABICA)).items() if f], idx)
    p["robusta_roll"] = _flag(
        [d for d, f in rolls.roll_flags(rb_dates, rolls.switch_days(rb_cal, rolls.ROBUSTA)).items() if f], idx)
    for name, col in CHECKED.items():
        p[f"{name}_stale"] = quality.stale_flag(p[col])
        p[f"{name}_near_stale"] = quality.near_stale_flag(p[col])
    for name, threshold in OUTLIER.items():
        p[f"{name}_outlier"] = quality.extreme_flag(p[CHECKED[name]], threshold)

    # each extreme TGJU move, checked against Bonbast (information only, D-008)
    from coffee_risk.data import checks
    ex = checks.usd_extremes(p)
    p["usd_outlier_check"] = None
    if not ex.empty:
        p.loc[ex["date"], "usd_outlier_check"] = ex["verdict"].values

    if save:
        paths.PROCESSED.mkdir(parents=True, exist_ok=True)
        p.to_parquet(paths.PANEL)
        build_official(save=True)
        weekly.build(p, save=True)
    return p


def build_official(save: bool = True) -> pd.DataFrame:
    """Official-channel USD series in rial, one column per TGJU key
    (information only; not part of the model panel)."""
    cols = {}
    for key in tgju.OFFICIAL_KEYS:
        if tgju.has("table", key):
            d = tgju.series(key)
            cols[key] = pd.Series(d["close"].values, index=pd.to_datetime(d["date"]))
    out = pd.DataFrame(cols).sort_index()
    out.index.name = "date"
    out.insert(0, "jdate", [to_jalali(d.date()) for d in out.index])
    if save:
        paths.PROCESSED.mkdir(parents=True, exist_ok=True)
        out.to_parquet(OFFICIAL)
    return out


def load() -> pd.DataFrame:
    return pd.read_parquet(paths.PANEL)
