"""D2: measured facts the simulator (D3) will be built on. Measure only: no
simulator, no strategies, no tuning.

Input is the weekly panel. Two kinds of numbers are built from it:

  * weekly statistics (volatility, correlations, autocorrelation, the
    bootstrap pool) use the panel's weekly returns as they are: Bonbast USD,
    roll-excluded coffee returns, D-018 empty weeks, ICO-filled robusta weeks;
  * multi-week changes (13 weeks = the horizon) are differences of a log
    index. For the dollar and the fund the index is the log level; a change
    whose first or last Wednesday has a stale level is left empty. For coffee
    the index is the running sum of the SAME weekly returns, roll days out and
    ICO-filled weeks in (D-022), so a multi-week coffee change is always the
    sum of its weekly returns; a window that contains a week with no return
    is left empty.
"""
from __future__ import annotations

import datetime as dt
from typing import Iterable

import numpy as np
import pandas as pd

from coffee_risk.dates import from_jalali, jalali_year

HORIZON = 13                                   # weeks = 3 months
WEEKS_PER_YEAR = 52
BACKTEST_START = from_jalali(1394, 2, 30)      # D-015: Wednesday 2015-05-20. The D2 sample; the backtest
                                               # itself starts later (D-041, engine.calibration.BACKTEST_START)
WAR_START_WEEK = dt.date(2026, 2, 25)          # last Wednesday before 9 Esfand 1404 (D-005)

# name -> weekly panel column
SERIES = {"usd": "usd_free", "arabica": "arabica_usd_kg", "robusta": "robusta_usd_kg",
          "fund": "fund_tr_index"}
RISKY = ["usd", "arabica", "robusta"]
MIXES = {"arabica": 1.0, "robusta": 0.0, "mix": 0.5}       # weight of arabica in the coffee part

# D-005 episodes: (key, kind, first day, last day) of the search range
EPISODES = [
    ("fx_1397", "fx", from_jalali(1397, 1, 1), from_jalali(1397, 12, 29)),
    ("fx_1401_1402", "fx", from_jalali(1401, 1, 1), from_jalali(1402, 12, 29)),
    ("coffee_2021", "coffee", dt.date(2021, 1, 1), dt.date(2021, 12, 31)),
    ("coffee_2024", "coffee", dt.date(2024, 1, 1), dt.date(2025, 3, 31)),
    ("war", "war", WAR_START_WEEK, None),
]
PERIODS = [("2012-10 .. 2017", dt.date(2012, 10, 1), dt.date(2017, 12, 31)),
           ("2018 .. 2022", dt.date(2018, 1, 1), dt.date(2022, 12, 31)),
           ("2023 .. now", dt.date(2023, 1, 1), None)]


def jyears(index) -> list[int]:
    return [jalali_year(d.date()) for d in index]


def sample(x, name: str):
    """'full' = everything; 'backtest' = from the week of 1394-02-30 (D-015)."""
    if name == "full":
        return x
    if name == "backtest":
        return x[x.index >= pd.Timestamp(BACKTEST_START)]
    raise ValueError(name)


def sample_windows(ch, name: str, weeks: int = HORIZON):
    """Multi-week changes (indexed by window END) that belong to a sample: for
    'backtest' the whole window must lie in it, i.e. it STARTS on or after the
    backtest start."""
    if name == "full":
        return ch
    if name == "backtest":
        return ch[ch.index >= pd.Timestamp(BACKTEST_START) + pd.Timedelta(weeks=weeks)]
    raise ValueError(name)


# ------------------------------------------------------------ weekly returns
def weekly_returns(wk: pd.DataFrame) -> pd.DataFrame:
    """The panel's weekly log returns, one column per series name."""
    return pd.DataFrame({name: wk[f"{col}_ret"] for name, col in SERIES.items()})


def mix_return(r_arabica, r_robusta, w_arabica: float = 0.5):
    """Weekly log return of a coffee basket rebalanced to the weights every
    week: log(w * e^rA + (1 - w) * e^rR)."""
    if w_arabica == 1.0:
        return r_arabica
    if w_arabica == 0.0:
        return r_robusta
    return np.log(w_arabica * np.exp(r_arabica) + (1.0 - w_arabica) * np.exp(r_robusta))


def rial_cost_returns(r: pd.DataFrame) -> pd.DataFrame:
    """Weekly log change of the rial cost of coffee = USD return + coffee
    return, for 100% arabica, 100% robusta and the 50/50 basket."""
    out = {}
    for name, w in MIXES.items():
        coffee = mix_return(r["arabica"], r["robusta"], w)
        out[f"coffee_{name}"] = coffee
        out[f"cost_{name}"] = r["usd"] + coffee
    return pd.DataFrame(out)


# ------------------------------------------------------------- log indices
def log_index(wk: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(log index, stale, gaps) per Wednesday for usd, arabica, robusta, fund,
    the three coffee baskets and the three rial-cost mixes.

    usd, fund: the index is the log level; `stale` marks Wednesdays whose
    level is more than 5 days old, so no change may start or end there.
    coffee (D-022): the index is the running sum of the panel's weekly returns
    (roll days out, ICO-filled weeks in). `gaps` counts the weeks so far that
    have no return: a change is valid only if that count is the same at both
    ends, i.e. no week inside the window is missing. Coffee is never "stale":
    its changes do not use the level.
    """
    idx = pd.DataFrame(index=wk.index)
    stale = pd.DataFrame(False, index=wk.index, columns=[])
    gaps = pd.DataFrame(0, index=wk.index, columns=[])
    for name in ("usd", "fund"):
        idx[name] = np.log(wk[SERIES[name]])
        stale[name] = wk[f"{SERIES[name]}_old"] | wk[SERIES[name]].isna()
        gaps[name] = 0
    r = weekly_returns(wk)
    started = {n: wk[SERIES[n]].notna() for n in ("arabica", "robusta")}

    def running(ret: pd.Series, ok: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
        has_prev = ok & ok.shift(1, fill_value=False)          # a return can exist this week
        missing = has_prev & ret.isna()
        return ret.fillna(0.0).cumsum().where(ok), ~ok, missing.cumsum()

    for name in ("arabica", "robusta"):
        idx[name], stale[name], gaps[name] = running(r[name], started[name])
    both = started["arabica"] & started["robusta"]
    for name, w in MIXES.items():
        ok = started["arabica"] if w == 1.0 else started["robusta"] if w == 0.0 else both
        coffee = mix_return(r["arabica"], r["robusta"], w)
        idx[f"coffee_{name}"], stale[f"coffee_{name}"], gaps[f"coffee_{name}"] = running(coffee, ok)
        idx[f"cost_{name}"] = idx["usd"] + idx[f"coffee_{name}"]
        stale[f"cost_{name}"] = stale[f"coffee_{name}"] | stale["usd"]
        gaps[f"cost_{name}"] = gaps[f"coffee_{name}"]
    return idx, stale, gaps


def horizon_changes(idx: pd.DataFrame, stale: pd.DataFrame, gaps: pd.DataFrame,
                    weeks: int = HORIZON) -> pd.DataFrame:
    """Log change over `weeks` weeks, indexed by the window's END Wednesday.
    Empty where the first or the last Wednesday is stale, or where a week
    inside the window has no return."""
    ch = idx - idx.shift(weeks)
    bad = stale | stale.shift(weeks, fill_value=True) | (gaps != gaps.shift(weeks))
    return ch.where(~bad)


def window_start(end: pd.Timestamp, weeks: int = HORIZON) -> pd.Timestamp:
    return end - pd.Timedelta(weeks=weeks)


# ------------------------------------------------------------- A. volatility
def rolling_vol(r: pd.Series, window: int = 26, min_periods: int | None = None) -> pd.Series:
    """Annualised rolling std of weekly log returns (x sqrt(52))."""
    mp = min_periods if min_periods is not None else int(window * 0.8)
    return r.rolling(window, min_periods=mp).std() * np.sqrt(WEEKS_PER_YEAR)


def current_vol(r: pd.Series, window: int) -> dict:
    """Latest rolling volatility and its percentile in the history of the
    same rolling series (share of past values at or below it)."""
    v = rolling_vol(r, window).dropna()
    last = v.iloc[-1]
    return {"window": window, "vol": float(last), "as_of": v.index[-1],
            "percentile": float((v <= last).mean()), "history": len(v),
            "median": float(v.median())}


# ----------------------------------------------------------------- B. carry
def carry(idx: pd.DataFrame, stale: pd.DataFrame, gaps: pd.DataFrame,
          weeks: int = HORIZON) -> pd.DataFrame:
    """13-week carry for every weekly start t: log USD change from t to t+13
    minus log fund return over the same weeks. Indexed by the START week."""
    cols = ["usd", "fund"]
    ch = horizon_changes(idx[cols], stale[cols], gaps[cols], weeks)
    out = pd.DataFrame({"usd": ch["usd"], "fund": ch["fund"], "carry": ch["usd"] - ch["fund"]})
    out.index = out.index - pd.Timedelta(weeks=weeks)
    return out.dropna(subset=["carry"])


def jensen_gap(log_changes: pd.Series) -> dict:
    """Mean simple change against the simple change of the mean log change.

    For log changes x: mean(e^x - 1) is the average percentage change;
    e^mean(x) - 1 is what the average log growth gives. The first is larger,
    by about half the variance of x (exactly e^mu * (e^(s^2/2) - 1) for a
    normal x). That gap is why equal log growth is not equal expected cost.
    """
    x = log_changes.dropna()
    mean_simple = float(np.expm1(x).mean())
    from_mean_log = float(np.expm1(x.mean()))
    return {"n": len(x), "mean_log": float(x.mean()), "std_log": float(x.std()),
            "mean_simple": mean_simple, "from_mean_log": from_mean_log,
            "gap": mean_simple - from_mean_log, "half_variance": float(x.var() / 2)}


def describe_carry(c: pd.Series) -> dict:
    return {"n": len(c), "mean": c.mean(), "median": c.median(), "p5": c.quantile(0.05),
            "p95": c.quantile(0.95), "share_positive": float((c > 0).mean())}


# ---------------------------------------------------------- C. distribution
def shape(r: pd.Series) -> dict:
    x = r.dropna()
    return {"n": len(x), "mean": x.mean(), "std": x.std(), "skew": x.skew(),
            "excess_kurtosis": x.kurt(), "min": x.min(), "max": x.max()}


def quantiles(x: pd.Series, qs: Iterable[float] = (0.01, 0.05, 0.5, 0.95, 0.99)) -> dict:
    x = x.dropna()
    return {"n": len(x), **{f"q{int(q * 100):02d}": float(x.quantile(q)) for q in qs}}


# ------------------------------------------------------------ D. dependence
def nonoverlapping_corr(ch: pd.DataFrame, a: str, b: str, weeks: int = HORIZON) -> dict:
    """Correlation of non-overlapping `weeks`-week changes, for every start
    offset 0..weeks-1; the range over the offsets."""
    x = ch[[a, b]]
    vals, ns = [], []
    for off in range(weeks):
        block = x.iloc[off::weeks].dropna()
        if len(block) >= 8:
            vals.append(block[a].corr(block[b]))
            ns.append(len(block))
    return {"min": float(np.min(vals)), "max": float(np.max(vals)),
            "median": float(np.median(vals)), "blocks_min": int(np.min(ns)),
            "blocks_max": int(np.max(ns))}


def tail_comovement(r: pd.DataFrame, given: str, other: str, q: float = 0.95) -> dict:
    """Mean and median of `other` in the weeks where `given` is above its
    q-quantile, next to its mean and median in all weeks."""
    x = r[[given, other]].dropna()
    cut = x[given].quantile(q)
    tail = x[x[given] > cut]
    return {"weeks": len(tail), "threshold": float(cut),
            "mean": float(tail[other].mean()), "median": float(tail[other].median()),
            "share_positive": float((tail[other] > 0).mean()),
            "all_mean": float(x[other].mean()), "all_median": float(x[other].median()),
            "all_share_positive": float((x[other] > 0).mean())}


# ----------------------------------------------------------- E. persistence
def acf(r: pd.Series, lags: int = HORIZON) -> pd.Series:
    """Autocorrelation at lags 1..lags, using the pairs of weeks that both
    have a value (a missing week does not shift the lags)."""
    return pd.Series({k: r.autocorr(lag=k) for k in range(1, lags + 1)})


def acf_band(r: pd.Series) -> float:
    """Approximate 95% band for a single autocorrelation: 1.96 / sqrt(n)."""
    return 1.96 / np.sqrt(r.notna().sum())


def block_length(r: pd.Series) -> dict:
    """Politis-White optimal block length (arch), for the stationary and the
    circular bootstrap. Missing weeks are dropped first."""
    from arch.bootstrap import optimal_block_length
    out = optimal_block_length(r.dropna().values)
    return {"stationary": float(out["stationary"].iloc[0]),
            "circular": float(out["circular"].iloc[0])}


def variance_ratio(r: pd.Series, weeks: int = HORIZON) -> dict:
    """How do `weeks`-week sums compare with what independent weeks would give?

    variance ratio = Var(sum of `weeks` consecutive returns) / (weeks x Var(one
    week)): 1 for independent weeks, above 1 when moves tend to continue.
    Also the excess kurtosis of those sums next to the value independent
    weeks would give (weekly excess kurtosis / weeks): a larger measured value
    means calm and stormy weeks come in clusters. Overlapping sums; a sum with
    a missing week is left out.
    """
    x = r.dropna()
    sums = r.rolling(weeks, min_periods=weeks).sum().dropna()
    return {"n_sums": len(sums), "variance_ratio": float(sums.var() / (weeks * x.var())),
            "kurtosis_sums": float(sums.kurt()), "kurtosis_if_independent": float(x.kurt() / weeks)}


def vol_predictability(r: pd.Series, lookback: int = 26, ahead: int = HORIZON) -> dict:
    """Does recent volatility say anything about the next quarter's? Rank
    correlation between the std of the last `lookback` weeks and the std of
    the next `ahead` weeks, on non-overlapping steps of `ahead` weeks."""
    past = r.rolling(lookback, min_periods=int(lookback * 0.8)).std()
    nxt = r[::-1].rolling(ahead, min_periods=int(ahead * 0.8)).std()[::-1].shift(-1)
    both = pd.DataFrame({"past": past, "next": nxt}).dropna().iloc[::ahead]
    return {"n": len(both), "spearman": float(both["past"].corr(both["next"], method="spearman")),
            "pearson": float(both["past"].corr(both["next"]))}


# ---------------------------------------------------------------- F, G. windows
def top_nonoverlapping(ch: pd.Series, k: int = 10, weeks: int = HORIZON,
                       largest: bool = True) -> pd.Series:
    """The `k` most extreme `weeks`-week changes whose windows do not overlap.
    `ch` is indexed by window end. Greedy: take the most extreme, drop every
    window that shares a week with it, repeat."""
    left = ch.dropna().sort_values(ascending=not largest)
    picked = []
    span = pd.Timedelta(weeks=weeks)
    while len(left) and len(picked) < k:
        end = left.index[0]
        picked.append(end)
        left = left[(left.index <= end - span) | (left.index >= end + span)]
    return ch.loc[picked]


def best_window(ch: pd.Series, first: dt.date, last: dt.date | None,
                weeks: int = HORIZON) -> pd.Timestamp | None:
    """End of the window with the largest change among the windows that lie
    wholly inside [first, last] (both its first and last Wednesday)."""
    x = ch.dropna()
    start = x.index - pd.Timedelta(weeks=weeks)
    inside = start >= pd.Timestamp(first)
    if last is not None:
        inside &= x.index <= pd.Timestamp(last)
    x = x[inside]
    return x.idxmax() if len(x) else None


def window_row(ch: pd.DataFrame, end: pd.Timestamp, weeks: int = HORIZON) -> dict:
    """Every series' change over the window that ends at `end`."""
    return {"start": window_start(end, weeks), "end": end, **ch.loc[end].to_dict()}


def path_change(idx: pd.DataFrame, stale: pd.DataFrame, gaps: pd.DataFrame,
                start: pd.Timestamp, end: pd.Timestamp) -> dict:
    """Log change of every index from `start` to `end`; empty if an endpoint
    is stale or a week in between has no return."""
    out = {"start": start, "end": end, "weeks": int((end - start).days // 7)}
    for c in idx.columns:
        bad = bool(stale.loc[start, c] or stale.loc[end, c]
                   or gaps.loc[start, c] != gaps.loc[end, c])
        out[c] = np.nan if bad else float(idx.loc[end, c] - idx.loc[start, c])
    return out


# ---------------------------------------------------------------- H. pool
def pool(r: pd.DataFrame, cols: Iterable[str] = RISKY) -> pd.Series:
    """Weeks in which every series of the joint pool has a weekly return."""
    return r[list(cols)].notna().all(axis=1)
