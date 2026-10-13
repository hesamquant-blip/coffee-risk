"""Arabica futures (ICE US "KC", US cents/lb) from Yahoo Finance via yfinance.

KC=F is Yahoo's continuous front-month series. Single contracts use tickers
like KCZ26.NYB (month code + two-digit year + ".NYB"); Yahoo serves them only
while the contract is listed, so only the contracts alive today are available
as roll evidence.

yfinance does its own HTTP, so there is no raw body to keep: the "raw" file is
the frame it returns, written to CSV without any change (auto_adjust=False).

Fallback if Yahoo fails: Stooq's kc.f CSV.
"""
from __future__ import annotations

import datetime as dt
import io
from pathlib import Path

import pandas as pd

from coffee_risk.data import http

SOURCE = "yahoo"
CONTINUOUS = "KC=F"
MONTH_CODES = {3: "H", 5: "K", 7: "N", 9: "U", 12: "Z"}     # KC delivery months
STOOQ_URL = "https://stooq.com/q/d/l/?s=kc.f&i=d"


def contract_ticker(year: int, month: int) -> str:
    return f"KC{MONTH_CODES[month]}{year % 100:02d}.NYB"


def contracts_around(today: dt.date, back: int = 2, ahead: int = 5) -> list[tuple[int, int]]:
    """(year, month) of the `back` most recent delivery months before today's
    month and the `ahead` next ones (today's month included)."""
    months = sorted(MONTH_CODES)
    seq = [(y, m) for y in range(today.year - 2, today.year + 4) for m in months]
    i = next(k for k, (y, m) in enumerate(seq) if (y, m) >= (today.year, today.month))
    return seq[i - back:i + ahead]


def _file(ticker: str) -> str:
    return ticker.replace("=", "_") + ".csv"


def parse(body: bytes) -> pd.DataFrame:
    """-> date (exchange calendar date), open, high, low, close, volume."""
    df = pd.read_csv(io.BytesIO(body))
    df.columns = [c.strip().lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"].astype(str).str[:10]).dt.date
    keep = ["date", "open", "high", "low", "close", "volume"]
    return df[keep].sort_values("date", kind="stable").reset_index(drop=True)


def _summary(body: bytes) -> http.Summary:
    df = parse(body)
    if df.empty:
        raise ValueError("no rows")
    return http.Summary(len(df), str(df["date"].min()), str(df["date"].max()))


def _download(ticker: str, route: str) -> bytes:
    import yfinance as yf
    http.pause()
    with http.proxy_env(route):
        df = yf.Ticker(ticker).history(period="max", interval="1d", auto_adjust=False)
    if df is None or df.empty:
        raise RuntimeError("yfinance returned no rows (ticker not served)")
    return df.to_csv().encode("utf-8")


def fetch_ticker(ticker: str, refresh: bool = False, route: str | None = None) -> Path:
    path = http.raw_path(SOURCE, _file(ticker))
    if path.exists() and not refresh:
        return path
    endpoint = f"yfinance.Ticker('{ticker}').history(period='max', interval='1d', auto_adjust=False)"
    errors = []
    for r in ((route,) if route else ("direct", "proxy")):
        try:
            if r == "proxy" and not http.proxy_configured():
                raise RuntimeError("no proxy variables set in the environment")
            body = _download(ticker, r)
            out = http.save_raw(SOURCE, _file(ticker), endpoint, r, body, _summary)
        except Exception as exc:
            msg = f"{type(exc).__name__}: {exc}"
            errors.append(f"[{r}] {msg}")
            http.log_attempt(SOURCE, _file(ticker), endpoint, r, "error", msg)
            continue
        http.log_attempt(SOURCE, _file(ticker), endpoint, r, "ok", f"{len(body)} bytes")
        return out
    raise http.FetchFailed(f"{SOURCE}/{_file(ticker)}: " + " | ".join(errors))


def _stooq_summary(body: bytes) -> http.Summary:
    if b"<html" in body[:400].lower():
        raise ValueError("Stooq served a JavaScript browser check, not a CSV")
    return _summary(body)


def fetch_stooq(symbol: str = "kc.f", refresh: bool = False,
                route: str | None = None) -> Path:
    return http.fetch_raw("stooq", symbol.replace(".", "_") + ".csv",
                          STOOQ_URL.replace("kc.f", symbol), _stooq_summary,
                          refresh=refresh, only_route=route)


def fetch(refresh: bool = False, route: str | None = None,
          today: dt.date | None = None) -> dict:
    """Continuous series (Stooq if Yahoo fails) plus the listed contracts."""
    out = {"continuous": None, "contracts": {}, "contract_errors": {}}
    try:
        out["continuous"] = fetch_ticker(CONTINUOUS, refresh, route)
    except http.FetchFailed as exc:
        print(f"  Yahoo failed ({exc}); trying Stooq")
        out["continuous"] = fetch_stooq("kc.f", refresh, route)
    for y, m in contracts_around(today or dt.date.today()):
        tk = contract_ticker(y, m)
        cached = http.raw_path(SOURCE, _file(tk)).exists()
        if not cached and not refresh and http.attempted(SOURCE, _file(tk)):
            out["contract_errors"][tk] = "not served on an earlier run (use --refresh to retry)"
            continue
        try:
            out["contracts"][tk] = fetch_ticker(tk, refresh, route)
        except http.FetchFailed as exc:
            out["contract_errors"][tk] = str(exc)
    return out


def load(ticker: str = CONTINUOUS) -> pd.DataFrame:
    path = http.raw_path(SOURCE, _file(ticker))
    if ticker == CONTINUOUS and not path.exists():
        path = http.raw_path("stooq", "kc_f.csv")
    return parse(path.read_bytes())


def continuous_source() -> str:
    return "yahoo" if http.raw_path(SOURCE, _file(CONTINUOUS)).exists() else "stooq"


def loaded_contracts() -> dict[str, pd.DataFrame]:
    out = {}
    for p in sorted((http.raw_path(SOURCE, "x").parent).glob("KC???.NYB.csv")):
        out[p.name[:-4]] = parse(p.read_bytes())
    return out
