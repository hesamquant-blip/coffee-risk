"""TSETMC: listed fixed-income funds (ETFs) and their daily history.

The HTTP rules and the instrument-list layout are adapted (minimal version)
from an earlier project by the author:
  * session set-up: browser User-Agent, proxy variables ignored, an empty 200
    is never trusted on first sight;
  * the 18-field layout of the bulk instrument list (field 17 = market code,
    305 = funds);
  * GetClosingPriceDailyList returns an instrument's whole daily history in
    one call.
Only code was reused; no data from that project is read.

Fund class. In the bulk list, sector 68 = exchange-traded funds and
sub-sector 6812 = fixed-income funds (checked by name: every 6812 row is a
"-ثابت"/"-د" fund).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

from coffee_risk.data import http

SOURCE = "tsetmc"
LIST_URL = "http://service.tsetmc.com/tsev2/data/TseClient2.aspx?t=Instrument&a=0"
DAILY_URL = "https://cdn.tsetmc.com/api/ClosingPrice/GetClosingPriceDailyList/{ins}/0"
LIST_FILE = "instrument_list.txt"

FIELDS = ["ins_code", "isin", "latin_ticker", "latin_name", "latin_base",
          "ticker", "name", "company_isin", "last_update", "flow", "base_name",
          "f11", "board_code", "f13", "board_tier", "sector", "sub_sector",
          "market_code"]
FUND_MARKET_CODE = "305"
FIXED_INCOME_SUB_SECTOR = "6812"
ROUTES = ("direct",)          # TSETMC answers on the direct route only


def _clean(s: str) -> str:
    """Arabic yeh/kaf -> Persian, trimmed (TSETMC mixes both forms)."""
    return s.replace("ي", "ی").replace("ك", "ک").strip()


# ------------------------------------------------------------ instrument list
def parse_list(body: bytes) -> pd.DataFrame:
    text = body.decode("utf-8", errors="replace")
    rows = [r.split(",") for r in text.split(";") if r.strip()]
    rows = [r for r in rows if len(r) == len(FIELDS)]
    if len(rows) < 1000:
        raise ValueError(f"instrument list too short: {len(rows)} rows")
    df = pd.DataFrame(rows, columns=FIELDS)
    for c in FIELDS:
        df[c] = df[c].str.strip()
    df["ticker"] = df["ticker"].map(_clean)
    df["name"] = df["name"].map(_clean)
    return df


def fetch_list(refresh: bool = False) -> Path:
    return http.fetch_raw(SOURCE, LIST_FILE, LIST_URL,
                          lambda b: http.Summary(rows=len(parse_list(b))),
                          routes=ROUTES, refresh=refresh)


def fixed_income_funds() -> pd.DataFrame:
    df = parse_list(http.raw_path(SOURCE, LIST_FILE).read_bytes())
    m = (df["market_code"] == FUND_MARKET_CODE) & (df["sub_sector"] == FIXED_INCOME_SUB_SECTOR)
    return df[m].reset_index(drop=True)


# -------------------------------------------------------------- daily history
def parse_daily(body: bytes) -> pd.DataFrame:
    """-> one row per day TSETMC lists, oldest first.

    close = pClosing (volume-weighted closing price), last = pDrCotVal (last
    trade), prev_close = priceYesterday (the reference TSETMC itself uses for
    the day's change; it is restated after a cash distribution), volume,
    value, trades.
    """
    d = json.loads(body)
    rows = d.get("closingPriceDaily")
    if rows is None:
        raise ValueError(f"unexpected daily payload: keys {list(d)[:5]}")
    if not rows:
        return pd.DataFrame(columns=["date", "close", "last", "prev_close", "open",
                                     "high", "low", "volume", "value", "trades"])
    df = pd.DataFrame(rows)
    out = pd.DataFrame({
        "date": pd.to_datetime(df["dEven"].astype(str), format="%Y%m%d").dt.date,
        "close": df["pClosing"].astype(float),
        "last": df["pDrCotVal"].astype(float),
        "prev_close": df["priceYesterday"].astype(float),
        "open": df["priceFirst"].astype(float),
        "high": df["priceMax"].astype(float),
        "low": df["priceMin"].astype(float),
        "volume": df["qTotTran5J"].astype(float),
        "value": df["qTotCap"].astype(float),
        "trades": df["zTotTran"].astype(float),
    })
    return out.sort_values("date", kind="stable").reset_index(drop=True)


def _daily_summary(body: bytes) -> http.Summary:
    df = parse_daily(body)
    if df.empty:
        return http.Summary(rows=0)
    return http.Summary(len(df), str(df["date"].min()), str(df["date"].max()))


def fetch_daily(ins_code: str, refresh: bool = False) -> Path:
    """Whole daily history of one instrument. An empty answer is re-requested
    twice before it is believed (TSETMC sometimes serves an empty 200)."""
    file = f"daily_{ins_code}.json"
    path = http.raw_path(SOURCE, file)
    if path.exists() and not refresh:
        return path
    url = DAILY_URL.format(ins=ins_code)
    last_exc = None
    for attempt in range(3):
        try:
            body = http.get(url, route="direct").content
            if _daily_summary(body).rows == 0 and attempt < 2:
                time.sleep(3)
                continue
            out = http.save_raw(SOURCE, file, url, "direct", body, _daily_summary)
            http.log_attempt(SOURCE, file, url, "direct", "ok", f"{len(body)} bytes")
            return out
        except Exception as exc:
            last_exc = exc
            http.log_attempt(SOURCE, file, url, "direct", "error",
                             f"{type(exc).__name__}: {exc}")
            time.sleep(3)
    raise http.FetchFailed(f"{SOURCE}/{file}: {type(last_exc).__name__}: {last_exc}")


def load_daily(ins_code: str) -> pd.DataFrame:
    return parse_daily(http.raw_path(SOURCE, f"daily_{ins_code}.json").read_bytes())


def has_daily(ins_code: str) -> bool:
    return http.raw_path(SOURCE, f"daily_{ins_code}.json").exists()


def fetch(refresh: bool = False, route: str | None = None) -> dict:
    """Instrument list, then the daily history of every fixed-income fund."""
    if route == "proxy":
        raise http.FetchFailed("tsetmc: only the direct route is supported "
                               "(TSETMC does not answer through a proxy)")
    fetch_list(refresh)
    funds = fixed_income_funds()
    ok, failed = 0, {}
    for ins in funds["ins_code"]:
        try:
            fetch_daily(ins, refresh)
            ok += 1
        except http.FetchFailed as exc:
            failed[ins] = str(exc)
    return {"funds": len(funds), "ok": ok, "failed": failed}
