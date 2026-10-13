"""TGJU fetchers and parsers (D-007: the history-table endpoint).

  table  https://api.tgju.org/v1/market/indicator/summary-table-data/<key>
         the JSON behind the table on www.tgju.org/profile/<key>/history. One
         row per date; each row carries its own Gregorian and Jalali date. The
         whole history comes back in one request.
  page   https://www.tgju.org/profile/<key>/history
         the history page itself; its HTML carries the newest ~30 rows. Used
         only to check the table.

D1 used TGJU's chart endpoint (platform.tgju.org/fa/tvdata/history). It was
dropped in D1b: that host's robots.txt disallows everything, and it serves two
bars for some dates. The table reproduces the D1 series exactly. The raw chart
files of D1 stay under data/raw/tgju/chart_*.json as a record; no code reads
them.

Since D1b TGJU's dollar is the check, not the primary series (D-001 revised).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from coffee_risk.data import http

SOURCE = "tgju"
TABLE_URL = ("https://api.tgju.org/v1/market/indicator/summary-table-data/{key}"
             "?lang=fa&order_dir=&draw=1&start=0&length=20000&search="
             "&order_col=&order_dir=&from=&to=&convert_to_ad=1")
PAGE_URL = "https://www.tgju.org/profile/{key}/history"

USD_KEY = "price_dollar_rl"
ROBUSTA_KEY = "commodities-london-coffee"
# Official-channel USD series (information only, spec D1-G).
OFFICIAL_KEYS = {
    "nima_sell_usd": "NIMA, sell",
    "sana_sell_usd": "SANA, sell",
    "bank_usd": "bank rate",
    "ice_transfer_usd_sell": "exchange centre, remittance, sell",
    "ice_currency_usd_sell": "exchange centre, banknote, sell",
}

_TAG = re.compile(r"<[^>]+>")
_ROW = re.compile(r"<tr[^>]*>\s*((?:<td[^>]*>.*?</td>\s*){8})</tr>", re.S)
_CELL = re.compile(r"<td[^>]*>(.*?)</td>", re.S)
_GDATE = re.compile(r"^\d{4}/\d{2}/\d{2}$")


# -------------------------------------------------------------------- parsing
def _num(cell: str) -> float:
    return float(_TAG.sub("", str(cell)).replace(",", "").strip())


def _frame(rows: list[list[str]]) -> pd.DataFrame:
    """Row layout: open, low, high, close, change, change %, Gregorian date,
    Jalali date."""
    return pd.DataFrame({
        "date": [pd.Timestamp(r[6].replace("/", "-")).date() for r in rows],
        "jdate": [r[7].replace("/", "-") for r in rows],
        "open": [_num(r[0]) for r in rows],
        "low": [_num(r[1]) for r in rows],
        "high": [_num(r[2]) for r in rows],
        "close": [_num(r[3]) for r in rows],
    })


def parse_table(body: bytes) -> pd.DataFrame:
    """Table JSON -> one row per table row, oldest first. Nothing is dropped:
    if TGJU ever lists a date twice, both rows are returned."""
    d = json.loads(body)
    rows = d.get("data")
    if not rows:
        raise ValueError("table response has no rows")
    if int(d.get("recordsTotal", len(rows))) != len(rows):
        raise ValueError(f"table truncated: {len(rows)} of {d.get('recordsTotal')} rows")
    return _frame(rows).iloc[::-1].reset_index(drop=True)


def parse_page(body: bytes) -> pd.DataFrame:
    """The rows TGJU prints in the HTML of its history page (the newest ~30),
    same layout as the table endpoint. Used only to check the table."""
    text = body.decode("utf-8", errors="replace")
    rows = []
    for m in _ROW.finditer(text):
        cells = [_TAG.sub("", c).strip() for c in _CELL.findall(m.group(1))]
        if len(cells) == 8 and _GDATE.match(cells[6]):
            rows.append(cells)
    if not rows:
        raise ValueError("no history rows found in the page")
    return _frame(rows).sort_values("date", kind="stable").reset_index(drop=True)


def _summary(parser):
    def run(body: bytes) -> http.Summary:
        df = parser(body)
        return http.Summary(len(df), str(df["date"].min()), str(df["date"].max()))
    return run


# ------------------------------------------------------------------- fetching
def fetch_table(key: str, refresh: bool = False, route: str | None = None) -> Path:
    return http.fetch_raw(SOURCE, f"table_{key}.json", TABLE_URL.format(key=key),
                          _summary(parse_table), refresh=refresh, only_route=route)


def fetch_page(key: str, refresh: bool = False, route: str | None = None) -> Path:
    return http.fetch_raw(SOURCE, f"page_{key}.html", PAGE_URL.format(key=key),
                          _summary(parse_page), refresh=refresh, only_route=route)


def load_table(key: str) -> pd.DataFrame:
    return parse_table(http.raw_path(SOURCE, f"table_{key}.json").read_bytes())


def load_page(key: str) -> pd.DataFrame:
    return parse_page(http.raw_path(SOURCE, f"page_{key}.html").read_bytes())


def has(kind: str, key: str) -> bool:
    ext = "html" if kind == "page" else "json"
    return http.raw_path(SOURCE, f"{kind}_{key}.{ext}").exists()


def series(key: str) -> pd.DataFrame:
    """One row per date for a key: date, close.

    The table has one row per date for every key we use (checked in D1b). If
    TGJU ever lists a date twice this stops instead of picking a row silently.
    """
    t = load_table(key)
    n_dup = int(t["date"].duplicated().sum())
    if n_dup:
        raise ValueError(f"TGJU table for {key!r} lists {n_dup} date(s) more than once; "
                         "decide how to treat them before building the panel")
    return t.sort_values("date", kind="stable").reset_index(drop=True)[["date", "close"]]
