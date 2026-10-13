"""Bonbast free-market USD/IRR, from the public archive on GitHub.

Repository SamadiPour/rial-exchange-rates-archive (MIT licence), branch
`data`, file currency/usd.json: {"1391/07/18": {"sell": 3550, "buy": 3300}, ...}
Keys are Jalali dates; rates are in TOMAN. We use the sell rate (what a buyer
of dollars pays) and convert to rial.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from coffee_risk.dates import parse_jalali
from coffee_risk.units import toman_to_rial
from coffee_risk.data import http

SOURCE = "bonbast"
FILE = "usd.json"
URL = ("https://raw.githubusercontent.com/SamadiPour/"
       "rial-exchange-rates-archive/data/currency/usd.json")


def parse(body: bytes) -> pd.DataFrame:
    """-> date, jdate, sell_toman, buy_toman, sell (rial), buy (rial)."""
    d = json.loads(body)
    if not isinstance(d, dict) or not d:
        raise ValueError("unexpected archive layout")
    rows = [{"date": parse_jalali(k), "jdate": k.replace("/", "-"),
             "sell_toman": v.get("sell"), "buy_toman": v.get("buy")}
            for k, v in d.items()]
    df = pd.DataFrame(rows).sort_values("date", kind="stable").reset_index(drop=True)
    df["sell"] = toman_to_rial(pd.to_numeric(df["sell_toman"], errors="coerce"))
    df["buy"] = toman_to_rial(pd.to_numeric(df["buy_toman"], errors="coerce"))
    return df


def _summary(body: bytes) -> http.Summary:
    df = parse(body)
    return http.Summary(len(df), str(df["date"].min()), str(df["date"].max()))


def fetch(refresh: bool = False, route: str | None = None) -> Path:
    return http.fetch_raw(SOURCE, FILE, URL, _summary, ua=http.SCRIPT_UA,
                          refresh=refresh, only_route=route)


def load() -> pd.DataFrame:
    return parse(http.raw_path(SOURCE, FILE).read_bytes())
