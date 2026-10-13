"""IMF monthly coffee indicators from FRED (CSV, no key).

PCOFFOTMUSDM  Other Mild Arabicas, US cents per pound, monthly average
PCOFFROBUSDM  Robustas,            US cents per pound, monthly average
"""
from __future__ import annotations

import io
from pathlib import Path

import pandas as pd

from coffee_risk.data import http

SOURCE = "fred"
URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
SERIES = {"PCOFFOTMUSDM": "arabica", "PCOFFROBUSDM": "robusta"}


def parse(body: bytes) -> pd.DataFrame:
    """-> month (first day of month, date), value (US cents/lb)."""
    df = pd.read_csv(io.BytesIO(body))
    if df.shape[1] != 2:
        raise ValueError(f"unexpected FRED layout: {list(df.columns)}")
    df.columns = ["month", "value"]
    df["month"] = pd.to_datetime(df["month"]).dt.date
    df["value"] = pd.to_numeric(df["value"], errors="coerce")   # "." = missing
    return df


def _summary(body: bytes) -> http.Summary:
    df = parse(body).dropna(subset=["value"])
    return http.Summary(len(df), str(df["month"].min()), str(df["month"].max()))


def fetch(refresh: bool = False, route: str | None = None) -> list[Path]:
    return [http.fetch_raw(SOURCE, f"{s}.csv", URL.format(series=s), _summary,
                           ua=http.SCRIPT_UA, refresh=refresh, only_route=route)
            for s in SERIES]


def load() -> pd.DataFrame:
    """-> month, arabica_imf_clb, robusta_imf_clb."""
    out = None
    for s, name in SERIES.items():
        df = parse(http.raw_path(SOURCE, f"{s}.csv").read_bytes())
        df = df.rename(columns={"value": f"{name}_imf_clb"})
        out = df if out is None else out.merge(df, on="month", how="outer")
    return out.sort_values("month").reset_index(drop=True)
