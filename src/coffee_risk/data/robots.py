"""robots.txt of every host this project talks to (spec: read and report).

Each file is saved unchanged under data/raw/robots/. `check()` then asks
Python's standard robots parser whether the path we request is allowed for a
generic client ("*"). A host with no robots.txt states no restriction.
"""
from __future__ import annotations

import urllib.robotparser

import pandas as pd

from coffee_risk.data import http

SOURCE = "robots"
# host -> (what we use it for, one path we request or would request)
HOSTS = {
    "platform.tgju.org": ("TGJU chart endpoint", "/fa/tvdata/history?symbol=PRICE_DOLLAR_RL"),
    "api.tgju.org": ("TGJU history-table endpoint",
                     "/v1/market/indicator/summary-table-data/price_dollar_rl"),
    "www.tgju.org": ("TGJU history page (read by eye)", "/profile/price_dollar_rl/history"),
    "raw.githubusercontent.com": ("Bonbast archive",
                                  "/SamadiPour/rial-exchange-rates-archive/data/currency/usd.json"),
    "query1.finance.yahoo.com": ("Yahoo Finance (via yfinance)", "/v8/finance/chart/KC=F"),
    "fred.stlouisfed.org": ("FRED CSV", "/graph/fredgraph.csv?id=PCOFFROBUSDM"),
    "cdn.tsetmc.com": ("TSETMC daily history",
                       "/api/ClosingPrice/GetClosingPriceDailyList/66818022341772870/0"),
    "service.tsetmc.com": ("TSETMC instrument list", "/tsev2/data/TseClient2.aspx?t=Instrument&a=0"),
    "ico.org": ("ICO indicator-price PDFs", "/documents/prices/I-CIP_08.2026.pdf"),
    "stooq.com": ("Stooq (fallback, not used)", "/q/d/l/?s=kc.f&i=d"),
    "www.investing.com": ("Investing.com (manual download only)",
                          "/commodities/london-coffee-historical-data"),
}


def _scheme(host: str) -> str:
    return "http" if host == "service.tsetmc.com" else "https"


def _summary(body: bytes) -> http.Summary:
    text = body.decode("utf-8", errors="replace")
    if "<html" in text.lower()[:2000]:
        raise ValueError("an HTML page was served instead of robots.txt")
    return http.Summary(rows=len(text.splitlines()))


def fetch(refresh: bool = False, route: str | None = None) -> dict:
    out = {}
    for host in HOSTS:
        url = f"{_scheme(host)}://{host}/robots.txt"
        file = f"{host}.txt"
        if (not http.raw_path(SOURCE, file).exists() and not refresh
                and http.attempted(SOURCE, file)):
            out[host] = "not served on an earlier run (use --refresh to retry)"
            continue
        errors = []
        # an honest script User-Agent first; some hosts only answer a browser one
        for ua in (http.SCRIPT_UA, http.BROWSER_UA):
            try:
                http.fetch_raw(SOURCE, f"{host}.txt", url, _summary, ua=ua,
                               refresh=refresh, only_route=route)
                errors = []
                break
            except http.FetchFailed as exc:
                errors.append(str(exc))
        out[host] = "ok" if not errors else errors[-1]
    return out


def check() -> pd.DataFrame:
    """One row per host: is there a robots.txt, and does it allow our path?"""
    rows = []
    for host, (use, path) in HOSTS.items():
        p = http.raw_path(SOURCE, f"{host}.txt")
        if not p.exists():
            rows.append({"host": host, "use": use, "robots": "none served", "allowed": None})
            continue
        rp = urllib.robotparser.RobotFileParser()
        rp.parse(p.read_text(encoding="utf-8", errors="replace").splitlines())
        rows.append({"host": host, "use": use, "robots": "present",
                     "allowed": rp.can_fetch("*", f"{_scheme(host)}://{host}{path}")})
    return pd.DataFrame(rows)
