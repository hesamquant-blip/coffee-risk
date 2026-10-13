"""Command line for the data layer.

    python -m coffee_risk.data fetch <source> [--refresh] [--route direct|proxy]
    python -m coffee_risk.data build-panel      (daily, official and weekly panels)

Sources: tgju, bonbast, arabica, robusta, ico, fred, fund, official, robots, all.
A failed source prints its exact error and does not stop the others; the
error is also appended to data/raw/FETCH_LOG.csv.
"""
from __future__ import annotations

import argparse
import sys

from coffee_risk.data import bonbast, fred, http, ico, robots, tgju, tsetmc, yahoo


def _tgju(refresh, route):
    """TGJU free-market USD/IRR (the check series since D1b): the
    history-table endpoint, plus the history page to check the table."""
    for kind, fn in (("table", tgju.fetch_table), ("page", tgju.fetch_page)):
        p = fn(tgju.USD_KEY, refresh, route)
        print(f"  {kind}: {p.name}")


def _bonbast(refresh, route):
    print(f"  {bonbast.fetch(refresh, route).name}")


def _arabica(refresh, route):
    r = yahoo.fetch(refresh, route)
    print(f"  continuous: {r['continuous'].name}")
    print(f"  contracts served: {sorted(r['contracts'])}")
    for tk, err in r["contract_errors"].items():
        print(f"  contract not served: {tk}: {err}")


def _robusta(refresh, route):
    """Robusta: TGJU's London-coffee key."""
    print(f"  table: {tgju.fetch_table(tgju.ROBUSTA_KEY, refresh, route).name}")
    # Other free automatic sources (spec D1-D.2), each tried once and logged:
    # Yahoo has no ICE London robusta ticker, and Stooq (rm.f) answers scripts
    # with a JavaScript browser check that we do not bypass.
    for label, file, source, fn in (
            ("Yahoo RC=F", "RC_F.csv", yahoo.SOURCE, lambda: yahoo.fetch_ticker("RC=F", refresh, route)),
            ("Yahoo RM=F", "RM_F.csv", yahoo.SOURCE, lambda: yahoo.fetch_ticker("RM=F", refresh, route)),
            ("Stooq rm.f", "rm_f.csv", "stooq", lambda: yahoo.fetch_stooq("rm.f", refresh, route))):
        if http.attempted(source, file) and not refresh:
            continue
        try:
            fn()
            print(f"  alternative source answered: {label}")
        except http.FetchFailed as exc:
            print(f"  alternative source failed: {label}: {exc}")


def _ico(refresh, route):
    """ICO daily indicator prices (Robustas), monthly PDFs: fills robusta gaps (D-020)."""
    r = ico.fetch(refresh, route)
    print(f"  months loaded: {len(r['ok'])} ({r['ok'][0][0]}-{r['ok'][0][1]:02d} .. "
          f"{r['ok'][-1][0]}-{r['ok'][-1][1]:02d})" if r["ok"] else "  months loaded: 0")
    if r["missing"]:
        print("  months that did not load: " + ", ".join(f"{y}-{m:02d}" for y, m in r["missing"]))
    rep = ico.report()
    for _, x in rep[rep["loaded"] & ~rep["ok"]].iterrows():
        status = (f"accepted by {x['exception']}, {x['rows_dropped']} row(s) set to missing"
                  if x["used"] else "NOT used")
        print(f"  failed validation {x['file']}: {x['problems']} -> {status}")


def _fred(refresh, route):
    for p in fred.fetch(refresh, route):
        print(f"  {p.name}")


def _fund(refresh, route):
    r = tsetmc.fetch(refresh, route)
    print(f"  fixed-income funds listed: {r['funds']}, histories ok: {r['ok']}")
    for ins, err in r["failed"].items():
        print(f"  failed {ins}: {err}")


def _official(refresh, route):
    for key in tgju.OFFICIAL_KEYS:
        try:
            print(f"  {tgju.fetch_table(key, refresh, route).name}")
        except http.FetchFailed as exc:
            print(f"  FAILED {key}: {exc}")


def _robots(refresh, route):
    for host, res in robots.fetch(refresh, route).items():
        print(f"  {host}: {res}")


SOURCES = {"tgju": _tgju, "bonbast": _bonbast, "arabica": _arabica,
           "robusta": _robusta, "ico": _ico, "fred": _fred, "fund": _fund,
           "official": _official, "robots": _robots}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m coffee_risk.data")
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch", help="fetch one source (or all)")
    f.add_argument("source", choices=[*SOURCES, "all"])
    f.add_argument("--refresh", action="store_true",
                   help="fetch again even if a raw file is cached")
    f.add_argument("--route", choices=["direct", "proxy"], default=None,
                   help="force one network route (default: direct, then proxy)")
    sub.add_parser("build-panel", help="build the daily and weekly panels in data/processed/")
    args = ap.parse_args(argv)

    if args.cmd == "build-panel":
        from coffee_risk.data import panel, weekly
        df = panel.build()
        print(f"daily panel: {len(df)} rows, {df.index.min().date()} .. {df.index.max().date()}, "
              f"{df.shape[1]} columns")
        wk = weekly.load()
        print(f"weekly panel: {len(wk)} Wednesdays, {wk.index.min().date()} .. "
              f"{wk.index.max().date()}, {wk.shape[1]} columns")
        return 0

    names = list(SOURCES) if args.source == "all" else [args.source]
    failed = []
    for name in names:
        print(f"[{name}]")
        try:
            SOURCES[name](args.refresh, args.route)
        except Exception as exc:                 # record, report, move on
            failed.append(name)
            print(f"  FAILED: {type(exc).__name__}: {exc}")
    if failed:
        print(f"failed sources: {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
