"""Polite HTTP fetching with an unchanged raw cache and a CSV manifest.

Rules (spec D1):
  * few requests, a pause between them, a normal User-Agent;
  * the raw response body is saved unchanged; a cached file is reused unless
    --refresh is given, so every fetcher is idempotent;
  * every network attempt (success or failure, with the exact error) is
    appended to data/raw/FETCH_LOG.csv, and every saved file has one row in
    data/raw/MANIFEST.csv.

Routes. The shell may export HTTP(S)_PROXY variables. "direct" ignores
those variables, "proxy" uses them. A fetcher lists the routes it is willing
to try, in order; if one fails the error is logged and the next is tried, so
a single fetcher can also be re-run on another route.
"""
from __future__ import annotations

import contextlib
import csv
import datetime as dt
import hashlib
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional

import requests

from coffee_risk import paths

BROWSER_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
# FRED (Akamai) drops requests that claim to be a browser but are not one; it
# answers a plain, honest script User-Agent.
SCRIPT_UA = "coffee-risk/0.1 (personal research script; python-requests)"

PAUSE_SECONDS = 2.0
TIMEOUT_SECONDS = 45
_PROXY_VARS = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
               "http_proxy", "https_proxy", "all_proxy")

MANIFEST_COLS = ["source", "file", "endpoint", "route", "fetched_at_utc",
                 "bytes", "rows", "first_date", "last_date", "sha256"]
LOG_COLS = ["at_utc", "source", "file", "endpoint", "route", "outcome", "detail"]

_last_request_at = 0.0


class FetchFailed(RuntimeError):
    """Every route failed for one raw file."""


@dataclass
class Summary:
    """What the manifest records about a raw file's content."""
    rows: Optional[int] = None
    first_date: Optional[str] = None
    last_date: Optional[str] = None


def _utcnow() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def pause() -> None:
    """Keep at least PAUSE_SECONDS between any two requests of this process."""
    global _last_request_at
    wait = _last_request_at + PAUSE_SECONDS - time.monotonic()
    if wait > 0:
        time.sleep(wait)
    _last_request_at = time.monotonic()


@contextlib.contextmanager
def proxy_env(route: str):
    """For libraries that read the proxy variables themselves (yfinance):
    hide them on the direct route, leave them on the proxy route."""
    saved = {v: os.environ.get(v) for v in _PROXY_VARS}
    try:
        if route == "direct":
            for v in _PROXY_VARS:
                os.environ.pop(v, None)
        yield
    finally:
        for v, val in saved.items():
            if val is None:
                os.environ.pop(v, None)
            else:
                os.environ[v] = val


def proxy_configured() -> bool:
    return any(os.environ.get(v) for v in _PROXY_VARS)


def get(url: str, route: str = "direct", ua: str = BROWSER_UA,
        timeout: int = TIMEOUT_SECONDS) -> requests.Response:
    """One GET on one route. Raises on network errors and on non-200."""
    if route not in ("direct", "proxy"):
        raise ValueError(f"unknown route {route!r}")
    if route == "proxy" and not proxy_configured():
        raise RuntimeError("no proxy variables set in the environment")
    pause()
    with requests.Session() as s:
        s.trust_env = route == "proxy"
        s.headers.update({"User-Agent": ua, "Accept": "*/*",
                          "Accept-Language": "en-US,en;q=0.9,fa;q=0.8"})
        r = s.get(url, timeout=timeout)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code} ({len(r.content)} bytes): "
                           f"{r.text[:120]!r}")
    if not r.content.strip():
        raise RuntimeError("HTTP 200 with an empty body")
    return r


# ----------------------------------------------------------------- bookkeeping
def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _write_csv(path: Path, cols: list[str], rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})
    tmp.replace(path)


def log_attempt(source: str, file: str, endpoint: str, route: str,
                outcome: str, detail: str = "") -> None:
    rows = _read_csv(paths.FETCH_LOG)
    rows.append({"at_utc": _utcnow(), "source": source, "file": file,
                 "endpoint": endpoint, "route": route, "outcome": outcome,
                 "detail": detail.replace("\n", " ")[:500]})
    _write_csv(paths.FETCH_LOG, LOG_COLS, rows)


def record_manifest(source: str, file: str, endpoint: str, route: str,
                    body: bytes, summary: Summary,
                    fetched_at: Optional[str] = None) -> None:
    rows = [r for r in _read_csv(paths.MANIFEST)
            if not (r["source"] == source and r["file"] == file)]
    rows.append({
        "source": source, "file": file, "endpoint": endpoint, "route": route,
        "fetched_at_utc": fetched_at or _utcnow(), "bytes": len(body),
        "rows": "" if summary.rows is None else summary.rows,
        "first_date": summary.first_date or "", "last_date": summary.last_date or "",
        "sha256": hashlib.sha256(body).hexdigest(),
    })
    rows.sort(key=lambda r: (r["source"], r["file"]))
    _write_csv(paths.MANIFEST, MANIFEST_COLS, rows)


def raw_path(source: str, file: str) -> Path:
    return paths.RAW / source / file


def save_raw(source: str, file: str, endpoint: str, route: str, body: bytes,
             summarize: Callable[[bytes], Summary]) -> Path:
    """Write a raw body unchanged and record it. `summarize` also validates:
    if it raises, nothing is written and an existing cache stays in place."""
    summary = summarize(body)
    path = raw_path(source, file)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(body)
    tmp.replace(path)
    record_manifest(source, file, endpoint, route, body, summary)
    return path


def fetch_raw(source: str, file: str, url: str,
              summarize: Callable[[bytes], Summary],
              routes: tuple[str, ...] = ("direct", "proxy"),
              ua: str = BROWSER_UA, refresh: bool = False,
              only_route: Optional[str] = None) -> Path:
    """Return the cached raw file, fetching it first if needed."""
    path = raw_path(source, file)
    if path.exists() and not refresh:
        return path
    if only_route:
        routes = (only_route,)
    errors = []
    for route in routes:
        try:
            body = get(url, route=route, ua=ua).content
            out = save_raw(source, file, url, route, body, summarize)
        except Exception as exc:                      # exact error goes to the log
            msg = f"{type(exc).__name__}: {exc}"
            errors.append(f"[{route}] {msg}")
            log_attempt(source, file, url, route, "error", msg)
            continue
        log_attempt(source, file, url, route, "ok", f"{len(body)} bytes")
        return out
    raise FetchFailed(f"{source}/{file}: " + " | ".join(errors))


def attempted(source: str, file: str) -> bool:
    """Has this raw file been tried before (successfully or not)?"""
    return any(r["source"] == source and r["file"] == file
               for r in _read_csv(paths.FETCH_LOG))


def failures() -> list[dict]:
    """Logged failed attempts for files that are still not in the cache."""
    return [r for r in _read_csv(paths.FETCH_LOG)
            if r["outcome"] == "error" and not raw_path(r["source"], r["file"]).exists()]

