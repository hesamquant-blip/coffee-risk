"""ICO daily indicator prices, from the ICO's public monthly PDFs (D-020).

One PDF per month, one row per trading day, in US cents/lb:
    I-CIP, Colombian Milds, Other Milds, Brazilian Naturals, Robustas
followed by the rows Average, High and Low.

We read the Robustas column only. It is a PHYSICAL indicator price, not the
ICE futures price, so it is used for returns only, never for levels: it fills
the weekly robusta returns that TGJU's data holes leave empty.

File names on the ICO site are not uniform, so each month is tried under the
names and hosts the author gave, in order:
    https://ico.org/documents/prices/I-CIP_MM.YYYY.pdf
    https://ico.org/documents/prices/I-CIP_<Month>YYYY.pdf
    the same two under https://icocoffee.org/documents/prices/
A month that loads under none of them is skipped and listed.

Validation (per month, before any use): the mean, the max and the min of the
parsed Robustas rows must equal the printed Average / High / Low to 0.01, and
the month in the PDF's title must be the month asked for. A month that fails
is not used.

Exceptions. The author may accept one failing file by a recorded decision.
An exception names the decision, the exact file (sha256) and the rows to set
to missing; it does not change the rule, and it stops applying the moment the
file on the ICO site changes. See EXCEPTIONS.
"""
from __future__ import annotations

import calendar
import datetime as dt
import hashlib
import io
import re
from pathlib import Path

import pandas as pd

from coffee_risk.data import http

SOURCE = "ico"
HOSTS = ("https://ico.org/documents/prices/", "https://icocoffee.org/documents/prices/")
FIRST_MONTH, LAST_MONTH = (2025, 1), (2026, 9)        # the range asked for in the spec
TOLERANCE = 0.01
ROUTES = ("direct",)       # ico.org answers directly; 4 names per month is already enough requests

# Documented exceptions to the validation rule: (year, month) -> what was decided.
EXCEPTIONS = {
    # D-021 (2026-10-13). The 23 rows of July 2026 are read correctly (an
    # independent read by the author gives the same rows) and match the printed
    # High and Low, but their mean is 184.78 against a printed 184.87. The
    # mismatch is inside the PDF. It sits in the one row that repeats the
    # previous day's value, 17 July, which no needed Wednesday uses. The month
    # is accepted with that row set to missing.
    (2026, 7): {"decision": "D-021",
                "sha256": "7afbfc861d65e36010f7a56079ae392c51a4aa2385ff4803cc0591bbb5d6f6dd",
                "drop": (dt.date(2026, 7, 17),)},
}

_MONTHS = {m.lower(): i for i, m in enumerate(calendar.month_name) if m}
_ABBR = {m.lower(): i for i, m in enumerate(calendar.month_abbr) if m}
_NUM = r"(-?\d[\d,]*\.\d+|-?\d[\d,]*)"
_ROW = re.compile(rf"^(\d{{1,2}})-([A-Za-z]{{3}})\s+{_NUM}\s+{_NUM}\s+{_NUM}\s+{_NUM}\s+{_NUM}\s*$")
_STAT = re.compile(rf"^(Average|High|Low)\s+{_NUM}\s+{_NUM}\s+{_NUM}\s+{_NUM}\s+{_NUM}\s*$")
_TITLE = re.compile(r"ICO Indicator Prices\s*-\s*([A-Za-z]+)\s+(\d{4})")


def months() -> list[tuple[int, int]]:
    out, (y, m) = [], FIRST_MONTH
    while (y, m) <= LAST_MONTH:
        out.append((y, m))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def file_name(year: int, month: int) -> str:
    return f"I-CIP_{year}-{month:02d}.pdf"


def candidate_urls(year: int, month: int) -> list[str]:
    names = [f"I-CIP_{month:02d}.{year}.pdf", f"I-CIP_{calendar.month_name[month]}{year}.pdf"]
    return [host + n for host in HOSTS for n in names]


# -------------------------------------------------------------------- parsing
def _f(x: str) -> float:
    return float(x.replace(",", ""))


def parse_text(text: str) -> tuple[pd.DataFrame, dict]:
    """Text of one monthly PDF -> (daily rows, printed facts).

    rows: date, robustas (US cents/lb).
    facts: title_year, title_month, and the printed average / high / low of
    the Robustas column.
    """
    title = _TITLE.search(text)
    if not title or title.group(1).lower() not in _MONTHS:
        raise ValueError("no 'ICO Indicator Prices - <Month> <Year>' title found")
    year, month = int(title.group(2)), _MONTHS[title.group(1).lower()]
    facts = {"title_year": year, "title_month": month}
    rows = []
    for line in text.splitlines():
        line = line.strip()
        m = _ROW.match(line)
        if m:
            mon = _ABBR.get(m.group(2).lower())
            if mon is None:
                raise ValueError(f"unknown month in row: {line!r}")
            rows.append({"date": dt.date(year, mon, int(m.group(1))), "robustas": _f(m.group(7))})
            continue
        s = _STAT.match(line)
        if s:
            facts[s.group(1).lower()] = _f(s.group(6))
    if not rows:
        raise ValueError("no daily rows found")
    df = pd.DataFrame(rows).drop_duplicates().sort_values("date", kind="stable").reset_index(drop=True)
    return df, facts


def parse(body: bytes) -> tuple[pd.DataFrame, dict]:
    import pdfplumber
    with pdfplumber.open(io.BytesIO(body)) as pdf:
        text = "\n".join(page.extract_text() or "" for page in pdf.pages)
    return parse_text(text)


def validate(df: pd.DataFrame, facts: dict, year: int, month: int) -> dict:
    """Compare the parsed Robustas rows with the printed Average/High/Low."""
    out = {"rows": len(df), "mean": float(df["robustas"].mean()),
           "max": float(df["robustas"].max()), "min": float(df["robustas"].min()),
           "printed_average": facts.get("average"), "printed_high": facts.get("high"),
           "printed_low": facts.get("low"),
           "title": f"{facts['title_year']}-{facts['title_month']:02d}"}
    problems = []
    if (facts["title_year"], facts["title_month"]) != (year, month):
        problems.append(f"title says {out['title']}")
    if any(d.year != year or d.month != month for d in df["date"]):
        problems.append("a row is dated outside the month")
    if df["date"].duplicated().any():
        problems.append("a date appears twice")
    for label, got, key in (("average", out["mean"], "average"), ("high", out["max"], "high"),
                            ("low", out["min"], "low")):
        printed = facts.get(key)
        if printed is None:
            problems.append(f"no printed {label}")
        elif abs(got - printed) > TOLERANCE + 1e-9:
            problems.append(f"{label}: parsed {got:.4f} vs printed {printed:.2f}")
    out["ok"] = not problems
    out["problems"] = "; ".join(problems)
    return out


def exception_for(year: int, month: int, body: bytes) -> dict | None:
    """The recorded exception for this month, if it is for exactly this file."""
    exc = EXCEPTIONS.get((year, month))
    if exc and hashlib.sha256(body).hexdigest() == exc["sha256"]:
        return exc
    return None


def _summary(body: bytes) -> http.Summary:
    if not body.startswith(b"%PDF"):
        raise ValueError("not a PDF")
    df, _ = parse(body)
    return http.Summary(len(df), str(df["date"].min()), str(df["date"].max()))


# ------------------------------------------------------------------- fetching
def fetch_month(year: int, month: int, refresh: bool = False,
                route: str | None = None) -> Path:
    """Download one month, trying each candidate name. Raises FetchFailed if
    none loads; a month that failed before is not retried without --refresh."""
    file = file_name(year, month)
    path = http.raw_path(SOURCE, file)
    if path.exists() and not refresh:
        return path
    if not refresh and http.attempted(SOURCE, file):
        raise http.FetchFailed(f"{SOURCE}/{file}: did not load on an earlier run "
                               "(use --refresh to retry)")
    errors = []
    for url in candidate_urls(year, month):
        try:
            return http.fetch_raw(SOURCE, file, url, _summary, routes=ROUTES, ua=http.SCRIPT_UA,
                                  refresh=True, only_route=route)
        except http.FetchFailed as exc:
            errors.append(str(exc))
    raise http.FetchFailed(f"{SOURCE}/{file}: no candidate name loaded ({len(errors)} tried)")


def fetch(refresh: bool = False, route: str | None = None) -> dict:
    ok, missing = [], []
    for y, m in months():
        try:
            fetch_month(y, m, refresh, route)
            ok.append((y, m))
        except http.FetchFailed:
            missing.append((y, m))
    return {"ok": ok, "missing": missing}


# -------------------------------------------------------------------- loading
def report() -> pd.DataFrame:
    """One row per month asked for: loaded?, rows, parsed vs printed, `ok`
    (passes validation), the exception applied if any, and `used`."""
    rows = []
    for y, m in months():
        path = http.raw_path(SOURCE, file_name(y, m))
        row = {"year": y, "month": m, "file": file_name(y, m), "loaded": path.exists(),
               "ok": False, "exception": "", "rows_dropped": 0, "used": False}
        if path.exists():
            body = path.read_bytes()
            try:
                df, facts = parse(body)
                row.update(validate(df, facts, y, m))
            except Exception as exc:                     # unreadable PDF = failed month
                row.update({"ok": False, "problems": f"{type(exc).__name__}: {exc}"})
            else:
                exc = None if row["ok"] else exception_for(y, m, body)
                if exc:
                    row["exception"] = exc["decision"]
                    row["rows_dropped"] = int(df["date"].isin(exc["drop"]).sum())
                row["used"] = bool(row["ok"] or exc)
        rows.append(row)
    return pd.DataFrame(rows)


def load() -> pd.DataFrame:
    """Daily ICO Robustas (US cents/lb) from the months that pass validation,
    plus any month accepted by a recorded exception, without the rows that
    exception sets to missing."""
    frames = []
    for _, r in report().iterrows():
        if r["used"]:
            body = http.raw_path(SOURCE, r["file"]).read_bytes()
            df, _ = parse(body)
            exc = None if r["ok"] else exception_for(int(r["year"]), int(r["month"]), body)
            if exc:
                df = df[~df["date"].isin(exc["drop"])]
            frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["date", "robustas"])
    return pd.concat(frames).sort_values("date", kind="stable").reset_index(drop=True)


def available() -> bool:
    return any(http.raw_path(SOURCE, file_name(y, m)).exists() for y, m in months())
