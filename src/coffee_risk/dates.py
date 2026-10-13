"""Tehran calendar dates and Jalali conversion.

Iran observed daylight saving time until 2022, so a unix timestamp is turned
into a Tehran calendar date through the tz database, never with a fixed
+03:30 offset.
"""
from __future__ import annotations

import datetime as dt
from zoneinfo import ZoneInfo

import jdatetime

TEHRAN = ZoneInfo("Asia/Tehran")


def unix_to_tehran_date(ts: int | float) -> dt.date:
    """Unix timestamp (seconds, UTC) -> calendar date in Tehran."""
    return dt.datetime.fromtimestamp(ts, tz=TEHRAN).date()


def to_jalali(d: dt.date) -> str:
    """Gregorian date -> 'YYYY-MM-DD' Jalali string."""
    j = jdatetime.date.fromgregorian(date=d)
    return f"{j.year:04d}-{j.month:02d}-{j.day:02d}"


def jalali_year(d: dt.date) -> int:
    return jdatetime.date.fromgregorian(date=d).year


def jalali_month(d: dt.date) -> int:
    return jdatetime.date.fromgregorian(date=d).month


def from_jalali(year: int, month: int, day: int) -> dt.date:
    return jdatetime.date(year, month, day).togregorian()


def parse_jalali(s: str) -> dt.date:
    """'1391/07/18' or '1391-07-18' -> Gregorian date."""
    y, m, d = (int(x) for x in s.strip().replace("/", "-").split("-"))
    return from_jalali(y, m, d)
