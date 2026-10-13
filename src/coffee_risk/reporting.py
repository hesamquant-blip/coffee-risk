"""Small helpers shared by the report scripts: Persian number and date
formatting, Markdown tables, and one figure style.

Stage reports are in Persian (project rule 2). Numbers are written with
Persian digits, a thousands separator and the sign after the number, which is
how they read correctly in right-to-left text. Gregorian dates in parentheses
keep Latin digits so the two calendars cannot be confused.
"""
from __future__ import annotations

import datetime as dt

import numpy as np

from coffee_risk.dates import to_jalali

_FA = str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹")


def fa(x) -> str:
    """Latin digits -> Persian digits."""
    return str(x).translate(_FA)


def _nan(x) -> bool:
    if x is None:
        return True
    try:
        return bool(np.isnan(x))
    except TypeError:
        return False


def num(x, d: int = 0) -> str:
    if _nan(x):
        return "—"
    x = round(float(x), d) + 0.0                      # no "minus zero" after rounding
    return fa(f"{abs(x):,.{d}f}".replace(",", "٬")) + ("−" if x < 0 else "")


def pct(x, d: int = 2, sign: bool = False) -> str:
    """A fraction as a percentage: 0.0123 -> ٪۱.۲۳ (sign written after it)."""
    if _nan(x):
        return "—"
    v = round(float(x) * 100, d) + 0.0
    tail = "−" if v < 0 else ("+" if sign and v > 0 else "")
    return "٪" + fa(f"{abs(v):,.{d}f}".replace(",", "٬")) + tail


def billions(x, d: int = 1) -> str:
    """Rials as billions of rials."""
    return num(float(x) / 1e9, d)


def strategy_fa(name) -> str:
    """Persian name of an engine strategy (now, staged_n, payment_day, partial_NN)."""
    if name is None:
        return "هیچ"
    if name == "now":
        return "همه امروز"
    if name == "payment_day":
        return "همه روز پرداخت"
    if name.startswith("staged"):
        return "سه‌پله‌ای"
    return f"{pct(int(name.split('_')[1]) / 100, 0)} امروز"


def as_date(d) -> dt.date:
    return d.date() if hasattr(d, "date") else d


def jd(d) -> str:
    """Jalali date in Persian digits."""
    return fa(to_jalali(as_date(d)))


def jg(d) -> str:
    """Jalali date with the Gregorian date in parentheses."""
    return f"{jd(d)} ({as_date(d)})"


def table(headers: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


# --------------------------------------------------------------------- figures
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"   # validated categorical slots 1-3
RED, NEUTRAL = "#e34948", "#f0efec"                    # diverging pair: blue <-> red around a gray midpoint
YELLOW = "#eda100"                                     # categorical slot 4


def style(plt) -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "text.color": INK,
        "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID,
        "axes.axisbelow": True,
        "grid.linewidth": 0.8, "grid.linestyle": "-", "axes.spines.top": False,
        "axes.spines.right": False, "axes.titlesize": 11, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "font.size": 9, "legend.frameon": False,
        "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
    })


def end_label(ax, x, y, text) -> None:
    ax.annotate(text, (x, y), xytext=(5, 0), textcoords="offset points",
                va="center", fontsize=8, color=INK2)
