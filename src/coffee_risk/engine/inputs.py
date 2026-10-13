"""Inputs of the engine and the FX-purchase strategies.

One dataclass holds everything a run needs. Every default is a PLACEHOLDER:
the three values taken from data (s0, market_rate, premium_share) are filled
by `Inputs.from_data`; the rest are the author's working assumptions (spec D3)
and are labelled as such in the report.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional, Union

import pandas as pd

WEEKS_PER_YEAR = 52
PREMIUM_MONTHS = 36           # months of futures / IMF ratios behind the premium default
IMF_LAG_MONTHS = 3            # the IMF month M is taken as known from month M + 3 on

Drift = Union[str, float]     # "neutral" | "historical" | "zero" | an annual simple rate


@dataclass(frozen=True)
class Inputs:
    # --- taken from data at the decision date (no placeholder possible) -----
    s0: float                         # free-market USD, rial: Bonbast sell at as_of
    market_rate: float                # annual; the fund's trailing-13-week return, annualised
    premium_share: float              # physical premium as a share of the invoice, fixed at signing

    # --- the purchase (D-034) ------------------------------------------------
    usd_amount: float = 100_000.0
    payment_week: int = 13
    channel_discount: float = 0.0     # official channel = free rate x (1 - discount), same dynamics
    contract: str = "floating"        # "floating": priced at payment; "fixed": priced now (D-032)
    arabica_share: float = 0.5        # D-003

    # --- rates and frictions -------------------------------------------------
    funds_rate: Optional[float] = None   # the firm's annual cost of rial funds, as a level;
                                         # None = market_rate + funds_spread (D-030)
    funds_spread: float = 0.0         # points over the market rate, e.g. 0.10 = market + 10 pp (D-039)
    fx_holding_cost: float = 0.0      # annual, of the rial value at purchase (D-031)
    sellback_spread: float = 0.01     # excess USD is sold at S_T x (1 - spread)

    # --- revenue side and the constraint (D-033, D-035) ----------------------
    planned_margin: float = 0.25
    margin_floor: float = 0.10
    confidence: float = 0.95
    pass_through: float = 0.0         # share of an unexpected FX move passed into selling prices
    pass_through_mode: str = "up_only"   # "up_only": prices rise with USD but do not fall (D-043)
                                         # "symmetric": they follow it both ways
    demand_sd: float = 0.0            # sd of a lognormal revenue factor with mean 1
    budget_factor: float = 1.10       # budget = factor x planned cost
    tie_tolerance: float = 0.0025     # mean costs closer than this share of the plan are a tie (D-037)

    # --- simulation choices (D-024..D-029) -----------------------------------
    drift: Drift = "neutral"          # D-028
    usd_vol: str = "fhs"              # "fhs" (filtered historical simulation) | "plain"
    half_life: float = 13.0           # weeks, of the EWMA volatility
    usd_vol_scale: float = 1.0        # multiplier of the simulated USD volatility (the calculator's slider)
    mean_block: float = 13.0          # weeks, stationary bootstrap

    def __post_init__(self):
        if self.contract not in ("floating", "fixed"):
            raise ValueError(f"contract must be 'floating' or 'fixed', not {self.contract!r}")
        if self.usd_vol not in ("fhs", "plain"):
            raise ValueError(f"usd_vol must be 'fhs' or 'plain', not {self.usd_vol!r}")
        if isinstance(self.drift, str) and self.drift not in ("neutral", "historical", "zero"):
            raise ValueError(f"unknown drift {self.drift!r}")
        if not 0.0 <= self.arabica_share <= 1.0:
            raise ValueError("arabica_share must be between 0 and 1")
        if self.pass_through_mode not in ("up_only", "symmetric"):
            raise ValueError(f"pass_through_mode must be 'up_only' or 'symmetric', "
                             f"not {self.pass_through_mode!r}")
        if self.funds_rate is not None and self.funds_spread != 0.0:
            raise ValueError("give funds_rate (a level) or funds_spread (over the market rate), not both")
        if self.tie_tolerance < 0:
            raise ValueError("tie_tolerance cannot be negative")
        if self.usd_vol_scale < 0:
            raise ValueError("usd_vol_scale cannot be negative")

    @property
    def cost_of_funds(self) -> float:
        if self.funds_rate is not None:
            return self.funds_rate
        return self.market_rate + self.funds_spread

    def with_(self, **changes) -> "Inputs":
        return replace(self, **changes)

    # ------------------------------------------------------------ from data
    @classmethod
    def from_data(cls, as_of, weekly: pd.DataFrame, daily: Optional[pd.DataFrame] = None,
                  imf: Optional[pd.DataFrame] = None, **overrides) -> "Inputs":
        """Inputs as of a decision date, with the three data-based defaults.
        Only data dated on or before `as_of` is used."""
        as_of = pd.Timestamp(as_of)
        w = float(overrides.get("arabica_share", cls.arabica_share))
        values = {"s0": spot(weekly, as_of), "market_rate": market_rate(weekly, as_of)}
        if "premium_share" not in overrides:
            if daily is None or imf is None:
                raise ValueError("premium_share needs the daily panel and the IMF series, "
                                 "or pass premium_share=...")
            values["premium_share"] = premium_share(daily, imf, as_of, w)
        values.update(overrides)
        return cls(**values)


# ------------------------------------------------------ data-based defaults
def spot(weekly: pd.DataFrame, as_of: pd.Timestamp) -> float:
    """Bonbast sell at the decision date, rial per USD."""
    return float(weekly.loc[as_of, "usd_free"])


def market_rate(weekly: pd.DataFrame, as_of: pd.Timestamp, weeks: int = 13) -> float:
    """The fund's return over the last `weeks` weeks, annualised (simple)."""
    level = weekly["fund_tr_index"]
    now, before = level.loc[as_of], level.loc[as_of - pd.Timedelta(weeks=weeks)]
    return float((now / before) ** (WEEKS_PER_YEAR / weeks) - 1.0)


def premium_by_coffee(daily: pd.DataFrame, imf: pd.DataFrame, as_of: pd.Timestamp,
                      months: int = PREMIUM_MONTHS, min_days: int = 15,
                      lag_months: int = IMF_LAG_MONTHS) -> dict:
    """1 - median(futures / IMF physical indicator) over the last `months`
    months known at `as_of`, for arabica and robusta.

    `imf` has columns month (first day), arabica_imf_clb, robusta_imf_clb.
    The monthly futures level is the mean of the daily series (USD/kg) in
    months with at least `min_days` values. The IMF publishes with a delay:
    a month counts as known `lag_months` months later, so that a past decision
    date never sees a figure that was not out yet."""
    from coffee_risk.units import cents_lb_to_usd_kg
    m = imf.copy()
    m.index = pd.to_datetime(m.pop("month"))
    start_of_month = as_of.normalize().replace(day=1) - pd.DateOffset(months=lag_months - 1)
    out = {}
    for name in ("arabica", "robusta"):
        d = daily.loc[:as_of, f"{name}_usd_kg"]
        g = d.groupby(d.index.to_period("M")).agg(["mean", "count"])
        g.index = g.index.to_timestamp()
        fut = g.loc[(g["count"] >= min_days) & (g.index < start_of_month), "mean"]
        phys = cents_lb_to_usd_kg(m[f"{name}_imf_clb"])
        ratio = (fut / phys.reindex(fut.index)).dropna().tail(months)
        out[name] = {"premium": float(1.0 - ratio.median()), "months": len(ratio),
                     "first": ratio.index.min(), "last": ratio.index.max()}
    return out


def premium_share(daily: pd.DataFrame, imf: pd.DataFrame, as_of: pd.Timestamp,
                  arabica_share: float) -> float:
    """Mix-weighted default of the physical premium share."""
    p = premium_by_coffee(daily, imf, as_of)
    return arabica_share * p["arabica"]["premium"] + (1.0 - arabica_share) * p["robusta"]["premium"]


# ------------------------------------------------------------------ strategies
@dataclass(frozen=True)
class Strategy:
    """When the USD is bought. `tranches` = (week, share of usd_amount) for
    purchases BEFORE the payment week; whatever is still missing at the
    payment week is bought then, and any excess is sold back."""
    name: str
    tranches: tuple[tuple[int, float], ...] = ()

    @property
    def share_before_payment(self) -> float:
        return float(sum(s for _, s in self.tranches))

    @property
    def share_at_week0(self) -> float:
        return float(sum(s for t, s in self.tranches if t == 0))


def now() -> Strategy:
    return Strategy("now", ((0, 1.0),))


def payment_day() -> Strategy:
    return Strategy("payment_day", ())


def staged(n: int = 3, payment_week: int = 13) -> Strategy:
    """n equal tranches at weeks round(k * T / n), k = 0..n-1 (0, 4, 9 for n = 3, T = 13)."""
    weeks = [int(round(k * payment_week / n)) for k in range(n)]
    return Strategy(f"staged_{n}", tuple((t, 1.0 / n) for t in weeks))


def partial(h: float) -> Strategy:
    """Share h at week 0 and the rest on the payment day."""
    name = f"partial_{int(round(h * 100)):d}"
    return Strategy(name, ((0, float(h)),) if h > 0 else ())


def partial_family(step: float = 0.10) -> list[Strategy]:
    n = int(round(1.0 / step))
    return [partial(k / n) for k in range(n + 1)]


def strategy_set(payment_week: int = 13) -> list[Strategy]:
    """The strategies the decision rule compares: now, staged, payment day and
    the partial family in 10% steps (0% and 100% are payment day and now)."""
    family = [s for s in partial_family() if 0.0 < s.share_at_week0 < 1.0]
    return [now(), staged(3, payment_week), payment_day()] + family


__all__ = ["Inputs", "Strategy", "now", "payment_day", "staged", "partial", "partial_family",
           "strategy_set", "spot", "market_rate", "premium_share", "premium_by_coffee",
           "WEEKS_PER_YEAR", "IMF_LAG_MONTHS"]
