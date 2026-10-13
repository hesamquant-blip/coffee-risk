"""The calculator kit: the files the web calculator is built from (D-014).

    calculator_paths.json   20,000 sampled paths x 13 weeks (D-052), as the engine
                            samples them and before any drift or volatility
                            is applied, so that the page can apply the
                            user's own:
                              usd_z     the sampled standardised USD residuals
                                        (zero mean over the pool)
                              arabica   the roll-excluded weekly log returns
                              robusta   of the same sampled weeks
                              demand_z  standard-normal draws, one per path
                            plus s0, the market rate, the current weekly EWMA
                            volatility, the premium of each coffee, the
                            past-only mean weekly USD return, and the settings.
    scenarios.json          S1..S7 as cumulative changes at the staged weeks
                            and the payment week. Aggregates only.
    parity.json             input cases with the results the page must
                            reproduce (engine.calculator on these same files).

No file holds a price series or the dates of the sampled weeks; the scenario
file names its windows and nothing else. Everything is written under
data/export/, which is never committed.
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from coffee_risk import paths as P
from coffee_risk.engine import calculator as calc
from coffee_risk.engine import cost as costs
from coffee_risk.engine import simulator as sim
from coffee_risk.engine import stress
from coffee_risk.engine.inputs import Inputs, premium_by_coffee, staged

EXPORT_DIR = P.DATA / "export"                            # under data/: never committed
SCALE = 1e-5                                              # arrays are stored as integers x 1e-5
SCENARIO_KEYS = {"fx_1397": "S1", "fx_1401_1402": "S2", "coffee_2021": "S3", "coffee_2024": "S4",
                 "war": "S5", "war_full": "S6", "perfect_storm": "S7"}
SCENARIO_LABELS = {"S1": "FX jump of 1397", "S2": "FX jump of 1401-1402", "S3": "coffee rally of 2021",
                   "S4": "coffee rally of 2024", "S5": "war, worst 13 weeks", "S6": "war, full path",
                   "S7": "perfect storm (hypothetical): USD of S1 with coffee of S4"}

# The twelve parity cases of spec D6-prep: name -> changes to the base inputs.
CASES = {
    "base": {},
    "fixed": {"contract": "fixed"},
    "funds_plus_10pp": {"funds_spread": 0.10},
    "fx_holding_3pct": {"fx_holding_cost": 0.03},
    "drift_zero": {"drift": "zero"},
    "drift_historical": {"drift": "historical"},
    "arabica_share_0": {"arabica_share": 0.0},
    "arabica_share_1": {"arabica_share": 1.0},
    "pass_through_0.5_up_only": {"pass_through": 0.5, "pass_through_mode": "up_only"},
    "demand_sd_10pct": {"demand_sd": 0.10},
    "usd_vol_x0.6": {"usd_vol_scale": 0.6},
    "margin_floor_5pct": {"margin_floor": 0.05},
}


def _ints(a: np.ndarray) -> list:
    return np.rint(np.asarray(a, dtype=float) / SCALE).astype(np.int64).ravel().tolist()


def paths_kit(as_of, n: int = 20000, seed: int = 0, horizon: int = 13, options: Optional[Inputs] = None,
              weekly: Optional[pd.DataFrame] = None, daily: Optional[pd.DataFrame] = None,
              imf: Optional[pd.DataFrame] = None) -> dict:
    """The content of calculator_paths.json as of a decision date. The sampled
    weeks are the ones `simulate(as_of, n, horizon, seed, options)` uses, so
    the page and the engine see the same paths (up to the rounding of the file)."""
    as_of = pd.Timestamp(as_of)
    if weekly is None or daily is None or imf is None:
        from coffee_risk.data import fred, panel, weekly as weekly_panel
        weekly = weekly_panel.load() if weekly is None else weekly
        daily = panel.load() if daily is None else daily
        imf = fred.load() if imf is None else imf
    if options is None:
        options = Inputs.from_data(as_of, weekly, daily, imf)
    if options.usd_vol != "fhs":
        raise ValueError("the calculator file is built from the filtered simulation (D-049)")
    prep = sim.prepare(weekly, as_of, options.with_(usd_vol_scale=1.0))
    rng = np.random.default_rng(seed)                       # the same first draws as simulate()
    idx = sim.stationary_indices(rng, len(prep["weeks"]), n, horizon, options.mean_block)
    prem = premium_by_coffee(daily, imf, as_of)
    return {
        "format": "coffee-risk calculator paths, version 2",
        "as_of": str(as_of.date()), "n_paths": n, "horizon": horizon, "seed": int(seed),
        "layout": "each array: integers x scale, path by path (n_paths rows of horizon weeks); "
                  "demand_z: one value per path",
        "scale": SCALE,
        "s0": float(options.s0), "market_rate": float(options.market_rate),
        "usd_vol_weekly": float(prep["info"]["sigma_as_of"]),
        "usd_vol_method": "fhs", "half_life": float(options.half_life), "mean_block": float(options.mean_block),
        "hist_mean_weekly_usd_return": float(np.expm1(prep["usd_raw"]).mean()),
        "premium": {"arabica": float(prem["arabica"]["premium"]), "robusta": float(prem["robusta"]["premium"])},
        "usd_z": _ints(prep["usd_z"][idx]), "arabica": _ints(prep["arabica"][idx]),
        "robusta": _ints(prep["robusta"][idx]), "demand_z": _ints(costs.demand_normals(n, seed)),
    }


def scenarios_kit(weekly: pd.DataFrame) -> dict:
    """The content of scenarios.json: for S1..S7 the cumulative simple changes
    of USD, arabica and robusta at the staged weeks and at the payment week."""
    out = []
    for sc in stress.scenarios(weekly):
        T = sc.horizon
        weeks = [w for w, _ in staged(3, T).tranches if w > 0] + [T]
        level = {n: np.concatenate([[1.0], np.exp(np.cumsum(sc.returns[n].values.astype(float)))])
                 for n in stress.SERIES}
        key = SCENARIO_KEYS[sc.key]
        row = {"key": key, "id": sc.key, "label": SCENARIO_LABELS[key], "hypothetical": bool(sc.hypothetical),
               "payment_week": T, "weeks": weeks,
               "usd_window": [str(pd.Timestamp(d).date()) for d in sc.usd_window],
               "coffee_window": [str(pd.Timestamp(d).date()) for d in sc.coffee_window]}
        row.update({n: [round(float(level[n][w] - 1.0), 6) for w in weeks] for n in stress.SERIES})
        out.append(row)
    return {"format": "coffee-risk scenarios, version 1",
            "note": "cumulative simple changes from the first week of the window, at the staged weeks and "
                    "at the payment week; applied to today's levels (replay from today)",
            "scenarios": out}


def input_record(inputs: Inputs, kit: dict) -> dict:
    """Every input of a case, spelled out for the page."""
    return {"usd_amount": inputs.usd_amount, "payment_week": inputs.payment_week, "s0": inputs.s0,
            "market_rate": inputs.market_rate, "channel_discount": inputs.channel_discount,
            "contract": inputs.contract, "arabica_share": inputs.arabica_share,
            "premium_arabica": kit["premium"]["arabica"], "premium_robusta": kit["premium"]["robusta"],
            "premium_share": inputs.premium_share, "funds_rate": inputs.cost_of_funds,
            "fx_holding_cost": inputs.fx_holding_cost, "sellback_spread": inputs.sellback_spread,
            "planned_margin": inputs.planned_margin, "margin_floor": inputs.margin_floor,
            "confidence": inputs.confidence, "pass_through": inputs.pass_through,
            "pass_through_mode": inputs.pass_through_mode, "demand_sd": inputs.demand_sd,
            "budget_factor": inputs.budget_factor, "tie_tolerance": inputs.tie_tolerance,
            "drift": inputs.drift, "usd_vol_scale": inputs.usd_vol_scale}


def parity_kit(kit: dict, scenarios: dict, cases: Optional[dict] = None) -> dict:
    """The content of parity.json: each case's inputs and the results of
    `calculator.evaluate` on exactly the two files given."""
    cases = CASES if cases is None else cases
    out = []
    for name, changes in cases.items():
        inputs = calc.kit_inputs(kit, **changes)
        out.append({"name": name, "changes": changes, "inputs": input_record(inputs, kit),
                    "outputs": calc.evaluate(kit, inputs, scenarios["scenarios"])})
    return {"format": "coffee-risk parity, version 1",
            "note": "costs (mean, median, var95, cvar95) are shares of the planned cost of the case; "
                    "probabilities are shares of the paths; null = does not exist",
            "computed_on": {"as_of": kit["as_of"], "n_paths": kit["n_paths"], "seed": kit["seed"]},
            "cases": out}


def write_json(doc: dict, path: Path) -> dict:
    """Write compact JSON; return its size, plain and gzipped."""
    body = json.dumps(doc, separators=(",", ":"), allow_nan=False, ensure_ascii=True).encode("ascii")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)
    return {"path": path, "bytes": len(body), "gzip_bytes": len(gzip.compress(body, 9))}


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="ascii"))
