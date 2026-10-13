"""The engine: simulate joint paths, price FX-purchase strategies, measure
risk, apply the decision rule.

    inputs     Inputs (every assumption in one place) and the strategies
    simulator  simulate(as_of, n_paths, ...) -> Paths
    cost       cost_on_paths, revenue, margin
    decision   metrics, evaluate, decide, minimum hedge ratio, run
    export     the calculator kit: sampled paths, scenarios, parity cases
    calculator the calculator's arithmetic on the kit's own files
    stress     replay of historical windows, reverse stress, stress floor
    backtest   the rolling backtest on realised paths
    calibration  how often realised USD moves fell outside the simulated range

`cost` and `decision` take any Paths: bootstrap paths, one stress path or one
realised historical path, so the stress runs and the backtest reuse them.
"""
from coffee_risk.engine.inputs import (Inputs, Strategy, now, partial, partial_family,
                                       payment_day, staged, strategy_set)
from coffee_risk.engine.simulator import Paths, simulate
from coffee_risk.engine.cost import cost_on_paths, margin, planned_cost, revenue
from coffee_risk.engine.decision import (Decision, Result, decide, evaluate, hedge_curve,
                                         metrics, minimum_hedge_ratio, run)

__all__ = ["Inputs", "Strategy", "now", "partial", "partial_family", "payment_day", "staged",
           "strategy_set", "Paths", "simulate", "cost_on_paths", "margin", "planned_cost",
           "revenue", "Decision", "Result", "decide", "evaluate", "hedge_curve", "metrics",
           "minimum_hedge_ratio", "run"]
