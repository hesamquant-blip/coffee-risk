"""Run settings shared by the report scripts, so that every report quotes the
same base case and the same random draws."""

REPORT_PATHS = 20_000         # the base case and its sensitivities (D3)
REPORT_SEED = 20261013
BACKTEST_PATHS = 2_000        # per decision date in the calibration preview and the backtest
EXPORT_PATHS = 20_000         # the calculator's path file: the same sample as the reports (D-052)
