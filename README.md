<div dir="rtl">

**ریسک ارز و قیمت قهوه برای یک واردکننده** — نمونه‌کار برای پرسش آگهی «تحلیلگر کمّی و ریسک بازار» آرگن کافی.

**پرسش:** شرکت باید سه ماه دیگر ۱۰۰٬۰۰۰ دلار برای یک محمولهٔ قهوه بپردازد، ولی درآمدش ریالی است. دلار را کِی و چقدر بخرد؟

**پاسخ:** با فرض‌های پایه، قیمت قهوه را همین امروز با فروشنده ثابت کنید و دست‌کم ۸۴٪ دلار را همین امروز بخرید.

- [یادداشت مدیریتی](https://claude.ai/artifact/DiUPTYGQP4S8kQYR9t69gE)
- [ماشین‌حساب](https://claude.ai/artifact/9RL7fyU573a7rNwFvT4ko5)
- [گزارش‌های مرحله‌به‌مرحله](reports/)

</div>

# coffee-risk

FX and coffee price risk for a rial-revenue coffee importer that must pay for
a USD 100,000 shipment in three months.

**Status:** all stages are done: D0 (repo), D1 (data), D1b (data review), D2
(data understanding), D3 (the engine), D4 (stress scenarios), D5 (backtest)
and D6-prep (the calculator kit). The calculator page and the memo exist;
their links are at the top.

- D1 report: [reports/d1_data/README.md](reports/d1_data/README.md) (Persian).
  It is frozen as written; it describes the data layer at the end of that stage.
- D1b addendum: [reports/d1_data/addendum_d1b.md](reports/d1_data/addendum_d1b.md)
  (Persian). It records what changed after the author's review: Bonbast is now
  the primary dollar series and TGJU the check (D-001 revised), TGJU is read
  from its history-table endpoint (D-007), model returns are weekly (D-011),
  and the holes in the robusta data are filled from the ICO indicator (D-020).
  It is frozen as reviewed.
- D2 report: [reports/d2_eda/README.md](reports/d2_eda/README.md) (Persian).
  Measured facts for the simulator: volatility over time, drift and carry,
  distribution shape, dependence, persistence and block length, the rial cost
  of coffee, stress windows, and the bootstrap pool. It ends with the
  decisions D3 needs.
- D3 report: [reports/d3_engine/README.md](reports/d3_engine/README.md) (Persian).
  The engine's base case as of the latest week, the price of insurance, the
  sensitivities, the risk split and a calibration preview (how often the
  realised 13-week USD change fell outside the simulated 5%-95% range). Every
  input default is a placeholder; the report ends with the decisions taken in
  review and the ones still needed. It was rebuilt after the D3 review (D3b)
  and says at the top what changed. Frozen as reviewed.
- D4 report: [reports/d4_stress/README.md](reports/d4_stress/README.md) (Persian).
  Seven crisis paths replayed from today's position (five real 13-week
  windows, the full war path, one hypothetical combination), priced for every
  strategy under both contracts, and a reverse stress test: the USD rise that
  breaks the margin floor, with how often history produced it.
- D5 report: [reports/d5_backtest/README.md](reports/d5_backtest/README.md) (Persian).
  Every Wednesday from 1394-03-27 with a known 13-week outcome is a decision
  date. What each simple strategy really cost and how often the margin broke
  the floor; what following the model's minimum hedge ratio would have done
  against its promise of at most 5% risk; calibration of the simulated range
  by volatility regime, for the filtered method and the plain bootstrap.
  Rebuilt in D5b with an illustration of the stress floor (D-051): never hedge
  less than what survives the largest USD rise seen so far.

## How this was made

The code in this repository was written with Claude Code from the author's
specifications and reviewed by the author. Decisions, their dates and their
reasons are recorded in [DECISIONS.md](DECISIONS.md). This repository is a
snapshot of the working repository; its commit history is not included (D-054).

## Setup

```
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -e . --no-deps
.venv/bin/python -m pytest
```

`pyproject.toml` only makes the `src/coffee_risk` package importable (so
`python -m coffee_risk.data ...` works); the pinned dependencies live in
`requirements.txt`.

## Data

Public data only. Nothing under `data/` is committed (D-006); the fetchers
rebuild it. Each fetcher runs on its own and reuses its cached raw file unless
`--refresh` is given.

```
.venv/bin/python -m coffee_risk.data fetch all        # or one of: tgju bonbast arabica robusta ico fred fund official robots
.venv/bin/python -m coffee_risk.data build-panel      # data/processed/panel_daily.parquet, panel_weekly.parquet
.venv/bin/python scripts/d2_report.py                 # reports/d2_eda/README.md + figures
.venv/bin/python scripts/d4_report.py                 # reports/d4_stress/README.md + figures
.venv/bin/python scripts/d5_report.py                 # reports/d5_backtest/README.md + figures (about a minute)
.venv/bin/python scripts/d6_kit.py                    # data/export/: the calculator kit (not committed)
```

`scripts/d1_report.py` and `scripts/d1b_addendum.py` built the two reviewed
D1 documents and are kept for the record: they ran on the code of their stage
and do not run on the current code, because later stages changed the data
they read. `scripts/d3_report.py` is kept the same way: its last step wrote
the calculator's path file in a format the kit has replaced.

| Source | What | Notes |
|---|---|---|
| `bonbast` | free-market USD/IRR, sell rate (primary, D-001 revised) | public archive on GitHub, toman converted to rial; column `usd_free` |
| `tgju` | free-market USD/IRR, key `price_dollar_rl` (check) | history-table endpoint (D-007), plus the history page to check it; column `usd_tgju` |
| `arabica` | ICE arabica front month `KC=F` and the listed contracts | Yahoo Finance via yfinance |
| `robusta` | ICE London robusta front month | TGJU key `commodities-london-coffee` (weekday rows) |
| `ico` | ICO Robustas daily indicator price, monthly PDFs | a physical price, used only to fill the weekly robusta returns that TGJU's data holes leave empty (D-020); never for levels |
| `fred` | IMF monthly coffee indicators | controls only |
| `fund` | listed fixed-income funds on TSETMC | rial opportunity-cost index |
| `official` | official-channel USD series from TGJU | information only (D-002) |
| `robots` | robots.txt of every host used | reported in the D1 report |

Panels (both under `data/processed/`, never committed):

- `panel_daily.parquet`: one row per calendar day, a cell filled only where
  that series has an observation; flag columns for rolls, stale and near-stale
  values and outliers.
- `panel_weekly.parquet`: one row per Wednesday (D-011): level = last
  observation on or before that Wednesday, weekly log return, and flags. The
  coffee return leaves out futures roll days. A return that is not a real
  one-week move (stale level, or the first observation after a data hole) is
  left empty (D-018); for robusta such weeks are filled from the ICO indicator
  and marked in `robusta_return_source` (D-020).

Raw responses are saved unchanged under `data/raw/<source>/`, listed in
`data/raw/MANIFEST.csv`; every network attempt is logged in
`data/raw/FETCH_LOG.csv`.

## Reports

One folder per stage under `reports/`. Stage reports are written in Persian.
New ideas that are out of scope are parked in [reports/ideas.md](reports/ideas.md).

## The engine (D3)

`src/coffee_risk/engine/` answers one question for a decision date and a set
of inputs: what does each way of buying the dollars cost in rials, how risky
is it, and which one does the decision rule pick.

```python
from coffee_risk.data import fred, panel, weekly
from coffee_risk import engine

wk = weekly.load()
as_of = wk.index[-1]
inputs = engine.Inputs.from_data(as_of, wk, panel.load(), fred.load())     # every default is a placeholder
paths = engine.simulate(as_of, 20_000, seed=1, options=inputs, weekly=wk)  # only weeks up to as_of are used
result = engine.run(paths, inputs)      # result.table, result.decision, result.min_hedge, result.curve
```

- `inputs.py`: the `Inputs` dataclass (all assumptions in one place) and the
  strategies (all now, staged, all on the payment day, a share now and the rest
  on the payment day).
- `simulator.py`: joint stationary bootstrap of the weekly returns up to and
  including the decision week (D-038); filtered historical simulation for USD
  (each week standardised by a volatility from earlier weeks only); drift set
  on the mean (D-024..D-029).
- `cost.py`: rial cost at the payment week on any set of paths; revenue (with
  symmetric or up-only pass-through, D-043) and margin.
- `decision.py`: risk metrics, the decision rule (D-035, ties by D-037) and
  the minimum hedge ratio, which is the headline output.
- `calibration.py`: the calibration preview and the start of the backtest
  (D-041).
- `stress.py`: replay of a historical window from today's levels and the
  reverse stress test (D4).
- `backtest.py`: the rolling backtest (D5): realised paths (with the fund's
  realised growth as the cost of funds), the engine run past-only at each
  date, breach rates, and calibration with a pinball score.
- `export.py`: the calculator kit (D-014), written under `data/export/` and
  never committed: undated sampled paths (standardised USD residuals, coffee
  returns, demand draws), the scenarios as aggregates, and parity cases.
- `calculator.py`: the calculator's arithmetic in Python on those same files;
  its results are the parity cases the web page must reproduce.

## The calculator kit (D6-prep)

The web calculator is a single page in JavaScript. It reads three files that
`scripts/d6_kit.py` writes to `data/export/`: `calculator_paths.json` (20,000
sampled paths, the same sample as the reports, D-052), `scenarios.json` (S1..S7 as aggregates) and `parity.json`
(twelve input cases with the results to reproduce).
[docs/calculator_spec.md](docs/calculator_spec.md) lists every formula in the
order the page applies them. The tests re-implement that document from
scratch and compare it with the parity cases.

## The calculator page

[calculator/](calculator/) holds the page (Persian, right-to-left): `engine.js`
(the formulas of the spec, usable in the browser and in Node), `template.html`
and `build.py`, which fills the template with the engine and the kit's files.
It was written with Claude Code from the spec; see
[calculator/README.md](calculator/README.md). The built page embeds the
simulated paths and is not committed.

```
python3 calculator/build.py          # data/export/coffee-fx-calculator.html and preview.html
node calculator/test_parity.js       # engine.js against data/export/parity.json
```

`tests/test_calculator_parity.py` runs the Node check inside the test suite;
it is skipped when Node or the kit files are missing.
