# Decision log

Each entry: id, date, who decided, the decision, and the reason.
Provisional defaults chosen during a stage are listed in that stage's report
under "Decisions needed" and move here once the author settles them.

Reasons are recorded as the author gave them. Where the author decided without
restating one, the column cites the report section the decision followed (see
Notes).

| ID | Date | Decided by | Decision | Reason |
|---|---|---|---|---|
| D-001 | 2026-10-12 | author | **Superseded by D-001 (revised) below.** Original: free-market USD/IRR from TGJU (key `price_dollar_rl`) as the primary source; Bonbast (sell rate) as an independent check. | Original reason (drafted by Claude from the spec): TGJU has the longest daily free-market history; a second, independently collected series is needed to trust it. |
| D-001 (revised) | 2026-10-12 | author | Primary free-market USD/IRR = Bonbast sell rate (toman × 10). TGJU `price_dollar_rl` becomes the check. Known Bonbast limits: one snapshot a day; repeated values on holidays/outages (e.g. 1404/12/23–27). | TGJU shows managed or frozen quotes in exactly the stress windows the model needs (1394 H2, 1397, Dey 1404, early war); Bonbast is US-hosted and kept updating. (Author's finding in the D1 review, reproduced in `reports/d1_data/addendum_d1b.md`, section 3.) |
| D-002 | 2026-10-12 | author | FX channel: the base case is the free market. The official channel appears later only as an optional "discount vs free market" input. | The importer's real access to official-rate FX is unknown and changes with policy; the free market is the rate it can always get. |
| D-003 | 2026-10-12 | author | The arabica/robusta mix of the importer is unknown: it becomes a calculator input (default 50/50). Both series are fetched. | The mix is not public; the result should not depend on a guess. |
| D-004 | 2026-10-12 | author | Robusta daily: try automatic sources first; the fallback is a CSV the author downloads manually from Investing.com. **The fallback was dropped by D-020:** Investing.com's download ignores the chosen date range. | No clearly free automatic source for ICE London robusta is known in advance. |
| D-005 | 2026-10-12 | author | Stress scenarios (exact dates set in D2): FX jump 1397, FX jump 1401–1402, coffee 2021, coffee 2024, and the war period from 9 Esfand 1404 (2026-02-28) to date. The war period is important. | These are the episodes a reader in Iran will ask about; the war period is the most recent and most relevant. |
| D-006 | 2026-10-12 | author | Raw and processed data are never committed. Only code, tests, reports and figures go into git. | Copyright of TGJU / Investing.com / ICO. |
| D-007 | 2026-10-12 | author | TGJU via the history-table endpoint (`api.tgju.org`). The chart endpoint and the "last bar" rule are removed from the code; the existing raw chart files are kept as a record. | Same data as the chart endpoint; platform.tgju.org disallows bots in robots.txt, the table endpoint does not. |
| D-008 | 2026-10-12 | author | Unconfirmed TGJU jumps need no handling. `usd_outlier_check` is kept as information. | TGJU is only the check now, so its unconfirmed jumps no longer enter the model. |
| D-009 | 2026-10-12 | author | The robusta roll rule is applied to the full history. | The rule is verified for arabica; for robusta the evidence is clear from 2021, and applying it earlier drops only ~2% of returns. |
| D-010 | 2026-10-12 | author | Weekend-dated TGJU robusta rows count as unobserved. | ICE London does not trade at weekends; rows with an unclear date are not used. |
| D-011 | 2026-10-12 | author | Model returns are weekly: last observation on or before each Wednesday (Tehran calendar date), log return Wednesday to Wednesday. | Equal-length returns, no move lost, absorbs Bonbast's <= 1-day snapshot timing. |
| D-012 | 2026-10-12 | author | The calculator gets an input "physical premium over futures" (USD/kg), default from the futures-vs-IMF table. Price risk is modelled on the futures part only. | Importers usually pay futures plus a differential fixed in the contract, so price risk sits in the futures part; the premium only sets the cost level. |
| D-013 | 2026-10-12 | author | The backtest starts in 1394. | The opportunity-cost index starts 1393-12-24 (start week refined by D-015). |
| D-014 | 2026-10-12 | author | No raw or processed price series in anything public. The calculator may ship simulated, undated return paths and estimated parameters. | TGJU requires written consent for copying, the IMF series are copyrighted, and Investing.com has its own terms. |
| D-015 | 2026-10-12 | author | The backtest starts in the week of 1394-02-30 (Wednesday 2015-05-20). Refines D-013. Start week superseded by D-041; the reviewed D2 report keeps this sample. | It is the first week after the fund began trading regularly. |
| D-016 | 2026-10-12 | author | **Superseded by D-020** (the Investing route was dropped). Was: "Investing agrees with TGJU" = median absolute level difference <= 1% and daily-return correlation >= 0.90. | Thresholds were fixed before seeing the full file. |
| D-017 | 2026-10-12 | author | **Superseded by D-020** (the Investing route was dropped). Was: a row dated on the download day is dropped. | It is an intraday price, not a close. |
| D-018 | 2026-10-13 | author | The first weekly return after any gap longer than 5 calendar days is missing, unless filled as in D-020. Read as in D-023. | A multi-week move booked in one week would be a fake jump in the bootstrap pool. |
| D-019 | — | — | **Number unused.** It was never sent. | — |
| D-020 | 2026-10-13 | author | Robusta gap fill from the ICO's public daily indicator prices (Robustas, monthly PDFs): where the weekly robusta return is missing because of a hole in TGJU's data, it is taken from ICO Robustas (last ICO observation on or before each Wednesday), provided the agreement test passes (weekly returns 2025-01 to 2026-09, fresh prices, no TGJU roll in the week: correlation >= 0.90 and volatility ratio ICO/TGJU between 0.8 and 1.25; fixed before looking). Such weeks are marked `robusta_return_source = "ico"`. Levels stay TGJU. A month whose parsed rows do not reproduce the printed Average/High/Low to 0.01 is not used. The Investing.com route is dropped. | Fills a 5-week hole in the war period from an official public source, validated against TGJU before use; the physical-vs-futures basis matters for levels, not for weekly returns, and the test checks this. |
| D-021 | 2026-10-13 | author | ICO July 2026 is accepted with one exception: the 17-Jul row is set to missing. A documented exception, not a change of the validation rule. | The parse and an independent read by the author give the same 23 rows, so the mismatch with the printed average is inside the PDF; it sits in a row that repeats the previous day and is not used by any needed Wednesday. |
| D-022 | 2026-10-13 | author | Coffee multi-week changes are built the same way everywhere: sums of roll-excluded weekly returns, including ICO-filled weeks. Replaces "13-week changes use TGJU endpoints" from the gap-fill message. | That wording let the 2026-07-28 roll back in. |
| D-023 | 2026-10-13 | author | Claude's reading of D-018 is accepted: a weekly return is empty (a) when the level at the end of the week is more than 5 calendar days old, and (b) in the week that holds the first observation after a hole of more than 5 calendar days; for every series. | — |
| D-024 | 2026-10-13 | author | Bootstrap pool = all weeks from 1391-07; inside the backtest, only weeks before each decision date. | The largest sample, containing both calm and crisis regimes; past-only use keeps the backtest honest. |
| D-025 | 2026-10-13 | author | Stationary bootstrap, mean block length 13 weeks; sensitivity at 4 and 8 weeks. | USD volatility clusters last longer than the horizon and the Politis–White estimate for USD returns is unstable; tying the block length to the horizon avoids both the instability and look-ahead. |
| D-026 | 2026-10-13 | author | Filtered historical simulation for USD (volatility estimated from past data only); plain bootstrap for coffee; plain-bootstrap results also reported. | USD volatility is persistent (rank correlation 0.63 between past and next-quarter volatility) and is now ~1.4× its average; coffee volatility is near normal and less persistent. |
| D-027 | 2026-10-13 | author | Stress set = the five windows in section 7 of the D2 report plus the full war path. | Covers both FX episodes, both coffee episodes and the current war; the extra coffee windows overlap the chosen ones heavily. |
| D-028 | 2026-10-13 | author | USD drift is a user input. Default "neutral": the expected simple USD change to each week equals the growth of rials at the market rate (the fund's return) over the same weeks, so before frictions the mean rial cost of buying now equals that of buying on the payment day. Defined on the mean, not on the log/median. The past-only historical mean and zero drift are also reported. | D2b: "equal log growth" and "equal expected cost" differ by 1.1–1.8% per quarter. |
| D-029 | 2026-10-13 | author | Coffee drift: zero in expectation (E[F_k/F_0] = 1). | The futures price is the market's forecast. |
| D-030 | 2026-10-13 | author | The firm's cost of rial funds is an input separate from drift. Default = the fund's return (idle cash); sensitivity 40% (illustrative borrowing rate, labelled as an assumption). | Buying FX early costs the firm its own funding rate, which can exceed the market rate. |
| D-031 | 2026-10-13 | author | FX holding cost (fees and counterparty risk of holding USD or a havaleh until payment), annual %, is an input. Default 0, sensitivity 3%. | No public data; stated openly. |
| D-032 | 2026-10-13 | author | Two levers: when FX is bought, and whether the coffee price is fixed with the supplier now ("fixed") or at payment ("floating"). | No exchange-traded hedge (futures/options) is available to an Iranian importer. |
| D-033 | 2026-10-13 | author | Revenue side: the selling plan is priced on the expected cost under the chosen drift; pass-through of unexpected FX moves into selling prices within the horizon is an input (default 0, conservative); revenue/demand uncertainty is an input (default off). Other costs are folded into net revenue and treated as fixed in rials. | Keep the revenue side simple and explicit. |
| D-034 | 2026-10-13 | author | Inventory: limited stock is why the purchase is mandatory and the payment date fixed; it enters as the payment week (default 13), not as a separate model. | — |
| D-035 | 2026-10-13 | author | Decision rule: among all strategies, keep those with P(margin >= floor) >= confidence; choose the lowest mean rial cost; tie-break on lower CVaR95. If none qualifies, say so and report the best achievable probability and the highest floor achievable at the chosen confidence. | — |
| D-036 | 2026-10-13 | author | The memo's base case stays "floating"; "fixed" is shown next to it. | The main finding is that FX timing alone cannot protect the margin while the coffee price is open; both levers (D-032) are needed. |
| D-037 | 2026-10-13 | author | Mean-cost differences below 0.25% of plan count as ties, broken by lower CVaR95. The headline output is the minimum hedge ratio; the chosen strategy is shown next to it. Refines D-035. | Under neutral drift the means are equal by construction, and differences below Monte-Carlo noise should not pick the answer. |
| D-038 | 2026-10-13 | author | The week ending on the decision date enters the pool and the volatility estimate. | Its return is known on that date. |
| D-039 | 2026-10-13 | author | The funds-rate sensitivity is market rate + 10 percentage points (replaces the fixed 40% of D-030). | A fixed level means different things at different dates. |
| D-040 | 2026-10-13 | author | Volatility details stay as implemented: EWMA with mean zero, at least 13 prior returns, floor = past-only 5th percentile. | — |
| D-041 | 2026-10-13 | author | The backtest starts 1394-03-27 (Wednesday 2015-06-17). Supersedes the start week of D-015. | It is the first week with a 13-week fund return. |
| D-042 | 2026-10-13 | author | The IMF 3-month publication lag and the historical-drift definition stay as implemented. | — |
| D-043 | 2026-10-13 | author | Pass-through has two modes: "symmetric" and "up only" (selling prices rise with unexpected USD increases but do not fall when USD falls). When pass_through > 0, "up only" is the default. | Retail prices in Iran are sticky downward; the symmetric mode makes full hedging look risky for a reason that may not exist. |
| D-044 | 2026-10-13 | author | Up-only pass-through keeps the planned revenue as is; the higher mean revenue it implies is stated in the memo. | A firm that can pass FX jumps on does earn more on average; removing that would need a new assumption. |
| D-045 | 2026-10-13 | author | FHS stays the default (D-026), with plain-bootstrap figures reported next to it. The D5 calibration splits starts into high- and low-volatility halves and adds a width-aware score. | The preview could not separate the two methods overall; they differ in when the band opens and closes, and today is a high-volatility week. In 1404-1405 the plain bootstrap missed 13.2% of upside moves and FHS none (6 independent windows: a sign, not proof). |
| D-046 | 2026-10-13 | author | Excess USD stays valued at the market rate; the memo says in one sentence that the S1 margin above plan is a gain from holding USD, not the strategy's aim. | — |
| D-047 | 2026-10-13 | author | Scenarios keep today's funds rate. | Realized fund returns are covered by the backtest. |
| D-048 | 2026-10-13 | author | The engine is not retuned. The memo reports both numbers (promised 5%, realized 7.6%) and where the model fails (calm weeks before a jump). | — |
| D-049 | 2026-10-13 | author | FHS stays the default. | — |
| D-050 | 2026-10-13 | author | No separate coffee calibration. The memo's floating-contract statements use backtest facts, not model probabilities; the calculator shows a warning when the contract is floating. | — |
| D-051 | 2026-10-13 | author | "Stress floor" as a policy option: hedge = max(model h*, the share needed to survive a stress scenario chosen by management). It goes into the calculator and is illustrated in the backtest with an in-sample caveat. | The model cannot see a jump coming out of a calm period (D5); a stress floor is set by a business choice, not fitted to the backtest. |
| D-052 | 2026-10-13 | author | The calculator kit holds 20,000 paths: the reports' own sample (same seed 20261013, same decision date, same simulation settings). Only `calculator_paths.json` and the `parity.json` computed from it change. | The page and the reports must give the same numbers; Monte-Carlo error at 5,000 paths moved the minimum hedge ratio by up to 14 points (pass-through 0.5 up only: 78% on the page against 64% in the reports). Cost: a file about four times as large and about four times the computing in the browser. |
| D-053 | 2026-10-13 | author | The calculator's default stress floor is the largest real 13-week USD jump seen up to the decision date: the rule of the D5b backtest. The page takes it from `scenarios.json` (non-hypothetical scenarios with payment week 13, the largest USD change at the payment week). Today that is S1 (+141%), so the base-case suggestion is 84%. S2 stays in the list. | D5b, fixed contract, 578 dates, target 5%: the model alone broke the floor in 7.6% of weeks; with the worst-seen floor 3.8% (mean hedge 65%, mean cost 101.4% of plan); with S2 as the floor from the date it had happened 6.4%, which misses the target. The rule was chosen after the model-only result was seen, so the backtest is not pre-registered. The page takes "worst seen" from the curated scenario list, so a bigger future jump needs a new scenario in the kit. |
| D-054 | 2026-10-13 | author | Publication: the repository is published as a snapshot of the working repository, without its commit history. Left out: the coding-assistant instruction file. Comments and the D1 report name the two network routes neutrally ("direct", "proxy"). D-014 covers data files and raw or processed price series; figures of public prices in the reports are published, and the few format rows in tests/test_parsers.py are test fixtures. No number or result changed. | Privacy before going public; a portfolio needs the code, reports and figures, not the working history. |

Notes

- 2026-10-12, D1 review: the author accepted the "Reason" texts Claude had
  drafted for D-002..D-006.
- 2026-10-12, D1 review: the author confirmed the official-channel regime
  dates of D1 report section 8 and added two: Dey 1404 (preferential rate
  abolished in the budget) and Ordibehesht 1405 (preferential rate restored for
  a few essentials; coffee not among them). The full list is in
  `coffee_risk.data.checks.OFFICIAL_REGIMES`.
- 2026-10-12, D1b review: the addendum was accepted. The author's own check of
  the 13-week correlation used one start week; over all start weeks it is
  0.95-0.99 outside 1397.
- 2026-10-13, D2 review: the report was accepted; the author rebuilt five of
  its numbers from the raw files.
- 2026-10-13, D2b review: accepted. The author supplied the reasons for D-018
  and D-024..D-027.
- 2026-10-13, D-053: before it was written the author asked whether the D5b
  rule, run on 2026-10-07, really picks S1. It does: the largest 13-week rise
  of the Bonbast series up to that date is +140.5%, from 1397-04-06
  (2018-06-27) to 1397-07-04 (2018-09-26), the S1 window; the war weeks are
  included and their largest 13-week rise (+47.6%, ending 2026-10-07) is below
  it. The author had first proposed S2 as the default; the D5b backtest had
  not used S2, and S2 applied from the date it happened missed the 5% target.
- 2026-10-13, D4/D5 review: accepted. The author rebuilt the fixed-contract
  backtest rows from the raw Bonbast and fund files (mean cost 100.0 / 101.6 /
  105.1, breach 0.0 / 1.6 / 13.5%, worst margin 21 / -2 / -72% for all today /
  staged / payment day); they match. Decisions D-046..D-051. The stress floor
  (D-051) was thought of after the D5 results were seen: its backtest (D5b) is
  an illustration, not out-of-sample evidence.
- 2026-10-13, D3b review: accepted. Decisions D-044 and D-045. The
  volatility split of the D5 calibration (EWMA USD volatility at the start
  above or below the median over all start dates) was fixed by the author in
  the D5 spec, before any result was seen.
- 2026-10-13, D3 review: accepted, including the fixed-contract sensitivity
  table that the spec had not asked for. Decisions D-036..D-043; the D3 report
  was rebuilt in D3b with a "what changed" note. D-041 moves the start of the
  backtest; `coffee_risk.eda.BACKTEST_START` still holds the D-015 week
  because the reviewed D2 report is defined on it, and the engine uses
  `coffee_risk.engine.calibration.BACKTEST_START`.

## Open (not decided yet)

Nothing from the stage reports is open: the D4 and D5 items were decided in
D-046..D-051. Open points of the calculator kit (what the page shows as "the
hedge" when the model has no minimum hedge ratio; the stress floor ignores
coffee; no plain bootstrap on the page) are in section 14 of
`docs/calculator_spec.md`; its first point, the Monte-Carlo error of 5,000
paths, was settled by D-052.

- ICO PDFs for January-August 2025 are not on the ICO site under either file
  name; the ICO agreement test therefore starts in September 2025.
