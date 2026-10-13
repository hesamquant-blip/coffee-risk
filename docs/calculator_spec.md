# Calculator specification

Every formula the engine uses, in the order the page must apply them. The page
is right when it reproduces `parity.json` from `calculator_paths.json` and
`scenarios.json`. The Python reference is `src/coffee_risk/engine/calculator.py`
(it calls the engine's own cost and decision functions on the same files).

`scripts/d6_kit.py` writes the three JSON files and a copy of this document to
`data/export/`. Nothing there is committed (D-006, D-014).

Conventions: weeks are numbered 0 (today) to T (payment week); N = number of
paths; H = 13 weeks in the path file; 52 weeks in a year; all rates are annual
simple rates unless a line says "weekly"; `^` is a power; arithmetic is in
double precision.

---

## 1. Files

### 1.1 `calculator_paths.json`

| Field | Meaning |
|---|---|
| `as_of` | decision date of the file (the only date in it) |
| `n_paths` (N), `horizon` (H) | 20,000 and 13. The sample is the one the reports use: same seed, same decision date, same settings (D-052) |
| `scale` | 1e-5: every array holds integers; value = integer × scale |
| `usd_z` | N × H: sampled standardised USD residuals, zero mean over the pool. Path by path: element of path p (0-based), week k (1..H) is at index `p*H + (k-1)` |
| `arabica`, `robusta` | N × H, same layout: roll-excluded weekly log returns of the same sampled weeks |
| `demand_z` | N standard-normal draws, one per path |
| `s0` | USD/IRR today, rial (Bonbast sell on `as_of`) |
| `market_rate` | the fund's trailing 13-week return, annualised |
| `usd_vol_weekly` | current weekly EWMA volatility of USD (σ) |
| `hist_mean_weekly_usd_return` | past-only mean simple weekly USD return of the pool (m_hist), for the "historical" drift |
| `premium.arabica`, `premium.robusta` | default physical premium as a share of the invoice |
| `half_life`, `mean_block`, `usd_vol_method`, `seed`, `format`, `layout` | how the file was sampled; not used in formulas |

The file holds no price series and no dates of the sampled weeks.

### 1.2 `scenarios.json`

`scenarios`: seven objects (S1..S7). `payment_week` is T_s (13; 32 for S6).
`weeks` = the two later staged weeks and the payment week (`[4, 9, 13]`;
S6: `[11, 21, 32]`). `usd`, `arabica`, `robusta`: cumulative simple change
from the start of the window at each of those weeks (level ratio = 1 + change).
`hypothetical` is true for S7 (USD of S1 with coffee of S4). Aggregates only.

### 1.3 `parity.json`

`cases`: twelve objects with `name`, `inputs` (every input of section 2,
spelled out) and `outputs` (section 12).

---

## 2. Inputs

Names are the keys of `inputs` in `parity.json`. Every default is a
placeholder of the project except the three read from the file.

| Input | Default | Notes |
|---|---|---|
| `usd_amount` | 100,000 | USD |
| `payment_week` (T) | 13 | parity uses 13; the formulas hold for any T ≤ H |
| `s0` | file | rial per USD |
| `market_rate` | file | drives the neutral drift |
| `channel_discount` (d) | 0 | official channel = free rate × (1 − d), same dynamics |
| `contract` | `"floating"` | or `"fixed"` |
| `arabica_share` (w) | 0.5 | |
| `premium_share` (π) | w·premium.arabica + (1 − w)·premium.robusta | the user may type another value |
| `funds_rate` (f) | = `market_rate` | the "+10 pp" option is `market_rate + 0.10` |
| `fx_holding_cost` (c) | 0 | per year, on the rial value at purchase |
| `sellback_spread` (b) | 0.01 | |
| `planned_margin` (μ) | 0.25 | |
| `margin_floor` (φ) | 0.10 | |
| `confidence` (α) | 0.95 | |
| `pass_through` (ρ) | 0 | |
| `pass_through_mode` | `"up_only"` | or `"symmetric"`; matters only when ρ > 0 |
| `demand_sd` | 0 | |
| `budget_factor` | 1.10 | budget = factor × planned cost |
| `tie_tolerance` | 0.0025 | share of the planned cost (D-037) |
| `drift` | `"neutral"` | `"zero"`, `"historical"`, or a number (annual simple rate) |
| `usd_vol_scale` (v) | 1 | the volatility slider |

---

## 3. Paths

For path p = 0..N−1 and week k = 1..H:

1. **Decode.** `z[p][k] = usd_z[p*H + k-1] * scale`; `a[p][k]`, `r[p][k]`
   likewise from `arabica`, `robusta`; `n[p] = demand_z[p] * scale`.
2. **Volatility rescale (USD only).** `u[p][k] = z[p][k] * usd_vol_weekly * v`.
3. **Cumulate.** `LS[p][k] = exp(u[p][1] + … + u[p][k])`, and `LA`, `LR` from
   `a`, `r` the same way. Week 0 is 1 for every path.
4. **Target growth of USD**, k = 0..H:
   - neutral: `g[k] = (1 + market_rate)^(k/52)`
   - zero: `g[k] = 1`
   - historical: `g[k] = (1 + hist_mean_weekly_usd_return)^k`
   - a number δ: `g[k] = (1 + δ)^(k/52)`
5. **Drift normalisation, week by week.** With `mean_k(X) = (1/N) Σ_p X[p][k]`:
   - `S[p][k] = LS[p][k] * g[k] / mean_k(LS)`
   - `A[p][k] = LA[p][k] / mean_k(LA)`
   - `R[p][k] = LR[p][k] / mean_k(LR)`

   So the mean of `S[·][k]` is exactly `g[k]` and the mean of each coffee is 1
   (D-028, D-029). `S`, `A`, `R` are level ratios to today.

The normalisation must be redone whenever the volatility scale or the drift
changes. The coffee arrays do not depend on any input.

---

## 4. Dollars needed at the payment week

- fixed contract: `need[p] = usd_amount`
- floating contract: `need[p] = usd_amount * ((1 − π) * (w * A[p][T] + (1 − w) * R[p][T]) + π)`

---

## 5. Strategies

A strategy is a list of tranches (week t, share q of `usd_amount`) bought
before the payment week; whatever is missing at T is bought then.

| Name in parity | Tranches |
|---|---|
| `now` | (0, 1) |
| `payment_day` | none |
| `staged` | (round(k·T/3), 1/3) for k = 0, 1, 2 → weeks 0, 4, 9 for T = 13 |
| `h_X` (X = 0, 10, …, 100) | (0, X/100); no tranche when X = 0 |

`h_0` is `payment_day` and `h_100` is `now`.

---

## 6. Rial cost at the payment week, per path

`rate[p][k] = s0 * (1 − d) * S[p][k]`.

```
cost[p]   = 0
bought    = 0
for each tranche (t, q):
    usd    = q * usd_amount
    years  = (T − t) / 52
    cost[p] += usd * rate[p][t] * ((1 + f)^years + c * years)     # funding + holding cost
    bought += usd
gap[p]  = need[p] − bought
cost[p] += rate[p][T] * (gap[p] if gap[p] >= 0 else gap[p] * (1 − b))   # shortfall bought, excess sold back
```

All rial flows are valued at T. An excess (gap < 0) lowers the cost by what
it is sold for: the rate of T less the sell-back spread (D-046).

---

## 7. Plan, revenue, margin

```
C_plan   = usd_amount * s0 * (1 − d) * g[T]
R_plan   = C_plan / (1 − μ)
x[p]     = S[p][T] / g[T] − 1                    # the unexpected USD move
x[p]     = max(x[p], 0)   if pass_through_mode == "up_only"
D[p]     = 1                                      if demand_sd == 0
D[p]     = exp(−s2/2 + sqrt(s2) * n[p]),  s2 = ln(1 + demand_sd^2)     otherwise
R[p]     = R_plan * (1 + ρ * x[p]) * D[p]
margin[p] = 1 − cost[p] / R[p]
```

With "up only" and ρ > 0 the mean of R is above R_plan (D-044): say so.

---

## 8. Metrics of one strategy

Quantile Q(X, q): sort X ascending into y[0..N−1]; `pos = q * (N − 1)`;
`i = floor(pos)`; `Q = y[i] + (pos − i) * (y[i+1] − y[i])` (y[N−1] when
i = N−1). This is the default of numpy.

| Metric | Formula | In parity (as `strategies.<name>`) |
|---|---|---|
| mean cost | mean of `cost` | `mean` = mean / C_plan |
| median | Q(cost, 0.5) | `median` = … / C_plan |
| VaR | Q(cost, α) | `var95` = … / C_plan |
| CVaR | mean of the K largest costs, `K = max(1, ceil(N * (1 − α) − 1e-9))` | `cvar95` = … / C_plan |
| P(cost > budget) | share of paths with `cost > budget_factor * C_plan` | `p_over_budget` |
| P(margin < floor) | share of paths with `margin < φ` | `p_margin_below_floor` |
| low margin | Q(margin, 1 − α) | `margin_q05` |

---

## 9. Decision rule (D-035, D-037) and minimum hedge ratio

**Candidates**, in this order: `now`, `staged`, `payment_day`, `h_10`, `h_20`,
…, `h_90` (twelve; `h_0` and `h_100` are already there under other names).

1. `prob[s] = 1 − P(margin < floor)` for each candidate.
2. `best_probability` = the largest `prob` and `best_probability_strategy` its
   candidate; `highest_floor` = the largest `margin_q05` and
   `highest_floor_strategy` its candidate. On equal values the first in the
   order wins.
3. Qualifying = candidates with `prob[s] >= α − 1e-12`.
4. If none qualifies: `qualifies = false`, `chosen = null`. Report
   `best_probability` and `highest_floor`.
5. Otherwise let m* be the lowest mean cost among the qualifying. Tied = the
   qualifying with `(mean[s] − m*) / C_plan < max(tie_tolerance, 1e-9)`.
   `chosen` = the tied candidate with the lowest CVaR (first in the order on
   equal values).

**Minimum hedge ratio** (the headline output): for i = 0, 1, …, 100, with
h = i/100 and the strategy "(0, h), rest on the payment day", the smallest h
with `1 − P(margin < floor) >= α − 1e-12`; `null` if no h does.

---

## 10. Reverse stress and stress floor

Units of `usd_amount * s0 * (1 − d)`, so the amount, the rate and the channel
discount cancel. For a payment week T (13, or the scenario's):

```
g = g[T]                                      # section 3, step 4 (for T = 32 use the same formula with k = 32)
F = (1 + f)^(T/52) + c * T/52                 # what a unit bought today has cost by T
L = (1 − φ) / (1 − μ)                         # allowed cost per unit of plan
allowed(x) = L * g * (1 + ρ * s),   s = x/g − 1,  and s = max(s, 0) in the "up_only" mode
```

Coffee is taken as flat (fixed contract) and demand noise is left out.

**Reverse stress**: the USD level x = S_T/S_0 at which the margin falls below
the floor when the share h is bought today and the rest at T
(cost = h·F + (1 − h)·x):

- ρ = 0: `x* = (L*g − h*F) / (1 − h)`; for h = 1: never (`null`) if F ≤ L·g.
  With f = market rate and c = 0 this is a rise of `(L − 1)/(1 − h)` above
  the expected level: `0.2/(1 − h)` for μ = 0.25, φ = 0.10.
- ρ > 0 (a rise above the expected level g; both modes agree there):
  if `h*F + (1 − h)*g > L*g` the plan already fails at the expected level:
  `x* = g`; else if `1 − h − L*ρ <= 0`: never (`null`), revenue rises at least
  as fast as cost; else `x* = (L*g*(1 − ρ) − h*F) / (1 − h − L*ρ)`.

The rise versus today is `x* − 1`.

**Stress floor** (D-051): for a scenario whose USD ends at
`x = 1 + (last value of its usd list)`, with T = the scenario's payment week:

```
if x <= allowed(x):      share = 0
elif F > allowed(x):     share = null        # even buying everything today does not survive
else:                    share = min((x − allowed(x)) / (x − F), 1)
```

The hedge with a stress floor is `max(minimum hedge ratio, share)`.

**Default scenario** (D-053): the largest real 13-week USD jump seen up to
the decision date, the rule the D5b backtest tested. From `scenarios.json`:
among the scenarios with `hypothetical == false` and `payment_week == 13`,
the one with the largest last value of `usd` (the first in the file on equal
values). The other scenarios and "none" stay selectable. "Seen" means "in
the scenario list": a larger jump after the kit was built becomes the default
only when the kit is rebuilt with a scenario for it.

---

## 11. Replaying a scenario

One path. For a scenario with payment week T_s and `weeks = [w1, w2, T_s]`:
`S[w1] = 1 + usd[0]`, `S[w2] = 1 + usd[1]`, `S[T_s] = 1 + usd[2]`, `S[0] = 1`;
`A`, `R` likewise. Use sections 4, 6 and 7 with T = T_s, `g[k]` of section 3
step 4 extended to k = T_s, and D = 1 (no demand noise). The staged strategy
buys at weeks 0, w1, w2. Outputs: cost / C_plan, margin, and whether the
margin is below the floor. The funds rate, the premium and the plan are
today's (D-047).

---

## 12. `parity.json`

For each case, `outputs` holds:

| Key | Content |
|---|---|
| `plan_growth`, `plan_rial`, `planned_revenue_rial` | g[T], C_plan, R_plan |
| `strategies` | for `now`, `staged`, `payment_day`, `h_0` … `h_100`: the seven metrics of section 8 |
| `min_hedge` | section 9; `null` if none |
| `qualifies`, `chosen`, `best_probability`, `best_probability_strategy`, `highest_floor`, `highest_floor_strategy` | section 9 |
| `stress_floor` | share for S1 and S2 (section 10); `null` if none |
| `reverse_stress` | x* for h = 0, 0.25, 0.5, 0.75 (keys `h_0` …); `null` = never |
| `scenarios` | for S1..S7: cost (share of the scenario's plan), margin, `breaks` for `now`, `staged`, `payment_day` (section 11) |

Cases: base; fixed contract; funds rate + 10 pp; FX holding cost 3%; drift
zero; drift historical; arabica share 0; arabica share 1; pass-through 0.5 up
only; demand sd 10%; USD volatility × 0.6; margin floor 5%. In the two
arabica-share cases the premium share is the default mix for that share.

**Tolerances.** Costs and margins: relative 1e-9 (different `exp`/`pow`
libraries differ in the last digits). Probabilities are multiples of 1/N and
should match exactly; `min_hedge`, `chosen` and the `*_strategy` names must
match exactly.

---

## 13. What the page must say

- **Floating contract** (D-050): show a warning. The model's probabilities
  for a floating contract were optimistic in the backtest; statements about
  it rest on backtest facts, not on these probabilities.
- **The promise and the record** (D-048): the model promises at most 5% risk;
  following its minimum hedge ratio in the backtest broke the floor more
  often than that (the figure is in the D5 report). It fails in calm weeks
  before a jump. The stress floor is the answer offered for that.
- **Placeholders**: every default except `s0`, `market_rate` and the premiums
  is an assumption.
- **Official channel**: modelled as the free rate less a discount, with the
  same dynamics; a managed rate that moves in steps is not modelled.
- **Up-only pass-through** raises the mean revenue above the planned revenue
  (D-044).

---

## 14. Open points

1. **Monte-Carlo error of 5,000 paths: resolved by D-052.** The first kit
   held 5,000 paths and its minimum hedge ratio differed from the reports'
   (20,000 paths) by up to 14 points where the constraint is met only
   narrowly. The file now holds the reports' own 20,000 paths, so the page
   and the reports give the same numbers; `kit_summary.md` shows the
   comparison case by case. The cost is a file four times as large and about
   four times the computing time in the browser.
2. **No minimum hedge ratio, but a stress floor.** With a floating contract
   the model has no minimum hedge ratio while the stress-floor share exists.
   This document defines the two numbers; what the page shows as "the hedge"
   in that case is a choice (suggested: the stress-floor share, with the
   statement that the 95% constraint cannot be met).
3. **The stress floor ignores coffee.** It uses the fixed-contract formula
   (section 10) for every case, as D5b did. With a floating contract a
   scenario's coffee move also matters; section 11 shows it for `now`,
   `staged` and `payment_day`, but no "share that survives" is defined there.
4. **Plain bootstrap.** The file holds standardised residuals, so the page
   can rescale the volatility but cannot reproduce the plain-bootstrap
   figures of the reports (D-049 keeps the filtered method as the default).
