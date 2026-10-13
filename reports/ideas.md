# Ideas (not in scope yet)

New ideas are parked here, not built. Each line: date, idea, why it might matter.

| Date | Idea | Why it might matter | Status |
|---|---|---|---|
| 2026-10-12 | Build our own continuous arabica series from single contracts, switching before the delivery month instead of at expiry. | `KC=F` follows the expiring contract for about 13 thinly traded days before each roll; its prices there are a little noisier than the liquid contract. Needs a source for expired contracts (Yahoo has none). | open |
| 2026-10-12 | A panel flag for days on which TGJU and Bonbast differ by more than 2%. | The reports only give the count; a flag would let D2 condition on source disagreement directly. | open |
| 2026-10-12 | A weekly (Wednesday to Wednesday) return grid for the joint bootstrap. | Common-calendar returns mix 1-day and 4-5-day steps (about 40 common dates in 3 months). | done in D1b (D-011) |
| 2026-10-12 | Persian labels in the figures (needs a shaping library for right-to-left text). | The memo is in Persian; report figures use English labels with Persian captions. | open |
| 2026-10-12 | A "premium over futures" input in the calculator, with a default from the IMF physical indicators. | Futures are 4-26% below the IMF arabica indicator; an importer pays the physical price. | decided (D-012), to build with the calculator |
| 2026-10-12 | A second, independent check of the robusta dates for early 2026 once a full Investing.com file exists (the arabica-based check in the D1b addendum is indirect). | TGJU's robusta has weekend-dated rows from Feb 2026; the weekday rows test as correctly dated against arabica, but only a same-instrument source can confirm it. | open, waits for the file |
| 2026-10-13 | Fixing only a share of the coffee price with the supplier (a "coffee hedge ratio" next to the FX one). | D3: with a floating contract no FX strategy meets the margin constraint, with a fixed one a partial purchase today is enough; the useful answer may be in between. | open |
| 2026-10-13 | The planned margin (or selling price) that makes the floating contract feasible, as an output. | D3: with a floating contract the highest floor that can be met at 95% is below the 10% target; the importer may prefer to reprice than to fix the coffee price. | open |
| 2026-10-13 | A two-way table: minimum hedge ratio by USD drift and funds rate. | D3 varies one input at a time; these two interact (both set the expected cost of buying early). | open |
| 2026-10-13 | At each date use the wider of the two USD bands (filtered simulation, plain bootstrap). | D5: the filtered band is too narrow in calm weeks and the plain one too narrow in volatile weeks; the wider of the two would cover both failures. A new rule, so it needs its own out-of-sample test. | open |
| 2026-10-13 | A floor on the minimum hedge ratio in calm regimes. | D5: before the first jump of 1396-97 the model asked for no hedge at all; the breaches cluster at the start of jumps. Any number must not be tuned on the same backtest. | open |
| 2026-10-13 | A calibration check of the coffee paths alone (no USD). | D5: with a floating contract realised costs fell above the simulated 95th percentile about twice as often as 5%; it is not yet clear how much of that is coffee. | open |
| 2026-10-13 | Record the probability-integral transform (the simulated rank of the realised value) for every date. | D5 counts only the two tails; the full rank shows where in the distribution the model is off. | open |
