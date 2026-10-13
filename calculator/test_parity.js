// Check engine.js against parity.json (tolerances from calculator_spec.md section 12).
const fs = require("fs");
const E = require("./engine.js");
const paths = JSON.parse(fs.readFileSync(__dirname + "/../data/export/calculator_paths.json"));
const scen = JSON.parse(fs.readFileSync(__dirname + "/../data/export/scenarios.json")).scenarios;
const parity = JSON.parse(fs.readFileSync(__dirname + "/../data/export/parity.json"));
const P = E.prepare(paths);

let checked = 0, failed = 0, maxRel = 0;
function num(path, a, b) {
  checked++;
  if (a === null || b === null || a === undefined || b === undefined) {
    if (a !== b && !(a == null && b == null)) { failed++; console.log("NULL MISMATCH", path, a, b); }
    return;
  }
  const rel = Math.abs(a - b) / Math.max(1e-12, Math.abs(b));
  const abs = Math.abs(a - b);
  if (rel > maxRel && abs > 1e-15) maxRel = rel;
  if (rel > 1e-9 && abs > 1e-12) { failed++; console.log("NUM", path, a, b, rel); }
}
function exact(path, a, b) {
  checked++;
  if (a !== b) { failed++; console.log("EXACT", path, a, b); }
}
for (const c of parity.cases) {
  const inp = c.inputs, exp = c.outputs;
  const out = E.runAll(P, inp, scen);
  num(c.name + ".plan_growth", out.plan_growth, exp.plan_growth);
  num(c.name + ".plan_rial", out.plan_rial, exp.plan_rial);
  num(c.name + ".planned_revenue_rial", out.planned_revenue_rial, exp.planned_revenue_rial);
  for (const s in exp.strategies) {
    for (const m of ["mean", "median", "var95", "cvar95", "margin_q05"]) num(`${c.name}.${s}.${m}`, out.strategies[s][m], exp.strategies[s][m]);
    for (const m of ["p_over_budget", "p_margin_below_floor"]) exact(`${c.name}.${s}.${m}`, out.strategies[s][m], exp.strategies[s][m]);
  }
  exact(c.name + ".min_hedge", out.min_hedge, exp.min_hedge);
  exact(c.name + ".qualifies", out.qualifies, exp.qualifies);
  exact(c.name + ".chosen", out.chosen, exp.chosen);
  exact(c.name + ".best_probability", out.best_probability, exp.best_probability);
  exact(c.name + ".best_probability_strategy", out.best_probability_strategy, exp.best_probability_strategy);
  num(c.name + ".highest_floor", out.highest_floor, exp.highest_floor);
  exact(c.name + ".highest_floor_strategy", out.highest_floor_strategy, exp.highest_floor_strategy);
  for (const k in exp.stress_floor) num(`${c.name}.stress_floor.${k}`, out.stress_floor[k], exp.stress_floor[k]);
  for (const k in exp.reverse_stress) num(`${c.name}.reverse_stress.${k}`, out.reverse_stress[k], exp.reverse_stress[k]);
  for (const sk in exp.scenarios) for (const st in exp.scenarios[sk]) {
    const e = exp.scenarios[sk][st], o = out.scenarios[sk][st];
    num(`${c.name}.sc.${sk}.${st}.cost`, o.cost, e.cost);
    num(`${c.name}.sc.${sk}.${st}.margin`, o.margin, e.margin);
    exact(`${c.name}.sc.${sk}.${st}.breaks`, o.breaks, e.breaks);
  }
  // any extra keys in parity we have not checked?
  const known = new Set(["plan_growth","plan_rial","planned_revenue_rial","strategies","min_hedge","qualifies","chosen","best_probability","best_probability_strategy","highest_floor","highest_floor_strategy","stress_floor","reverse_stress","scenarios"]);
  for (const k in exp) if (!known.has(k)) console.log("unchecked key in parity:", c.name, k);
}
console.log(`checked ${checked} values, failed ${failed}, largest relative difference ${maxRel.toExponential(2)}`);
const t0 = Date.now(); for (let i = 0; i < 5; i++) E.runAll(P, Object.assign({}, parity.cases[0].inputs, { usd_vol_scale: 1 + i * 0.01 }), scen);
console.log("ms per full run:", (Date.now() - t0) / 5);
