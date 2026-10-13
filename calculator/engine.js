// Coffee FX-risk calculator engine — a line-by-line port of docs/calculator_spec.md.
// Works in the browser (window.CoffeeEngine) and in Node (module.exports).
(function (root) {
  "use strict";

  // ---------- 3. Paths ----------
  function prepare(pathsFile) {
    const N = pathsFile.n_paths, H = pathsFile.horizon, sc = pathsFile.scale;
    const z = new Float64Array(N * H), LA = new Float64Array(N * (H + 1)), LR = new Float64Array(N * (H + 1));
    const n = new Float64Array(N);
    for (let p = 0; p < N; p++) {
      let ca = 0, cr = 0;
      LA[p * (H + 1)] = 1; LR[p * (H + 1)] = 1;
      for (let k = 1; k <= H; k++) {
        const i = p * H + (k - 1);
        z[i] = pathsFile.usd_z[i] * sc;
        ca += pathsFile.arabica[i] * sc;
        cr += pathsFile.robusta[i] * sc;
        LA[p * (H + 1) + k] = Math.exp(ca);
        LR[p * (H + 1) + k] = Math.exp(cr);
      }
      n[p] = pathsFile.demand_z[p] * sc;
    }
    // coffee normalisation does not depend on inputs: do it once
    const A = new Float64Array(N * (H + 1)), R = new Float64Array(N * (H + 1));
    for (let k = 0; k <= H; k++) {
      let ma = 0, mr = 0;
      for (let p = 0; p < N; p++) { ma += LA[p * (H + 1) + k]; mr += LR[p * (H + 1) + k]; }
      ma /= N; mr /= N;
      for (let p = 0; p < N; p++) { A[p * (H + 1) + k] = LA[p * (H + 1) + k] / ma; R[p * (H + 1) + k] = LR[p * (H + 1) + k] / mr; }
    }
    return { N, H, z, A, R, n, file: pathsFile, _cacheKey: null, S: null };
  }

  function growth(inp, file, k) {
    const d = inp.drift;
    if (d === "neutral") return Math.pow(1 + inp.market_rate, k / 52);
    if (d === "zero") return 1;
    if (d === "historical") return Math.pow(1 + file.hist_mean_weekly_usd_return, k);
    return Math.pow(1 + Number(d), k / 52);
  }

  function usdPaths(P, inp) {
    const key = [inp.usd_vol_scale, inp.drift, inp.market_rate].join("|");
    if (P._cacheKey === key) return P.S;
    const N = P.N, H = P.H, sig = P.file.usd_vol_weekly * inp.usd_vol_scale;
    const LS = new Float64Array(N * (H + 1));
    for (let p = 0; p < N; p++) {
      let c = 0; LS[p * (H + 1)] = 1;
      for (let k = 1; k <= H; k++) { c += P.z[p * H + (k - 1)] * sig; LS[p * (H + 1) + k] = Math.exp(c); }
    }
    const S = new Float64Array(N * (H + 1));
    for (let k = 0; k <= H; k++) {
      let m = 0;
      for (let p = 0; p < N; p++) m += LS[p * (H + 1) + k];
      m /= N;
      const g = growth(inp, P.file, k);
      for (let p = 0; p < N; p++) S[p * (H + 1) + k] = LS[p * (H + 1) + k] * g / m;
    }
    P._cacheKey = key; P.S = S;
    return S;
  }

  // ---------- 5. Strategies ----------
  function strategyTranches(name, T) {
    if (name === "now") return [[0, 1]];
    if (name === "payment_day") return [];
    if (name === "staged") return [0, 1, 2].map(k => [Math.round(k * T / 3), 1 / 3]);
    if (name.startsWith("h_")) { const x = Number(name.slice(2)) / 100; return x > 0 ? [[0, x]] : []; }
    throw new Error("unknown strategy " + name);
  }

  // ---------- 4, 6, 7 on many paths ----------
  function simulate(P, inp) {
    const N = P.N, H = P.H, T = inp.payment_week, W = H + 1;
    const S = usdPaths(P, inp);
    const d = inp.channel_discount, f = inp.funds_rate, c = inp.fx_holding_cost, b = inp.sellback_spread;
    const gT = growth(inp, P.file, T);
    const Cplan = inp.usd_amount * inp.s0 * (1 - d) * gT;
    const Rplan = Cplan / (1 - inp.planned_margin);
    const w = inp.arabica_share, pi = inp.premium_share;
    const need = new Float64Array(N), rateT = new Float64Array(N), rate0 = new Float64Array(N), Rev = new Float64Array(N);
    const s2 = Math.log(1 + inp.demand_sd * inp.demand_sd);
    for (let p = 0; p < N; p++) {
      need[p] = inp.contract === "fixed" ? inp.usd_amount
        : inp.usd_amount * ((1 - pi) * (w * P.A[p * W + T] + (1 - w) * P.R[p * W + T]) + pi);
      rateT[p] = inp.s0 * (1 - d) * S[p * W + T];
      rate0[p] = inp.s0 * (1 - d) * S[p * W];
      let x = S[p * W + T] / gT - 1;
      if (inp.pass_through_mode === "up_only") x = Math.max(x, 0);
      const D = inp.demand_sd === 0 ? 1 : Math.exp(-s2 / 2 + Math.sqrt(s2) * P.n[p]);
      Rev[p] = Rplan * (1 + inp.pass_through * x) * D;
    }
    function cost(name) {
      const tr = strategyTranches(name, T);
      const out = new Float64Array(N);
      for (let p = 0; p < N; p++) {
        let cst = 0, bought = 0;
        for (const [t, q] of tr) {
          const usd = q * inp.usd_amount, years = (T - t) / 52;
          cst += usd * (inp.s0 * (1 - d) * S[p * W + t]) * (Math.pow(1 + f, years) + c * years);
          bought += usd;
        }
        const gap = need[p] - bought;
        cst += rateT[p] * (gap >= 0 ? gap : gap * (1 - b));
        out[p] = cst;
      }
      return out;
    }
    // fast P(margin < floor) for "(0,h), rest at T"
    const F0 = Math.pow(1 + f, T / 52) + c * T / 52;
    function pBelowForH(h) {
      let cnt = 0;
      for (let p = 0; p < N; p++) {
        const usd = h * inp.usd_amount;
        let cst = h > 0 ? usd * rate0[p] * F0 : 0;
        const gap = need[p] - (h > 0 ? usd : 0);
        cst += rateT[p] * (gap >= 0 ? gap : gap * (1 - b));
        if (1 - cst / Rev[p] < inp.margin_floor) cnt++;
      }
      return cnt / N;
    }
    return { N, T, gT, Cplan, Rplan, Rev, cost, pBelowForH, S, need };
  }

  // ---------- 8. Metrics ----------
  function quantileSorted(y, q) {
    const N = y.length, pos = q * (N - 1), i = Math.floor(pos);
    if (i >= N - 1) return y[N - 1];
    return y[i] + (pos - i) * (y[i + 1] - y[i]);
  }
  function metrics(sim, inp, costArr) {
    const N = sim.N, a = inp.confidence;
    let sum = 0, over = 0, below = 0;
    const margin = new Float64Array(N);
    const budget = inp.budget_factor * sim.Cplan;
    for (let p = 0; p < N; p++) {
      sum += costArr[p];
      if (costArr[p] > budget) over++;
      margin[p] = 1 - costArr[p] / sim.Rev[p];
      if (margin[p] < inp.margin_floor) below++;
    }
    const y = Float64Array.from(costArr).sort();
    const K = Math.max(1, Math.ceil(N * (1 - a) - 1e-9));
    let tail = 0;
    for (let i = N - K; i < N; i++) tail += y[i];
    const ms = Float64Array.from(margin).sort();
    return {
      mean: sum / N / sim.Cplan,
      median: quantileSorted(y, 0.5) / sim.Cplan,
      var95: quantileSorted(y, a) / sim.Cplan,
      cvar95: tail / K / sim.Cplan,
      p_over_budget: over / N,
      p_margin_below_floor: below / N,
      margin_q05: quantileSorted(ms, 1 - a),
      _sorted: y, _margin: margin
    };
  }

  const CANDIDATES = ["now", "staged", "payment_day", "h_10", "h_20", "h_30", "h_40", "h_50", "h_60", "h_70", "h_80", "h_90"];
  const ALL = ["now", "staged", "payment_day"].concat([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100].map(x => "h_" + x));

  // ---------- 9. Decision rule ----------
  function decide(strats, sim, inp) {
    const a = inp.confidence;
    let bp = -1, bps = null, hf = -Infinity, hfs = null;
    const prob = {};
    for (const s of CANDIDATES) {
      prob[s] = 1 - strats[s].p_margin_below_floor;
      if (prob[s] > bp) { bp = prob[s]; bps = s; }
      if (strats[s].margin_q05 > hf) { hf = strats[s].margin_q05; hfs = s; }
    }
    const qual = CANDIDATES.filter(s => prob[s] >= a - 1e-12);
    let chosen = null;
    if (qual.length) {
      let m = Math.min(...qual.map(s => strats[s].mean));
      const tol = Math.max(inp.tie_tolerance, 1e-9);
      const tied = qual.filter(s => (strats[s].mean - m) < tol); // means are already / C_plan
      let bestC = Infinity;
      for (const s of tied) if (strats[s].cvar95 < bestC) { bestC = strats[s].cvar95; chosen = s; }
    }
    let minH = null;
    for (let i = 0; i <= 100; i++) {
      if (1 - sim.pBelowForH(i / 100) >= a - 1e-12) { minH = i / 100; break; }
    }
    return { qualifies: qual.length > 0, chosen, best_probability: bp, best_probability_strategy: bps,
             highest_floor: hf, highest_floor_strategy: hfs, min_hedge: minH };
  }

  // ---------- 10. Reverse stress and stress floor ----------
  function stressParts(inp, file, T) {
    const g = growth(inp, file, T);
    const F = Math.pow(1 + inp.funds_rate, T / 52) + inp.fx_holding_cost * T / 52;
    const L = (1 - inp.margin_floor) / (1 - inp.planned_margin);
    return { g, F, L };
  }
  function reverseStress(inp, file, h, T) {
    T = T || inp.payment_week;
    const { g, F, L } = stressParts(inp, file, T), rho = inp.pass_through;
    if (rho === 0) {
      if (h >= 1) return F <= L * g ? null : 0;
      return (L * g - h * F) / (1 - h);
    }
    if (h * F + (1 - h) * g > L * g) return g;
    if (1 - h - L * rho <= 0) return null;
    return (L * g * (1 - rho) - h * F) / (1 - h - L * rho);
  }
  function stressFloor(inp, file, scen) {
    const T = scen.payment_week;
    const { g, F, L } = stressParts(inp, file, T);
    const x = 1 + scen.usd[scen.usd.length - 1];
    let s = x / g - 1;
    if (inp.pass_through_mode === "up_only") s = Math.max(s, 0);
    const allowed = L * g * (1 + inp.pass_through * s);
    if (x <= allowed) return 0;
    if (F > allowed) return null;
    return Math.min((x - allowed) / (x - F), 1);
  }

  // ---------- 11. Replay a scenario ----------
  function replay(inp, file, scen) {
    const T = scen.payment_week, [w1, w2] = scen.weeks;
    const lev = arr => ({ 0: 1, [w1]: 1 + arr[0], [w2]: 1 + arr[1], [T]: 1 + arr[2] });
    const S = lev(scen.usd), A = lev(scen.arabica), R = lev(scen.robusta);
    const d = inp.channel_discount, f = inp.funds_rate, c = inp.fx_holding_cost, b = inp.sellback_spread;
    const gT = growth(inp, file, T);
    const Cplan = inp.usd_amount * inp.s0 * (1 - d) * gT, Rplan = Cplan / (1 - inp.planned_margin);
    const w = inp.arabica_share, pi = inp.premium_share;
    const need = inp.contract === "fixed" ? inp.usd_amount
      : inp.usd_amount * ((1 - pi) * (w * A[T] + (1 - w) * R[T]) + pi);
    let x = S[T] / gT - 1; if (inp.pass_through_mode === "up_only") x = Math.max(x, 0);
    const Rev = Rplan * (1 + inp.pass_through * x);
    const trs = {
      now: [[0, 1]], payment_day: [], staged: [[0, 1 / 3], [w1, 1 / 3], [w2, 1 / 3]]
    };
    const out = {};
    function run(tr) {
      let cst = 0, bought = 0;
      for (const [t, q] of tr) {
        const usd = q * inp.usd_amount, years = (T - t) / 52;
        cst += usd * inp.s0 * (1 - d) * S[t] * (Math.pow(1 + f, years) + c * years);
        bought += usd;
      }
      const gap = need - bought, rT = inp.s0 * (1 - d) * S[T];
      cst += rT * (gap >= 0 ? gap : gap * (1 - b));
      const margin = 1 - cst / Rev;
      return { cost: cst / Cplan, margin, breaks: margin < inp.margin_floor };
    }
    for (const k in trs) out[k] = run(trs[k]);
    out._hedge = h => run(h > 0 ? [[0, h]] : []);
    return out;
  }

  // ---------- full run ----------
  function runAll(P, inp, scenarios, opts) {
    opts = opts || {};
    const sim = simulate(P, inp);
    const strategies = {};
    for (const s of ALL) strategies[s] = metrics(sim, inp, sim.cost(s));
    const dec = decide(strategies, sim, inp);
    const out = Object.assign({ plan_growth: sim.gT, plan_rial: sim.Cplan, planned_revenue_rial: sim.Rplan, strategies }, dec);
    out.stress_floor = {};
    for (const sc of scenarios) out.stress_floor[sc.key] = stressFloor(inp, P.file, sc);
    out.reverse_stress = {};
    for (const h of [0, 0.25, 0.5, 0.75]) out.reverse_stress["h_" + Math.round(h * 100)] = reverseStress(inp, P.file, h);
    out.scenarios = {};
    for (const sc of scenarios) out.scenarios[sc.key] = replay(inp, P.file, sc);
    out._sim = sim;
    return out;
  }

  // price-of-insurance curve: mean cost / Cplan and P(margin<floor) for h on a grid
  function insuranceCurve(sim, inp, step) {
    step = step || 0.02;
    const N = sim.N, pts = [];
    const S = sim.S, W = (sim.S.length / N);
    const F0 = Math.pow(1 + inp.funds_rate, sim.T / 52) + inp.fx_holding_cost * sim.T / 52;
    const d = inp.channel_discount, b = inp.sellback_spread;
    for (let h = 0; h <= 1 + 1e-9; h += step) {
      let sum = 0, below = 0;
      for (let p = 0; p < N; p++) {
        const usd = h * inp.usd_amount;
        const r0 = inp.s0 * (1 - d) * S[p * W], rT = inp.s0 * (1 - d) * S[p * W + sim.T];
        let cst = h > 0 ? usd * r0 * F0 : 0;
        const gap = sim.need[p] - (h > 0 ? usd : 0);
        cst += rT * (gap >= 0 ? gap : gap * (1 - b));
        sum += cst;
        if (1 - cst / sim.Rev[p] < inp.margin_floor) below++;
      }
      pts.push({ h: Math.round(h * 100) / 100, mean: sum / N / sim.Cplan, pBelow: below / N });
    }
    return pts;
  }

  const api = { prepare, growth, simulate, metrics, decide, reverseStress, stressFloor, replay, runAll, insuranceCurve, CANDIDATES, ALL };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.CoffeeEngine = api;
})(typeof window !== "undefined" ? window : this);
