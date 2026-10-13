#!/usr/bin/env python
"""Write the calculator kit to data/export/ (D6-prep).

    calculator_paths.json   20,000 sampled paths x 13 weeks, the sample of the reports (D-052)
    scenarios.json          S1..S7 as aggregates
    parity.json             twelve input cases with the results the page must reproduce
    calculator_spec.md      a copy of docs/calculator_spec.md
    kit_summary.md          sizes and the checks printed below

Nothing under data/ is committed. The page is built by the author from
calculator_spec.md and tested against parity.json.

    .venv/bin/python scripts/d6_kit.py
"""
from __future__ import annotations

import gzip
import json
import shutil

import numpy as np

from coffee_risk import paths
from coffee_risk.data import fred, panel, weekly
from coffee_risk.engine import calculator as calc
from coffee_risk.engine import decision as dec
from coffee_risk.engine import export, settings
from coffee_risk.engine import simulator as sim
from coffee_risk.engine.inputs import Inputs

OUT = export.EXPORT_DIR
SMALL_KIT = 5_000             # the size of the first kit, kept for the comparison of D-052
SPEC = paths.ROOT / "docs" / "calculator_spec.md"


def build() -> str:
    wk, daily, imf = weekly.load(), panel.load(), fred.load()
    as_of = wk.index[-1]
    base = Inputs.from_data(as_of, wk, daily, imf)
    n, seed = settings.EXPORT_PATHS, settings.REPORT_SEED

    files = {}
    files["calculator_paths.json"] = export.write_json(
        export.paths_kit(as_of, n, seed, options=base, weekly=wk, daily=daily, imf=imf), OUT / "calculator_paths.json")
    files["scenarios.json"] = export.write_json(export.scenarios_kit(wk), OUT / "scenarios.json")
    # parity is computed on what is on disk, not on what is in memory
    kit, scen = export.read_json(OUT / "calculator_paths.json"), export.read_json(OUT / "scenarios.json")
    parity = export.parity_kit(kit, scen)
    files["parity.json"] = export.write_json(parity, OUT / "parity.json")
    shutil.copyfile(SPEC, OUT / "calculator_spec.md")
    assert export.parity_kit(export.read_json(OUT / "calculator_paths.json"),
                             export.read_json(OUT / "scenarios.json")) == export.read_json(OUT / "parity.json")

    # ---- checks
    engine = sim.simulate(as_of, n, kit["horizon"], seed, base, wk)
    page = calc.kit_paths(kit, calc.kit_inputs(kit))
    gap = {name: float(np.abs(getattr(page, name) / getattr(engine, name) - 1).max())
           for name in ("usd", "arabica", "robusta")}
    by = {c["name"]: c["outputs"] for c in parity["cases"]}

    # The reports' numbers: the engine itself, on its own unrounded paths (20,000 paths,
    # the reports' seed), for the same twelve cases. The page works on the file.
    assert n == settings.REPORT_PATHS, "D-052: the page and the reports use the same sample"
    rows, differences = [], []
    for name, changes in export.CASES.items():
        i = calc.kit_inputs(kit, **changes)
        r = dec.run(sim.simulate(as_of, settings.REPORT_PATHS, kit["horizon"], seed, i, wk), i)
        p_out = by[name]
        chosen = {"staged_3": "staged"}.get(r.decision.chosen, r.decision.chosen)
        chosen = chosen.replace("partial_", "h_") if chosen else chosen
        same = (p_out["min_hedge"] == r.min_hedge and p_out["chosen"] == chosen
                and abs(p_out["best_probability"] - r.decision.best_probability) < 0.5 / n)
        rows.append((name, p_out, r, chosen, same))
        if not same:
            differences.append(name)

    def share(x):
        return "none" if x is None else f"{x * 100:.0f}%"

    L = ["# Calculator kit\n",
         f"Decision date {kit['as_of']}; {kit['n_paths']:,} paths x {kit['horizon']} weeks; seed {kit['seed']}; "
         f"s0 = {kit['s0']:,.0f} rial; market rate {kit['market_rate'] * 100:.1f}%; weekly USD volatility "
         f"{kit['usd_vol_weekly'] * 100:.2f}%; premium arabica {kit['premium']['arabica'] * 100:.1f}%, robusta "
         f"{kit['premium']['robusta'] * 100:.1f}%; past-only mean weekly USD return "
         f"{kit['hist_mean_weekly_usd_return'] * 100:.3f}%.\n",
         "## Files\n", "| File | Size | Gzipped |", "|---|---|---|"]
    for name, f in files.items():
        L.append(f"| `{name}` | {f['bytes'] / 1e3:,.0f} kB | {f['gzip_bytes'] / 1e3:,.0f} kB |")
    L.append(f"| `calculator_spec.md` | {(OUT / 'calculator_spec.md').stat().st_size / 1e3:,.0f} kB | |")
    L += ["", "## Checks\n",
          "- `parity.json` was computed from the two JSON files as written to disk and recomputes to the same "
          "content.",
          f"- Paths rebuilt from the file against the engine's own paths (same seed): largest relative "
          f"difference USD {gap['usd']:.1e}, arabica {gap['arabica']:.1e}, robusta {gap['robusta']:.1e} "
          "(the rounding of the file to 5 decimals).",
          f"- Scenarios: {len(scen['scenarios'])}; cases in parity: {len(parity['cases'])}.", "",
          "## The page against the reports (D-052)\n",
          f"Both now use the same {n:,} paths: the page reads them from the file (rounded to 5 decimals), the "
          "reports take them from the engine. \"Page\" is `parity.json`; \"reports\" is the engine run on the "
          "same inputs.\n"]
    if differences:
        L.append(f"**They do not match in {len(differences)} of {len(rows)} cases: " + ", ".join(differences)
                 + ".** The rows are marked below.\n")
    else:
        L.append(f"**They match exactly in all {len(rows)} cases:** the same minimum hedge ratio, the same chosen "
                 f"strategy, and the same best probability to the last path (1 in {n:,}).\n")
    L += ["| Case | Minimum hedge ratio: page | reports | Chosen: page | reports | Best P(margin >= floor): page | "
          "reports | Same |", "|---|---|---|---|---|---|---|---|"]
    for name, p_out, r, chosen, same in rows:
        L.append(f"| {name} | {share(p_out['min_hedge'])} | {share(r.min_hedge)} | {p_out['chosen'] or 'none'} | "
                 f"{chosen or 'none'} | {p_out['best_probability'] * 100:.3f}% | "
                 f"{r.decision.best_probability * 100:.3f}% | {'yes' if same else '**no**'} |")
    # ---- what D-052 bought and what it costs: the 5,000-path kit, built in memory only
    small = json.loads(json.dumps(export.paths_kit(as_of, SMALL_KIT, seed, options=base, weekly=wk, daily=daily,
                                                   imf=imf)))
    small_body = json.dumps(small, separators=(",", ":")).encode("ascii")
    small_by = {c["name"]: c["outputs"] for c in export.parity_kit(small, scen)["cases"]}
    moved = {name: abs(small_by[name]["min_hedge"] - by[name]["min_hedge"]) for name in by
             if small_by[name]["min_hedge"] is not None and by[name]["min_hedge"] is not None}
    L += ["", f"## What D-052 changed: {SMALL_KIT:,} paths before, {n:,} now\n",
          f"| | {SMALL_KIT:,} paths | {n:,} paths |", "|---|---|---|",
          f"| `calculator_paths.json` | {len(small_body) / 1e3:,.0f} kB | {files['calculator_paths.json']['bytes'] / 1e3:,.0f} kB |",
          f"| gzipped | {len(gzip.compress(small_body, 9)) / 1e3:,.0f} kB | "
          f"{files['calculator_paths.json']['gzip_bytes'] / 1e3:,.0f} kB |"]
    for name in moved:
        L.append(f"| minimum hedge ratio, {name} | {share(small_by[name]['min_hedge'])} | {share(by[name]['min_hedge'])} |")
    L += ["", f"Largest move of the minimum hedge ratio: {max(moved.values()) * 100:.0f} points "
              f"({max(moved, key=moved.get)}). The page's work per recalculation grows with the number of paths, "
              f"about {n / SMALL_KIT:.0f} times."]
    default = calc.default_stress_scenario(scen["scenarios"])
    L += ["", "## Stress floor in the base case\n",
          "The page's default (D-053) is the largest real 13-week USD jump seen up to the decision date: the "
          "non-hypothetical scenario with payment week 13 and the largest USD change.\n",
          "| Scenario | USD change at the payment week | Share that survives it | Default |", "|---|---|---|---|"]
    for s in scen["scenarios"]:
        if s["key"] in by["base"]["stress_floor"]:
            L.append(f"| {s['key']} {s['label']} | {s['usd'][-1] * 100:+.0f}% | "
                     f"{share(by['base']['stress_floor'][s['key']])} | {'yes' if s['key'] == default['key'] else ''} |")
    text = "\n".join(L) + "\n"
    (OUT / "kit_summary.md").write_text(text, encoding="utf-8")
    return text


def main() -> None:
    print(build())


if __name__ == "__main__":
    main()
