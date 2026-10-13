# Calculator page

Single-page web calculator (Persian, RTL). It runs the engine formulas of `docs/calculator_spec.md` in JavaScript on the exported simulated paths.

- `engine.js`: the formulas, line by line from the spec. Browser global `CoffeeEngine`, or `require` in Node.
- `template.html`: the page (inputs, answer, tables, charts). `build.py` fills in `engine.js` and the files of `data/export/`.
- `test_parity.js`: `node calculator/test_parity.js` checks the engine against `data/export/parity.json` (spec tolerances).

Build: `python3 calculator/build.py` writes `data/export/coffee-fx-calculator.html` (not in git: it embeds the simulated paths) and a local `preview.html`.

Written with Claude Code from the spec. The live page: https://claude.ai/artifact/9RL7fyU573a7rNwFvT4ko5
