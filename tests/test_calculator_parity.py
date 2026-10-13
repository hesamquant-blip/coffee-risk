"""The calculator's JavaScript engine against parity.json.

`calculator/test_parity.js` runs `calculator/engine.js` on the kit in
data/export/ and compares every value with parity.json (tolerances of
docs/calculator_spec.md, section 12). It always exits 0 and reports the
number of failures in its last lines, so both are checked here.
"""
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
KIT = ROOT / "data" / "export"
KIT_FILES = ("calculator_paths.json", "scenarios.json", "parity.json")


def test_javascript_engine_reproduces_parity():
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not on PATH")
    missing = [name for name in KIT_FILES if not (KIT / name).exists()]
    if missing:                                    # data/ is never committed (D-006): build the kit first
        pytest.skip("calculator kit not built, run scripts/d6_kit.py (missing: " + ", ".join(missing) + ")")
    run = subprocess.run([node, str(ROOT / "calculator" / "test_parity.js")], capture_output=True, text=True,
                         cwd=ROOT, timeout=900)
    out = run.stdout + run.stderr
    assert run.returncode == 0, out
    assert "failed 0" in out and re.search(r"\bfailed 0\b", out), out
    checked = re.search(r"checked (\d+) values", out)
    assert checked and int(checked.group(1)) > 0, out
