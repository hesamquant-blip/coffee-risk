"""Assemble the calculator page: template + engine.js + the kit's JSON files."""
import json, pathlib, sys

d = pathlib.Path(__file__).parent
data = d.parent / "data" / "export"
paths_file = data / (sys.argv[1] if len(sys.argv) > 1 else "calculator_paths.json")
tpl = (d / "template.html").read_text(encoding="utf-8")
engine = (d / "engine.js").read_text(encoding="utf-8")
paths = json.loads(paths_file.read_text())
scen = json.loads((data / "scenarios.json").read_text())
for s in (json.dumps(paths, separators=(",", ":")), json.dumps(scen, separators=(",", ":"))):
    assert "</script" not in s
page = (tpl.replace("__ENGINE__", engine)
           .replace("__PATHS__", json.dumps(paths, separators=(",", ":")))
           .replace("__SCEN__", json.dumps(scen, separators=(",", ":"), ensure_ascii=False)))
out = data / "coffee-fx-calculator.html"
out.write_text(page, encoding="utf-8")
# local preview copy with a minimal skeleton (the Artifact host adds its own)
(data / "preview.html").write_text(
    '<!doctype html><html lang="fa" dir="rtl"><head><meta charset="utf-8">'
    '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">'
    '<style>body{margin:0}</style></head><body>' + page + "</body></html>", encoding="utf-8")
print(out, round(out.stat().st_size / 1e6, 2), "MB")
