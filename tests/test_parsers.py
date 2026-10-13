import datetime as dt
import json

import numpy as np
import pytest

from coffee_risk.data import bonbast, ico, panel, tgju

def test_table_parser_reads_dates_and_column_order():
    body = json.dumps({"recordsTotal": 2, "data": [
        ["2,635,850", "2,631,000", "2,671,000", "2,643,050",
         '<span class="low" dir="ltr">7950</span>', "0.3%", "2026/10/11", "1405/07/19"],
        ["13,700", "13,600", "13,800", "13,750", "0", "0%", "2011/11/26", "1390/09/05"],
    ]}).encode()
    df = tgju.parse_table(body)
    assert list(df["date"]) == [dt.date(2011, 11, 26), dt.date(2026, 10, 11)]
    last = df.iloc[-1]
    assert (last["open"], last["low"], last["high"], last["close"]) == \
        (2635850.0, 2631000.0, 2671000.0, 2643050.0)
    assert last["jdate"] == "1405-07-19"


def test_bonbast_parser_converts_toman_to_rial_and_jalali_dates():
    body = json.dumps({"1391/07/19": {"sell": 3550},
                       "1391/07/18": {"sell": 3500, "buy": 3400}}).encode()
    df = bonbast.parse(body)
    assert list(df["date"]) == [dt.date(2012, 10, 9), dt.date(2012, 10, 10)]
    assert df["sell"].tolist() == [35000, 35500]
    assert df["buy"].iloc[0] == 34000 and np.isnan(df["buy"].iloc[1])


def test_history_page_parser_reads_the_printed_rows():
    html = """<table><tr><th>x</th></tr>
    <tr><td>2,635,850</td><td>2,631,000</td><td>2,671,000</td><td>2,643,050</td>
        <td><span class="low">7950</span></td><td>0.3%</td><td>2026/10/11</td><td>1405/07/19</td></tr>
    <tr><td>2,675,050</td><td>2,650,000</td><td>2,678,100</td><td>2,651,000</td>
        <td>11000</td><td>0.41%</td><td>2026/10/10</td><td>1405/07/18</td></tr>
    <tr><td>a</td><td>b</td></tr></table>""".encode()
    df = tgju.parse_page(html)
    assert list(df["date"]) == [dt.date(2026, 10, 10), dt.date(2026, 10, 11)]
    assert df["close"].tolist() == [2651000.0, 2643050.0]
    with pytest.raises(ValueError):
        tgju.parse_page(b"<html><body>nothing here</body></html>")


def test_tgju_series_refuses_a_repeated_date(monkeypatch):
    import pandas as pd
    table = pd.DataFrame({"date": [dt.date(2026, 1, 3), dt.date(2026, 1, 4), dt.date(2026, 1, 4)],
                          "close": [1.0, 2.0, 3.0]})
    monkeypatch.setattr(tgju, "load_table", lambda key: table)
    with pytest.raises(ValueError):
        tgju.series("any_key")
    monkeypatch.setattr(tgju, "load_table", lambda key: table.iloc[:2])
    assert tgju.series("any_key")["close"].tolist() == [1.0, 2.0]


def test_robusta_series_is_tgju_weekday_rows_only(monkeypatch):
    import pandas as pd
    d = lambda day: dt.date(2026, 7, day)
    #        Fri 17            Sat 18 (weekend row)  Mon 20
    tg = pd.DataFrame([(d(17), 3800.0), (d(18), 3811.0), (d(20), 3879.0)], columns=["date", "close"])
    monkeypatch.setattr(tgju, "series", lambda key: tg.copy())
    series, served = panel.robusta_series()
    assert list(series["date"]) == [d(17), d(20)]                    # no Saturday row
    assert set(series["source"]) == {"tgju"}
    assert served["weekend"].tolist() == [False, True, False]        # the row is still there, marked


# ------------------------------------------------------------------ ICO PDFs
ICO_TEXT = """ICO Indicator Prices - August 2026 (I-CIP)
In US cents/lb
Colombian Brazilian
I-CIP Other Milds Robustas
Milds Naturals
3-Aug 283.11 374.87 350.86 314.34 186.11
4-Aug 282.18 374.89 350.82 311.47 186.02
5-Aug 286.64 1,380.12 356.09 318.60 187.44
Average 283.98 376.63 352.59 314.80 186.52
High 286.64 380.12 356.09 318.60 187.44
Low 282.18 374.87 350.82 311.47 186.02
DoD Change 1.8% 2.3% 1.8% 2.4% 0.5%
"""


def test_ico_text_parser_reads_the_robustas_column():
    df, facts = ico.parse_text(ICO_TEXT)
    assert list(df["date"]) == [dt.date(2026, 8, 3), dt.date(2026, 8, 4), dt.date(2026, 8, 5)]
    assert df["robustas"].tolist() == [186.11, 186.02, 187.44]       # last of the five columns
    assert (facts["title_year"], facts["title_month"]) == (2026, 8)
    assert (facts["average"], facts["high"], facts["low"]) == (186.52, 187.44, 186.02)


def test_ico_month_passes_only_if_rows_match_the_printed_statistics():
    df, facts = ico.parse_text(ICO_TEXT)
    assert ico.validate(df, facts, 2026, 8)["ok"]
    assert not ico.validate(df, facts, 2026, 7)["ok"]                # the title names another month
    off = dict(facts, average=facts["average"] + 0.09)               # printed average off by 0.09
    res = ico.validate(df, off, 2026, 8)
    assert not res["ok"] and "average" in res["problems"]
    near = dict(facts, average=facts["average"] + 0.005)             # inside the 0.01 tolerance
    assert ico.validate(df, near, 2026, 8)["ok"]
    assert not ico.validate(df.iloc[:-1], facts, 2026, 8)["ok"]      # a lost row changes mean and max


def test_ico_parser_refuses_a_page_without_rows():
    with pytest.raises(ValueError):
        ico.parse_text("ICO Indicator Prices - August 2026 (I-CIP)\nAverage 1 2 3 4 5\n")
    with pytest.raises(ValueError):
        ico.parse_text("nothing here")


def test_ico_candidate_names_and_months():
    urls = ico.candidate_urls(2025, 9)
    assert urls[0].endswith("/I-CIP_09.2025.pdf") and urls[1].endswith("/I-CIP_September2025.pdf")
    assert len(urls) == 4 and urls[2].startswith("https://icocoffee.org/")
    m = ico.months()
    assert m[0] == (2025, 1) and m[-1] == (2026, 9) and len(m) == 21


def test_ico_exception_is_tied_to_one_file_and_drops_only_its_rows(monkeypatch, tmp_path):
    """D-021: a failing month is used only through a recorded exception for
    exactly that file; the validation rule itself does not change."""
    import hashlib
    from coffee_risk.data import http
    bad = ICO_TEXT.replace("Average 283.98 376.63 352.59 314.80 186.52",
                           "Average 283.98 376.63 352.59 314.80 186.61")        # printed average off by 0.09
    files = {"I-CIP_2026-08.pdf": bad.encode()}
    monkeypatch.setattr(ico, "months", lambda: [(2026, 8)])
    monkeypatch.setattr(ico, "parse", lambda body: ico.parse_text(body.decode()))
    for name, body in files.items():
        (tmp_path / name).write_bytes(body)
    monkeypatch.setattr(http, "raw_path", lambda source, file: tmp_path / file)

    monkeypatch.setattr(ico, "EXCEPTIONS", {})
    rep = ico.report().iloc[0]
    assert not rep["ok"] and not rep["used"] and ico.load().empty                 # the rule: not used

    sha = hashlib.sha256(files["I-CIP_2026-08.pdf"]).hexdigest()
    exc = {(2026, 8): {"decision": "D-999", "sha256": sha, "drop": (dt.date(2026, 8, 4),)}}
    monkeypatch.setattr(ico, "EXCEPTIONS", exc)
    rep = ico.report().iloc[0]
    assert not rep["ok"] and rep["used"] and rep["exception"] == "D-999" and rep["rows_dropped"] == 1
    assert list(ico.load()["date"]) == [dt.date(2026, 8, 3), dt.date(2026, 8, 5)]    # the row is set to missing

    (tmp_path / "I-CIP_2026-08.pdf").write_bytes(bad.encode() + b"\n")              # the file changes
    rep = ico.report().iloc[0]
    assert not rep["used"] and rep["exception"] == "" and ico.load().empty           # exception no longer applies

    (tmp_path / "I-CIP_2026-08.pdf").write_bytes(ICO_TEXT.encode())                  # a corrected file
    rep = ico.report().iloc[0]
    assert rep["ok"] and rep["used"] and rep["rows_dropped"] == 0 and len(ico.load()) == 3


def test_the_recorded_ico_exception_is_the_one_decided():
    exc = ico.EXCEPTIONS[(2026, 7)]
    assert exc["decision"] == "D-021" and exc["drop"] == (dt.date(2026, 7, 17),)
    assert list(ico.EXCEPTIONS) == [(2026, 7)]                                       # and the only one

