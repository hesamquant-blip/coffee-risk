import datetime as dt

import pytest

from coffee_risk import dates, units


def _ts(y, m, d, hh=0, mm=0):
    return dt.datetime(y, m, d, hh, mm, tzinfo=dt.timezone.utc).timestamp()


def test_midnight_utc_bar_is_the_same_tehran_date():
    # TGJU stamps daily bars at 00:00 UTC = 03:30 in Tehran, same calendar day
    assert dates.unix_to_tehran_date(1322265600) == dt.date(2011, 11, 26)


def test_timestamp_just_after_tehran_midnight_winter():
    # 20:45 UTC = 00:15 next day in Tehran (UTC+3:30)
    assert dates.unix_to_tehran_date(_ts(2024, 1, 10, 20, 45)) == dt.date(2024, 1, 11)


def test_timestamp_just_before_tehran_midnight_winter():
    # 20:15 UTC = 23:45 same day in Tehran
    assert dates.unix_to_tehran_date(_ts(2024, 1, 10, 20, 15)) == dt.date(2024, 1, 10)


def test_daylight_saving_before_2022_moves_midnight():
    # Summer 2020: Tehran was UTC+4:30, so 19:45 UTC is already the next day.
    assert dates.unix_to_tehran_date(_ts(2020, 6, 1, 19, 45)) == dt.date(2020, 6, 2)
    # Summer 2024: no daylight saving any more (UTC+3:30), still the same day.
    assert dates.unix_to_tehran_date(_ts(2024, 6, 1, 19, 45)) == dt.date(2024, 6, 1)


def test_jalali_round_trip():
    assert dates.to_jalali(dt.date(2026, 2, 28)) == "1404-12-09"
    assert dates.parse_jalali("1391/07/18") == dt.date(2012, 10, 9)
    assert dates.jalali_year(dt.date(2026, 3, 21)) == 1405


def test_unit_conversions():
    assert units.cents_lb_to_usd_kg(100.0) == pytest.approx(2.20462)
    # definition check: 1 lb = 0.45359237 kg
    assert units.cents_lb_to_usd_kg(250.0) == pytest.approx(2.5 / 0.45359237, rel=1e-5)
    assert units.usd_tonne_to_usd_kg(4080.0) == pytest.approx(4.08)
    assert units.toman_to_rial(26_000) == 260_000
    # the spec's cross-check: 185 US cents/lb is about USD 4,080 per tonne
    assert units.cents_lb_to_usd_kg(185.0) * 1000 == pytest.approx(4078.5, abs=1.0)
