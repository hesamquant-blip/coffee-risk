"""Unit conversions. FX is always in rial (IRR), never toman."""
from __future__ import annotations

# 1 lb = 0.45359237 kg, so 1 US cent/lb = 0.01 / 0.45359237 USD/kg
CENTS_PER_LB_TO_USD_PER_KG = 0.0220462
KG_PER_TONNE = 1000.0
RIAL_PER_TOMAN = 10.0


def cents_lb_to_usd_kg(x):
    """US cents per pound -> USD per kilogram (arabica, IMF indicators)."""
    return x * CENTS_PER_LB_TO_USD_PER_KG


def usd_tonne_to_usd_kg(x):
    """USD per metric tonne -> USD per kilogram (robusta, ICE London)."""
    return x / KG_PER_TONNE


def toman_to_rial(x):
    return x * RIAL_PER_TOMAN
