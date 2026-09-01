"""
Exact liquidation price formulas — used when we KNOW (entry, size, collateral).
==============================================================================
Equity at price p for a LONG of `size` opened at `entry`:
    equity(p) = collateral + (p - entry) · size
Liquidation when equity = MMR · |notional at p|:
    collateral + (liq - entry) · size = MMR · size · liq
    →  liq_long = (size · entry − collateral) / (size · (1 − MMR))
                = entry · (1 − 1/L) / (1 − MMR)         where L = size·entry/collateral

For SHORT (symmetric):
    →  liq_short = (collateral + size · entry) / (size · (1 + MMR))
                 = entry · (1 + 1/L) / (1 + MMR)

These are used by the Hyperliquid tracker when on-chain liquidationPx is
unavailable yet (between clearinghouseState refreshes).
"""
from __future__ import annotations


def exact_liq_long(entry: float, size: float, collateral: float, mmr: float) -> float:
    if size <= 0 or entry <= 0 or collateral <= 0:
        return 0.0
    notional = size * entry
    denom = size * (1.0 - mmr)
    if denom <= 0:
        return 0.0
    return max((notional - collateral) / denom, 0.0)


def exact_liq_short(entry: float, size: float, collateral: float, mmr: float) -> float:
    if size <= 0 or entry <= 0 or collateral <= 0:
        return 0.0
    return (collateral + size * entry) / (size * (1.0 + mmr))


def implied_leverage(entry: float, size: float, collateral: float) -> float:
    if collateral <= 0 or size <= 0 or entry <= 0:
        return 0.0
    return (size * entry) / collateral


def liq_from_known_position(entry, size, collateral, is_long, mmr):
    L = implied_leverage(entry, size, collateral)
    if is_long:
        return exact_liq_long(entry, size, collateral, mmr), L
    return exact_liq_short(entry, size, collateral, mmr), L
