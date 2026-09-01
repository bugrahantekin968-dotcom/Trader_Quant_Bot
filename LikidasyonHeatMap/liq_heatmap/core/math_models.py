"""
Core mathematical models.
=========================
Liquidation formulas, time decay, leverage fan-out, OI weighting.
All functions are pure and NumPy-vectorized.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np


# ---------------------------------------------------------------------------
# 1. Liquidation prices  (exact isolated-margin formula)
# ---------------------------------------------------------------------------
# Derivation (long): liquidation when equity = MMR · notional
#   collateral/L·... → liq = entry · (1 − 1/L) / (1 − MMR)
# This is the exact isolated-margin formula, matching core/exact_liq.py and
# the model CoinGlass uses. The earlier `entry·(1−1/L+MMR)` was a first-order
# approximation that drifted ~0.04% at low leverage.
def liq_price_long(entry, leverage, mmr):
    """entry × (1 − 1/L) / (1 − MMR). Element-wise on NumPy arrays."""
    return entry * (1.0 - 1.0 / leverage) / (1.0 - mmr)


def liq_price_short(entry, leverage, mmr):
    """entry × (1 + 1/L) / (1 + MMR)."""
    return entry * (1.0 + 1.0 / leverage) / (1.0 + mmr)


# ---------------------------------------------------------------------------
# 2. Time decay  w = log(1+V) · exp(-λ·Δt)
# ---------------------------------------------------------------------------
def decay_weight(volume, age_hours, lam: float):
    """
    Logarithmic + exponential decay.

    - log1p(V) tames whale-spike candles so a single $100M print does not
      drown out the rest of the map.
    - exp(-λ·Δt) gradually erases stale positions; calibrate λ per timeframe
      so half-life ≈ 10% of window.
    """
    v = np.maximum(np.asarray(volume, dtype=np.float64), 0.0)
    return np.log1p(v) * np.exp(-lam * np.asarray(age_hours, dtype=np.float64))


# ---------------------------------------------------------------------------
# 3. Long / Short split  (exchange-reported account ratio)
# ---------------------------------------------------------------------------
def split_long_short(weighted_volume, long_ratio: float):
    lr = float(np.clip(long_ratio, 0.0, 1.0))
    return weighted_volume * lr, weighted_volume * (1.0 - lr)


# ---------------------------------------------------------------------------
# 4. Cross-exchange normalization  (OI-weighted)
# ---------------------------------------------------------------------------
def oi_weights(open_interest: dict[str, float]) -> dict[str, float]:
    total = sum(open_interest.values())
    if total <= 0:
        return {k: 0.0 for k in open_interest}
    return {k: v / total for k, v in open_interest.items()}


# ---------------------------------------------------------------------------
# 5. Projection: (entry, weighted_volume) → liq cells per leverage tier
# ---------------------------------------------------------------------------
def project_liquidations(
    entry_price: float,
    weighted_volume_long: float,
    weighted_volume_short: float,
    leverage_buckets: Sequence[int],
    leverage_distribution: dict,            # {leverage: fraction}
    mmr: float,
):
    """
    For one candle, fan out into liquidation cells across every leverage tier.

    Returns (long_prices, long_weights, short_prices, short_weights), each
    an array aligned to leverage_buckets.
    """
    L = np.asarray(leverage_buckets, dtype=np.float64)
    dist = np.asarray(
        [leverage_distribution[int(l)] for l in L], dtype=np.float64
    )

    long_prices  = liq_price_long(entry_price,  L, mmr)
    short_prices = liq_price_short(entry_price, L, mmr)

    long_weights  = weighted_volume_long  * dist
    short_weights = weighted_volume_short * dist

    return long_prices, long_weights, short_prices, short_weights
