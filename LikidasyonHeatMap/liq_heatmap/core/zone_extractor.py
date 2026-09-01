"""
Zone extractor — turn the bin matrix into actionable liquidation zones.
======================================================================
Pipeline:
    1. Sum each price bin across the time window  →  column_mass
    2. Find local maxima (manual peak detection, no scipy dep)
    3. Expand each peak outward until mass drops below `drop_ratio` × peak
    4. Merge overlapping zones
    5. Calibrate raw mass → dollars via OI-anchored scaling
    6. Rank, filter by min-share, return top N per side

Output per zone:
    side        — "long" or "short"
    price_low   — bottom of the zone (USD)
    price_high  — top of the zone
    price_center — peak price
    dollars     — calibrated liquidation $ inside the zone
    rank        — 1..N (largest first)
    pct_from_spot — distance from current spot (signed: -%X means below)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from core.percent_bins import PercentBins


@dataclass
class Zone:
    side: str                   # "long" or "short"
    price_low: float
    price_high: float
    price_center: float
    dollars: float
    rank: int = 0
    pct_from_spot: float = 0.0

    def to_dict(self) -> dict:
        return {
            "side": self.side,
            "rank": self.rank,
            "price_low":   round(self.price_low, 4),
            "price_high":  round(self.price_high, 4),
            "price_center": round(self.price_center, 4),
            "dollars":     round(self.dollars, 2),
            "pct_from_spot": round(self.pct_from_spot, 3),
        }


# ============================================================
# Peak detection (manual, no scipy)
# ============================================================
def _find_peaks(arr: np.ndarray, min_prominence_ratio: float = 0.08) -> list[int]:
    """
    Return indices i where arr[i] > arr[i-1] AND arr[i] > arr[i+1] AND
    arr[i] > min_prominence_ratio × max(arr).
    Flat plateaus: only the leftmost cell is picked.
    """
    if arr.size < 3:
        return []
    threshold = float(arr.max()) * min_prominence_ratio
    if threshold <= 0:
        return []
    peaks = []
    for i in range(1, arr.size - 1):
        v = arr[i]
        if v <= threshold:
            continue
        # Left side strict, right side ≥ to handle plateaus
        if v > arr[i - 1] and v >= arr[i + 1]:
            peaks.append(int(i))
    return peaks


def _expand_zone(arr: np.ndarray, peak_idx: int, drop_ratio: float) -> tuple[int, int]:
    """
    Walk outward from peak_idx until mass drops below peak × drop_ratio.
    Returns (lo_idx, hi_idx) inclusive.
    """
    cutoff = float(arr[peak_idx]) * drop_ratio
    lo = peak_idx
    while lo > 0 and arr[lo - 1] >= cutoff:
        lo -= 1
    hi = peak_idx
    while hi < arr.size - 1 and arr[hi + 1] >= cutoff:
        hi += 1
    return lo, hi


def _merge_overlapping(arr: np.ndarray,
                       intervals: list[tuple[int, int, int]]) -> list[tuple[int, int, int]]:
    """
    intervals: list of (lo, hi, peak_idx). Merge if they overlap, keeping
    the STRONGEST peak (highest arr value) as the merged center.
    """
    if not intervals:
        return []
    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0]]
    for lo, hi, p in intervals[1:]:
        last_lo, last_hi, last_p = merged[-1]
        if lo <= last_hi:                          # overlap
            best_p = p if arr[p] > arr[last_p] else last_p
            merged[-1] = (last_lo, max(last_hi, hi), best_p)
        else:
            merged.append((lo, hi, p))
    return merged


# ============================================================
# Calibration: raw mass → dollars
# ============================================================
def _calibration_factor(side_mat: np.ndarray, target_dollars: float) -> float:
    """Scale the raw matrix so its total equals target_dollars."""
    raw = float(side_mat.sum())
    if raw <= 0 or target_dollars <= 0:
        return 0.0
    return target_dollars / raw


# ============================================================
# Main entry point
# ============================================================
def extract_zones(
    dollar_bins: PercentBins,
    current_price: float,
    total_oi_usd: float,
    long_ratio: float,
    max_zones_per_side: int = 5,
    min_dollar_share: float = 0.03,
    drop_ratio: float = 0.30,
    peak_prominence: float = 0.08,
    dex_positions: list[tuple[float, float, bool]] | None = None,
    liquidatable_frac: float = 1.0,
    zoom_range: float | None = None,
) -> dict[str, list[Zone]]:
    """
    Returns {"long": [Zone...], "short": [Zone...]} sorted by dollars desc.

    Calibration target per side = OI × side_ratio × liquidatable_frac.

    zoom_range: if set, only price cells within ±zoom_range of current_price are
    considered (the rest are masked). Lets every timeframe read ONE shared master
    dollar map but display its own zoom band, so the same price shows the SAME $
    across timeframes — only the visible width differs.

    dex_positions: optional (liq_price, exact_dollars, is_long) — EXACT $, added
    on top of the calibrated CEX estimate.
    """
    out = {"long": [], "short": []}
    if dollar_bins is None or current_price <= 0:
        return out

    frac = max(0.0, min(1.0, liquidatable_frac))
    long_target  = total_oi_usd * long_ratio * frac
    short_target = total_oi_usd * (1.0 - long_ratio) * frac

    long_scale  = _calibration_factor(dollar_bins.long_mat,  long_target)
    short_scale = _calibration_factor(dollar_bins.short_mat, short_target)

    ticks = dollar_bins.price_ticks
    n_bins = dollar_bins.n_price_bins

    # Pre-bin DEX exact dollars onto the same grid (added AFTER calibration)
    dex_long  = np.zeros(n_bins, dtype=np.float64)
    dex_short = np.zeros(n_bins, dtype=np.float64)
    if dex_positions:
        for liq_px, dollars, is_long in dex_positions:
            b = dollar_bins.price_to_bin(liq_px)
            if b < 0:
                continue
            if is_long:
                dex_long[b]  += dollars
            else:
                dex_short[b] += dollars

    for side, mat, scale, dex_vec in [
        ("long",  dollar_bins.long_mat,  long_scale,  dex_long),
        ("short", dollar_bins.short_mat, short_scale, dex_short),
    ]:
        # Calibrated CEX dollars per price bin + exact DEX dollars on top
        col = mat.sum(axis=0) * scale + dex_vec     # now in real USD

        # ── PHYSICAL CONSTRAINT ───────────────────────────────────────────
        # A pending LONG liquidation can only sit BELOW the current price
        # (if it were above, price would have already swept it on the way
        # down → the position is already dead). Symmetrically, a pending
        # SHORT liquidation can only sit ABOVE the current price.
        # Zero the physically-impossible (already-swept) cells.
        col = col.copy()
        if side == "long":
            col[ticks >= current_price] = 0.0
        else:  # short
            col[ticks <= current_price] = 0.0

        # ── ZOOM ──────────────────────────────────────────────────────────
        # Restrict to ±zoom_range of spot so each timeframe shows its own band
        # while reading the same shared master map (consistent $ across TFs).
        if zoom_range is not None and zoom_range > 0:
            lo = current_price * (1.0 - zoom_range)
            hi = current_price * (1.0 + zoom_range)
            col[(ticks < lo) | (ticks > hi)] = 0.0

        if col.sum() <= 0:
            continue

        # 1. Peak detection
        peaks = _find_peaks(col, min_prominence_ratio=peak_prominence)
        if not peaks:
            continue

        # 2. Zone expansion
        intervals = []
        for p in peaks:
            lo, hi = _expand_zone(col, p, drop_ratio=drop_ratio)
            intervals.append((lo, hi, p))

        # 3. Merge overlapping zones (keep strongest peak)
        merged = _merge_overlapping(col, intervals)

        # 4. Build Zone objects — col is already in USD
        candidates: list[Zone] = []
        total_dollars = float(col.sum())
        for lo, hi, p in merged:
            zone_dollars = float(col[lo:hi + 1].sum())
            if zone_dollars < min_dollar_share * total_dollars:
                continue
            half_cell = dollar_bins.pct_bucket * dollar_bins.anchor_spot / 2
            price_low    = float(ticks[lo]) - half_cell
            price_high   = float(ticks[hi]) + half_cell
            price_center = float(ticks[p])
            pct = (price_center / current_price - 1.0) * 100.0
            candidates.append(Zone(
                side=side,
                price_low=price_low,
                price_high=price_high,
                price_center=price_center,
                dollars=zone_dollars,
                pct_from_spot=pct,
            ))

        # 5. Sort and rank
        candidates.sort(key=lambda z: z.dollars, reverse=True)
        candidates = candidates[:max_zones_per_side]
        for r, z in enumerate(candidates, start=1):
            z.rank = r

        out[side] = candidates

    return out


def nearest_zone_above(zones_short: list[Zone], current_price: float,
                       min_dist_pct: float = 0.0) -> Zone | None:
    """Closest short-liq zone above current price (the upside magnet).
    min_dist_pct: ignore magnets closer than this (e.g. 0.01 = 1%) so swing
    targets sit at a tradeable distance, not right on top of spot."""
    floor = current_price * (1.0 + min_dist_pct)
    cand = [z for z in zones_short if z.price_center >= floor]
    if not cand:
        return None
    return min(cand, key=lambda z: z.price_center - current_price)


def nearest_zone_below(zones_long: list[Zone], current_price: float,
                       min_dist_pct: float = 0.0) -> Zone | None:
    """Closest long-liq zone below current price (the downside magnet)."""
    ceil = current_price * (1.0 - min_dist_pct)
    cand = [z for z in zones_long if z.price_center <= ceil]
    if not cand:
        return None
    return min(cand, key=lambda z: current_price - z.price_center)
