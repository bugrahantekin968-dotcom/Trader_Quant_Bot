"""
Fixed 1%-slice liquidation packet.
==================================
Instead of variable-width peak zones (which overlap visually and have
unequal thickness), this produces a FIXED grid of 1%-wide slices:
    - `n_below` slices below spot   (long-liquidation dominant region)
    - `n_above` slices above spot   (short-liquidation dominant region)

Each slice reports BOTH long_liq_usd and short_liq_usd explicitly, so the
consumer (LLM / scorer) always knows which side pops at that level — no
ambiguity from overlapping bars.

This is the packet the trading bot's AI ingests to decide targets.
"""
from __future__ import annotations

import numpy as np

from core.percent_bins import PercentBins


def _calib(side_mat: np.ndarray, target: float) -> float:
    raw = float(side_mat.sum())
    if raw <= 0 or target <= 0:
        return 0.0
    return target / raw


def build_slice_packet(
    dollar_bins: PercentBins,
    current_price: float,
    total_oi_usd: float,
    long_ratio: float,
    dex_positions: list[tuple[float, float, bool]] | None = None,
    n_above: int = 20,
    n_below: int = 20,
    slice_pct: float | None = None,
    liquidatable_frac: float = 1.0,
) -> dict:
    """
    Returns a fixed-count slice packet (n_below + n_above rows).

    slice_pct: if None, derived from the grid's pct_range so the slices span
    exactly the timeframe's band (e.g. ±10% over 20 slices → 0.5% each on 24h,
    ±35% over 20 slices → 1.75% each on 1mo). Pass an explicit value to force
    a fixed width (e.g. 0.01 for literal 1% slices).
    """
    if dollar_bins is None or current_price <= 0:
        return {"current_price": current_price, "slices": []}

    if slice_pct is None:
        slice_pct = dollar_bins.pct_range / max(n_below, 1)

    # 1. Calibrate CEX dollar columns to real OI (× in-range fraction)
    frac = max(0.0, min(1.0, liquidatable_frac))
    long_scale  = _calib(dollar_bins.long_mat,  total_oi_usd * long_ratio * frac)
    short_scale = _calib(dollar_bins.short_mat, total_oi_usd * (1.0 - long_ratio) * frac)

    long_col  = dollar_bins.long_mat.sum(axis=0)  * long_scale     # USD per 0.1% cell
    short_col = dollar_bins.short_mat.sum(axis=0) * short_scale
    ticks = dollar_bins.price_ticks

    # 2. Add DEX exact dollars (already in USD, not scaled)
    if dex_positions:
        for liq_px, dollars, is_long in dex_positions:
            b = dollar_bins.price_to_bin(liq_px)
            if b < 0:
                continue
            if is_long:
                long_col[b]  += dollars
            else:
                short_col[b] += dollars

    # ── PHYSICAL CONSTRAINT ────────────────────────────────────────────────
    # Pending LONG liquidations exist ONLY below the current price; pending
    # SHORT liquidations exist ONLY above it. Anything on the wrong side has
    # already been swept (the position is dead) and must not appear as a
    # forward-looking magnet. Zero those cells.
    long_col[ticks >= current_price] = 0.0
    short_col[ticks <= current_price] = 0.0

    # 3. Aggregate 0.1% cells into fixed 1% slices, indexed by % offset from spot
    slices = []
    tot_long = tot_short = 0.0
    for i in range(-n_below, n_above):          # -20 .. +19
        pct_lo = i * slice_pct
        pct_hi = (i + 1) * slice_pct
        price_lo = current_price * (1 + pct_lo)
        price_hi = current_price * (1 + pct_hi)
        mask = (ticks >= price_lo) & (ticks < price_hi)
        l = float(long_col[mask].sum())
        s = float(short_col[mask].sum())
        tot_long  += l
        tot_short += s
        dominant = "none"
        if l > s and l > 0:   dominant = "long"
        elif s > l and s > 0: dominant = "short"
        slices.append({
            "idx": i,
            "pct_lo": round(pct_lo, 4),
            "pct_hi": round(pct_hi, 4),
            "price_lo": round(price_lo, 6),
            "price_hi": round(price_hi, 6),
            "price_mid": round((price_lo + price_hi) / 2, 6),
            "long_liq_usd":  round(l, 2),
            "short_liq_usd": round(s, 2),
            "total_usd":     round(l + s, 2),
            "dominant": dominant,
        })

    return {
        "current_price": current_price,
        "slice_pct": slice_pct,
        "n_above": n_above,
        "n_below": n_below,
        "total_long_liq_usd":  round(tot_long, 2),
        "total_short_liq_usd": round(tot_short, 2),
        "slices": slices,
    }


def top_magnets(packet: dict, k: int = 3) -> dict:
    """Extract the k biggest magnet slices above and below spot.
    A magnet above = short-liq cluster (upside target); below = long-liq (downside)."""
    slices = packet.get("slices", [])
    above = [s for s in slices if s["idx"] >= 0]
    below = [s for s in slices if s["idx"] < 0]
    # Above: rank by short_liq (the side that pops when price rises into it)
    above_sorted = sorted(above, key=lambda s: s["short_liq_usd"], reverse=True)[:k]
    # Below: rank by long_liq
    below_sorted = sorted(below, key=lambda s: s["long_liq_usd"], reverse=True)[:k]
    return {"above": above_sorted, "below": below_sorted}


def format_packet_text(packet: dict, coin: str, tf: str) -> str:
    """Human/LLM-readable 40-row table."""
    cp = packet["current_price"]
    lines = []
    lines.append(f"LIKIDASYON MIKNATIS HARITASI — {coin} {tf}  (spot ${cp:,.4f})")
    lines.append("  %offset |   fiyat aralığı        | LONG patlama | SHORT patlama")
    lines.append("  " + "-" * 66)
    # above first (high → low), then below (high → low) so it reads top-to-bottom
    above = [s for s in packet["slices"] if s["idx"] >= 0][::-1]
    below = [s for s in packet["slices"] if s["idx"] < 0][::-1]

    def fmt_usd(v):
        if v >= 1e9: return f"${v/1e9:.2f}B"
        if v >= 1e6: return f"${v/1e6:.1f}M"
        if v >= 1e3: return f"${v/1e3:.0f}k"
        return f"${v:.0f}"

    for s in above:
        lines.append(
            f"  {s['pct_hi']*100:+5.0f}%  | "
            f"{s['price_lo']:>10,.2f}-{s['price_hi']:>10,.2f} | "
            f"{'':>12} | {fmt_usd(s['short_liq_usd']):>12}"
        )
    lines.append(f"  ►►►►►  SPOT ${cp:,.4f}  ◄◄◄◄◄")
    for s in below:
        lines.append(
            f"  {s['pct_lo']*100:+5.0f}%  | "
            f"{s['price_lo']:>10,.2f}-{s['price_hi']:>10,.2f} | "
            f"{fmt_usd(s['long_liq_usd']):>12} | {'':>12}"
        )
    return "\n".join(lines)
