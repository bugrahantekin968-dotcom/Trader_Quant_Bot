"""
Liquidation recorder — ground-truth data collection for calibration.
====================================================================
Every real liquidation that prints on the exchange WS feeds (forceOrder /
allLiquidation / liquidation-orders) is appended to a JSONL file together
with a REFERENCE price (the recent swing high for long-liqs, swing low for
short-liqs). The reference lets the offline fitter back out the IMPLIED
LEVERAGE of the liquidated position:

    long  liquidated at P, came down from recent high H  → L ≈ H / (H − P)
    short liquidated at P, came up   from recent low  Lo → L ≈ Lo / (P − Lo)

Accumulating these over days yields the EMPIRICAL leverage distribution that
replaces the assumed LEVERAGE_DISTRIBUTION — making the model self-calibrating
against what the market actually does.

This recorder does the cheap part (logging). `calibration/fit_leverage.py`
does the fitting offline.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

logger = logging.getLogger("liqrec")


class LiquidationRecorder:
    def __init__(self, path: str = "./_calib/liquidations.jsonl"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._count = 0

    def record(self, coin: str, side: str, price: float, qty_usd: float,
               ref_high: float, ref_low: float, ts_ms: int,
               exchange: str = "") -> None:
        """
        side: "LONG_LIQ" or "SHORT_LIQ" (normalized).
        qty_usd: USD value of the liquidation.
        ref_high / ref_low: recent swing extremes from the engine's 24h candles.
        """
        if price <= 0 or qty_usd <= 0:
            return
        rec = {
            "ts": ts_ms, "coin": coin, "side": side,
            "price": round(price, 6), "qty_usd": round(qty_usd, 2),
            "ref_high": round(ref_high, 6), "ref_low": round(ref_low, 6),
            "exchange": exchange,
        }
        try:
            with open(self.path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            self._count += 1
            if self._count % 500 == 0:
                logger.info("recorded %d liquidations → %s", self._count, self.path)
        except Exception as e:
            logger.warning("record failed: %s", e)


# Side normalization: exchanges report the SIDE OF THE LIQUIDATION ORDER.
# A long position is liquidated by a SELL order; a short by a BUY order.
def normalize_liq_side(raw_side: str) -> str:
    s = (raw_side or "").strip().upper()
    if s in ("SELL", "S", "SELL_LONG"):
        return "LONG_LIQ"      # a forced SELL = a long getting liquidated
    if s in ("BUY", "B", "BUY_SHORT"):
        return "SHORT_LIQ"     # a forced BUY = a short getting liquidated
    # Some feeds already say long/short
    if "LONG" in s:
        return "LONG_LIQ"
    if "SHORT" in s:
        return "SHORT_LIQ"
    return "UNKNOWN"
