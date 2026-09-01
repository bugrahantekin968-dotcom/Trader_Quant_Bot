"""
Timeframe specs.
================
Encapsulates one heatmap "view" (24h, 3d, 1w, 1m). The engine maintains
one bin grid per (symbol, timeframe) pair.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class TimeframeSpec:
    label: str            # "24h"
    window_sec: int       # 86400
    candle_sec: int       # 360
    n_candles: int        # 240
    lambda_per_hour: float
    pct_range: float = 0.20   # ± price band for this timeframe's grid

    @property
    def half_life_hours(self) -> float:
        if self.lambda_per_hour <= 0:
            return float("inf")
        return math.log(2.0) / self.lambda_per_hour

    @property
    def window_hours(self) -> float:
        return self.window_sec / 3600.0

    def candle_index_for_ts(self, ts_ms: int) -> int:
        """Round to candle bucket open-time."""
        return (ts_ms // (self.candle_sec * 1000)) * (self.candle_sec * 1000)


def load_timeframes(raw: list[tuple]) -> list[TimeframeSpec]:
    out = []
    for row in raw:
        if len(row) == 6:
            l, w, c, n, lam, rng = row
        else:                       # backward-compat: 5-field rows
            l, w, c, n, lam = row
            rng = 0.20
        out.append(TimeframeSpec(label=l, window_sec=w, candle_sec=c,
                                 n_candles=n, lambda_per_hour=lam, pct_range=rng))
    return out
