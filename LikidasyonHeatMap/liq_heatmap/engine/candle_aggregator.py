"""
Multi-timeframe candle aggregator.
===================================
One incoming trade fans out into ALL four timeframe streams. Each stream
keeps a single in-progress candle and yields the previous one when a
new bucket begins.

Closed candles are returned via a callback so the engine can immediately
project them into the bin grid.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from core.timeframe import TimeframeSpec


@dataclass
class Candle:
    open_ms: int
    close_ms: int
    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    close: float = 0.0
    volume_quote: float = 0.0   # in USDT
    volume_base:  float = 0.0   # in coin
    vwap_num: float = 0.0       # Σ price·quote
    vwap_den: float = 0.0       # Σ quote
    n_trades: int = 0

    def add_trade(self, price: float, qty_base: float) -> None:
        if self.n_trades == 0:
            self.open = price
            self.high = price
            self.low  = price
        else:
            if price > self.high: self.high = price
            if price < self.low:  self.low  = price
        self.close = price
        quote = price * qty_base
        self.volume_base  += qty_base
        self.volume_quote += quote
        self.vwap_num += price * quote
        self.vwap_den += quote
        self.n_trades += 1

    @property
    def vwap(self) -> float:
        return self.vwap_num / self.vwap_den if self.vwap_den > 0 else 0.0


class SingleTFAggregator:
    """Holds one in-progress candle for one timeframe."""

    def __init__(self, tf: TimeframeSpec):
        self.tf = tf
        self.current: Candle | None = None

    def add_trade(self, ts_ms: int, price: float, qty_base: float) -> Candle | None:
        """
        Returns the just-closed candle (or None) when this trade triggers
        a candle rollover.
        """
        bucket_open = (ts_ms // (self.tf.candle_sec * 1000)) * (self.tf.candle_sec * 1000)
        bucket_close = bucket_open + self.tf.candle_sec * 1000

        closed_candle: Candle | None = None
        if self.current is None:
            self.current = Candle(open_ms=bucket_open, close_ms=bucket_close)
        elif bucket_open != self.current.open_ms:
            closed_candle = self.current
            self.current = Candle(open_ms=bucket_open, close_ms=bucket_close)

        self.current.add_trade(price, qty_base)
        return closed_candle

    def force_close(self) -> Candle | None:
        c = self.current
        self.current = None
        return c


class MultiTFAggregator:
    """Bundles N single-TF aggregators behind one trade-tick entry point."""

    def __init__(self, timeframes: list[TimeframeSpec]):
        self.aggs = {tf.label: SingleTFAggregator(tf) for tf in timeframes}

    def add_trade(self, ts_ms: int, price: float, qty_base: float):
        """Returns list[(tf_label, closed_candle)] — usually empty/1 item."""
        out = []
        for label, agg in self.aggs.items():
            closed = agg.add_trade(ts_ms, price, qty_base)
            if closed is not None:
                out.append((label, closed))
        return out
