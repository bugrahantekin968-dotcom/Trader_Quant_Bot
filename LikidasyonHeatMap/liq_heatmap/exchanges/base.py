"""
Exchange base layer.
====================
Shared dataclasses + abstract client interface. Every CEX client must
expose:
  - async stream(on_trade, on_liq)            → live WS feed
  - async fetch_context(session, coin)         → MarketContext
  - async fetch_klines(session, coin, tf)      → list[Kline]   (for backfill)
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Trade:
    exchange: str
    coin: str            # bare coin name: "BTC", "ETH", "XRP"
    price: float
    qty: float           # in base units (BTC), NOT quote
    ts_ms: int
    is_buyer_maker: bool


@dataclass
class LiquidationEvent:
    exchange: str
    coin: str
    side: str            # "BUY"=>short liq, "SELL"=>long liq (Binance convention)
    price: float
    qty: float
    ts_ms: int


@dataclass
class MarketContext:
    """Snapshot of REST-derived per-(exchange, coin) metrics."""
    exchange: str
    coin: str
    open_interest_usd: float
    funding_rate: float
    long_ratio: float
    mark_price: float
    ts_ms: int


@dataclass
class Kline:
    """Historical candle used during backfill."""
    exchange: str
    coin: str
    open_ms: int
    close_ms: int
    open: float
    high: float
    low: float
    close: float
    volume_base: float       # in BTC etc.
    volume_quote: float      # in USDT
