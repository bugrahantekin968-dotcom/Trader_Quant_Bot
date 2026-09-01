"""
Exchange registry.
==================
Single source of truth for which exchanges are wired in. The engine
walks this dict at startup.

To add a new venue:
  1. Implement a client following the BinanceClient pattern (stream +
     fetch_context + fetch_klines).
  2. Add an entry to REGISTRY below with its tier.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import config


class Tier(str, Enum):
    DEX_EXACT = "tier_1_dex_exact"     # per-position visibility (Hyperliquid, dYdX v4, GMX…)
    CEX_AGG   = "tier_2_cex_aggregate" # aggregate-only public data (Binance, Bybit…)


@dataclass
class ExchangeSpec:
    name: str
    tier: Tier


REGISTRY: dict[str, ExchangeSpec] = {
    "hyperliquid": ExchangeSpec("Hyperliquid",  Tier.DEX_EXACT),
    "binance":     ExchangeSpec("Binance USDS", Tier.CEX_AGG),
    "bybit":       ExchangeSpec("Bybit Linear", Tier.CEX_AGG),
    "okx":         ExchangeSpec("OKX Swap",     Tier.CEX_AGG),
    "bitget":      ExchangeSpec("Bitget USDT",  Tier.CEX_AGG),
    # New venues — structurally wired, endpoints need live verification.
    "gate":        ExchangeSpec("Gate USDT",    Tier.CEX_AGG),
    "mexc":        ExchangeSpec("MEXC Futures",  Tier.CEX_AGG),
    "htx":         ExchangeSpec("HTX Linear",   Tier.CEX_AGG),
    "kraken":      ExchangeSpec("Kraken Fut",   Tier.CEX_AGG),
}

# Tested-and-verified venues actually started by default. The 4 new ones are
# included in the registry for documentation but only spun up when
# include_unverified=True, so an unverified endpoint can't silently break a
# live run. Flip them on after verifying each client's endpoints.
VERIFIED = {"hyperliquid", "binance", "bybit", "okx", "bitget"}


def build_clients(coins: list[str], include_unverified: bool = False) -> dict:
    """Instantiate every registered exchange. Returns {name: client}.

    By default only VERIFIED venues are started. Pass include_unverified=True
    to also spin up Gate/MEXC/HTX/Kraken once their endpoints are confirmed.
    """
    from exchanges.binance_client     import BinanceClient
    from exchanges.bybit_client       import BybitClient
    from exchanges.okx_client         import OKXClient
    from exchanges.bitget_client      import BitgetClient
    from exchanges.hyperliquid_client import HyperliquidClient

    clients = {
        "hyperliquid": HyperliquidClient(coins),
        "binance":     BinanceClient(coins),
        "bybit":       BybitClient(coins),
        "okx":         OKXClient(coins),
        "bitget":      BitgetClient(coins),
    }
    if include_unverified:
        from exchanges.gate_client   import GateClient
        from exchanges.mexc_client   import MexcClient
        from exchanges.htx_client    import HtxClient
        from exchanges.kraken_client import KrakenClient
        clients.update({
            "gate":   GateClient(coins),
            "mexc":   MexcClient(coins),
            "htx":    HtxClient(coins),
            "kraken": KrakenClient(coins),
        })
    return clients
