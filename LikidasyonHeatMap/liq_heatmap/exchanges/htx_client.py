"""
Htx futures client.
═══════════════════════════════════════════════════════════════════════
⚠️  ENDPOINTS NEED LIVE VERIFICATION. Written to the same interface as the
    tested Binance/Bybit/OKX/Bitget clients, but the sandbox cannot reach
    Htx's servers, so the exact WS topics, REST paths and JSON field
    names must be checked against current Htx API docs before production.
    The STRUCTURE (stream / fetch_context / fetch_klines) is correct and
    matches the engine's expectations; fill in the parse bodies after
    verifying the live payload shapes.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time

import aiohttp

import config
from exchanges.base import Trade, LiquidationEvent, MarketContext, Kline

logger = logging.getLogger("htx")

_INTERVAL_MAP   = {"24h": "5m", "3d": "30m", "1w": "2h", "1m": "8h"}
_INTERVAL_LIMIT = {"24h": 288, "3d": 144, "1w": 84, "1m": 90}


class HtxClient:
    name = "htx"

    def __init__(self, coins: list[str]):
        self.coins = coins
        ep = config.EXCHANGE_ENDPOINTS.get("htx", {})
        self.ws_url   = ep.get("ws", "")
        self.rest_url = ep.get("rest", "")

    async def stream(self, on_trade, on_liq) -> None:
        # VERIFY: subscription message shape + topic names per Htx docs.
        if not self.ws_url:
            logger.warning("htx: no ws endpoint configured; stream disabled")
            return
        async with aiohttp.ClientSession() as session:
            while True:
                try:
                    async with session.ws_connect(self.ws_url, heartbeat=20) as ws:
                        for coin in self.coins:
                            sym = f"{coin}USDT"
                            await ws.send_str(json.dumps(
                                {"method": "subscribe", "params": [f"trades.{sym}"]}))
                        async for msg in ws:
                            if msg.type != aiohttp.WSMsgType.TEXT:
                                continue
                            # VERIFY: parse trades + liquidations from Htx payload,
                            # then call on_trade(Trade(...)) / on_liq(LiquidationEvent(...)).
                            _ = json.loads(msg.data)
                except Exception as e:
                    logger.warning("WS error: %s — reconnect 5s", e)
                    await asyncio.sleep(5)

    async def fetch_context(self, session, coin):
        # VERIFY: Htx ticker / open-interest / funding / long-short endpoints.
        # Must return MarketContext(exchange, coin, open_interest_usd, funding_rate,
        # long_ratio, mark_price, ts_ms) or None.
        logger.debug("htx: fetch_context stub for %s — verify endpoints", coin)
        return None

    async def fetch_klines(self, session, coin, tf_label):
        # VERIFY: Htx kline endpoint + row layout → list[Kline].
        return []
