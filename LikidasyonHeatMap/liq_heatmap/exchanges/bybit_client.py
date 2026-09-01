"""
Bybit v5 USDT-perpetual client.
================================
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Callable

import aiohttp

import config
from exchanges.base import Trade, LiquidationEvent, MarketContext, Kline

logger = logging.getLogger("bybit")

_INTERVAL_MAP = {"24h": "5", "3d": "30", "1w": "120", "1m": "240"}  # Bybit uses minutes as strings
_INTERVAL_LIMIT = {"24h": 288, "3d": 144, "1w": 84, "1m": 180}


class BybitClient:
    name = "bybit"

    def __init__(self, coins: list[str]):
        self.coins = coins
        self.ws_url   = config.EXCHANGE_ENDPOINTS["bybit"]["ws"]
        self.rest_url = config.EXCHANGE_ENDPOINTS["bybit"]["rest"]

    async def stream(self, on_trade, on_liq) -> None:
        args = []
        for coin in self.coins:
            sym = config.symbol_for("bybit", coin)
            args.append(f"publicTrade.{sym}")
            args.append(f"allLiquidation.{sym}")
        sub = {"op": "subscribe", "args": args}

        async with aiohttp.ClientSession() as session:
            while True:
                try:
                    async with session.ws_connect(self.ws_url, heartbeat=20) as ws:
                        await ws.send_str(json.dumps(sub))
                        logger.info("WS connected, subs=%d", len(args))
                        async for msg in ws:
                            if msg.type != aiohttp.WSMsgType.TEXT:
                                continue
                            data = json.loads(msg.data)
                            topic = data.get("topic", "")
                            if topic.startswith("publicTrade."):
                                for t in data.get("data", []):
                                    coin = t["s"].replace("USDT", "")
                                    on_trade(Trade(
                                        exchange="bybit", coin=coin,
                                        price=float(t["p"]),
                                        qty=float(t["v"]),
                                        ts_ms=int(t["T"]),
                                        is_buyer_maker=(t["S"] == "Sell"),
                                    ))
                            elif topic.startswith("allLiquidation."):
                                # Bybit v5 publishes batched liquidations
                                rows = data.get("data", [])
                                if isinstance(rows, dict):
                                    rows = [rows]
                                for d in rows:
                                    coin = d.get("symbol", "").replace("USDT", "")
                                    on_liq(LiquidationEvent(
                                        exchange="bybit", coin=coin,
                                        side=str(d.get("side", "")).upper(),
                                        price=float(d.get("price", 0)),
                                        qty=float(d.get("size", 0)),
                                        ts_ms=int(d.get("updatedTime", time.time() * 1000)),
                                    ))
                except Exception as e:
                    logger.warning("WS error: %s — reconnect 5s", e)
                    await asyncio.sleep(5)

    async def fetch_context(self, session, coin):
        sym = config.symbol_for("bybit", coin)
        tick_u  = f"{self.rest_url}/v5/market/tickers?category=linear&symbol={sym}"
        ratio_u = (f"{self.rest_url}/v5/market/account-ratio"
                   f"?category=linear&symbol={sym}&period=5min&limit=1")
        try:
            tick_j, ratio_j = await asyncio.gather(
                self._json(session, tick_u),
                self._json(session, ratio_u),
            )
            row = tick_j["result"]["list"][0]
            oi_contracts = float(row.get("openInterest", 0))
            mark         = float(row.get("markPrice", 0))
            funding      = float(row.get("fundingRate", 0))

            long_ratio = 0.5
            try:
                r0 = ratio_j["result"]["list"][0]
                buy  = float(r0.get("buyRatio", 0.5))
                sell = float(r0.get("sellRatio", 0.5))
                if buy + sell > 0:
                    long_ratio = buy / (buy + sell)
            except (KeyError, IndexError, TypeError):
                pass

            return MarketContext(
                exchange="bybit", coin=coin,
                open_interest_usd=oi_contracts * mark,
                funding_rate=funding,
                long_ratio=long_ratio,
                mark_price=mark,
                ts_ms=int(time.time() * 1000),
            )
        except Exception as e:
            logger.warning("ctx %s failed: %s", coin, e)
            return None

    async def fetch_klines(self, session, coin, tf_label):
        interval = _INTERVAL_MAP.get(tf_label)
        limit    = _INTERVAL_LIMIT.get(tf_label, 90)
        if not interval:
            return []
        sym = config.symbol_for("bybit", coin)
        url = f"{self.rest_url}/v5/market/kline?category=linear&symbol={sym}&interval={interval}&limit={limit}"
        try:
            raw = await self._json(session, url)
            out = []
            for r in raw.get("result", {}).get("list", []):
                # Bybit returns rows oldest-to-newest? Actually newest-first; reverse
                open_ms = int(r[0])
                o,h,l,c = float(r[1]), float(r[2]), float(r[3]), float(r[4])
                vol_base  = float(r[5])
                vol_quote = float(r[6])
                close_ms  = open_ms + int(interval) * 60_000
                out.append(Kline(
                    exchange="bybit", coin=coin,
                    open_ms=open_ms, close_ms=close_ms,
                    open=o, high=h, low=l, close=c,
                    volume_base=vol_base, volume_quote=vol_quote,
                ))
            out.sort(key=lambda k: k.open_ms)
            return out
        except Exception as e:
            logger.warning("klines %s/%s failed: %s", coin, tf_label, e)
            return []

    @staticmethod
    async def _json(session, url):
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as r:
            r.raise_for_status()
            return await r.json()
