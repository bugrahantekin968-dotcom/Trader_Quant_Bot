"""
Binance USDS-M Futures client.
==============================
- WebSocket: aggTrade + forceOrder
- REST:      open interest, funding/mark, top-trader L/S ratio
- Backfill:  klines + historical OI for cold-start population
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

logger = logging.getLogger("binance")

_INTERVAL_MAP = {
    "24h": "5m",      # closest standard candle to our 6-min internal bucket
    "3d":  "30m",
    "1w":  "2h",
    "1m":  "4h",
}
_INTERVAL_LIMIT = {"24h": 288, "3d": 144, "1w": 84, "1m": 180}


class BinanceClient:
    name = "binance"

    def __init__(self, coins: list[str]):
        self.coins = coins
        self.ws_url   = config.EXCHANGE_ENDPOINTS["binance"]["ws"]
        self.rest_url = config.EXCHANGE_ENDPOINTS["binance"]["rest"]

    # ----------------------------------------------------------- WS
    async def stream(
        self,
        on_trade: Callable[[Trade], None],
        on_liq:   Callable[[LiquidationEvent], None],
    ) -> None:
        streams = []
        for coin in self.coins:
            sym = config.symbol_for("binance", coin).lower()
            streams.append(f"{sym}@aggTrade")
            streams.append(f"{sym}@forceOrder")
        url = f"{self.ws_url}/{'/'.join(streams)}"

        async with aiohttp.ClientSession() as session:
            while True:
                try:
                    async with session.ws_connect(url, heartbeat=30) as ws:
                        logger.info("WS connected (%d streams)", len(streams))
                        async for msg in ws:
                            if msg.type != aiohttp.WSMsgType.TEXT:
                                continue
                            data = json.loads(msg.data)
                            ev = data.get("e")
                            if ev == "aggTrade":
                                coin = data["s"].replace("USDT", "")
                                on_trade(Trade(
                                    exchange="binance", coin=coin,
                                    price=float(data["p"]),
                                    qty=float(data["q"]),
                                    ts_ms=int(data["T"]),
                                    is_buyer_maker=bool(data["m"]),
                                ))
                            elif ev == "forceOrder":
                                o = data["o"]
                                coin = o["s"].replace("USDT", "")
                                on_liq(LiquidationEvent(
                                    exchange="binance", coin=coin,
                                    side=o["S"],
                                    price=float(o["ap"]),
                                    qty=float(o["q"]),
                                    ts_ms=int(o["T"]),
                                ))
                except Exception as e:
                    logger.warning("WS error: %s — reconnect 5s", e)
                    await asyncio.sleep(5)

    # ----------------------------------------------------------- REST
    async def fetch_context(self, session: aiohttp.ClientSession, coin: str) -> MarketContext | None:
        sym = config.symbol_for("binance", coin)
        oi_u    = f"{self.rest_url}/fapi/v1/openInterest?symbol={sym}"
        prem_u  = f"{self.rest_url}/fapi/v1/premiumIndex?symbol={sym}"
        ls_u    = (f"{self.rest_url}/futures/data/topLongShortAccountRatio"
                   f"?symbol={sym}&period=5m&limit=1")
        try:
            oi, prem, ls = await asyncio.gather(
                self._json(session, oi_u),
                self._json(session, prem_u),
                self._json(session, ls_u),
            )
            contracts = float(oi["openInterest"])
            mark      = float(prem["markPrice"])
            funding   = float(prem["lastFundingRate"])
            long_ratio = 0.5
            if isinstance(ls, list) and ls:
                long_ratio = float(ls[0]["longAccount"])
            return MarketContext(
                exchange="binance", coin=coin,
                open_interest_usd=contracts * mark,
                funding_rate=funding,
                long_ratio=long_ratio,
                mark_price=mark,
                ts_ms=int(time.time() * 1000),
            )
        except Exception as e:
            logger.warning("ctx %s failed: %s", coin, e)
            return None

    async def fetch_klines(self, session: aiohttp.ClientSession,
                           coin: str, tf_label: str) -> list[Kline]:
        interval = _INTERVAL_MAP.get(tf_label)
        limit    = _INTERVAL_LIMIT.get(tf_label, 90)
        if not interval:
            return []
        sym = config.symbol_for("binance", coin)
        url = f"{self.rest_url}/fapi/v1/klines?symbol={sym}&interval={interval}&limit={limit}"
        try:
            raw = await self._json(session, url)
            out = []
            for r in raw:
                open_ms = int(r[0]); close_ms = int(r[6])
                o,h,l,c = float(r[1]), float(r[2]), float(r[3]), float(r[4])
                vol_base  = float(r[5])
                vol_quote = float(r[7])
                out.append(Kline(
                    exchange="binance", coin=coin,
                    open_ms=open_ms, close_ms=close_ms,
                    open=o, high=h, low=l, close=c,
                    volume_base=vol_base, volume_quote=vol_quote,
                ))
            return out
        except Exception as e:
            logger.warning("klines %s/%s failed: %s", coin, tf_label, e)
            return []

    @staticmethod
    async def _json(session: aiohttp.ClientSession, url: str):
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as r:
            r.raise_for_status()
            return await r.json()
