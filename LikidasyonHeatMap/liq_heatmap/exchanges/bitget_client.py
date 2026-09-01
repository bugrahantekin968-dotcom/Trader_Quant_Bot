"""
Bitget v2 USDT-perpetual client.
================================
Bitget uses 'productType=USDT-FUTURES' for linear perps.
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

logger = logging.getLogger("bitget")

_INTERVAL_MAP   = {"24h": "5m", "3d": "30m", "1w": "1H", "1m": "4H"}  # Bitget valid: 1H/4H/6H/12H (no 2H/8H)
_INTERVAL_LIMIT = {"24h": 288, "3d": 144, "1w": 168, "1m": 180}


class BitgetClient:
    name = "bitget"

    def __init__(self, coins: list[str]):
        self.coins = coins
        self.ws_url   = config.EXCHANGE_ENDPOINTS["bitget"]["ws"]
        self.rest_url = config.EXCHANGE_ENDPOINTS["bitget"]["rest"]

    async def stream(self, on_trade, on_liq) -> None:
        args = []
        for coin in self.coins:
            sym = config.symbol_for("bitget", coin)
            args.append({"instType": "USDT-FUTURES", "channel": "trade", "instId": sym})
            # Bitget public liquidation channel
            args.append({"instType": "USDT-FUTURES", "channel": "liquidation-orders", "instId": sym})
        sub = {"op": "subscribe", "args": args}

        async with aiohttp.ClientSession() as session:
            while True:
                try:
                    async with session.ws_connect(self.ws_url, heartbeat=25) as ws:
                        await ws.send_str(json.dumps(sub))
                        logger.info("WS connected, subs=%d", len(args))
                        async for msg in ws:
                            if msg.type != aiohttp.WSMsgType.TEXT:
                                continue
                            data = json.loads(msg.data)
                            arg  = data.get("arg", {})
                            chan = arg.get("channel")
                            inst = arg.get("instId", "")
                            coin = inst.replace("USDT", "")
                            rows = data.get("data", [])
                            if chan == "trade":
                                for t in rows:
                                    # Bitget format: [ts, price, size, side]
                                    on_trade(Trade(
                                        exchange="bitget", coin=coin,
                                        price=float(t.get("price", 0)),
                                        qty=float(t.get("size", 0)),
                                        ts_ms=int(t.get("ts", time.time() * 1000)),
                                        is_buyer_maker=(t.get("side") == "sell"),
                                    ))
                            elif chan == "liquidation-orders":
                                for d in rows:
                                    on_liq(LiquidationEvent(
                                        exchange="bitget", coin=coin,
                                        side=str(d.get("side", "")).upper(),
                                        price=float(d.get("price", 0)),
                                        qty=float(d.get("size", 0)),
                                        ts_ms=int(d.get("ts", time.time() * 1000)),
                                    ))
                except Exception as e:
                    logger.warning("WS error: %s — reconnect 5s", e)
                    await asyncio.sleep(5)

    async def fetch_context(self, session, coin):
        sym = config.symbol_for("bitget", coin)
        # Ticker has mark, OI and funding all in one
        url = (f"{self.rest_url}/api/v2/mix/market/ticker"
               f"?productType=USDT-FUTURES&symbol={sym}")
        ls_url = (f"{self.rest_url}/api/v2/mix/market/account-long-short"
                  f"?productType=USDT-FUTURES&symbol={sym}&period=5m&limit=1")
        try:
            tick_j, ls_j = await asyncio.gather(
                self._json(session, url),
                self._json(session, ls_url),
            )
            row = (tick_j.get("data") or [{}])[0]
            mark    = float(row.get("markPrice") or row.get("lastPr") or 0)
            oi_ccy  = float(row.get("holdingAmount") or 0)
            funding = float(row.get("fundingRate") or 0)
            long_ratio = 0.5
            try:
                ls_row = (ls_j.get("data") or [{}])[0]
                long_acct  = float(ls_row.get("longAccountRatio", 0.5))
                short_acct = float(ls_row.get("shortAccountRatio", 0.5))
                if long_acct + short_acct > 0:
                    long_ratio = long_acct / (long_acct + short_acct)
            except (KeyError, IndexError, TypeError):
                pass
            return MarketContext(
                exchange="bitget", coin=coin,
                open_interest_usd=oi_ccy * mark,
                funding_rate=funding,
                long_ratio=long_ratio,
                mark_price=mark,
                ts_ms=int(time.time() * 1000),
            )
        except Exception as e:
            logger.warning("ctx %s failed: %s", coin, e)
            return None

    async def fetch_klines(self, session, coin, tf_label):
        gran = _INTERVAL_MAP.get(tf_label)
        lim  = _INTERVAL_LIMIT.get(tf_label, 90)
        if not gran:
            return []
        sym = config.symbol_for("bitget", coin)
        url = (f"{self.rest_url}/api/v2/mix/market/candles"
               f"?productType=USDT-FUTURES&symbol={sym}&granularity={gran}&limit={lim}")
        try:
            raw = await self._json(session, url)
            out = []
            for r in raw.get("data", []):
                open_ms = int(r[0])
                o,h,l,c = float(r[1]), float(r[2]), float(r[3]), float(r[4])
                vol_base  = float(r[5])
                vol_quote = float(r[6]) if len(r) > 6 else vol_base * c
                if gran.endswith("m"):
                    dur = int(gran[:-1]) * 60_000
                elif gran.endswith("H"):
                    dur = int(gran[:-1]) * 3600_000
                else:
                    dur = 60_000
                out.append(Kline(
                    exchange="bitget", coin=coin,
                    open_ms=open_ms, close_ms=open_ms + dur,
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
