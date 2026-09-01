"""
OKX v5 swap client.
===================
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

logger = logging.getLogger("okx")

_INTERVAL_MAP   = {"24h": "5m", "3d": "30m", "1w": "2H", "1m": "4H"}
_INTERVAL_LIMIT = {"24h": 288, "3d": 144, "1w": 84, "1m": 180}


class OKXClient:
    name = "okx"

    def __init__(self, coins: list[str]):
        self.coins = coins
        self.ws_url   = config.EXCHANGE_ENDPOINTS["okx"]["ws"]
        self.rest_url = config.EXCHANGE_ENDPOINTS["okx"]["rest"]
        self._inst_to_coin = {config.symbol_for("okx", c): c for c in coins}

    async def stream(self, on_trade, on_liq) -> None:
        args = []
        for coin in self.coins:
            inst = config.symbol_for("okx", coin)
            args.append({"channel": "trades", "instId": inst})
        # One subscription per instType covers all liquidations
        args.append({"channel": "liquidation-orders", "instType": "SWAP"})
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
                            rows = data.get("data", [])
                            if chan == "trades":
                                for t in rows:
                                    inst = t["instId"]
                                    coin = self._inst_to_coin.get(inst)
                                    if not coin:
                                        continue
                                    on_trade(Trade(
                                        exchange="okx", coin=coin,
                                        price=float(t["px"]),
                                        qty=float(t["sz"]),
                                        ts_ms=int(t["ts"]),
                                        is_buyer_maker=(t["side"] == "sell"),
                                    ))
                            elif chan == "liquidation-orders":
                                for r in rows:
                                    inst = r.get("instId")
                                    coin = self._inst_to_coin.get(inst)
                                    if not coin:
                                        continue
                                    for d in r.get("details", []):
                                        on_liq(LiquidationEvent(
                                            exchange="okx", coin=coin,
                                            side=("BUY" if d.get("side") == "buy" else "SELL"),
                                            price=float(d.get("bkPx", 0)),
                                            qty=float(d.get("sz", 0)),
                                            ts_ms=int(d.get("ts", time.time() * 1000)),
                                        ))
                except Exception as e:
                    logger.warning("WS error: %s — reconnect 5s", e)
                    await asyncio.sleep(5)

    async def fetch_context(self, session, coin):
        inst = config.symbol_for("okx", coin)
        tick_u  = f"{self.rest_url}/api/v5/market/ticker?instId={inst}"
        oi_u    = f"{self.rest_url}/api/v5/public/open-interest?instId={inst}"
        fund_u  = f"{self.rest_url}/api/v5/public/funding-rate?instId={inst}"
        ratio_u = f"{self.rest_url}/api/v5/rubik/stat/contracts/long-short-account-ratio?ccy={coin}"
        try:
            tick_j, oi_j, fund_j, ratio_j = await asyncio.gather(
                self._json(session, tick_u),
                self._json(session, oi_u),
                self._json(session, fund_u),
                self._json(session, ratio_u),
            )
            mark    = float(tick_j["data"][0]["last"])
            # OKX returns: oi (CONTRACTS), oiCcy (in base coin), oiUsd (USD).
            # Using `oi` (contracts) × mark over-counts by the contract size
            # (e.g. 0.01 BTC/contract → 100×). Prefer oiUsd, else oiCcy × mark.
            oi_row = oi_j["data"][0]
            if oi_row.get("oiUsd") not in (None, ""):
                oi_usd = float(oi_row["oiUsd"])
            else:
                oi_usd = float(oi_row.get("oiCcy", 0.0)) * mark
            funding = float(fund_j["data"][0]["fundingRate"])
            long_ratio = 0.5
            try:
                row = ratio_j["data"][0]
                ratio = float(row[1])
                long_ratio = ratio / (ratio + 1.0)
            except (KeyError, IndexError, TypeError):
                pass
            return MarketContext(
                exchange="okx", coin=coin,
                open_interest_usd=oi_usd,
                funding_rate=funding,
                long_ratio=long_ratio,
                mark_price=mark,
                ts_ms=int(time.time() * 1000),
            )
        except Exception as e:
            logger.warning("ctx %s failed: %s", coin, e)
            return None

    async def fetch_klines(self, session, coin, tf_label):
        bar = _INTERVAL_MAP.get(tf_label)
        lim = _INTERVAL_LIMIT.get(tf_label, 90)
        if not bar:
            return []
        inst = config.symbol_for("okx", coin)
        url = f"{self.rest_url}/api/v5/market/candles?instId={inst}&bar={bar}&limit={lim}"
        try:
            raw = await self._json(session, url)
            out = []
            for r in raw.get("data", []):
                open_ms = int(r[0])
                o,h,l,c = float(r[1]), float(r[2]), float(r[3]), float(r[4])
                vol_base  = float(r[5])
                vol_quote = float(r[6]) if len(r) > 6 else vol_base * c
                # candle bar duration parse
                if bar.endswith("m"):
                    dur = int(bar[:-1]) * 60_000
                elif bar.endswith("H"):
                    dur = int(bar[:-1]) * 3600_000
                else:
                    dur = 60_000
                out.append(Kline(
                    exchange="okx", coin=coin,
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
