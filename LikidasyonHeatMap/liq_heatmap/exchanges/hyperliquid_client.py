"""
Hyperliquid position tracker — Tier-1 DEX with EXACT per-position data.
=======================================================================
Public trade feed includes both user addresses → we reconstruct each
trader's net position from fills, then periodically pull
clearinghouseState to learn exact leverage + liquidationPx.

Output: liq_distribution(coin) → list[(liq_price, abs_size, is_long)]
which the engine scatters directly into the bin grid (no statistical
fan-out needed because every entry is a REAL position).
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable

import aiohttp

import config
from exchanges.base import Trade, MarketContext

logger = logging.getLogger("hyperliquid")


@dataclass
class HLPosition:
    user: str
    coin: str
    size: float = 0.0           # signed: +long, -short
    avg_entry: float = 0.0
    leverage: float = 0.0
    margin_used: float = 0.0
    liq_price: float = 0.0
    last_fill_ms: int = 0
    last_refresh_ms: int = 0


class HyperliquidClient:
    name = "hyperliquid"

    def __init__(self, coins: list[str]):
        self.coins = coins
        self.ws_url   = config.EXCHANGE_ENDPOINTS["hyperliquid"]["ws"]
        self.rest_url = config.EXCHANGE_ENDPOINTS["hyperliquid"]["rest"]

        # user → {coin → HLPosition}
        self.positions: dict[str, dict[str, HLPosition]] = defaultdict(dict)
        # users with size changes since last refresh
        self.dirty: set[str] = set()

        # also feed standard `on_trade` so the engine accumulates volume
        # into its CEX-style candle aggregator
        self._on_trade: Callable[[Trade], None] | None = None
        self._on_liq = None  # Hyperliquid has dedicated liquidation events

    # ============================================================
    # Position bookkeeping
    # ============================================================
    def _pos(self, user: str, coin: str) -> HLPosition:
        if coin not in self.positions[user]:
            self.positions[user][coin] = HLPosition(user=user, coin=coin)
        return self.positions[user][coin]

    def _apply_fill(self, user: str, coin: str, signed_qty: float, px: float, ts: int) -> None:
        p = self._pos(user, coin)
        new_size = p.size + signed_qty

        if abs(p.size) < 1e-12:                              # opening from flat
            p.avg_entry = px
        elif (p.size > 0 and signed_qty > 0) or (p.size < 0 and signed_qty < 0):
            tot = abs(p.size) + abs(signed_qty)
            p.avg_entry = (abs(p.size) * p.avg_entry + abs(signed_qty) * px) / tot
        elif abs(new_size) < 1e-12:                          # fully closed
            p.avg_entry = 0.0
            new_size = 0.0
        elif (p.size > 0) != (new_size > 0):                 # direction flip
            p.avg_entry = px
        # else: partial reduction, keep entry

        p.size = new_size
        p.last_fill_ms = ts

        if abs(p.size) > 1e-12:
            self.dirty.add(user)
        else:
            p.liq_price = 0.0
            p.leverage = 0.0

    def _on_hl_trade(self, t: dict) -> None:
        coin = t.get("coin")
        if coin not in self.coins:
            return
        users = t.get("users") or []
        if len(users) < 2:
            return
        try:
            px = float(t["px"])
            sz = float(t["sz"])
            ts = int(t.get("time", time.time() * 1000))
        except (KeyError, ValueError, TypeError):
            return

        maker_user, taker_user = users[0], users[1]
        taker_delta = sz if t.get("side") == "B" else -sz
        self._apply_fill(taker_user, coin, taker_delta, px, ts)
        self._apply_fill(maker_user, coin, -taker_delta, px, ts)

        # Also publish to the engine's candle aggregator
        if self._on_trade:
            self._on_trade(Trade(
                exchange="hyperliquid", coin=coin,
                price=px, qty=sz, ts_ms=ts,
                is_buyer_maker=(t.get("side") == "A"),
            ))

    # ============================================================
    # WebSocket loop
    # ============================================================
    async def stream(self, on_trade, on_liq) -> None:
        self._on_trade = on_trade
        self._on_liq   = on_liq
        sub_msgs = [
            {"method": "subscribe", "subscription": {"type": "trades", "coin": c}}
            for c in self.coins
        ]
        async with aiohttp.ClientSession() as session:
            while True:
                try:
                    async with session.ws_connect(self.ws_url, heartbeat=20) as ws:
                        for m in sub_msgs:
                            await ws.send_str(json.dumps(m))
                        logger.info("WS connected, coins=%s", self.coins)
                        async for msg in ws:
                            if msg.type != aiohttp.WSMsgType.TEXT:
                                continue
                            data = json.loads(msg.data)
                            if data.get("channel") == "trades":
                                for tr in data.get("data") or []:
                                    self._on_hl_trade(tr)
                except Exception as e:
                    logger.warning("WS error: %s — reconnect 5s", e)
                    await asyncio.sleep(5)

    # ============================================================
    # Periodic clearinghouseState refresh
    # ============================================================
    async def refresh_loop(self, interval_sec: int | None = None) -> None:
        interval = interval_sec or config.DEX_REFRESH_SEC
        batch_size = 25
        async with aiohttp.ClientSession() as session:
            while True:
                await asyncio.sleep(interval)
                users = list(self.dirty)[:batch_size]
                self.dirty.difference_update(users)
                if not users:
                    continue
                await asyncio.gather(
                    *[self._refresh_user(session, u) for u in users],
                    return_exceptions=True,
                )

    async def _refresh_user(self, session, user: str) -> None:
        try:
            body = {"type": "clearinghouseState", "user": user}
            async with session.post(self.rest_url, json=body,
                                     timeout=aiohttp.ClientTimeout(total=10)) as r:
                r.raise_for_status()
                state = await r.json()
        except Exception:
            return

        now_ms = int(time.time() * 1000)
        for ap in state.get("assetPositions", []):
            p_raw = ap.get("position", {})
            coin = p_raw.get("coin")
            if coin not in self.coins:
                continue
            pos = self._pos(user, coin)
            try:
                pos.leverage    = float(p_raw.get("leverage", {}).get("value", 0))
                pos.margin_used = float(p_raw.get("marginUsed", 0))
                liq = p_raw.get("liquidationPx")
                pos.liq_price   = float(liq) if liq not in (None, "") else 0.0
            except (TypeError, ValueError):
                pass
            pos.last_refresh_ms = now_ms

    # ============================================================
    # MarketContext for engine (mark, OI, funding)
    # ============================================================
    async def fetch_context(self, session, coin):
        try:
            body = {"type": "metaAndAssetCtxs"}
            async with session.post(self.rest_url, json=body,
                                     timeout=aiohttp.ClientTimeout(total=10)) as r:
                r.raise_for_status()
                payload = await r.json()
            # payload = [meta, [asset_ctxs...]] aligned with meta.universe
            meta, ctxs = payload[0], payload[1]
            universe = meta.get("universe", [])
            idx = next((i for i, u in enumerate(universe) if u.get("name") == coin), None)
            if idx is None:
                return None
            ctx = ctxs[idx]
            mark    = float(ctx.get("markPx") or ctx.get("oraclePx") or 0)
            oi      = float(ctx.get("openInterest", 0))    # base units
            funding = float(ctx.get("funding", 0))
            return MarketContext(
                exchange="hyperliquid", coin=coin,
                open_interest_usd=oi * mark,
                funding_rate=funding,
                long_ratio=0.5,             # HL doesn't publish L/S ratio; rely on others
                mark_price=mark,
                ts_ms=int(time.time() * 1000),
            )
        except Exception as e:
            logger.warning("ctx %s failed: %s", coin, e)
            return None

    async def fetch_klines(self, session, coin, tf_label):
        """Hyperliquid candle snapshot endpoint."""
        intervals = {"24h": "5m", "3d": "30m", "1w": "2h", "1m": "4h"}
        bar = intervals.get(tf_label)
        if not bar:
            return []
        # Hyperliquid: POST /info with type=candleSnapshot
        end = int(time.time() * 1000)
        start_offsets = {"24h": 86400, "3d": 3*86400, "1w": 7*86400, "1m": 30*86400}
        start = end - start_offsets[tf_label] * 1000
        body = {
            "type": "candleSnapshot",
            "req": {"coin": coin, "interval": bar, "startTime": start, "endTime": end},
        }
        try:
            async with session.post(self.rest_url, json=body,
                                     timeout=aiohttp.ClientTimeout(total=10)) as r:
                r.raise_for_status()
                raw = await r.json()
            from exchanges.base import Kline
            out = []
            for c in raw:
                open_ms = int(c["t"])
                close_ms = int(c.get("T", open_ms))
                out.append(Kline(
                    exchange="hyperliquid", coin=coin,
                    open_ms=open_ms, close_ms=close_ms,
                    open=float(c["o"]), high=float(c["h"]),
                    low=float(c["l"]),  close=float(c["c"]),
                    volume_base=float(c.get("v", 0)),
                    volume_quote=float(c.get("v", 0)) * float(c["c"]),
                ))
            out.sort(key=lambda k: k.open_ms)
            return out
        except Exception as e:
            logger.warning("klines %s/%s failed: %s", coin, tf_label, e)
            return []

    # ============================================================
    # Output for engine: exact liq distribution
    # ============================================================
    def liq_distribution(self, coin: str) -> list[tuple[float, float, bool]]:
        out = []
        for user, by_coin in self.positions.items():
            pos = by_coin.get(coin)
            if pos is None or abs(pos.size) < 1e-12 or pos.liq_price <= 0:
                continue
            out.append((pos.liq_price, abs(pos.size), pos.size > 0))
        return out

    def stats(self) -> dict:
        n_total = sum(1 for u in self.positions.values()
                          for p in u.values() if abs(p.size) > 0)
        n_with_liq = sum(1 for u in self.positions.values()
                              for p in u.values()
                              if abs(p.size) > 0 and p.liq_price > 0)
        return {
            "tracked_users": len(self.positions),
            "open_positions": n_total,
            "with_known_liq_price": n_with_liq,
            "dirty_queue": len(self.dirty),
        }
