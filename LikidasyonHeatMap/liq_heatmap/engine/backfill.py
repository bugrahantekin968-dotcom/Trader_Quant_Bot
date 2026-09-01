"""
Historical backfill.
====================
1m and 1w heatmaps cannot wait for live data to accumulate. On startup
we fetch klines from every exchange via REST and replay them through
the engine's projection pipeline.

For each kline we synthesize a Candle whose vwap≈close and volume=quote
volume. Decay is applied based on the kline's age, so 30-day-old data
naturally fades on the 1m timeframe (and is effectively zero on the 24h
timeframe).
"""
from __future__ import annotations

import asyncio
import logging
import time

import aiohttp

logger = logging.getLogger("backfill")


async def backfill_all(engine) -> None:
    """
    For every (exchange, coin, timeframe) triple, pull klines and replay
    them into the engine. Engine handles decay automatically via the
    candle's age relative to now.
    """
    from engine.candle_aggregator import Candle

    async with aiohttp.ClientSession() as session:
        tasks = []
        for ex_name, client in engine.clients.items():
            # Hyperliquid uses POST so it's slower; do CEX in parallel
            for coin in engine.coins:
                for tf in engine.timeframes:
                    tasks.append(_one(engine, session, client, coin, tf, Candle))
        await asyncio.gather(*tasks, return_exceptions=True)
    logger.info("Backfill complete")


async def _one(engine, session, client, coin: str, tf, Candle):
    try:
        klines = await client.fetch_klines(session, coin, tf.label)
    except Exception as e:
        logger.debug("backfill %s/%s/%s err: %s", client.name, coin, tf.label, e)
        return

    if not klines:
        return

    # Walk klines oldest-to-newest; for each, build a synthetic candle and
    # project it through the engine. Engine uses the candle's open_ms to
    # compute age → decay weight.
    now_ms = int(time.time() * 1000)
    for k in klines:
        # Skip klines outside the timeframe window
        age_sec = (now_ms - k.open_ms) / 1000.0
        if age_sec > tf.window_sec * 1.5:
            continue
        candle = Candle(
            open_ms=k.open_ms,
            close_ms=k.close_ms,
            volume_quote=k.volume_quote,
            volume_base=k.volume_base,
            vwap_num=k.close * k.volume_quote,
            vwap_den=k.volume_quote,
            n_trades=1,
        )
        # Project through the engine as if this candle just closed.
        # Caller marks this exchange's contribution to this coin/tf bin.
        engine.project_synthetic_candle(client.name, coin, tf.label, candle, age_ms=now_ms - k.open_ms)
