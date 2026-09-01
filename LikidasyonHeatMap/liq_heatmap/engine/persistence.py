"""
Persistence — periodic JSON snapshot of every bin matrix.
=========================================================
Snapshot format: one JSON file per (coin, timeframe), overwritten each
cycle. Bot can poll these files instead of WebSocket-ing the engine.

Parquet would be more compact but adds a dependency; JSON keeps the
system lean and human-debuggable. Switch to pyarrow only if disk size
becomes an issue (~2 MB per snapshot at full 12-grid layout).
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path

import config

logger = logging.getLogger("snapshot")


async def snapshot_loop(engine) -> None:
    out = Path(config.SNAPSHOT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    while True:
        await asyncio.sleep(config.SNAPSHOT_SEC)
        try:
            n = _write_snapshots(engine, out)
            logger.info("Wrote %d snapshots → %s", n, out)
        except Exception as e:
            logger.warning("snapshot: %s", e)


def _write_snapshots(engine, out: Path) -> int:
    from viz.zone_report import format_json
    n = 0
    for coin in engine.coins:
        st = engine.state[coin]
        ctx = engine.context_summary(coin)
        current_price = engine.current_price(coin)
        for tf_label, bs in st.bins_by_tf.items():
            zones = engine.extract_zones_for(coin, tf_label)
            payload = format_json(
                coin, tf_label, current_price,
                ctx["total_oi_usd"], ctx["avg_funding"], ctx["avg_long_ratio"],
                zones,
            )
            payload["dollar_bins"] = bs.dollar_bins.snapshot()
            with open(out / f"{coin.lower()}_{tf_label}.json", "w") as f:
                json.dump(payload, f)
            n += 1

            # Fixed 1%-slice packet — the trading bot reads THIS file
            packet = engine.slice_packet_for(coin, tf_label)
            with open(out / f"{coin.lower()}_{tf_label}_slices.json", "w") as f:
                json.dump(packet, f)
            n += 1
    return n
