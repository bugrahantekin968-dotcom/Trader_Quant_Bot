"""
main.py — entry point for the live system.

Pipeline:
    1. Build clients (Binance + Bybit + OKX + Bitget + Hyperliquid)
    2. Prime REST contexts
    3. Run backfill (replay 30 days of klines)
    4. Start WS streams
    5. Start periodic loops: zone report writer, snapshot, signal emit
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import config
from engine.heatmap_engine import HeatmapEngine
from engine.backfill        import backfill_all
from engine.persistence     import snapshot_loop
from exchanges.registry     import build_clients
from viz.zone_report        import write_all_outputs
from strategy.signals       import analyze_swing

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)-7s | %(name)-12s | %(message)s",
)
log = logging.getLogger("main")


async def zone_report_loop(engine, out_dir: Path) -> None:
    while True:
        await asyncio.sleep(config.HEATMAP_RENDER_SEC)
        try:
            summary = write_all_outputs(engine, out_dir)
            log.info("Wrote zone reports: %d txt + %d json + %d html",
                     len(summary["text_files"]),
                     len(summary["json_files"]),
                     len(summary["html_files"]))
        except Exception as e:
            log.warning("zone reports: %s", e)


async def signal_loop(engine) -> None:
    while True:
        await asyncio.sleep(config.SIGNAL_EMIT_SEC)
        for coin in engine.coins:
            sig = analyze_swing(engine, coin)
            if sig is None:
                continue
            tgt = "—"
            if sig.target:
                tgt = f"${sig.target.price_center:,.2f} (${sig.target.dollars/1e6:.1f}M)"
            rr = f"R:R={sig.rr_estimate:.2f}" if sig.rr_estimate else ""
            log.info(
                f"SIGNAL {sig.coin} @ ${sig.price:,.2f} | "
                f"{sig.consensus_bias.value} conf={sig.consensus_confidence:.2f} | "
                f"tgt={tgt} | {rr}"
            )


async def amain() -> None:
    clients = build_clients(config.SYMBOLS)
    engine = HeatmapEngine(config.SYMBOLS, clients)

    # Turn on ground-truth liquidation recording for the calibration loop.
    # This costs almost nothing (one appended JSONL line per real liquidation)
    # and builds the dataset that calibration/fit_leverage.py consumes.
    engine.enable_recorder()

    out_dir = Path(config.OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info("Liquidation Zone Engine starting")
    log.info("Symbols      : %s", config.SYMBOLS)
    log.info("Timeframes   : %s", [tf[0] for tf in config.TIMEFRAMES])
    log.info("Grid         : ±%.0f%% / %.1f%% cells = %d bins",
             config.PCT_RANGE * 100, config.PCT_BUCKET * 100, config.N_PRICE_BINS)
    log.info("Exchanges    : %s", list(clients.keys()))
    log.info("Output       : %s", out_dir.resolve())
    log.info("=" * 60)

    poll_task = asyncio.create_task(engine.poll_contexts())
    await asyncio.sleep(8)

    log.info("Running backfill…")
    try:
        await backfill_all(engine)
    except Exception as e:
        log.warning("backfill: %s", e)

    try:
        write_all_outputs(engine, out_dir)
    except Exception as e:
        log.warning("initial report: %s", e)

    tasks = [
        poll_task,
        asyncio.create_task(engine.dex_refresh_loop()),
        asyncio.create_task(engine.recenter_loop()),
        asyncio.create_task(zone_report_loop(engine, out_dir)),
        asyncio.create_task(signal_loop(engine)),
        asyncio.create_task(snapshot_loop(engine)),
    ]
    for name, client in clients.items():
        tasks.append(asyncio.create_task(
            client.stream(engine.on_trade, engine.on_liquidation)
        ))

    await asyncio.gather(*tasks)


if __name__ == "__main__":
    try:
        asyncio.run(amain())
    except KeyboardInterrupt:
        log.info("Shutting down.")
