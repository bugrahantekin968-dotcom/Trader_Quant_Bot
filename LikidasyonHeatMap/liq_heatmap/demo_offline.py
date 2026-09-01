"""
Offline end-to-end demo with realistic clustering simulation.
"""
import logging
import math
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(level=logging.INFO, format="%(name)-12s | %(message)s")

import asyncio
import config
from exchanges.base         import MarketContext
from engine.candle_aggregator import Candle
from engine.heatmap_engine  import HeatmapEngine
from viz.zone_report        import write_all_outputs, format_text_report
from strategy.signals       import analyze_swing


class StubClient:
    def __init__(self, name, coins, share):
        self.name = name
        self.coins = coins
        self.share = share

    async def stream(self, on_trade, on_liq):
        return

    async def fetch_context(self, session, coin):
        oi = {"BTC": 14e9, "ETH": 6e9, "XRP": 1.2e9}.get(coin, 1e8)
        mark = {"BTC": 65_000, "ETH": 3_200, "XRP": 2.45}[coin]
        # mild bias per coin
        long_ratio = {"BTC": 0.56, "ETH": 0.49, "XRP": 0.47}[coin]
        return MarketContext(
            exchange=self.name, coin=coin,
            open_interest_usd=oi * self.share,
            funding_rate=0.0001,
            long_ratio=long_ratio,
            mark_price=mark,
            ts_ms=int(time.time() * 1000),
        )

    async def fetch_klines(self, session, coin, tf_label):
        return []


def mean_reverting_series(base, n, vol=0.005, half_life=20, seed=None):
    """Ornstein-Uhlenbeck-like prices: random shocks + mean reversion.
    Produces stable price action that yields CLEAR liquidation clusters."""
    rng = random.Random(seed)
    decay = math.exp(-math.log(2) / half_life)
    log_p = math.log(base)
    log_mean = log_p
    series = []
    for _ in range(n):
        log_p = log_mean * (1 - decay) + log_p * decay + rng.gauss(0, vol)
        series.append(math.exp(log_p))
    return series


def main():
    coins = config.SYMBOLS
    # Exchange OI shares roughly match real-world (Binance dominant)
    clients = {
        "binance": StubClient("binance", coins, 0.42),
        "bybit":   StubClient("bybit",   coins, 0.24),
        "okx":     StubClient("okx",     coins, 0.18),
        "bitget":  StubClient("bitget",  coins, 0.16),
    }
    engine = HeatmapEngine(coins, clients)

    # Prime contexts synchronously
    loop = asyncio.new_event_loop()
    for coin in coins:
        for ex_name, client in clients.items():
            ctx = loop.run_until_complete(client.fetch_context(None, coin))
            engine.state[coin].contexts[ex_name] = ctx
    loop.close()

    spots = {"BTC": 65_000.0, "ETH": 3_200.0, "XRP": 2.45}
    for coin, px in spots.items():
        engine.state[coin].last_price = px

    # ----- Inject synthetic candles for EVERY timeframe directly -----
    # This is much faster than simulating per-trade and produces clean clustering.
    print("Generating synthetic candles per (coin, tf)…")
    now_ms = int(time.time() * 1000)
    for coin in coins:
        base_price = spots[coin]
        for tf in engine.timeframes:
            prices = mean_reverting_series(
                base_price, tf.n_candles,
                vol=0.004, half_life=tf.n_candles // 4,
                seed=hash((coin, tf.label)) & 0xFFFFFFFF,
            )
            for i, price in enumerate(prices):
                # candle i is at age (n - i - 1) * candle_sec back from now
                age_steps = tf.n_candles - i - 1
                open_ms = now_ms - age_steps * tf.candle_sec * 1000
                close_ms = open_ms + tf.candle_sec * 1000
                # Avg volume per candle for this TF
                vol_quote = random.uniform(2e7, 8e7) * (tf.candle_sec / 360)
                candle = Candle(
                    open_ms=open_ms, close_ms=close_ms,
                    open=price * (1 - 0.001), high=price * 1.002,
                    low=price * 0.998, close=price,
                    volume_quote=vol_quote, volume_base=vol_quote / price,
                    vwap_num=price * vol_quote, vwap_den=vol_quote,
                    n_trades=100,
                )
                engine.project_synthetic_candle(
                    "binance", coin, tf.label, candle,
                    age_ms=age_steps * tf.candle_sec * 1000,
                )
            # Update last_price to last candle's close
            engine.state[coin].last_price = prices[-1]
        print(f"  {coin}: spot ${engine.state[coin].last_price:,.2f}")

    # ----- Show zone reports inline -----
    print("\n" + "=" * 70)
    print("PRIMARY OUTPUT — Zone Reports")
    print("=" * 70)
    for coin in coins:
        ctx = engine.context_summary(coin)
        for tf in [t.label for t in engine.timeframes]:
            zones = engine.extract_zones_for(coin, tf)
            text = format_text_report(
                coin, tf, engine.current_price(coin),
                ctx["total_oi_usd"], ctx["avg_funding"], ctx["avg_long_ratio"],
                zones,
            )
            print(text)
            print()

    # ----- Persist to disk -----
    out_dir = Path("/mnt/user-data/outputs/zone_reports")
    summary = write_all_outputs(engine, out_dir)
    print(f"\nFiles written:  {len(summary['text_files'])} txt, "
          f"{len(summary['json_files'])} json, "
          f"{len(summary['html_files'])} html  →  {out_dir}")

    # ----- Swing signals -----
    print("\n" + "=" * 70)
    print("SWING SIGNALS")
    print("=" * 70)
    for coin in coins:
        sig = analyze_swing(engine, coin)
        if sig is None:
            continue
        target_str = "—"
        if sig.target:
            target_str = f"${sig.target.price_center:,.2f} (${sig.target.dollars/1e6:.1f}M)"
        rr_str = f"R:R={sig.rr_estimate:.2f}" if sig.rr_estimate else ""
        print(f"  {coin}: {sig.consensus_bias.value:12s}  conf={sig.consensus_confidence:.2f}  "
              f"target={target_str}  {rr_str}")


if __name__ == "__main__":
    main()
