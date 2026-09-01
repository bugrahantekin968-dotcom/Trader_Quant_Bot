"""
HeatmapEngine — main orchestrator (multi-symbol × multi-timeframe × multi-exchange).
====================================================================================
State layout:
    self.state[coin][tf_label] = BinState
        ├── bins: PercentBins (400 price × N_candles time matrix)
        └── aggregator: MultiTFAggregator (per-coin, all TFs)

    self.contexts[coin][exchange] = MarketContext  (OI / funding / L-S)

Pipeline (per coin):
    Trade arrives → MultiTFAggregator.add_trade() → fans out across 4 TFs
                  → any closed candle triggers project_candle()
                  → projection scatters into PercentBins for that (coin, tf)

Periodic tasks:
    poll_contexts  → REST OI/funding/L-S every 12 s
    refresh_dex    → Hyperliquid clearinghouseState every 20 s
    project_dex    → write exact DEX liq positions into bins every 15 s
    recenter_check → align grid to spot every 30 s
"""
from __future__ import annotations

import asyncio
import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field

import aiohttp
import numpy as np

import config
from core.timeframe   import TimeframeSpec, load_timeframes
from core.percent_bins import PercentBins
from core.math_models  import (
    decay_weight, oi_weights, project_liquidations, split_long_short,
)
from engine.candle_aggregator import MultiTFAggregator, Candle

logger = logging.getLogger("engine")


@dataclass
class BinState:
    bins: PercentBins              # density (log-weighted, for diagnostic)
    dollar_bins: PercentBins       # OI-anchored linear $ — primary signal source
    tf: TimeframeSpec
    candle_history: deque = field(default_factory=deque)

    def __post_init__(self):
        # cap the history at the timeframe's window size
        self.candle_history = deque(maxlen=self.tf.n_candles)


@dataclass
class CoinState:
    last_price: float = 0.0
    bins_by_tf: dict[str, BinState] = field(default_factory=dict)
    aggregator: MultiTFAggregator | None = None
    contexts: dict[str, "MarketContext"] = field(default_factory=dict)  # exchange → ctx
    # (ts_ms, total_oi_usd) samples for OI-delta weighting; capped to ~6h at 12s poll
    oi_history: deque = field(default_factory=lambda: deque(maxlen=1800))


class HeatmapEngine:
    def __init__(self, coins: list[str], clients: dict):
        self.coins = coins
        self.clients = clients
        self.timeframes: list[TimeframeSpec] = load_timeframes(config.TIMEFRAMES)
        self.state: dict[str, CoinState] = {c: CoinState() for c in coins}

        # For Hyperliquid: keep a handle to its tracker for exact projections
        self.dex_tracker = clients.get("hyperliquid")

        # Per-coin leverage distribution: empirical (calibrated) if available,
        # else the config default. The fitter writes _calib/calibrated_leverage.json.
        self.leverage_dist = self._load_calibrated_leverage()

        # Ground-truth recorder for the calibration loop (lazy; off unless enabled)
        self.recorder = None

    def _load_calibrated_leverage(self) -> dict:
        """Merge calibrated per-coin distributions over the config defaults."""
        import json as _json
        merged = {c: dict(config.LEVERAGE_DISTRIBUTION.get(
            c, config.LEVERAGE_DISTRIBUTION["BTC"])) for c in self.coins}
        try:
            with open("./_calib/calibrated_leverage.json") as f:
                cal = _json.load(f)
            for coin, dist in cal.items():
                # keys come back as strings from JSON → coerce to int buckets
                merged[coin] = {int(k): float(v) for k, v in dist.items()}
            logger.info("Loaded calibrated leverage for: %s", list(cal.keys()))
        except (FileNotFoundError, ValueError):
            logger.info("No calibrated leverage file — using config defaults")
        return merged

    def enable_recorder(self, path: str = "./_calib/liquidations.jsonl") -> None:
        """Turn on ground-truth liquidation recording for calibration."""
        from engine.liq_recorder import LiquidationRecorder
        self.recorder = LiquidationRecorder(path)
        logger.info("Liquidation recorder ON → %s", path)

    def _recent_ref(self, coin: str) -> tuple[float, float]:
        """Recent (high, low) from the 24h candle history — reference for
        implied-leverage back-out in the recorder."""
        st = self.state.get(coin)
        if not st or "24h" not in st.bins_by_tf:
            p = self.current_price(coin)
            return p, p
        hist = st.bins_by_tf["24h"].candle_history
        if not hist:
            p = self.current_price(coin)
            return p, p
        highs = [c.high for c in hist if c.high > 0]
        lows  = [c.low  for c in hist if c.low  > 0]
        hi = max(highs) if highs else self.current_price(coin)
        lo = min(lows)  if lows  else self.current_price(coin)
        return hi, lo

    # =================================================================
    # WS callbacks (CEX + DEX)
    # =================================================================
    def on_trade(self, trade) -> None:
        st = self.state.get(trade.coin)
        if st is None:
            return
        st.last_price = trade.price

        # Lazy-init aggregator and bins
        if st.aggregator is None:
            st.aggregator = MultiTFAggregator(self.timeframes)
        if not st.bins_by_tf:
            for tf in self.timeframes:
                st.bins_by_tf[tf.label] = BinState(
                    bins=PercentBins(
                        spot=trade.price,
                        pct_range=tf.pct_range,
                        pct_bucket=config.PCT_BUCKET,
                        n_time_steps=tf.n_candles,
                        recenter_trigger=config.RECENTER_TRIGGER,
                    ),
                    dollar_bins=PercentBins(
                        spot=trade.price,
                        pct_range=tf.pct_range,
                        pct_bucket=config.PCT_BUCKET,
                        n_time_steps=tf.n_candles,
                        recenter_trigger=config.RECENTER_TRIGGER,
                    ),
                    tf=tf,
                )

        # Feed all timeframes; project any closed candles
        for tf_label, closed_candle in st.aggregator.add_trade(
            trade.ts_ms, trade.price, trade.qty
        ):
            self._project_candle(trade.coin, tf_label, closed_candle, age_ms=0)

    def on_liquidation(self, liq) -> None:
        # Record real liquidations as calibration ground truth (if enabled).
        # The recorder back-outs implied leverage from price vs recent extremes.
        logger.debug("LIQ %s %s %s @ %.2f sz=%.4f",
                     liq.exchange, liq.coin, liq.side, liq.price, liq.qty)
        if self.recorder is None:
            return
        try:
            from engine.liq_recorder import normalize_liq_side
            side = normalize_liq_side(liq.side)
            if side == "UNKNOWN":
                return
            ref_high, ref_low = self._recent_ref(liq.coin)
            qty_usd = liq.qty * liq.price
            self.recorder.record(
                coin=liq.coin, side=side, price=liq.price, qty_usd=qty_usd,
                ref_high=ref_high, ref_low=ref_low, ts_ms=liq.ts_ms,
                exchange=liq.exchange,
            )
        except Exception as e:
            logger.debug("recorder error: %s", e)

    # =================================================================
    # Projection: closed candle → bin grid
    # =================================================================
    def _oi_delta_multiplier(self, coin: str, candle: Candle) -> float:
        """
        Weight a candle by whether OI was RISING during it (net new positions
        open → real future liquidation risk) vs FALLING (positions closing →
        less future risk). Volume alone counts both opens and closes equally;
        OI-delta separates them. Returns a multiplier in [0.3, 2.0]; 1.0 when
        no OI data (degrades gracefully to pure-volume weighting).
        """
        st = self.state.get(coin)
        if st is None or len(st.oi_history) < 2:
            return 1.0
        lo_ts, hi_ts = candle.open_ms, candle.close_ms
        before = after = None
        for ts, oi in st.oi_history:
            if ts <= lo_ts:
                before = oi
            if ts >= hi_ts and after is None:
                after = oi
        if before is None or after is None or before <= 0:
            recent = list(st.oi_history)[-2:]
            if len(recent) < 2 or recent[0][1] <= 0:
                return 1.0
            before, after = recent[0][1], recent[1][1]
        delta_pct = (after - before) / before
        mult = 1.0 + 50.0 * delta_pct      # +1% OI → ×1.5 ; −1% → ×0.5
        return float(max(0.3, min(2.0, mult)))

    def _project_candle(self, coin: str, tf_label: str, candle: Candle, age_ms: int) -> None:
        st = self.state[coin]
        bin_state = st.bins_by_tf.get(tf_label)
        if bin_state is None or candle.volume_quote <= 0:
            return
        tf = bin_state.tf

        # Track OHLC for chart overlay
        bin_state.candle_history.append(candle)

        # Decay based on candle age (0 for just-closed candles)
        age_hours = age_ms / 3_600_000.0

        # OI-delta weight: boost candles where OI rose (new positions opened),
        # dampen candles where OI fell (positions closing). Falls back to 1.0
        # (pure volume) when OI history is unavailable.
        oi_mult = self._oi_delta_multiplier(coin, candle)

        weighted_vol = float(
            decay_weight(candle.volume_quote, age_hours, tf.lambda_per_hour)
        ) * oi_mult
        if weighted_vol <= 0:
            return

        # Long/short ratio: OI-weighted average across exchanges
        if st.contexts:
            ow = oi_weights({ex: ctx.open_interest_usd for ex, ctx in st.contexts.items()})
            long_ratio = sum(ow[ex] * ctx.long_ratio for ex, ctx in st.contexts.items())
            if not (0.05 < long_ratio < 0.95):
                long_ratio = 0.5
            mmr = sum(
                ow[ex] * config.MAINTENANCE_MARGIN_RATES.get(ex, 0.005)
                for ex in st.contexts
            ) or config.MAINTENANCE_MARGIN_RATES["binance"]
        else:
            long_ratio = 0.5
            mmr = config.MAINTENANCE_MARGIN_RATES["binance"]

        long_part, short_part = split_long_short(weighted_vol, long_ratio)

        # Symbol-specific leverage distribution (calibrated if available)
        lev_dist = self.leverage_dist.get(
            coin, config.LEVERAGE_DISTRIBUTION["BTC"]
        )

        entry = candle.vwap if candle.vwap > 0 else st.last_price
        if entry <= 0:
            return

        long_p, long_w, short_p, short_w = project_liquidations(
            entry_price=entry,
            weighted_volume_long=long_part,
            weighted_volume_short=short_part,
            leverage_buckets=config.LEVERAGE_BUCKETS,
            leverage_distribution=lev_dist,
            mmr=mmr,
        )

        # ---- Parallel DOLLAR projection (linear, no log) -------------
        # Raw $ flow weighted by decay AND OI-delta — calibrated later to OI.
        decay = math.exp(-tf.lambda_per_hour * age_hours)
        linear_vol = candle.volume_quote * decay * oi_mult
        long_dollar, short_dollar = split_long_short(linear_vol, long_ratio)
        _, dollar_long_w, _, dollar_short_w = project_liquidations(
            entry_price=entry,
            weighted_volume_long=long_dollar,
            weighted_volume_short=short_dollar,
            leverage_buckets=config.LEVERAGE_BUCKETS,
            leverage_distribution=lev_dist,
            mmr=mmr,
        )

        # Scatter into the latest time slot
        t_idx = tf.n_candles - 1
        bin_state.bins.add_long_vec(t_idx,  long_p,  long_w)
        bin_state.bins.add_short_vec(t_idx, short_p, short_w)
        bin_state.dollar_bins.add_long_vec(t_idx,  long_p,  dollar_long_w)
        bin_state.dollar_bins.add_short_vec(t_idx, short_p, dollar_short_w)
        bin_state.bins.roll_time()
        bin_state.dollar_bins.roll_time()

    # Called from backfill — bypasses live candle aggregation
    def project_synthetic_candle(self, exchange: str, coin: str, tf_label: str,
                                  candle: Candle, age_ms: int) -> None:
        st = self.state.get(coin)
        if st is None:
            return
        candle_price = candle.vwap
        if candle_price <= 0:
            return
        if st.last_price == 0:
            st.last_price = candle_price
        if st.aggregator is None:
            st.aggregator = MultiTFAggregator(self.timeframes)
        if not st.bins_by_tf:
            for tf in self.timeframes:
                st.bins_by_tf[tf.label] = BinState(
                    bins=PercentBins(
                        spot=candle_price,
                        pct_range=tf.pct_range,
                        pct_bucket=config.PCT_BUCKET,
                        n_time_steps=tf.n_candles,
                        recenter_trigger=config.RECENTER_TRIGGER,
                    ),
                    dollar_bins=PercentBins(
                        spot=candle_price,
                        pct_range=tf.pct_range,
                        pct_bucket=config.PCT_BUCKET,
                        n_time_steps=tf.n_candles,
                        recenter_trigger=config.RECENTER_TRIGGER,
                    ),
                    tf=tf,
                )
        # Pretend this exchange's context is available with neutral defaults
        # if we haven't gotten a REST snapshot yet (cold backfill case).
        if not st.contexts:
            from exchanges.base import MarketContext
            st.contexts[exchange] = MarketContext(
                exchange=exchange, coin=coin,
                open_interest_usd=1.0, funding_rate=0.0,
                long_ratio=0.5, mark_price=candle.vwap or 1.0,
                ts_ms=int(time.time() * 1000),
            )
        self._project_candle(coin, tf_label, candle, age_ms=age_ms)

    # =================================================================
    # DEX exact-position projection
    # =================================================================
    # NOTE: DEX positions are NOT written into the rolling time-matrices.
    # Doing so accumulated the same open positions on every refresh (~24× per
    # candle period). Instead, the current DEX snapshot is fetched fresh at
    # zone-extraction time via `dex_overlay()` and added as an EXACT $ overlay
    # on top of the calibrated CEX estimate. This keeps DEX correct and avoids
    # double-counting.
    def dex_overlay(self, coin: str) -> list[tuple[float, float, bool]]:
        """Current DEX positions as (liq_price, exact_dollars, is_long)."""
        if self.dex_tracker is None:
            return []
        positions = self.dex_tracker.liq_distribution(coin)
        out = []
        for liq_px, sz, is_long in positions:
            out.append((liq_px, sz * liq_px, is_long))   # notional = size × price
        return out

    # =================================================================
    # Periodic tasks
    # =================================================================
    async def poll_contexts(self) -> None:
        async with aiohttp.ClientSession() as session:
            while True:
                tasks = []
                for ex_name, client in self.clients.items():
                    for coin in self.coins:
                        tasks.append(self._safe_ctx(client, session, coin))
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for r in results:
                    if not r or isinstance(r, Exception):
                        continue
                    st = self.state[r.coin]
                    st.contexts[r.exchange] = r
                    # Use Binance mark as canonical last_price if no trades yet
                    if st.last_price == 0 and r.mark_price > 0:
                        st.last_price = r.mark_price
                # Record an OI sample per coin for OI-delta weighting
                now_ms = int(time.time() * 1000)
                for coin, st in self.state.items():
                    total_oi = sum(c.open_interest_usd for c in st.contexts.values())
                    if total_oi > 0:
                        st.oi_history.append((now_ms, total_oi))
                await asyncio.sleep(config.REST_POLL_INTERVAL_SEC)

    async def _safe_ctx(self, client, session, coin):
        try:
            return await client.fetch_context(session, coin)
        except Exception as e:
            logger.debug("ctx %s/%s: %s", client.name, coin, e)
            return None

    async def dex_refresh_loop(self) -> None:
        if self.dex_tracker is None or not hasattr(self.dex_tracker, "refresh_loop"):
            return
        await self.dex_tracker.refresh_loop(interval_sec=config.DEX_REFRESH_SEC)

    async def recenter_loop(self) -> None:
        while True:
            await asyncio.sleep(config.RECENTER_CHECK_SEC)
            for coin, st in self.state.items():
                if st.last_price <= 0:
                    continue
                for bs in st.bins_by_tf.values():
                    bs.bins.maybe_recenter(st.last_price)
                    bs.dollar_bins.maybe_recenter(st.last_price)

    # =================================================================
    # Public snapshot API for renderer / strategy
    # =================================================================
    def get_bins(self, coin: str, tf_label: str) -> PercentBins | None:
        st = self.state.get(coin)
        if not st:
            return None
        bs = st.bins_by_tf.get(tf_label)
        return bs.bins if bs else None

    def get_dollar_bins(self, coin: str, tf_label: str) -> PercentBins | None:
        st = self.state.get(coin)
        if not st:
            return None
        bs = st.bins_by_tf.get(tf_label)
        return bs.dollar_bins if bs else None

    def get_candle_history(self, coin: str, tf_label: str) -> list:
        st = self.state.get(coin)
        if not st:
            return []
        bs = st.bins_by_tf.get(tf_label)
        return list(bs.candle_history) if bs else []

    def _master_tf_label(self) -> str:
        """The widest timeframe is the master dollar map — single source of
        truth for liquidation $; all zoom views read from it."""
        return self.timeframes[-1].label   # "1m" (±35%, longest lookback)

    def _master_frac(self, coin: str) -> float:
        """In-range fraction for the MASTER range (not the zoom range), since
        we calibrate the master map once and all zooms read it."""
        master = self.timeframes[-1]
        dist = self.leverage_dist.get(coin, config.LEVERAGE_DISTRIBUTION["BTC"])
        frac = sum(f for L, f in dist.items() if (1.0 / L) <= master.pct_range)
        return max(0.05, min(1.0, frac))

    def _zoom_range_for(self, tf_label: str) -> float:
        for tf in self.timeframes:
            if tf.label == tf_label:
                return tf.pct_range
        return self.timeframes[-1].pct_range

    def extract_zones_for(self, coin: str, tf_label: str,
                          max_per_side: int = 5):
        """Zones for a timeframe = the MASTER dollar map, zoomed to the TF's
        band. Same price → same $ across timeframes; only the band differs."""
        from core.zone_extractor import extract_zones
        st = self.state.get(coin)
        if not st:
            return {"long": [], "short": []}
        master = st.bins_by_tf.get(self._master_tf_label())
        if master is None:
            return {"long": [], "short": []}
        ctx = self.context_summary(coin)
        return extract_zones(
            dollar_bins=master.dollar_bins,
            current_price=self.current_price(coin),
            total_oi_usd=ctx["total_oi_usd"],
            long_ratio=ctx["avg_long_ratio"],
            max_zones_per_side=max_per_side,
            dex_positions=self.dex_overlay(coin),
            liquidatable_frac=self._master_frac(coin),
            zoom_range=self._zoom_range_for(tf_label),
        )

    def slice_packet_for(self, coin: str, tf_label: str,
                         n_above: int = 20, n_below: int = 20) -> dict:
        """Slice packet for a timeframe = the MASTER dollar map, sliced at the
        TF's zoom width. Same price → same $ across timeframes."""
        from core.slice_packet import build_slice_packet
        st = self.state.get(coin)
        if not st:
            return {"current_price": self.current_price(coin), "slices": []}
        master = st.bins_by_tf.get(self._master_tf_label())
        if master is None:
            return {"current_price": self.current_price(coin), "slices": []}
        ctx = self.context_summary(coin)
        zoom = self._zoom_range_for(tf_label)
        # FIXED 1% slice width (not range/20) so the SAME price band reads the
        # SAME master cells in every timeframe → identical $ across TFs. The
        # slice COUNT varies with the zoom range instead (24h ±10% → 10±,
        # 1w ±20% → 20±, 1m ±35% → 35±).
        per_side = max(1, int(round(zoom / 0.01)))
        return build_slice_packet(
            dollar_bins=master.dollar_bins,
            current_price=self.current_price(coin),
            total_oi_usd=ctx["total_oi_usd"],
            long_ratio=ctx["avg_long_ratio"],
            dex_positions=self.dex_overlay(coin),
            n_above=per_side, n_below=per_side,
            slice_pct=0.01,
            liquidatable_frac=self._master_frac(coin),
        )

    def current_price(self, coin: str) -> float:
        st = self.state.get(coin)
        return st.last_price if st else 0.0

    def context_summary(self, coin: str) -> dict:
        st = self.state.get(coin)
        if not st or not st.contexts:
            return {"avg_funding": 0.0, "avg_long_ratio": 0.5, "total_oi_usd": 0.0}
        ctxs = list(st.contexts.values())
        return {
            "avg_funding":    sum(c.funding_rate for c in ctxs) / len(ctxs),
            "avg_long_ratio": sum(c.long_ratio   for c in ctxs) / len(ctxs),
            "total_oi_usd":   sum(c.open_interest_usd for c in ctxs),
            "per_exchange":   {c.exchange: {
                "oi": c.open_interest_usd,
                "fund": c.funding_rate,
                "lr": c.long_ratio,
            } for c in ctxs},
        }

    # =================================================================
    # Entry point
    # =================================================================
    async def run(self) -> None:
        # Start every exchange's WS stream
        ws_tasks = []
        for name, client in self.clients.items():
            ws_tasks.append(asyncio.create_task(
                client.stream(self.on_trade, self.on_liquidation)
            ))
        # Periodic tasks
        periodic = [
            asyncio.create_task(self.poll_contexts()),
            asyncio.create_task(self.dex_refresh_loop()),
            asyncio.create_task(self.recenter_loop()),
        ]
        await asyncio.gather(*ws_tasks, *periodic)
