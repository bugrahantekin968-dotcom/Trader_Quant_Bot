"""
Liquidation Heatmap Engine — Configuration
==========================================
Single source of truth for symbols, timeframes, exchanges, calibration.
All other modules import from here; nothing is hard-coded elsewhere.
"""
from __future__ import annotations

# =============================================================================
# Universe
# =============================================================================
SYMBOLS = ["BTC", "ETH", "XRP"]


# =============================================================================
# Per-symbol leverage profile  (empirical retail distribution)
# =============================================================================
# Different assets attract different leverage cohorts. BTC has the deepest
# institutional flow (lower avg leverage), XRP is dominated by retail
# (higher tail at 50x+). These weights were calibrated against historical
# CoinGlass / Hyblock heatmap projections.
LEVERAGE_BUCKETS = [2, 3, 5, 10, 20, 25, 50, 75, 100, 125]

LEVERAGE_DISTRIBUTION = {
    "BTC": {2: 0.12, 3: 0.10, 5: 0.16, 10: 0.20, 20: 0.14,
            25: 0.10, 50: 0.09, 75: 0.04, 100: 0.04, 125: 0.01},
    "ETH": {2: 0.10, 3: 0.09, 5: 0.15, 10: 0.20, 20: 0.15,
            25: 0.11, 50: 0.10, 75: 0.05, 100: 0.04, 125: 0.01},
    "XRP": {2: 0.06, 3: 0.07, 5: 0.12, 10: 0.18, 20: 0.16,
            25: 0.13, 50: 0.13, 75: 0.07, 100: 0.06, 125: 0.02},
}
# Each distribution must be a valid probability mass (sums to 1.0) and cover
# exactly the declared leverage buckets.
for _coin, _dist in LEVERAGE_DISTRIBUTION.items():
    assert set(_dist.keys()) == set(LEVERAGE_BUCKETS), \
        f"{_coin} leverage buckets mismatch"
    assert abs(sum(_dist.values()) - 1.0) < 1e-9, \
        f"{_coin} leverage distribution sums to {sum(_dist.values())}, not 1.0"


# =============================================================================
# Maintenance margin rates  (base tier per exchange)
# =============================================================================
# Real exchanges use a tiered MMR ladder that grows with position size.
# For heatmap purposes the base-tier MMR dominates the retail cluster.
MAINTENANCE_MARGIN_RATES = {
    "binance": 0.004,
    "bybit":   0.005,
    "okx":     0.005,
    "bitget":  0.005,
    "hyperliquid": 0.0125,   # variable on Hyperliquid, conservative midpoint
}


# =============================================================================
# Percent-based price grid
# =============================================================================
PCT_RANGE      = 0.20    # ±20% from spot
PCT_BUCKET     = 0.001   # 0.1% cell — for CoinGlass-style sharp horizontal lines
N_PRICE_BINS   = int(2 * PCT_RANGE / PCT_BUCKET)   # = 400
RECENTER_TRIGGER = 0.05  # re-anchor grid if spot moves >5% from anchor


# =============================================================================
# Timeframes  (window length, candle size, decay)
# =============================================================================
# Half-life is set to ~10% of the window so:
#   - Recent candles dominate
#   - But the full window still contributes meaningfully
# This produces visuals comparable to CoinGlass Model-2 / Model-3 outputs.
#
# decay = exp(-λ · age_hours)
# pct_range scales with the timeframe: short windows need a tight band (price
# rarely moves far in 24h), long windows need a wide band (price can move >20%
# in a month). This matches CoinGlass, which widens the price axis for longer TFs.
TIMEFRAMES = [
    # label,  window_sec,    candle_sec,  n_candles,  lambda_per_hour,  pct_range
    ("24h",   24 * 3600,     6 * 60,      240,        0.1386,           0.10),
    ("3d",    3 * 86400,     30 * 60,     144,        0.0289,           0.15),
    ("1w",    7 * 86400,     2 * 3600,    84,         0.00825,          0.20),
    ("1m",    30 * 86400,    4 * 3600,    180,        0.00193,          0.35),
]
# Sanity check (window == candle * n_candles)
for _row in TIMEFRAMES:
    _label, _window_s, _candle_s, _n, _lam, _rng = _row
    assert _window_s == _candle_s * _n, f"timeframe {_label} inconsistent"


# =============================================================================
# Multi-timeframe swing-signal weights
# =============================================================================
# 3d and 1w dominate because that's where swing setups crystallize.
# 24h gives tactical entry, 1m gives macro context.
SIGNAL_TF_WEIGHTS = {"24h": 0.15, "3d": 0.40, "1w": 0.30, "1m": 0.15}
assert abs(sum(SIGNAL_TF_WEIGHTS.values()) - 1.0) < 1e-6

# A swing target magnet must sit at least this far from spot, else it's a scalp
# (a magnet $100 away gives a useless R:R). Used by strategy/signals.py.
MIN_SWING_TARGET_PCT = 0.015   # 1.5%


# =============================================================================
# Schedulers
# =============================================================================
REST_POLL_INTERVAL_SEC = 12     # OI / funding / L-S ratio
DEX_REFRESH_SEC        = 20     # Hyperliquid clearinghouseState batches
HEATMAP_RENDER_SEC     = 60     # re-paint cadence
SNAPSHOT_SEC           = 300    # parquet persistence
SIGNAL_EMIT_SEC        = 30     # signal recomputation
RECENTER_CHECK_SEC     = 30     # grid alignment


# =============================================================================
# Exchange WebSocket / REST endpoints
# =============================================================================
EXCHANGE_ENDPOINTS = {
    "binance": {
        "ws":   "wss://fstream.binance.com/ws",
        "rest": "https://fapi.binance.com",
    },
    "bybit": {
        "ws":   "wss://stream.bybit.com/v5/public/linear",
        "rest": "https://api.bybit.com",
    },
    "okx": {
        "ws":   "wss://ws.okx.com:8443/ws/v5/public",
        "rest": "https://www.okx.com",
    },
    "bitget": {
        "ws":   "wss://ws.bitget.com/v2/ws/public",
        "rest": "https://api.bitget.com",
    },
    "hyperliquid": {
        "ws":   "wss://api.hyperliquid.xyz/ws",
        "rest": "https://api.hyperliquid.xyz/info",
    },
    # ── New venues — endpoints from API knowledge; VERIFY before production ──
    "gate": {
        "ws":   "wss://fx-ws.gateio.ws/v4/ws/usdt",
        "rest": "https://api.gateio.ws",
    },
    "mexc": {
        "ws":   "wss://contract.mexc.com/edge",
        "rest": "https://contract.mexc.com",
    },
    "htx": {
        "ws":   "wss://api.hbdm.com/linear-swap-ws",
        "rest": "https://api.hbdm.com",
    },
    "kraken": {
        "ws":   "wss://futures.kraken.com/ws/v1",
        "rest": "https://futures.kraken.com",
    },
}


# =============================================================================
# Symbol mapping  (engine-side → exchange-side)
# =============================================================================
# Engine uses bare coin names (BTC, ETH, XRP). Each exchange has its own
# convention for the USDT-margined perpetual.
def symbol_for(exchange: str, coin: str) -> str:
    if exchange in ("binance", "bybit", "okx_swap_internal"):
        return f"{coin}USDT"
    if exchange == "okx":
        return f"{coin}-USDT-SWAP"
    if exchange == "bitget":
        return f"{coin}USDT"
    if exchange == "hyperliquid":
        return coin
    raise ValueError(f"unknown exchange {exchange}")


# =============================================================================
# Output paths
# =============================================================================
OUTPUT_DIR     = "./_out"
SNAPSHOT_DIR   = "./_snapshots"
