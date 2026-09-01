"""
backtest/datafeed.py
══════════════════════════════════════════════════════════════════════
Feeds historical OHLCV to the walk-forward engine WITHOUT look-ahead.

Design: everything is derived from a single 1h series (the finest TF used
in scoring after 15m was dropped). 4h and 1d are RESAMPLED from 1h, which
guarantees higher-TF bars are only "visible" once fully closed — no
timestamp-alignment bugs, no leakage.

Sources:
    load_csv(path)         — real OHLCV (e.g. exported from MT5). Columns:
                             time, open, high, low, close, volume  (time = UTC).
    generate_synthetic(...) — realistic regime-switching GBM so the engine
                              runs end-to-end for a demo. Clearly synthetic.

Look-ahead control (the cardinal rule):
    At the close of 1h bar T (wall-clock eval_time = open[T] + 1h), a bar of
    ANY timeframe is visible iff its close_time ≤ eval_time. slice_at(T)
    enforces this with searchsorted — O(log n) per step.
"""
from __future__ import annotations
import hashlib
import numpy as np
import pandas as pd


def _stable_seed(symbol: str, base: int) -> int:
    """Deterministic per-(symbol, base) seed — unlike hash(), stable across runs."""
    h = int(hashlib.md5(symbol.encode("utf-8")).hexdigest(), 16)
    return (base * 1_000_003 + h) % (2 ** 32)

_TF_DELTA = {"1h": pd.Timedelta(hours=1), "4h": pd.Timedelta(hours=4),
             "1d": pd.Timedelta(days=1)}


def load_csv(path: str) -> pd.DataFrame:
    """Load a 1h OHLCV CSV with a UTC 'time' column → DatetimeIndex frame."""
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]
    tcol = "time" if "time" in df.columns else df.columns[0]
    df[tcol] = pd.to_datetime(df[tcol], utc=True)
    df = df.set_index(tcol).sort_index()
    keep = [c for c in ("open", "high", "low", "close", "volume") if c in df.columns]
    if "volume" not in df.columns:
        df["volume"] = 0.0
        keep.append("volume")
    return df[keep]


def generate_synthetic(symbol: str, n_hours: int = 24 * 120, seed: int = 0,
                       start_price: float = 2000.0) -> pd.DataFrame:
    """
    Regime-switching geometric Brownian motion with volatility clustering.
    Produces ~`n_hours` hourly bars (default 120 days). SYNTHETIC — for
    pipeline validation only; swap in real data via load_csv for real results.
    """
    rng = np.random.default_rng(_stable_seed(symbol, seed))
    # Regime: drift + vol pairs (bull / bear / range), Markov-switched.
    regimes = [(+0.00018, 0.006), (-0.00018, 0.006), (0.0, 0.004), (+0.00040, 0.010)]
    out = np.empty(n_hours); out[0] = start_price
    r = 0
    closes = [start_price]
    highs, lows, opens, vols = [], [], [], []
    price = start_price
    for i in range(n_hours):
        if rng.random() < 0.012:               # ~ regime switch every ~83h
            r = rng.integers(0, len(regimes))
        drift, vol = regimes[r]
        ret = rng.normal(drift, vol)
        o = price
        c = max(1e-6, price * (1 + ret))
        intrabar = abs(rng.normal(0, vol)) * price
        hi = max(o, c) + intrabar * rng.random()
        lo = min(o, c) - intrabar * rng.random()
        v = abs(rng.normal(1.0, 0.4)) * 1_000 * (1 + 3 * abs(ret) / max(vol, 1e-9))
        opens.append(o); highs.append(hi); lows.append(lo); closes.append(c); vols.append(v)
        price = c
    idx = pd.date_range("2025-01-01", periods=n_hours, freq="1h", tz="UTC")
    return pd.DataFrame({"open": opens, "high": highs, "low": lows,
                         "close": closes[1:], "volume": vols}, index=idx)


def _resample(df_1h: pd.DataFrame, rule: str) -> pd.DataFrame:
    o = df_1h["open"].resample(rule).first()
    h = df_1h["high"].resample(rule).max()
    l = df_1h["low"].resample(rule).min()
    c = df_1h["close"].resample(rule).last()
    v = df_1h["volume"].resample(rule).sum()
    out = pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": v}).dropna()
    return out


class DataFeed:
    """Pre-resamples 1h→4h/1d once, then serves leak-free slices by step index."""
    # Live bot's fixed lookback depths (main.py candles_per_tf) — the backtest
    # must use the SAME rolling window or its S/R diverges from live over time.
    _DEPTHS = {"1d": 200, "4h": 600, "1h": 400}

    def __init__(self, df_1h: pd.DataFrame, depths: dict | None = None):
        self.df_1h = df_1h
        self.df_4h = _resample(df_1h, "4h")
        self.df_1d = _resample(df_1h, "1d")
        self.depths = depths or dict(self._DEPTHS)
        # close_time arrays for searchsorted (bar open + tf duration)
        self._ct = {
            "1h": (self.df_1h.index + _TF_DELTA["1h"]).values,
            "4h": (self.df_4h.index + _TF_DELTA["4h"]).values,
            "1d": (self.df_1d.index + _TF_DELTA["1d"]).values,
        }
        self._frames = {"1h": self.df_1h, "4h": self.df_4h, "1d": self.df_1d}

    def __len__(self):
        return len(self.df_1h)

    def slice_at(self, t: int) -> dict[str, pd.DataFrame]:
        """
        The last `depth` bars of each TF that have CLOSED by the close of 1h
        bar t — a fixed rolling window identical to what the live bot fetches
        (1d:200, 4h:600, 1h:400). Early on (before `depth` bars exist) it
        simply returns however many are available, exactly as the live bot
        would if it had only been running that long.
        """
        eval_time = (self.df_1h.index[t] + _TF_DELTA["1h"]).to_datetime64()
        out = {}
        for tf, frame in self._frames.items():
            n = int(np.searchsorted(self._ct[tf], eval_time, side="right"))
            lo = max(0, n - self.depths.get(tf, n))
            out[tf] = frame.iloc[lo:n]
        return out

    def price_at(self, t: int) -> float:
        return float(self.df_1h["close"].iloc[t])

    def bar(self, t: int):
        row = self.df_1h.iloc[t]
        return (float(row["open"]), float(row["high"]),
                float(row["low"]), float(row["close"]))

    def timestamp_at(self, t: int) -> str:
        return str(self.df_1h.index[t])
