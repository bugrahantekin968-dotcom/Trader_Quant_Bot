"""
backtest/fetch_binance_data.py
══════════════════════════════════════════════════════════════════════
Pull DEEP 1h OHLCV history from Binance (public data, no API key) and write
the exact CSV the walk-forward backtester expects:

    data/BTCUSD.csv  data/ETHUSD.csv  data/XRPUSD.csv
    columns: time(UTC ISO), open, high, low, close, volume

The engine derives 4h/1d by resampling the 1h series, so 1h is all we need.
Prices on Binance ≈ Eightcap (<~0.5% on the major zones we trade), so this is a
faithful proxy for the backtest; live execution still runs on Eightcap MT5.

USAGE (run locally, in the env that already has ccxt):
    python -m backtest.fetch_binance_data --years 3
    python -m backtest.fetch_binance_data --years 5 --out-dir ./data
    python -m backtest.fetch_binance_data --symbols BTC/USDT,ETH/USDT --years 2

Then run the backtest on the freshly-pulled data:
    python -m backtest.run_backtest --csv-dir ./data --threshold 49
    python -m backtest.run_backtest --csv-dir ./data --sweep 40 55 5

NOTE: ~3 years of 1h bars ≈ 26k rows/symbol; the fetch paginates 1000 bars per
call with rate-limiting, so a 3y/3-symbol pull takes a couple of minutes.
"""
from __future__ import annotations
import argparse
import os
import time

import pandas as pd

try:
    import ccxt
except ImportError:  # pragma: no cover
    raise SystemExit("ccxt gerekli:  pip install ccxt")

# Binance spot symbol -> MT5 symbol used for the CSV filename / backtest key.
DEFAULT_MAP = {
    "BTC/USDT": "BTCUSD",
    "ETH/USDT": "ETHUSD",
    "XRP/USDT": "XRPUSD",
}


def _make_exchange(name: str):
    """Build a public (keyless) ccxt exchange with rate-limiting on."""
    if not hasattr(ccxt, name):
        raise SystemExit(f"ccxt borsa bilinmiyor: {name}")
    ex = getattr(ccxt, name)({"enableRateLimit": True})
    ex.load_markets()
    return ex


def fetch_ohlcv_paginated(ex, symbol: str, timeframe: str, since_ms: int,
                          batch_limit: int = 1000, max_retries: int = 5) -> list:
    """Walk fetch_ohlcv forward from since_ms to now, returning all bars.

    Handles the per-call cap (batch_limit) and transient network/exchange
    errors with exponential backoff. Stops when a short/empty batch or a batch
    that does not advance the cursor is returned (end of history reached).
    """
    tf_ms = ex.parse_timeframe(timeframe) * 1000
    now_ms = ex.milliseconds()
    rows: list = []
    cursor = since_ms
    while cursor < now_ms:
        for attempt in range(max_retries):
            try:
                batch = ex.fetch_ohlcv(symbol, timeframe, since=cursor, limit=batch_limit)
                break
            except (ccxt.NetworkError, ccxt.ExchangeError) as e:  # type: ignore[attr-defined]
                wait = 2 ** attempt
                print(f"    [retry {attempt+1}/{max_retries}] {type(e).__name__}: {e} — {wait}s bekle")
                time.sleep(wait)
        else:
            print("    [warn] max retries — bu sembolde durduruldu")
            break

        if not batch:
            break
        rows.extend(batch)
        last_ts = batch[-1][0]
        next_cursor = last_ts + tf_ms
        if next_cursor <= cursor:        # no forward progress -> done
            break
        cursor = next_cursor
        done = len(rows)
        span_days = (last_ts - rows[0][0]) / 86_400_000
        print(f"    ...{done:>7,} bar  (son: {pd.to_datetime(last_ts, unit='ms', utc=True)}, ~{span_days:.0f} gün)", end="\r")
        if len(batch) < batch_limit:     # exchange returned a partial last page
            break
    print()
    return rows


def rows_to_frame(rows: list) -> pd.DataFrame:
    """OHLCV rows -> deduped, sorted DataFrame with a UTC 'time' column."""
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df = df.drop_duplicates(subset="ts").sort_values("ts").reset_index(drop=True)
    df["time"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df[["time", "open", "high", "low", "close", "volume"]]


def main(argv=None):
    p = argparse.ArgumentParser(description="Binance 1h OHLCV -> backtest CSV")
    p.add_argument("--symbols", type=str, default=",".join(DEFAULT_MAP),
                   help="virgülle ayrık Binance sembolleri (örn. BTC/USDT,ETH/USDT)")
    p.add_argument("--timeframe", type=str, default="1h", help="mum periyodu (varsayılan 1h)")
    p.add_argument("--years", type=float, default=3.0, help="kaç yıl geriye (varsayılan 3)")
    p.add_argument("--out-dir", type=str, default="data", help="CSV çıktı klasörü")
    p.add_argument("--exchange", type=str, default="binance",
                   help="ccxt borsa adı (erişim engeliyse: binanceus, kraken, bybit ...)")
    args = p.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)
    ex = _make_exchange(args.exchange)
    since_ms = ex.milliseconds() - int(args.years * 365 * 24 * 3600 * 1000)

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    print(f"[fetch] borsa={args.exchange} tf={args.timeframe} ~{args.years}y geriye -> {args.out_dir}/\n")
    _ranges: list[str] = []
    for sym in symbols:
        mt5_sym = DEFAULT_MAP.get(sym, sym.replace("/", ""))
        if sym not in ex.markets:
            print(f"[skip] {sym} bu borsada yok"); continue
        print(f"[{sym} -> {mt5_sym}.csv] çekiliyor...")
        rows = fetch_ohlcv_paginated(ex, sym, args.timeframe, since_ms)
        if not rows:
            print(f"[skip] {sym}: veri gelmedi"); continue
        df = rows_to_frame(rows)
        path = os.path.join(args.out_dir, f"{mt5_sym}.csv")
        df.to_csv(path, index=False)
        print(f"[done] {path}: {len(df):,} adet {args.timeframe} bar "
              f"({df['time'].iloc[0].date()} -> {df['time'].iloc[-1].date()})\n")
        _ranges.append(f"{mt5_sym}: {df['time'].iloc[0].date()} -> {df['time'].iloc[-1].date()} ({len(df):,} bar)")

    # VERI DAMGASI — her cekimde yenilenir; run_backtest bunu cikti klasorune
    # kopyalar. Kural: damgasi farkli iki kosu ASLA kiyaslanamaz (taze veri =
    # yeni pencere = yeni referans gerekir).
    if _ranges:
        _stamp = os.path.join(args.out_dir, "DATA_VERSION.txt")
        with open(_stamp, "w", encoding="utf-8") as f:
            f.write(f"fetched: {pd.Timestamp.utcnow().strftime('%Y-%m-%d %H:%M')} UTC  "
                    f"exchange={args.exchange} tf={args.timeframe}\n")
            for line in _ranges:
                f.write(line + "\n")
        print(f"[stamp] {_stamp} yazildi — bu damga her kosunun cikti klasorune kopyalanir")

    print("Bitti. Backtest:\n  python -m backtest.run_backtest --csv-dir "
          f"{args.out_dir} --threshold 49")


if __name__ == "__main__":
    main()
