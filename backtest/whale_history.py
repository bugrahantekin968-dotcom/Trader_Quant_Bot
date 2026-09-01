"""
backtest/whale_history.py
══════════════════════════════════════════════════════════════════════
Historical whale (D pillar) provider for the backtest.

Fetches the SAME CoinGlass v4 endpoints the live bot uses — just with bigger
limits so we get the full time series instead of only the latest point:
    L/S ratio : /api/futures/top-long-short-account-ratio/history
    OI        : /api/futures/open-interest/aggregated-history

Plan-aware granularity (your CoinGlass plan: 4h=180d, 1d=unlimited):
    L/S  → 1d  (unlimited) → covers the whole backtest window. D.1 contrarian
           is the core whale signal (10 pts) so we want maximum coverage.
    OI   → 4h  (180d) → true "OI change 4h" exactly like live D.3. Trades older
           than 180d simply get neutral OI (oi_change=None → D.3 1/1).

Funding (D.2, 8 pts): the live exchange-list endpoint is a SNAPSHOT (no
history) and live funding was ~always Neutral (3/3). So funding is defaulted
to neutral (funding_raw=0.0 → D.2 3/3) — a tiny, well-justified approximation.

last_4h_green (also D.3) is derived by the scorer from the PRICE feed, not
here — so this provider never needs price data.

Leak-free: at eval time T we return the most recent whale reading with
timestamp ≤ T (bisect). Never a future point.

Usage (two steps, like export_mt5.py):
    # 1) fetch once → data/whale_cache.json   (see fetch_whale.py)
    # 2) backtest:
    python -m backtest.run_backtest --csv-dir ./data \
           --whale-cache ./data/whale_cache.json --threshold 65
"""
from __future__ import annotations
import bisect, json, time, urllib.parse, urllib.request
from datetime import datetime

CG_BASE = "https://open-api-v4.coinglass.com"

# backtest symbol → (OI symbol, L/S pair)
_SYM = {
    "BTCUSD": ("BTC", "BTCUSDT"),
    "ETHUSD": ("ETH", "ETHUSDT"),
    "XRPUSD": ("XRP", "XRPUSDT"),
}

# CoinGlass interval forms — try v4 ("1d") then legacy ("d1") like the live bot
_INTERVAL_ALT = {"1d": "d1", "d1": "1d", "4h": "h4", "h4": "4h", "1h": "h1", "h1": "1h"}


# ──────────────────────────────────────────────────────────────────────
# Low-level GET (mirrors CoinglassManager._get: CG-API-KEY header, success/code)
# ──────────────────────────────────────────────────────────────────────
def _get(api_key: str, endpoint: str, params: dict):
    url = CG_BASE + endpoint + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={
        "CG-API-KEY": api_key, "accept": "application/json"})
    with urllib.request.urlopen(req, timeout=25) as r:
        payload = json.loads(r.read().decode("utf-8"))
    ok = (payload.get("success") is True or
          str(payload.get("code", "")) in ("0", "000000", "200"))
    if not ok:
        raise RuntimeError(f"CG fail {endpoint}: {payload.get('msg', '?')} "
                           f"(code {payload.get('code')})")
    data = payload.get("data")
    if isinstance(data, dict) and "list" in data:
        return data["list"]
    return data if isinstance(data, list) else (data or [])


def _get_history(api_key, endpoint, params, interval):
    """Try the requested interval form, then its alternate; return first non-empty."""
    for form in (interval, _INTERVAL_ALT.get(interval, interval)):
        p = dict(params); p["interval"] = form
        try:
            data = _get(api_key, endpoint, p)
            if data:
                return data
        except Exception as exc:
            last = exc
    return []


# ──────────────────────────────────────────────────────────────────────
# Field extractors (defensive multi-key, like the live _parse_* helpers)
# ──────────────────────────────────────────────────────────────────────
def _ts_of(item):
    for k in ("time", "t", "createTime", "ts", "timestamp", "endTime"):
        v = item.get(k)
        if v is not None:
            try:
                v = int(float(v))
            except (TypeError, ValueError):
                continue
            return v // 1000 if v > 1_000_000_000_000 else v   # ms → s
    return None

def _ls_of(item):
    for k in ("top_account_long_short_ratio", "longShortRatio", "ratio",
              "global_account_long_short_ratio", "long_short_ratio"):
        v = item.get(k)
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                pass
    return None

def _oi_of(item):
    for k in ("openInterest", "close", "c", "open_interest", "openInterestUsd"):
        v = item.get(k)
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                pass
    return None


# ──────────────────────────────────────────────────────────────────────
# Fetch + cache
# ──────────────────────────────────────────────────────────────────────
def fetch_symbol(api_key, oi_sym, ls_pair, ls_interval="1d", oi_interval="4h",
                 limit=4500, exchange="Binance"):
    ls_raw = _get_history(api_key, "/api/futures/top-long-short-account-ratio/history",
                          {"exchange": exchange, "symbol": ls_pair, "limit": limit},
                          ls_interval)
    ls = sorted((t, v) for t, v in
                ((_ts_of(x), _ls_of(x)) for x in ls_raw if isinstance(x, dict))
                if t is not None and v is not None)

    oi_raw = _get_history(api_key, "/api/futures/open-interest/aggregated-history",
                          {"symbol": oi_sym, "limit": limit}, oi_interval)
    oi = sorted((t, v) for t, v in
                ((_ts_of(x), _oi_of(x)) for x in oi_raw if isinstance(x, dict))
                if t is not None and v is not None)
    return {"ls": ls, "oi": oi}


def fetch_and_cache(api_key, symbols, out_path,
                    ls_interval="1d", oi_interval="4h", limit=4500, pause=0.6):
    cache = {"ls_interval": ls_interval, "oi_interval": oi_interval, "symbols": {}}
    for sym in symbols:
        oi_sym, ls_pair = _SYM.get(sym, (sym.replace("USD", ""), sym.replace("USD", "") + "USDT"))
        try:
            d = fetch_symbol(api_key, oi_sym, ls_pair, ls_interval, oi_interval, limit)
            cache["symbols"][sym] = d
            ls_span = (f"{datetime.utcfromtimestamp(d['ls'][0][0]).date()}→"
                       f"{datetime.utcfromtimestamp(d['ls'][-1][0]).date()}") if d["ls"] else "yok"
            print(f"[whale] {sym}: L/S {len(d['ls'])} nokta ({ls_span}) | OI {len(d['oi'])} nokta")
        except Exception as exc:
            print(f"[whale] {sym}: HATA → {exc}")
            cache["symbols"][sym] = {"ls": [], "oi": []}
        time.sleep(pause)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(cache, f)
    print(f"[whale] cache yazıldı → {out_path}")
    return cache


# ──────────────────────────────────────────────────────────────────────
# Provider (leak-free lookup)
# ──────────────────────────────────────────────────────────────────────
def _parse_ts(ts_str: str) -> int:
    dt = datetime.fromisoformat(str(ts_str).replace("Z", "").strip())
    if dt.tzinfo is not None:
        dt = dt.replace(tzinfo=None)        # treat as UTC-naive epoch
    epoch = datetime(1970, 1, 1)
    return int((dt - epoch).total_seconds())


class WhaleProvider:
    def __init__(self, cache: dict):
        self.sym = cache.get("symbols", {})
        self._ls_t = {s: [t for t, _ in d.get("ls", [])] for s, d in self.sym.items()}
        self._oi_t = {s: [t for t, _ in d.get("oi", [])] for s, d in self.sym.items()}

    def __call__(self, symbol: str, ts_str: str):
        d = self.sym.get(symbol)
        if not d:
            return None
        ts = _parse_ts(ts_str)

        # L/S: most recent ≤ ts
        ls_arr = self._ls_t.get(symbol, [])
        i = bisect.bisect_right(ls_arr, ts) - 1
        ls_ratio = d["ls"][i][1] if i >= 0 else None

        # OI change: last two points ≤ ts
        oi_arr = self._oi_t.get(symbol, [])
        j = bisect.bisect_right(oi_arr, ts) - 1
        oi_change = None
        if j >= 1:
            cur, prev = d["oi"][j][1], d["oi"][j - 1][1]
            if prev:
                oi_change = (cur - prev) / prev * 100.0

        return {
            "ls_ratio":          ls_ratio,
            "funding_raw":       0.0,           # neutral (live was ~always neutral)
            "oi_change_4h_pct":  oi_change,
            # last_4h_green injected by scorer from the price feed
        }


def load_provider(cache_path: str) -> WhaleProvider:
    with open(cache_path, encoding="utf-8") as f:
        return WhaleProvider(json.load(f))
