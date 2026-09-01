"""
calibration/fit_leverage.py
===========================
Offline fitter. Reads the JSONL produced by LiquidationRecorder and derives
the EMPIRICAL leverage distribution per coin, then writes
`_calib/calibrated_leverage.json` which the engine loads at startup.

Implied leverage from one liquidation:
    long  liq at P from recent high H : L = H / (H − P)
    short liq at P from recent low  Lo: L = Lo / (P − Lo)

Each liquidation's USD value is added to the NEAREST configured leverage
bucket. Per coin we then normalize the bucket USD sums to a probability mass.

Usage:
    python -m calibration.fit_leverage              # default paths
    python -m calibration.fit_leverage --min 2000   # require ≥2000 events/coin
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import config

BUCKETS = config.LEVERAGE_BUCKETS


def _nearest_bucket(lev: float) -> int:
    return min(BUCKETS, key=lambda b: abs(b - lev))


def implied_leverage(rec: dict) -> float | None:
    p = rec.get("price", 0.0)
    if p <= 0:
        return None
    side = rec.get("side")
    if side == "LONG_LIQ":
        h = rec.get("ref_high", 0.0)
        if h <= p:
            return None
        return h / (h - p)
    if side == "SHORT_LIQ":
        lo = rec.get("ref_low", 0.0)
        if lo <= 0 or p <= lo:
            return None
        return lo / (p - lo)
    return None


def fit(jsonl_path: str, min_events_per_coin: int = 1000) -> dict:
    """Returns {coin: {leverage: fraction}}. Coins below the event floor are
    left out (caller keeps the config default for them)."""
    # coin → bucket → usd
    acc: dict[str, dict[int, float]] = defaultdict(lambda: {b: 0.0 for b in BUCKETS})
    counts: dict[str, int] = defaultdict(int)

    path = Path(jsonl_path)
    if not path.exists():
        print(f"[fit] no data file at {path}")
        return {}

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            lev = implied_leverage(rec)
            if lev is None:
                continue
            lev = max(BUCKETS[0], min(BUCKETS[-1], lev))   # clamp to [2,125]
            b = _nearest_bucket(lev)
            acc[rec["coin"]][b] += float(rec.get("qty_usd", 0.0))
            counts[rec["coin"]] += 1

    out = {}
    for coin, buckets in acc.items():
        if counts[coin] < min_events_per_coin:
            print(f"[fit] {coin}: only {counts[coin]} events (< {min_events_per_coin}) — skipped")
            continue
        total = sum(buckets.values())
        if total <= 0:
            continue
        dist = {b: round(buckets[b] / total, 5) for b in BUCKETS}
        # renormalize rounding drift
        s = sum(dist.values())
        dist = {b: round(v / s, 5) for b, v in dist.items()}
        out[coin] = dist
        print(f"[fit] {coin}: {counts[coin]} events → {dist}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", default="./_calib/liquidations.jsonl")
    ap.add_argument("--out",   default="./_calib/calibrated_leverage.json")
    ap.add_argument("--min", type=int, default=1000,
                    help="min liquidation events per coin before trusting the fit")
    args = ap.parse_args()

    dist = fit(args.jsonl, min_events_per_coin=args.min)
    if not dist:
        print("[fit] nothing fitted — keeping config defaults")
        return
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(dist, f, indent=2)
    print(f"[fit] wrote {args.out}  ({len(dist)} coins)")


if __name__ == "__main__":
    main()
