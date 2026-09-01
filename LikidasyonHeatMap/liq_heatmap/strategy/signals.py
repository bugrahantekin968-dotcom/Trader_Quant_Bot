"""
Zone-anchored multi-timeframe swing signal.
============================================
Signal is now derived from extracted zones, not raw bin density.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import config
from core.zone_extractor import Zone, nearest_zone_above, nearest_zone_below


class Bias(str, Enum):
    HUNT_LONGS  = "HUNT_LONGS"
    HUNT_SHORTS = "HUNT_SHORTS"
    NEUTRAL     = "NEUTRAL"


@dataclass
class TFSignal:
    tf: str
    bias: Bias
    confidence: float
    up_magnet: Zone | None
    down_magnet: Zone | None


@dataclass
class SwingSignal:
    coin: str
    price: float
    funding: float
    long_ratio: float
    per_tf: dict[str, TFSignal]
    consensus_bias: Bias
    consensus_confidence: float
    target: Zone | None
    stop_anchor: Zone | None
    rr_estimate: float | None
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "coin": self.coin,
            "price": self.price,
            "funding": self.funding,
            "long_ratio": self.long_ratio,
            "consensus_bias": self.consensus_bias.value,
            "consensus_confidence": round(self.consensus_confidence, 3),
            "target":      self.target.to_dict()      if self.target      else None,
            "stop_anchor": self.stop_anchor.to_dict() if self.stop_anchor else None,
            "rr_estimate": round(self.rr_estimate, 2) if self.rr_estimate is not None else None,
            "per_tf": {
                tf: {
                    "bias": s.bias.value,
                    "confidence": round(s.confidence, 3),
                    "up_magnet":   s.up_magnet.to_dict()   if s.up_magnet   else None,
                    "down_magnet": s.down_magnet.to_dict() if s.down_magnet else None,
                } for tf, s in self.per_tf.items()
            },
            "reasons": self.reasons,
        }


def _tf_signal(tf: str, zones, current_price: float,
               funding: float, long_ratio: float) -> TFSignal:
    longs  = zones.get("long",  [])
    shorts = zones.get("short", [])
    up   = nearest_zone_above(shorts, current_price)
    down = nearest_zone_below(longs,  current_price)

    if up and not down:
        return TFSignal(tf, Bias.HUNT_SHORTS, 0.6, up, None)
    if down and not up:
        return TFSignal(tf, Bias.HUNT_LONGS, 0.6, None, down)
    if not up and not down:
        return TFSignal(tf, Bias.NEUTRAL, 0.0, None, None)

    up_strength   = up.dollars   / max(1.0, abs(up.pct_from_spot)   + 1)
    down_strength = down.dollars / max(1.0, abs(down.pct_from_spot) + 1)
    total = up_strength + down_strength
    if total <= 0:
        return TFSignal(tf, Bias.NEUTRAL, 0.0, up, down)

    up_share = up_strength / total

    crowded_longs  = funding > 0.0003 and long_ratio > 0.55
    crowded_shorts = funding < -0.0003 and long_ratio < 0.45
    if crowded_longs:
        up_share = max(0.0, up_share - 0.10)
    if crowded_shorts:
        up_share = min(1.0, up_share + 0.10)

    if up_share > 0.58:
        conf = min(1.0, (up_share - 0.5) * 2)
        return TFSignal(tf, Bias.HUNT_SHORTS, conf, up, down)
    if up_share < 0.42:
        conf = min(1.0, (0.5 - up_share) * 2)
        return TFSignal(tf, Bias.HUNT_LONGS, conf, up, down)
    return TFSignal(tf, Bias.NEUTRAL, 0.0, up, down)


def analyze_swing(engine, coin: str) -> SwingSignal | None:
    current_price = engine.current_price(coin)
    if current_price <= 0:
        return None
    ctx = engine.context_summary(coin)
    funding    = ctx["avg_funding"]
    long_ratio = ctx["avg_long_ratio"]

    per_tf: dict[str, TFSignal] = {}
    for tf in [t.label for t in engine.timeframes]:
        zones = engine.extract_zones_for(coin, tf)
        per_tf[tf] = _tf_signal(tf, zones, current_price, funding, long_ratio)

    score = 0.0
    for tf, sig in per_tf.items():
        w = config.SIGNAL_TF_WEIGHTS.get(tf, 0.0)
        if sig.bias == Bias.HUNT_SHORTS:
            score += w * sig.confidence
        elif sig.bias == Bias.HUNT_LONGS:
            score -= w * sig.confidence

    if score > 0.25:
        bias = Bias.HUNT_SHORTS
    elif score < -0.25:
        bias = Bias.HUNT_LONGS
    else:
        bias = Bias.NEUTRAL

    conf = min(1.0, abs(score))

    target = None
    stop_anchor = None
    min_dist = getattr(config, "MIN_SWING_TARGET_PCT", 0.015)
    for prefer in ("3d", "1w", "24h"):
        s = per_tf.get(prefer)
        if not s or s.bias != bias:
            continue
        zones = engine.extract_zones_for(coin, prefer)
        if bias == Bias.HUNT_SHORTS:
            tgt = nearest_zone_above(zones.get("short", []), current_price, min_dist)
            anc = nearest_zone_below(zones.get("long", []),  current_price, 0.0)
        else:  # HUNT_LONGS
            tgt = nearest_zone_below(zones.get("long", []),  current_price, min_dist)
            anc = nearest_zone_above(zones.get("short", []), current_price, 0.0)
        if tgt:
            target, stop_anchor = tgt, anc
            break

    rr = None
    if target and stop_anchor:
        reward = abs(target.price_center - current_price)
        risk   = abs(stop_anchor.price_center - current_price)
        if risk > 0:
            rr = reward / risk

    reasons = [
        f"score={score:+.3f}",
        f"funding={funding*100:+.4f}%",
        f"long_ratio={long_ratio:.3f}",
    ]
    for tf in ("3d", "1w"):
        if tf in per_tf:
            reasons.append(f"{tf}: {per_tf[tf].bias.value} ({per_tf[tf].confidence:.2f})")

    return SwingSignal(
        coin=coin, price=current_price,
        funding=funding, long_ratio=long_ratio,
        per_tf=per_tf, consensus_bias=bias,
        consensus_confidence=conf,
        target=target, stop_anchor=stop_anchor,
        rr_estimate=rr, reasons=reasons,
    )
