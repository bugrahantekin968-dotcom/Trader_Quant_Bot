"""
backtest/scorer.py
══════════════════════════════════════════════════════════════════════
DETERMINISTIC port of the live LLM scoring rubric (the A/B/C/D matrix in
build_llm_prompt). Given the SAME analyze_all() output the live bot feeds
the LLM, this reproduces the exact long_total / short_total / breakdown /
SL-TP plan WITHOUT calling the LLM.

Why this exists (see chat): the LLM was doing deterministic arithmetic.
Porting it means backtests cost $0, run at thousands of bars/sec, are
reproducible, and carry NO hindsight bias (Python can't "know" the future
the way an LLM trained on historical data can).

WHALE (D pillar) handling:
	whale=None	→ D scored as NEUTRAL ("no data" tier = +1/+1 each sub →
				  +3 both sides). Since it adds equally to both columns it
				  NEVER changes direction (long vs short), only the absolute
				  total — so the directional signal (A/B/C) is tested cleanly.
	whale=dict	→ real D tiers applied (for the day we source historical
				  sentiment). Expected keys:
					  {"ls_ratio": float, "funding_raw": float,
					   "oi_change_4h_pct": float}

The 10 sub-score keys match the live _SUBSCORE_CAPS exactly:
	macro_trend, momentum_rsi, momentum_ema, momentum_macd,
	sr_proximity, sr_quality, structural_events,
	ls_ratio, funding_rate, oi_behavior
"""
from __future__ import annotations
import math
from datetime import datetime, timedelta

# Sub-score ceilings — sum = 100
# ── REDESIGN (de-redundancy, see chat) ───────────────────────────────────
# Trend was triple-counted (macro_trend 25 + momentum_ema 10 + momentum_macd 5
# all point the same way in a clean trend ≈ 40 pts), which is WHY the system
# was trend-locked and structurally could not act on reversals.
#	• momentum_macd 5 → 0	(lagging trend/momentum, most redundant — DROPPED)
#	• momentum_ema	10 → 0	(price-vs-EMA ≈ macro_trend on a faster frame — DROPPED)
# The freed 15 pts go to the two MOST INDEPENDENT non-trend signals:
#	• sr_proximity	15 → 20 (location / "where" — the core entry signal)
#	• structural_events 5 → 15 (BOS/CHoCH/retest — genuine SMC structure that
#								was badly underweighted; the reversal trigger)
# Flow family (ls+funding+oi=25) is LEFT UNCHANGED — deliberately NOT inflated,
# because ls_ratio/funding/oi are mutually correlated (positioning data); piling
# weight onto all three would just swap trend-redundancy for flow-redundancy.
# funding_rate is RE-BANDED (below) to fire on extremes instead of sitting dead.
# These are PRINCIPLED (collinearity) allocations — NOT fitted to the backtest.
# The keys momentum_ema / momentum_macd are kept (cap 0) so CSV columns stay stable.
# FAITHFUL to the live LLM rubric (main.py). This Python scorer is now the
# SHARED scoring engine for BOTH the backtest AND live (live calls it instead
# of asking the LLM to do the table-lookup+sum; the LLM is reserved for the
# GO/NO-GO judgment on candidates that pass). Weights match the rubric exactly:
#	A macro 25 | B rsi 10 + ema 10 + macd 5 | C prox 15 + qual 5 + struct 5
#	D ls 10 + funding 8 + oi 7	 → sums to 100.
# (The de-redundant experiment — macd 0, ema 0, prox 20, struct 15 — lives on
#  behind a flag for OOS A/B testing; it is NOT the live behaviour.)
SUBSCORE_CAPS = {
	"macro_trend": 25, "momentum_rsi": 10, "momentum_ema": 10,
	"momentum_macd": 5, "sr_proximity": 15, "sr_quality": 5,
	"structural_events": 5, "ls_ratio": 10, "funding_rate": 8, "oi_behavior": 7,
}

_TF_MINUTES = {"1d": 1440, "4h": 240, "1h": 60, "15m": 15}

_LONG_CANDLES  = {"BULLISH_ENGULFING", "HAMMER", "INVERTED_HAMMER"}
_SHORT_CANDLES = {"BEARISH_ENGULFING", "SHOOTING_STAR", "HANGING_MAN"}


# ──────────────────────────────────────────────────────────────────────
# A — MACRO TREND (1D × 4H matrix)
# ──────────────────────────────────────────────────────────────────────
_MACRO_TABLE = {
	("BULLISH", "BULLISH"): (25, 0),
	("BULLISH", "RANGING"): (18, 3),
	("BULLISH", "BEARISH"): (20, 5),
	("RANGING", "RANGING"): (8, 8),
	("BEARISH", "BULLISH"): (5, 20),
	("BEARISH", "RANGING"): (3, 18),
	("BEARISH", "BEARISH"): (0, 25),
}

def _score_macro(t1d: str, t4h: str) -> tuple[int, int]:
	# Any combination not in the table (e.g. RANGING/BULLISH) → treat mixed.
	if (t1d, t4h) in _MACRO_TABLE:
		return _MACRO_TABLE[(t1d, t4h)]
	# Fallbacks for the 2 un-listed mixed cases, symmetric-ish:
	if t1d == "RANGING" and t4h == "BULLISH":
		return (12, 4)
	if t1d == "RANGING" and t4h == "BEARISH":
		return (4, 12)
	return (8, 8)


# ──────────────────────────────────────────────────────────────────────
# B.1 — RSI multi-band (avg of 1h & 4h)
# ──────────────────────────────────────────────────────────────────────
def _score_rsi(rsi_1h, rsi_4h) -> tuple[int, int]:
	vals = [r for r in (rsi_1h, rsi_4h) if isinstance(r, (int, float))]
	if not vals:
		return (1, 1)
	avg = sum(vals) / len(vals)
	if avg < 20:  return (10, 0)
	if avg < 30:  return (8, 0)
	if avg < 45:  return (3, 5)
	if avg <= 55: return (2, 2)
	if avg < 70:  return (5, 3)
	if avg <= 80: return (0, 8)
	return (0, 10)


# ──────────────────────────────────────────────────────────────────────
# B.2 — EMA confluence (1h, fallback 4h)
# ──────────────────────────────────────────────────────────────────────
_EMA_TABLE = {
	"BOUNCING_OFF_EMA":		 (10, 0),
	"FIRMLY_ABOVE_BOTH":	 (7, 0),
	"CHOPPING_BETWEEN_EMAS": (2, 2),
	"FIRMLY_BELOW_BOTH":	 (0, 7),
	"REJECTING_BELOW_EMA":	 (0, 10),
	"INSUFFICIENT_DATA":	 (1, 1),
}

def _score_ema(ema_1h, ema_4h) -> tuple[int, int]:
	# FAITHFUL to live rubric B.2: use 1h ema_confluence, fall back to 4h.
	conf = ema_1h if ema_1h in _EMA_TABLE else ema_4h
	return _EMA_TABLE.get(conf, (1, 1))


# ──────────────────────────────────────────────────────────────────────
# B.3 — MACD state (1h)
# ──────────────────────────────────────────────────────────────────────
_MACD_TABLE = {
	"BULL_BUILDING": (5, 0), "BULL_FADING": (3, 1),
	"FLAT_NEAR_ZERO": (1, 1), "BEAR_FADING": (1, 3),
	"BEAR_BUILDING": (0, 5), "INSUFFICIENT_DATA": (1, 1),
}

def _score_macd(macd_state) -> tuple[int, int]:
	# FAITHFUL to live rubric B.3: 1h macd_state → tier.
	return _MACD_TABLE.get(macd_state, (1, 1))


# ──────────────────────────────────────────────────────────────────────
# C.1 — Major S/R proximity (asymmetric, classify by closer)
# ──────────────────────────────────────────────────────────────────────
def _zone_dist(current, level):
	"""current -> seviyenin BOLGESINE uzaklik (icindeyse 0); zone yoksa noktaya duser."""
	p  = level.get("price")
	lo = level.get("zone_low", p)
	hi = level.get("zone_high", p)
	if lo is None or hi is None:
		lo = hi = p
	if current > hi:
		return (current - hi) / current
	if current < lo:
		return (lo - current) / current
	return 0.0


def _score_proximity(current, key_levels):
	"""Returns (long_pts, short_pts, side, nearest_level_obj_or_None)."""
	# zone-bazli alaka: destek, fiyat zone DIBININ altina dusmediyse gecerli;
	# direnc, fiyat zone TEPESININ ustune cikmadiysa gecerli (merkez degil bolge).
	# Step 2: drive proximity/quality from the dedicated 4H swing pool (ZigZag-
	# prominence, min 3 genuine touches). Fall back to legacy HTF levels only
	# if the H4 pool yields nothing for that side.
	_bl_s = key_levels.get("blended_supports")
	_bl_r = key_levels.get("blended_resistances")
	if _bl_s is not None or _bl_r is not None:
		_sups_src = _bl_s or []
		_ress_src = _bl_r or []
	else:
		_min_t = 3
		_h4s = [l for l in key_levels.get("h4_supports", []) if (l.get("touch_count", 0) or 0) >= _min_t]
		_h4r = [l for l in key_levels.get("h4_resistances", []) if (l.get("touch_count", 0) or 0) >= _min_t]
		_sups_src = _h4s if _h4s else key_levels.get("major_supports", [])
		_ress_src = _h4r if _h4r else key_levels.get("major_resistances", [])
	sups = [l for l in _sups_src if current >= l.get("zone_low", l["price"])]
	ress = [l for l in _ress_src if current <= l.get("zone_high", l["price"])]
	near_sup = max(sups, key=lambda l: l["price"]) if sups else None
	near_res = min(ress, key=lambda l: l["price"]) if ress else None

	d_sup = _zone_dist(current, near_sup) if near_sup else math.inf   # NOKTA degil BOLGE
	d_res = _zone_dist(current, near_res) if near_res else math.inf

	# DIRECTIONAL (Step 2b): short proximity from the nearest RESISTANCE above;
	# long proximity from the nearest SUPPORT below — independent, so drifting
	# toward a support no longer steals the short its resistance proximity. Tier
	# by zone distance, scaled by touch strength (MAJOR blended zones = high touch).
	def _tier(d):
		if d <= 0.003: return 15
		if d <= 0.008: return 12
		if d <= 0.015: return 9
		if d <= 0.025: return 5
		return 0

	def _strength_factor(level):
		t = (level.get("touch_count", 0) or 0) if level else 0
		if t >= 6: return 1.0
		if t >= 4: return 0.8
		if t >= 3: return 0.6
		return 0.5

	long_pts  = round(_tier(d_sup) * _strength_factor(near_sup)) if d_sup <= 0.025 else 0
	short_pts = round(_tier(d_res) * _strength_factor(near_res)) if d_res <= 0.025 else 0
	long_lvl  = near_sup if d_sup <= 0.025 else None
	short_lvl = near_res if d_res <= 0.025 else None
	return (long_pts, short_pts, long_lvl, short_lvl)


# ──────────────────────────────────────────────────────────────────────
# C.2 — S/R quality bonus (GATE on C.1; single highest + confluence)
# ──────────────────────────────────────────────────────────────────────
def _score_quality(side, level, key_levels):
	if side is None or level is None:
		return (0, 0)
	label = level.get("label", "")
	touches = level.get("touch_count", 0) or 0

	if (side == "long" and label == "SR_FLIP_SUPPORT") or \
	   (side == "short" and label == "SR_FLIP_RESISTANCE"):
		base = 5
	elif touches >= 6: base = 4
	elif touches >= 4: base = 3
	elif touches == 3: base = 2
	else:			   base = 0

	# Confluence: within 0.3% of POC or a psychological round number → +1
	price = level["price"]
	conf = 0
	poc = key_levels.get("poc_level")
	if poc and poc.get("price"):
		if abs(price - poc["price"]) / price <= 0.003:
			conf = 1
	if conf == 0:
		psychs = [(p.get("price") if isinstance(p, dict) else p)
				 for p in key_levels.get("psychological_above", [])] + \
				[(p.get("price") if isinstance(p, dict) else p)
				 for p in key_levels.get("psychological_below", [])]
		for pp in psychs:
			if pp and abs(price - pp) / price <= 0.003:
				conf = 1
				break

	total = min(5, base + conf)
	return (total, 0) if side == "long" else (0, total)


# ──────────────────────────────────────────────────────────────────────
# C.3 — Structural events (fakeout filter, fresh, candlestick + killzone)
# ──────────────────────────────────────────────────────────────────────
_LONG_TIER = [	 # (predicate, points) — first match wins, highest first
	# FAITHFUL to live rubric C.3 (cap 5). CHoCH + respected-retest is the top
	# tier; a bare CHoCH/BOS scores less (a bare CHoCH is fakeout-prone).
	(lambda e: e["type"] == "CHOCH_BULL" and e.get("retest_status") == "RETESTED_RESPECTED", 5),
	(lambda e: e["type"] == "CHOCH_BULL" and e.get("is_strong_close"), 4),
	(lambda e: e["type"] == "CHOCH_BULL", 3),
	(lambda e: e["type"] == "BOS_BULL" and e.get("is_strong_close"), 3),
	(lambda e: e["type"] == "BOS_BULL", 2),
]
_SHORT_TIER = [
	(lambda e: e["type"] == "CHOCH_BEAR" and e.get("retest_status") == "RETESTED_RESPECTED", 5),
	(lambda e: e["type"] == "CHOCH_BEAR" and e.get("is_strong_close"), 4),
	(lambda e: e["type"] == "CHOCH_BEAR", 3),
	(lambda e: e["type"] == "BOS_BEAR" and e.get("is_strong_close"), 3),
	(lambda e: e["type"] == "BOS_BEAR", 2),
]

def _event_utc_hour(analysis, tf, bars_since):
	"""Broker-TZ-independent event UTC hour from last_timestamp − bars_since×tf_min."""
	try:
		ts = analysis[tf].get("last_timestamp")
		if not ts:
			return None
		dt = datetime.fromisoformat(str(ts).replace("Z", "").strip())
		dt -= timedelta(minutes=_TF_MINUTES.get(tf, 60) * int(bars_since))
		return dt.hour
	except Exception:
		return None

def _in_killzone(hour):
	if hour is None:
		return False
	return (7 <= hour < 11) or (12 <= hour < 16)

def _score_structural(analysis, current):
	"""Returns (long_pts, short_pts). Scans 1h/4h/1d only (NOT 15m)."""
	best_long = (0, None, None)	  # (pts, tf, event)
	best_short = (0, None, None)
	for tf in ("1h", "4h", "1d"):
		a = analysis.get(tf) or {}
		for ev in a.get("structure_events", []) or []:
			# FAKEOUT FILTER
			if not ev.get("volume_backed", False):
				continue
			if ev.get("retest_status") == "RETESTED_BROKEN":
				continue
			if int(ev.get("bars_since", 999)) > 12:	  # fresh only
				continue
			for pred, pts in _LONG_TIER:
				if pred(ev) and pts > best_long[0]:
					best_long = (pts, tf, ev); break
			for pred, pts in _SHORT_TIER:
				if pred(ev) and pts > best_short[0]:
					best_short = (pts, tf, ev); break

	# Price-inside unmitigated OB / FVG on 1h (only if nothing stronger).
	# FAITHFUL to live rubric C.3: OB=+2, FVG=+1.
	a1h = analysis.get("1h") or {}
	if best_long[0] < 2:
		for ob in a1h.get("order_blocks", []) or []:
			if ob.get("type") == "BULL_OB" and ob.get("low", -math.inf) <= current <= ob.get("high", math.inf):
				best_long = max(best_long, (2, "1h", {"type": "BULL_OB"}), key=lambda x: x[0]); break
	if best_long[0] < 1:
		for fv in a1h.get("unmitigated_fvgs", []) or []:
			if fv.get("type") == "BULL_FVG" and fv.get("low", -math.inf) <= current <= fv.get("high", math.inf):
				best_long = max(best_long, (1, "1h", {"type": "BULL_FVG"}), key=lambda x: x[0]); break
	if best_short[0] < 2:
		for ob in a1h.get("order_blocks", []) or []:
			if ob.get("type") == "BEAR_OB" and ob.get("low", -math.inf) <= current <= ob.get("high", math.inf):
				best_short = max(best_short, (2, "1h", {"type": "BEAR_OB"}), key=lambda x: x[0]); break
	if best_short[0] < 1:
		for fv in a1h.get("unmitigated_fvgs", []) or []:
			if fv.get("type") == "BEAR_FVG" and fv.get("low", -math.inf) <= current <= fv.get("high", math.inf):
				best_short = max(best_short, (1, "1h", {"type": "BEAR_FVG"}), key=lambda x: x[0]); break

	def bonuses(pts, tf, ev, aligned_set):
		if pts <= 0 or ev is None or tf is None:
			return pts
		# Candlestick: aligned pattern on the SAME tf's most recent bar (+1)
		pats = (analysis.get(tf) or {}).get("candlestick_patterns", []) or []
		if any(p.get("pattern") in aligned_set for p in pats):
			pts = min(5, pts + 1)
		# Killzone: event utc hour (+1)
		bs = ev.get("bars_since", 0)
		if _in_killzone(_event_utc_hour(analysis, tf, bs)):
			pts = min(5, pts + 1)
		return pts

	lp = bonuses(*best_long, _LONG_CANDLES)
	sp = bonuses(*best_short, _SHORT_CANDLES)
	return (lp, sp)


# ──────────────────────────────────────────────────────────────────────
# D — WHALE (real tiers if data supplied, else neutral +1/+1)
# ──────────────────────────────────────────────────────────────────────
def _score_whale(whale):
	if not whale:
		return {"ls_ratio": (1, 1), "funding_rate": (1, 1), "oi_behavior": (1, 1)}
	out = {}
	ls = whale.get("ls_ratio")
	if ls is None:			  out["ls_ratio"] = (1, 1)
	elif ls < 0.5:			  out["ls_ratio"] = (10, 0)
	elif ls < 0.7:			 out["ls_ratio"] = (8, 0)
	elif ls < 1.2:			 out["ls_ratio"] = (4, 1)
	elif ls < 2.0:			 out["ls_ratio"] = (1, 4)
	elif ls <= 3.0:			 out["ls_ratio"] = (0, 8)
	else:					 out["ls_ratio"] = (0, 10)

	# FAITHFUL to live rubric D.2 (crypto-adjusted bands). Funding is contrarian:
	# very negative → shorts pay longs → crowd net-short → fade them LONG.
	# very positive → longs pay → crowd net-long → fade them SHORT. Narrow neutral
	# band so it only speaks at genuine extremes. NOTE: the backtest neutralizes
	# funding (funding_raw=0.0 → 3/3), so this re-banding affects the LIVE bot only
	# — the backtest cannot validate it without a funding-rate history feed.
	fr = whale.get("funding_raw")
	if   fr is None:        out["funding_rate"] = (1, 1)
	elif fr <  -0.0010:     out["funding_rate"] = (8, 0)
	elif fr <  -0.0003:     out["funding_rate"] = (6, 1)
	elif fr <=  0.0003:     out["funding_rate"] = (3, 3)   # neutral
	elif fr <=  0.0010:     out["funding_rate"] = (1, 5)
	elif fr <=  0.0020:     out["funding_rate"] = (0, 7)
	else:                   out["funding_rate"] = (0, 8)

	oi = whale.get("oi_change_4h_pct")
	grn = whale.get("last_4h_green")
	if oi is None:			  out["oi_behavior"] = (1, 1)
	elif oi > 3 and grn:	  out["oi_behavior"] = (7, 0)
	elif oi > 3 and not grn:  out["oi_behavior"] = (0, 7)
	elif oi < -3 and grn:	  out["oi_behavior"] = (1, 3)
	elif oi < -3 and not grn: out["oi_behavior"] = (3, 1)
	else:					  out["oi_behavior"] = (2, 2)
	return out


# ──────────────────────────────────────────────────────────────────────
# SL / TP construction (rubric steps 4a–4f, 5, 6)
# ──────────────────────────────────────────────────────────────────────
def _retest_anchor_entry(side, current, key_levels, max_pct=0.03):
	"""Anticipatory RETEST entry. If a fresh SR-flip (a broken level) sits on the
	trade's side within max_pct of price, anchor the entry as a LIMIT 0.5% on the
	APPROACH side of that level, so a retest that turns ~0.5% short of the exact
	level still fills. Returns (entry, flip_level) or (None, None) -> fall back to
	the default current+-0.2% entry.

	  LONG  <- nearest SR_FLIP_SUPPORT below price (former resistance, now support):
	            buy-limit at flip*1.005 (fills on a pull-back DOWN to the level).
	  SHORT <- nearest SR_FLIP_RESISTANCE above price (former support, now res):
	            sell-limit at flip*0.995 (fills on a pull-back UP to the level).
	Valid only when the limit sits on the correct side of price (a real pending
	order) AND price is within max_pct of the level (a FRESH retest, not a stale
	break price ran far from). Stop then sits just past the level = invalidation.
	"""
	flips = key_levels.get("sr_flips", []) or []
	if side == "long":
		cands = [f["price"] for f in flips
				 if f.get("label") == "SR_FLIP_SUPPORT" and (f.get("price") or 0) < current]
		if cands:
			flip = max(cands)
			entry = flip * 1.005
			if entry < current and (current - entry) / current <= max_pct:
				return entry, flip
	else:
		cands = [f["price"] for f in flips
				 if f.get("label") == "SR_FLIP_RESISTANCE" and (f.get("price") or 0) > current]
		if cands:
			flip = min(cands)
			entry = flip * 0.995
			if entry > current and (entry - current) / current <= max_pct:
				return entry, flip
	return None, None


def _build_plan(side, current, key_levels, score,
				min_sl=0.015, max_sl=0.04, min_rr=2.0, max_rr=3.5):
	"""Returns dict(entry, sl, tp, rr, position_usd) or None if WAIT-override."""
	# Step 2b consistency: SL/TP off the SAME blended 1D+4H pool the scorer/LLM use
	# (was legacy major_*, leaving this stale/empty -> backtest could not place a TP
	# and under-traded). Fall back to the 4H pool, then legacy HTF levels.
	_bl_s = key_levels.get("blended_supports")
	_bl_r = key_levels.get("blended_resistances")
	if _bl_s is not None or _bl_r is not None:
		_src_s = _bl_s or []
		_src_r = _bl_r or []
	else:
		_mt = 3
		_h4s = [l for l in key_levels.get("h4_supports", []) if (l.get("touch_count", 0) or 0) >= _mt]
		_h4r = [l for l in key_levels.get("h4_resistances", []) if (l.get("touch_count", 0) or 0) >= _mt]
		_src_s = _h4s if _h4s else key_levels.get("major_supports", [])
		_src_r = _h4r if _h4r else key_levels.get("major_resistances", [])
	sups = sorted([l["price"] for l in _src_s], reverse=True)
	ress = sorted([l["price"] for l in _src_r])
	# TP pool ALSO sees intraday (1H) + psychological levels — the SAME set the
	# live LLM uses for targets (key_levels feeds both). SL stays on the strong
	# blended pool (sups/ress above); only the TARGET search widens, so a nearer
	# valid-RR level can fill the same trade the live LLM would also take.
	_tp_intr = [l["price"] for l in key_levels.get("intraday_resistances", [])]
	_tp_ints = [l["price"] for l in key_levels.get("intraday_supports", [])]
	_tp_psa  = [p for p in key_levels.get("psychological_above", []) if isinstance(p, (int, float))]
	_tp_psb  = [p for p in key_levels.get("psychological_below", []) if isinstance(p, (int, float))]
	tp_ress = sorted(set(ress + _tp_intr + _tp_psa))
	tp_sups = sorted(set(sups + _tp_ints + _tp_psb), reverse=True)

	if side == "long":
		_re, _rf = _retest_anchor_entry("long", current, key_levels)
		if _re is not None:
			entry = _re
			structural_sl = _rf * 0.99       # stop just below the broken/flip level
		else:
			entry = current * 0.998
			near_sup = next((p for p in sups if p < entry), None)
			if near_sup is None:
				return None
			structural_sl = near_sup * 0.99
		sl_pct = abs(entry - structural_sl) / entry
		if sl_pct < min_sl:
			sl = entry * (1 - min_sl); sl_pct = min_sl
		else:
			sl = structural_sl
		if sl_pct > max_sl:
			return None
		tp = None
		for cand in [p for p in tp_ress if p > entry]:
			rr = abs(cand - entry) / abs(entry - sl)
			if rr >= min_rr:
				tp = cand; break
		if tp is None:
			return None
		rr = abs(tp - entry) / abs(entry - sl)
	else:  # short
		_re, _rf = _retest_anchor_entry("short", current, key_levels)
		if _re is not None:
			entry = _re
			structural_sl = _rf * 1.01       # stop just above the broken/flip level
		else:
			entry = current * 1.002
			near_res = next((p for p in ress if p > entry), None)
			if near_res is None:
				return None
			structural_sl = near_res * 1.01
		sl_pct = abs(entry - structural_sl) / entry
		if sl_pct < min_sl:
			sl = entry * (1 + min_sl); sl_pct = min_sl
		else:
			sl = structural_sl
		if sl_pct > max_sl:
			return None
		tp = None
		for cand in [p for p in tp_sups if p < entry]:
			rr = abs(entry - cand) / abs(entry - sl)
			if rr >= min_rr:
				tp = cand; break
		if tp is None:
			return None
		rr = abs(entry - tp) / abs(entry - sl)

	# RR cap (live postprocess ile ayni): uzak TP yerine max_rr mesafesi.
	if rr > max_rr:
		if side == "long":
			tp = entry + max_rr * abs(entry - sl)
		else:
			tp = entry - max_rr * abs(entry - sl)
		rr = max_rr
	if rr < min_rr:
		return None
	pos = 20000.0 if score >= 81 else 10000.0
	return {"entry": round(entry, 6), "sl": round(sl, 6), "tp": round(tp, 6),
			"rr": round(rr, 3), "sl_pct": round(sl_pct, 4), "position_usd": pos}


# ──────────────────────────────────────────────────────────────────────
# REVERSAL PLAYBOOK (experimental) — confirmed 4H CHoCH with a STRUCTURE stop
# ──────────────────────────────────────────────────────────────────────
# A separate, counter-trend decision path. The trend-gated A/B/C/D score can
# (by design) almost never fire a reversal: when a CHoCH first prints, macro_trend
# still points the old way and dominates. This playbook lets a HIGH-CONVICTION
# reversal trade through on its own terms:
#
#	Trigger (all mandatory):
#	  • 4H (high-TF) CHOCH_BULL / CHOCH_BEAR	 — character change on a TF that matters
#	  • retest_status == RETESTED_RESPECTED		  — price came back to the broken
#													level and HELD (a bare CHoCH is
#													fakeout-prone; the held retest is
#													what makes it tradeable)
#	  • volume_backed							  — fakeout filter
#	  • fresh (bars_since ≤ 12)
#
#	Stop (your spec — structure invalidation, not a flat %):
#	  • bullish CHoCH → stop BELOW the most recent swing LOW before the break
#		(the bottom the reversal is turning from; a new low = thesis dead)
#	  • bearish CHoCH → stop ABOVE the most recent swing HIGH before the break
#
#	The trigger is deliberately spartan (no extra RSI/flow knobs) to avoid
#	over-fitting the playbook itself. RETESTED_RESPECTED on 4H is already a high bar.
#
# ⚠️  This OPENS counter-trend trading, which is high-risk and, on our one-regime
#	  35-trade sample, effectively UN-VALIDATABLE (very few 4H CHoCH+retest setups
#	  exist in the window). Treat results as experimental. Disable with
#	  enable_reversal=False ( --no-reversal ).
def _detect_reversal_setup(analysis):
	"""Return ('long'|'short', event) for a confirmed fresh 4H CHoCH, else None."""
	a4h = analysis.get("4h") or {}
	best = None	  # (bars_since, side, event) — prefer the freshest
	for ev in a4h.get("structure_events", []) or []:
		if not ev.get("volume_backed", False):
			continue
		if ev.get("retest_status") != "RETESTED_RESPECTED":
			continue
		bs = int(ev.get("bars_since", 999))
		if bs > 12:
			continue
		et = ev.get("type")
		side = "long" if et == "CHOCH_BULL" else ("short" if et == "CHOCH_BEAR" else None)
		if side is None:
			continue
		if best is None or bs < best[0]:
			best = (bs, side, ev)
	return (best[1], best[2]) if best else None


def _build_reversal_plan(side, current, key_levels, swings_4h, choch_event,
						 min_sl=0.015, max_sl=0.04, min_rr=2.0, max_rr=3.5):
	"""Reversal entry with a STRUCTURE stop. Returns plan dict or None."""
	ev_idx = choch_event.get("bar_idx")
	lows  = [s for s in swings_4h if s.get("type") in ("SL", "LL", "HL")]
	highs = [s for s in swings_4h if s.get("type") in ("SH", "HH", "LH")]

	def _pivot_before(pool):
		cand = [s for s in pool if ev_idx is None or s.get("index", -1) < ev_idx]
		return max(cand, key=lambda s: s.get("index", -1)) if cand else None

	# Step 2b consistency: reversal TP pool from the blended 1D+4H pool too.
	_blr = key_levels.get("blended_resistances")
	_bls = key_levels.get("blended_supports")
	if _blr is None and _bls is None:
		_blr = key_levels.get("h4_resistances") or key_levels.get("major_resistances", [])
		_bls = key_levels.get("h4_supports") or key_levels.get("major_supports", [])
	_rev_r = ([l["price"] for l in (_blr or [])]
			  + [l["price"] for l in key_levels.get("intraday_resistances", [])]
			  + [p for p in key_levels.get("psychological_above", []) if isinstance(p, (int, float))])
	_rev_s = ([l["price"] for l in (_bls or [])]
			  + [l["price"] for l in key_levels.get("intraday_supports", [])]
			  + [p for p in key_levels.get("psychological_below", []) if isinstance(p, (int, float))])
	if side == "long":
		entry = current * 0.999
		piv = _pivot_before(lows)
		if piv is None:
			return None
		struct_sl = piv["price"] * 0.99
		if struct_sl >= entry:
			return None
		sl_pct = (entry - struct_sl) / entry
		if sl_pct < min_sl:
			sl = entry * (1 - min_sl); sl_pct = min_sl
		elif sl_pct > max_sl:
			return None								 # structure too far → bounded risk
		else:
			sl = struct_sl
		ress = sorted(set(p for p in _rev_r if p > entry))
		tp = next((p for p in ress if (p - entry) / (entry - sl) >= min_rr), None)
		if tp is None:
			return None
		rr = (tp - entry) / (entry - sl)
	else:  # short
		entry = current * 1.001
		piv = _pivot_before(highs)
		if piv is None:
			return None
		struct_sl = piv["price"] * 1.01
		if struct_sl <= entry:
			return None
		sl_pct = (struct_sl - entry) / entry
		if sl_pct < min_sl:
			sl = entry * (1 + min_sl); sl_pct = min_sl
		elif sl_pct > max_sl:
			return None
		else:
			sl = struct_sl
		sups = sorted(set(p for p in _rev_s if p < entry), reverse=True)
		tp = next((p for p in sups if (entry - p) / (sl - entry) >= min_rr), None)
		if tp is None:
			return None
		rr = (entry - tp) / (sl - entry)

	# RR cap (live postprocess ile ayni): uzak TP yerine max_rr mesafesi.
	if rr > max_rr:
		if side == "long":
			tp = entry + max_rr * abs(entry - sl)
		else:
			tp = entry - max_rr * abs(entry - sl)
		rr = max_rr
	if rr < min_rr:
		return None
	return {"entry": round(entry, 6), "sl": round(sl, 6), "tp": round(tp, 6),
			"rr": round(rr, 3), "sl_pct": round(sl_pct, 4),
			"position_usd": 10000.0,				 # experimental → conservative size
			"playbook": "reversal"}


# ──────────────────────────────────────────────────────────────────────
# PUBLIC — score one bar exactly like the live engine
# ──────────────────────────────────────────────────────────────────────
def score(analysis: dict, current_price: float, whale: dict | None = None,
		  threshold: float = 49.0, enable_reversal: bool = True,
		  min_sl: float = 0.015, max_sl: float = 0.04, min_rr: float = 2.0, max_rr: float = 3.5,
		  short_exh_veto: float = 0.0,
		  long_exh_veto: float = 0.0) -> dict:
	"""
	analysis	  : output of TechnicalAnalyzer.analyze_all (truncated to ≤T)
	current_price : live price at this bar's close
	whale		  : None → D neutral (A/B/C-only backtest); dict → real D
	threshold	  : decision gate (default 49 ≈ 65×75/100 for A/B/C-only)
	"""
	kl = analysis.get("key_levels", {}) or {}
	a1d = (analysis.get("1d") or {}).get("trend", "RANGING")
	a4h = (analysis.get("4h") or {}).get("trend", "RANGING")
	ind1h = (analysis.get("1h") or {}).get("indicators", {}) or {}
	ind4h = (analysis.get("4h") or {}).get("indicators", {}) or {}

	macro = _score_macro(a1d, a4h)
	rsi	  = _score_rsi(ind1h.get("rsi_14"), ind4h.get("rsi_14"))
	ema	  = _score_ema(ind1h.get("ema_confluence"), ind4h.get("ema_confluence"))
	macd  = _score_macd(ind1h.get("macd_state"))
	c1_l, c1_s, long_lvl, short_lvl = _score_proximity(current_price, kl)
	c2_l  = _score_quality("long", long_lvl, kl)[0]
	c2_s  = _score_quality("short", short_lvl, kl)[1]
	c3	  = _score_structural(analysis, current_price)
	# D.3 needs the last 4h candle direction — derive it from the price feed
	# (analysis), not from the whale source, so the provider stays price-free.
	if whale is not None and "last_4h_green" not in whale:
		dirn = (analysis.get("4h") or {}).get("last_candle_direction")
		if dirn in ("GREEN", "RED"):
			whale = {**whale, "last_4h_green": (dirn == "GREEN")}
	whale_pts = _score_whale(whale)

	long_bd = {
		"macro_trend": macro[0], "momentum_rsi": rsi[0], "momentum_ema": ema[0],
		"momentum_macd": macd[0], "sr_proximity": c1_l, "sr_quality": c2_l,
		"structural_events": c3[0], "ls_ratio": whale_pts["ls_ratio"][0],
		"funding_rate": whale_pts["funding_rate"][0], "oi_behavior": whale_pts["oi_behavior"][0],
	}
	short_bd = {
		"macro_trend": macro[1], "momentum_rsi": rsi[1], "momentum_ema": ema[1],
		"momentum_macd": macd[1], "sr_proximity": c1_s, "sr_quality": c2_s,
		"structural_events": c3[1], "ls_ratio": whale_pts["ls_ratio"][1],
		"funding_rate": whale_pts["funding_rate"][1], "oi_behavior": whale_pts["oi_behavior"][1],
	}
	# Clamp each to its cap, then sum (identical to live _sum_breakdown)
	long_total	= sum(min(v, SUBSCORE_CAPS[k]) for k, v in long_bd.items())
	short_total = sum(min(v, SUBSCORE_CAPS[k]) for k, v in short_bd.items())

	# ── R&D: alt-TF dönüş vetosu (sadece SHORT). exhaustion(short) >= eşik ise
	#    short açma (dönüşe/bounce'a short engellenir). 0.0 = kapalı (mevcut davranış).
	#    Backtest ve canlı AYNI scorer'ı kullandığı için ikisinde de geçerli olur.
	_short_vetoed = False
	if short_exh_veto > 0.0:
		try:
			_short_vetoed = float(exhaustion_score(analysis, "short").get("total", 0.0)) >= short_exh_veto
		except Exception:
			_short_vetoed = False
	_long_vetoed = False
	if long_exh_veto > 0.0:
		try:
			_long_vetoed = float(exhaustion_score(analysis, "long").get("total", 0.0)) >= long_exh_veto
		except Exception:
			_long_vetoed = False

	# ── Reversal playbook (experimental, opt-out via enable_reversal=False).
	#	 Checked FIRST: on a confirmed 4H CHoCH + held retest it OVERRIDES the
	#	 trend-gated decision — at a confirmed character change we stop following
	#	 the old trend. Uses a STRUCTURE stop, not the generic % stop. ──
	if enable_reversal:
		setup = _detect_reversal_setup(analysis)
		if setup:
			rside, rev = setup
			swings_4h = (analysis.get("4h") or {}).get("swing_points", []) or []
			rplan = _build_reversal_plan(rside, current_price, kl, swings_4h, rev,
										 min_sl=min_sl, max_sl=max_sl, min_rr=min_rr, max_rr=max_rr)
			if rplan and not (rside == "short" and _short_vetoed) and not (rside == "long" and _long_vetoed):
				decision = "LONG" if rside == "long" else "SHORT"
				return {
					"decision": decision,
					"long_total": long_total, "short_total": short_total,
					"long_breakdown": long_bd, "short_breakdown": short_bd,
					"plan": rplan, "playbook": "reversal",
					"chosen_side": rside,
					"nearest_level": rev.get("level"),
					"nearest_label": rev.get("type"),
				}

	# ── Trend-continuation (the de-redundant A/B/C/D score) ──
	decision = "WAIT"; plan = None
	if long_total >= threshold and long_total > short_total and not _long_vetoed:
		plan = _build_plan("long", current_price, kl, long_total,
						   min_sl=min_sl, max_sl=max_sl, min_rr=min_rr, max_rr=max_rr)
		decision = "LONG" if plan else "WAIT"
	elif short_total >= threshold and short_total > long_total and not _short_vetoed:
		plan = _build_plan("short", current_price, kl, short_total,
						   min_sl=min_sl, max_sl=max_sl, min_rr=min_rr, max_rr=max_rr)
		decision = "SHORT" if plan else "WAIT"

	return {
		"decision": decision,
		"long_total": long_total, "short_total": short_total,
		"long_breakdown": long_bd, "short_breakdown": short_bd,
		"plan": plan, "playbook": "continuation",
		"chosen_side": ("short" if short_total >= long_total else "long"),
		"nearest_level": ((short_lvl if short_total >= long_total else long_lvl) or {}).get("price"),
		"nearest_label": ((short_lvl if short_total >= long_total else long_lvl) or {}).get("label"),
	}

# ══════════════════════════════════════════════════════════════════════
# EXHAUSTION SCORER — "is the trend in our position's direction dying?"
# ══════════════════════════════════════════════════════════════════════
# A SEPARATE scoring template for OPEN positions (not entries). It answers a
# different question than score(): not "should we enter" but "has the trend we
# are riding run out of steam enough to bank the rest?".
#
# Design philosophy (the 'fine line'):
#   Trend exhaustion is NOT a single indicator touching a level. In a strong
#   trend RSI sits pinned at an extreme for many bars (continuation, NOT
#   reversal). Genuine exhaustion = momentum TURNING from an extreme + market
#   structure starting to break + trend support lost, CONFIRMED across 1H & 4H.
#   So we score a CONFLUENCE and only act above a high threshold (+ an LLM
#   judge), never on one weak signal.
#
# Two mirror-symmetric templates:
#   LONG  position  -> score how exhausted the UPTREND is (overbought rollover,
#                      bearish CHoCH, EMA loss, bear MACD).
#   SHORT position  -> the exact mirror (oversold rollover, bullish CHoCH, EMA
#                      reclaim, bull MACD).
#
# Returns dict(total, breakdown, hard_override, reasons):
#   total          0..100  — higher = more exhausted
#   breakdown      {structure, momentum, trend_support} — per-group points
#   hard_override  True if a 4H volume-backed CHoCH AGAINST the position printed
#                  (structure literally flipped — close now, no LLM needed)
#   reasons        human-readable list of which signals fired (LLM prompt + logs)
#
# Inputs:
#   analysis        {"1h": <analyze_timeframe out>, "4h": <...>}
#   side            "long" | "short"  (the OPEN position's direction)
#   prev_rsi_1h/4h  RSI from the PREVIOUS exhaustion check (to detect the TURN,
#                   not just the level). None on first check -> turn factors skip.
#
# Point caps per group keep any single group from dominating; tune the act
# threshold (main.py cfg.exhaustion_threshold) against a backtest.

_OB = ("STRONG_OVERBOUGHT", "EXTREME_OVERBOUGHT")
_OS = ("STRONG_OVERSOLD", "EXTREME_OVERSOLD")


def _macd_bull(state) -> bool:
    return state in ("BULL_BUILDING", "BULL_FADING")


def _macd_bear(state) -> bool:
    return state in ("BEAR_BUILDING", "BEAR_FADING")


def _freshest_event(events, types, max_bars=6):
    """Freshest structure event whose type is in `types` and bars_since<=max_bars."""
    best = None
    for ev in events or []:
        if ev.get("type") not in types:
            continue
        bs = ev.get("bars_since")
        if bs is None or bs > max_bars:
            continue
        cur = best.get("bars_since") if (best and best.get("bars_since") is not None) else 999
        if best is None or bs < cur:
            best = ev
    return best


def _exhaustion_long(a1h, a4h):
    """LONG position: how exhausted is the UPTREND? (mirror of _exhaustion_short)"""
    i1 = a1h.get("indicators", {}) or {}
    i4 = a4h.get("indicators", {}) or {}
    ev1 = a1h.get("structure_events", []) or []
    ev4 = a4h.get("structure_events", []) or []
    reasons = []
    hard = False

    # -- S — STRUCTURE: the uptrend is breaking (bearish CHoCH/BOS) ----------
    s = 0
    e4 = _freshest_event(ev4, ("CHOCH_BEAR",))
    if e4 is not None:
        if e4.get("volume_backed"):
            s += 35; hard = True
            reasons.append("4H CHOCH_BEAR fresh + volume-backed (structure flipped)")
        else:
            s += 20
            reasons.append("4H CHOCH_BEAR fresh")
    else:
        b4 = _freshest_event(ev4, ("BOS_BEAR",))
        if b4 is not None:
            s += 12
            reasons.append("4H BOS_BEAR fresh (lower low)")
    e1 = _freshest_event(ev1, ("CHOCH_BEAR",))
    if e1 is not None:
        if e1.get("volume_backed"):
            s += 16; reasons.append("1H CHOCH_BEAR fresh + volume-backed")
        else:
            s += 10; reasons.append("1H CHOCH_BEAR fresh")
    s = min(s, 45)

    # -- M — MOMENTUM: rolling DOWN from overbought + MACD flipping bearish (DELTA) --
    #   Delta-bazli: sadece " asiri bolgede" olmak puan vermez; RSI'nin DONMESI
    #   (prev'e gore dususe gecmesi) ve MACD'nin FRESH flip'i puan verir. Boylece
    #   saglikli trendde momentum ~0; gercek donuste sicrar (statik taban yok).
    m = 0
    r4, z4 = i4.get("rsi_14"), i4.get("rsi_zone")
    r1, z1 = i1.get("rsi_14"), i1.get("rsi_zone")
    prev_r4, prev_r1 = i4.get("prev_rsi_14"), i1.get("prev_rsi_14")          # ONCEKI BAR
    prev_mac1, prev_mac4 = i1.get("prev_macd_state"), i4.get("prev_macd_state")  # ONCEKI BAR
    if z4 in _OB and r4 is not None and prev_r4 is not None and r4 < prev_r4:
        m += 15; reasons.append(f"4H RSI {r4} rolling over from {z4}")
    if z1 in _OB and r1 is not None and prev_r1 is not None and r1 < prev_r1:
        m += 10; reasons.append(f"1H RSI {r1} rolling over from {z1}")
    mac1, mac4 = i1.get("macd_state"), i4.get("macd_state")
    if _macd_bear(mac1):
        if _macd_bull(prev_mac1):
            m += 12; reasons.append("1H MACD fresh flip to bearish")
        else:
            m += 4;  reasons.append("1H MACD bearish (sustained)")
    if _macd_bear(mac4):
        if _macd_bull(prev_mac4):
            m += 8;  reasons.append("4H MACD fresh flip to bearish")
        else:
            m += 3;  reasons.append("4H MACD bearish (sustained)")
    m = min(m, 35)

    # -- T — TREND SUPPORT: price losing the EMAs / 4H trend flipping --------
    t = 0
    emc1 = i1.get("ema_confluence")
    if emc1 == "FIRMLY_BELOW_BOTH":
        t += 12; reasons.append("1H price firmly below both EMAs")
    elif emc1 == "REJECTING_BELOW_EMA":
        t += 8; reasons.append("1H rejecting below EMA")
    elif emc1 == "CHOPPING_BETWEEN_EMAS":
        t += 5; reasons.append("1H chopping between EMAs")
    tr4 = a4h.get("trend")
    if tr4 == "BEARISH":
        t += 10; reasons.append("4H trend flipped BEARISH")
    elif tr4 == "RANGING":
        t += 5; reasons.append("4H trend now RANGING")
    t = min(t, 20)

    return {"total": s + m + t,
            "breakdown": {"structure": s, "momentum": m, "trend_support": t},
            "hard_override": hard, "reasons": reasons}


def _exhaustion_short(a1h, a4h):
    """SHORT position: how exhausted is the DOWNTREND? (mirror of _exhaustion_long)"""
    i1 = a1h.get("indicators", {}) or {}
    i4 = a4h.get("indicators", {}) or {}
    ev1 = a1h.get("structure_events", []) or []
    ev4 = a4h.get("structure_events", []) or []
    reasons = []
    hard = False

    # -- S — STRUCTURE: the downtrend is breaking (bullish CHoCH/BOS) --------
    s = 0
    e4 = _freshest_event(ev4, ("CHOCH_BULL",))
    if e4 is not None:
        if e4.get("volume_backed"):
            s += 35; hard = True
            reasons.append("4H CHOCH_BULL fresh + volume-backed (structure flipped)")
        else:
            s += 20
            reasons.append("4H CHOCH_BULL fresh")
    else:
        b4 = _freshest_event(ev4, ("BOS_BULL",))
        if b4 is not None:
            s += 12
            reasons.append("4H BOS_BULL fresh (higher high)")
    e1 = _freshest_event(ev1, ("CHOCH_BULL",))
    if e1 is not None:
        if e1.get("volume_backed"):
            s += 16; reasons.append("1H CHOCH_BULL fresh + volume-backed")
        else:
            s += 10; reasons.append("1H CHOCH_BULL fresh")
    s = min(s, 45)

    # -- M — MOMENTUM: turning UP from oversold + MACD flipping bullish (DELTA) --
    #   Ayna: RSI'nin DONMESI (prev'e gore yukselise gecmesi) + MACD FRESH flip
    #   puan verir; "asiri oversold'da takili durmak" (guclu dusus) puan VERMEZ.
    m = 0
    r4, z4 = i4.get("rsi_14"), i4.get("rsi_zone")
    r1, z1 = i1.get("rsi_14"), i1.get("rsi_zone")
    prev_r4, prev_r1 = i4.get("prev_rsi_14"), i1.get("prev_rsi_14")          # ONCEKI BAR
    prev_mac1, prev_mac4 = i1.get("prev_macd_state"), i4.get("prev_macd_state")  # ONCEKI BAR
    if z4 in _OS and r4 is not None and prev_r4 is not None and r4 > prev_r4:
        m += 15; reasons.append(f"4H RSI {r4} turning up from {z4}")
    if z1 in _OS and r1 is not None and prev_r1 is not None and r1 > prev_r1:
        m += 10; reasons.append(f"1H RSI {r1} turning up from {z1}")
    mac1, mac4 = i1.get("macd_state"), i4.get("macd_state")
    if _macd_bull(mac1):
        if _macd_bear(prev_mac1):
            m += 12; reasons.append("1H MACD fresh flip to bullish")
        else:
            m += 4;  reasons.append("1H MACD bullish (sustained)")
    if _macd_bull(mac4):
        if _macd_bear(prev_mac4):
            m += 8;  reasons.append("4H MACD fresh flip to bullish")
        else:
            m += 3;  reasons.append("4H MACD bullish (sustained)")
    m = min(m, 35)

    # -- T — TREND SUPPORT: price reclaiming the EMAs / 4H trend flipping ----
    t = 0
    emc1 = i1.get("ema_confluence")
    if emc1 == "FIRMLY_ABOVE_BOTH":
        t += 12; reasons.append("1H price firmly above both EMAs")
    elif emc1 == "BOUNCING_OFF_EMA":
        t += 8; reasons.append("1H bouncing off EMA (reclaim)")
    elif emc1 == "CHOPPING_BETWEEN_EMAS":
        t += 5; reasons.append("1H chopping between EMAs")
    tr4 = a4h.get("trend")
    if tr4 == "BULLISH":
        t += 10; reasons.append("4H trend flipped BULLISH")
    elif tr4 == "RANGING":
        t += 5; reasons.append("4H trend now RANGING")
    t = min(t, 20)

    return {"total": s + m + t,
            "breakdown": {"structure": s, "momentum": m, "trend_support": t},
            "hard_override": hard, "reasons": reasons}


def exhaustion_score(analysis, side):
    """
    Score trend exhaustion for an OPEN position. See module notes above.
      analysis : {"1h": analyze_timeframe(...), "4h": analyze_timeframe(...)}
      side     : "long" | "short"  (the OPEN position's direction)
    Returns dict(total, breakdown, hard_override, reasons).
    """
    a1h = analysis.get("1h") or {}
    a4h = analysis.get("4h") or {}
    sd = str(side).lower()
    if sd == "long":
        return _exhaustion_long(a1h, a4h)
    if sd == "short":
        return _exhaustion_short(a1h, a4h)
    return {"total": 0.0, "breakdown": {}, "hard_override": False,
            "reasons": [f"unknown side {side!r}"]}
