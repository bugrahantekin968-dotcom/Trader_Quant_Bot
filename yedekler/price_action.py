"""
price_action.py
===============
Hybrid Technical Analyzer for the FundedNext Swing Trade Bot.

Modules:
	1.	Classical indicators: EMA-50, EMA-200, RSI-14, ATR-14, MACD(12/26/9).
	2.	SMC / ICT structural analysis:
			- Swing High / Low detection
			- Trend classification (BULLISH / BEARISH / RANGING)
			- BOS (Break of Structure) and CHoCH (Change of Character) events
			- Fair Value Gap (FVG) detection
			- Order Block detection
			- Common candlestick patterns
	3.	Historical Major Support / Resistance clustering:
			- HTF (1D + 4H) swing point clustering with tolerance
			- Minimum touch filtering
			- S/R Flip identification (former support that became resistance,
			  former resistance that became support)

All public-facing keys, comments and log messages are in English.

Usage:
	from price_action import TechnicalAnalyzer

	analyzer = TechnicalAnalyzer(swing_order=5)
	results	 = analyzer.analyze_all(
		{"1d": df_1d, "4h": df_4h, "1h": df_1h, "15m": df_15m},
		current_price=75_500.0,
	)

	# results["15m"]["indicators"]["rsi_14"]	-> 42.3
	# results["4h"]["order_blocks"]				 -> [...]
	# results["key_levels"]["major_supports"]	 -> [...]
	# results["key_levels"]["sr_flips"]			 -> [...]
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# SECTION 1 — Classical Technical Indicators
# ═══════════════════════════════════════════════════════════════════

def _calc_ema(series: pd.Series, period: int) -> pd.Series:
	return series.ewm(span=period, adjust=False).mean()


def _calc_rsi(close: pd.Series, period: int = 14) -> pd.Series:
	delta	 = close.diff()
	gain	 = delta.clip(lower=0.0)
	loss	 = (-delta).clip(lower=0.0)
	avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
	avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()
	rs		 = avg_gain / avg_loss.replace(0.0, 1e-10)
	return 100.0 - (100.0 / (1.0 + rs))


def _calc_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
	high	   = df["high"]
	low		   = df["low"]
	prev_close = df["close"].shift(1)
	tr = pd.concat([
		high - low,
		(high - prev_close).abs(),
		(low  - prev_close).abs(),
	], axis=1).max(axis=1)
	return tr.ewm(alpha=1.0 / period, adjust=False).mean()


def _rsi_zone(rsi: float) -> str:
	"""
	Granular RSI zone classification used by the scoring matrix.
	Returns one of:
		EXTREME_OVERSOLD, STRONG_OVERSOLD, WEAK_BEARISH,
		NEUTRAL,
		WEAK_BULLISH, STRONG_OVERBOUGHT, EXTREME_OVERBOUGHT
	"""
	if rsi < 20:  return "EXTREME_OVERSOLD"
	if rsi < 30:  return "STRONG_OVERSOLD"
	if rsi < 45:  return "WEAK_BEARISH"
	if rsi <= 55: return "NEUTRAL"
	if rsi <= 70: return "WEAK_BULLISH"
	if rsi <= 80: return "STRONG_OVERBOUGHT"
	return "EXTREME_OVERBOUGHT"


def _ema_confluence(
	last_close: float,
	ema_50:		Optional[float],
	ema_200:	Optional[float],
	atr_pct:	Optional[float],
) -> str:
	"""
	Classifies price-vs-EMA confluence in granular tiers for the scoring matrix.
	Returns one of:
		FIRMLY_ABOVE_BOTH		 (price > both 50 and 200, clear margin)
		BOUNCING_OFF_EMA		 (price within 0.5 * atr_pct % of 50 or 200, last close above)
		CHOPPING_BETWEEN_EMAS	 (price between 50 and 200)
		REJECTING_BELOW_EMA		 (price within 0.5 * atr_pct % below 50 or 200, last close below)
		FIRMLY_BELOW_BOTH		 (price < both 50 and 200, clear margin)
		INSUFFICIENT_DATA
	"""
	if ema_50 is None or ema_200 is None or last_close <= 0:
		return "INSUFFICIENT_DATA"

	# Tolerance band — use 0.5 * ATR%, but at least 0.2%, at most 1%.
	base_pct = (atr_pct or 0.5) / 100.0
	tol		 = max(0.002, min(0.01, base_pct * 0.5))

	above_50_far  = last_close > ema_50	 * (1 + tol)
	above_200_far = last_close > ema_200 * (1 + tol)
	below_50_far  = last_close < ema_50	 * (1 - tol)
	below_200_far = last_close < ema_200 * (1 - tol)

	near_50_above  = ema_50	 * (1 - tol) <= last_close <= ema_50  * (1 + tol) and last_close >= ema_50
	near_200_above = ema_200 * (1 - tol) <= last_close <= ema_200 * (1 + tol) and last_close >= ema_200
	near_50_below  = ema_50	 * (1 - tol) <= last_close <= ema_50  * (1 + tol) and last_close < ema_50
	near_200_below = ema_200 * (1 - tol) <= last_close <= ema_200 * (1 + tol) and last_close < ema_200

	if above_50_far and above_200_far:
		return "FIRMLY_ABOVE_BOTH"
	if below_50_far and below_200_far:
		return "FIRMLY_BELOW_BOTH"
	if near_50_above or near_200_above:
		return "BOUNCING_OFF_EMA"
	if near_50_below or near_200_below:
		return "REJECTING_BELOW_EMA"
	return "CHOPPING_BETWEEN_EMAS"


def _compute_indicators(df: pd.DataFrame) -> dict:
	ind: dict  = {}
	n		   = len(df)
	last_close = float(df["close"].iloc[-1])

	# --- EMA-50 ---
	if n >= 50:
		ema50_val			  = round(float(_calc_ema(df["close"], 50).iloc[-1]), 6)
		ind["ema_50"]		  = ema50_val
		ind["price_vs_ema50"] = "ABOVE" if last_close > ema50_val else "BELOW"
	else:
		ind["ema_50"]		  = None
		ind["price_vs_ema50"] = "INSUFFICIENT_DATA"

	# --- EMA-200 ---
	if n >= 200:
		ema200_val			   = round(float(_calc_ema(df["close"], 200).iloc[-1]), 6)
		ind["ema_200"]		   = ema200_val
		ind["price_vs_ema200"] = "ABOVE" if last_close > ema200_val else "BELOW"
		e50					   = ind.get("ema_50")
		ind["ema_signal"]	   = "BULLISH" if (e50 is not None and e50 > ema200_val) else "BEARISH"
		ind["ema_golden_cross"] = (e50 is not None and e50 > ema200_val)
	else:
		ind["ema_200"]			= None
		ind["price_vs_ema200"]	= "INSUFFICIENT_DATA"
		ind["ema_signal"]		= "INSUFFICIENT_DATA"
		ind["ema_golden_cross"] = None

	# --- RSI-14 ---
	if n >= 16:
		_rsi_ser = _calc_rsi(df["close"], 14)
		rsi_val  = round(float(_rsi_ser.iloc[-1]), 2)
		ind["rsi_14"]	 = rsi_val
		ind["rsi_zone"]	 = _rsi_zone(rsi_val)
		ind["prev_rsi_14"] = round(float(_rsi_ser.iloc[-2]), 2) if n >= 17 else rsi_val
	else:
		ind["rsi_14"]	= None
		ind["rsi_zone"] = "INSUFFICIENT_DATA"
		ind["prev_rsi_14"] = None

	# --- ATR-14 ---
	if n >= 16:
		atr_val		  = float(_calc_atr(df, 14).iloc[-1])
		ind["atr_14"] = round(atr_val, 6)
		ind["atr_pct"] = round(atr_val / last_close * 100, 4) if last_close else None
	else:
		ind["atr_14"]  = None
		ind["atr_pct"] = None

	# --- MACD (12, 26, 9) — granular 5-state classification ---
	if n >= 35:
		try:
			ema12		= df["close"].ewm(span=12, adjust=False).mean()
			ema26		= df["close"].ewm(span=26, adjust=False).mean()
			macd_line	= ema12 - ema26
			macd_signal = macd_line.ewm(span=9, adjust=False).mean()
			macd_hist	= macd_line - macd_signal

			m_line_last = float(macd_line.iloc[-1])
			m_sig_last	= float(macd_signal.iloc[-1])
			m_hist_last = float(macd_hist.iloc[-1])
			m_hist_prev = float(macd_hist.iloc[-2])

			ind["macd_line"]   = round(m_line_last, 6)
			ind["macd_signal"] = round(m_sig_last,	6)
			ind["macd_hist"]   = round(m_hist_last, 6)

			# Granular state matching scoring matrix B.3 tiers.
			# FLAT_NEAR_ZERO threshold scales with ATR for cross-symbol parity.
			atr_val = ind.get("atr_14") or 0.0
			flat_threshold = 0.1 * atr_val
			if flat_threshold > 0 and abs(m_line_last) < flat_threshold:
				state = "FLAT_NEAR_ZERO"
			elif m_line_last > m_sig_last:
				# Bull side: expanding (BUILDING) vs contracting (FADING) histogram
				state = "BULL_BUILDING" if abs(m_hist_last) > abs(m_hist_prev) else "BULL_FADING"
			else:
				# Bear side
				state = "BEAR_BUILDING" if abs(m_hist_last) > abs(m_hist_prev) else "BEAR_FADING"
			ind["macd_state"] = state
			# prev BAR MACD state (iloc[-2] vs iloc[-3]) — bar-over-bar flip tespiti
			m_line_p2 = float(macd_line.iloc[-2])
			m_sig_p2  = float(macd_signal.iloc[-2])
			m_hist_p3 = float(macd_hist.iloc[-3]) if n >= 36 else m_hist_prev
			if flat_threshold > 0 and abs(m_line_p2) < flat_threshold:
				prev_state = "FLAT_NEAR_ZERO"
			elif m_line_p2 > m_sig_p2:
				prev_state = "BULL_BUILDING" if abs(m_hist_prev) > abs(m_hist_p3) else "BULL_FADING"
			else:
				prev_state = "BEAR_BUILDING" if abs(m_hist_prev) > abs(m_hist_p3) else "BEAR_FADING"
			ind["prev_macd_state"] = prev_state
		except Exception:
			ind["macd_line"]   = None
			ind["macd_signal"] = None
			ind["macd_hist"]   = None
			ind["macd_state"]  = "INSUFFICIENT_DATA"
			ind["prev_macd_state"]  = "INSUFFICIENT_DATA"
	else:
		ind["macd_line"]   = None
		ind["macd_signal"] = None
		ind["macd_hist"]   = None
		ind["macd_state"]  = "INSUFFICIENT_DATA"
		ind["prev_macd_state"]  = "INSUFFICIENT_DATA"

	# --- EMA confluence classification (used by scoring matrix) ---
	ind["ema_confluence"] = _ema_confluence(
		last_close=last_close,
		ema_50=ind.get("ema_50"),
		ema_200=ind.get("ema_200"),
		atr_pct=ind.get("atr_pct"),
	)

	return ind


# ═══════════════════════════════════════════════════════════════════
# SECTION 2 — SMC / ICT: Swing detection
# ═══════════════════════════════════════════════════════════════════

def _swing_highs(high: np.ndarray, order: int) -> list[int]:
	"""
	Detect swing highs over a sliding window of ±`order` bars.
	Equal highs at SAME price level: only the LEFTMOST gets registered (via argmax)
	to avoid inflating cluster touch_count with adjacent duplicates.
	Spaced double tops (separated > 2×order bars) still both register independently;
	the cluster step combines them by tolerance into a single level with touch_count=2.
	"""
	result = []
	for i in range(order, len(high) - order):
		window = high[i - order: i + order + 1]
		if high[i] == window.max() and np.argmax(window) == order:
			result.append(i)
	return result


def _swing_lows(low: np.ndarray, order: int) -> list[int]:
	"""
	Detect swing lows. Mirror of _swing_highs — equal lows handled same way.
	"""
	result = []
	for i in range(order, len(low) - order):
		window = low[i - order: i + order + 1]
		if low[i] == window.min() and np.argmin(window) == order:
			result.append(i)
	return result


def _zigzag_pivots(high: np.ndarray, low: np.ndarray, threshold_pct: float) -> list[dict]:
	"""
	ZigZag reversal pivots — prominence filter for S/R quality.
	Returns time-ordered, ALTERNATING pivots [{idx, price, kind}] (kind 'SH'/'SL')
	where every leg (move from the previous pivot) is >= threshold_pct (fraction,
	e.g. 0.02 = 2%). Wiggles smaller than the threshold (milim-milim chop) never
	become pivots, so a cluster touch_count reflects GENUINE rejections, not the
	time price spent loitering in a range. The final in-progress extreme (current
	direction) is appended so the most recent swing is not dropped.
	"""
	n = len(high)
	if n < 2 or threshold_pct <= 0:
		return []
	pivots: list[dict] = []
	hi_i, hi = 0, float(high[0])
	lo_i, lo = 0, float(low[0])
	trend = 0  # +1 up-leg (seeking high), -1 down-leg (seeking low), 0 undetermined
	for i in range(1, n):
		h = float(high[i]); l = float(low[i])
		if trend == 1:
			if h > hi:
				hi, hi_i = h, i
			elif l <= hi * (1.0 - threshold_pct):
				pivots.append({"idx": hi_i, "price": hi, "kind": "SH"})
				trend, lo, lo_i = -1, l, i
		elif trend == -1:
			if l < lo:
				lo, lo_i = l, i
			elif h >= lo * (1.0 + threshold_pct):
				pivots.append({"idx": lo_i, "price": lo, "kind": "SL"})
				trend, hi, hi_i = 1, h, i
		else:
			if h > hi:
				hi, hi_i = h, i
			if l < lo:
				lo, lo_i = l, i
			if l <= hi * (1.0 - threshold_pct):
				pivots.append({"idx": hi_i, "price": hi, "kind": "SH"})
				trend, lo, lo_i = -1, l, i
			elif h >= lo * (1.0 + threshold_pct):
				pivots.append({"idx": lo_i, "price": lo, "kind": "SL"})
				trend, hi, hi_i = 1, h, i
	if trend == 1:
		pivots.append({"idx": hi_i, "price": hi, "kind": "SH"})
	elif trend == -1:
		pivots.append({"idx": lo_i, "price": lo, "kind": "SL"})
	return pivots


def _classify_swings(
	df:		pd.DataFrame,
	sh_idx: list[int],
	sl_idx: list[int],
) -> list[dict]:
	"""
	Classifies swing points as HH / LH (higher / lower high) and HL / LL (higher / lower low).
	The very first swing of each kind is labeled simply 'SH' (Swing High) or 'SL' (Swing Low).
	"""
	result: list[dict] = []

	for k, i in enumerate(sh_idx):
		label = "SH" if k == 0 else (
			"HH" if df["high"].iloc[i] > df["high"].iloc[sh_idx[k - 1]] else "LH"
		)
		result.append({
			"index":	 i,
			"price":	 round(float(df["high"].iloc[i]), 6),
			"timestamp": str(df.index[i]),
			"type":		 label,
		})

	for k, i in enumerate(sl_idx):
		label = "SL" if k == 0 else (
			"HL" if df["low"].iloc[i] > df["low"].iloc[sl_idx[k - 1]] else "LL"
		)
		result.append({
			"index":	 i,
			"price":	 round(float(df["low"].iloc[i]), 6),
			"timestamp": str(df.index[i]),
			"type":		 label,
		})

	return sorted(result, key=lambda x: x["index"])


# ═══════════════════════════════════════════════════════════════════
# SECTION 3 — Trend determination and structure events
# ═══════════════════════════════════════════════════════════════════

def _determine_trend(swings: list[dict], last_close: Optional[float] = None,
                     ema_50: Optional[float] = None, ema_200: Optional[float] = None) -> str:
	"""
	Robust trend = confluence of two INDEPENDENT reads (BULLISH/BEARISH/RANGING):
	  1. STRUCTURE (swing HH/HL): the DOMINANT character of the last few swings,
	     not just the final pair (which can be stale). A decisive close beyond the
	     recent swing band OVERRIDES stale labels — a steady trend prints no fresh
	     pivots, so the last confirmed swing can sit far above/below price.
	  2. EMA FILTER (price vs EMA50 vs EMA200): a smooth read that tracks the
	     current price and therefore never goes stale.
	Combined: agreement -> that direction; one neutral -> the other; direct
	conflict -> RANGING (transitional, stay out). No single signal can force a
	wrong call alone. last_close / ema_* optional -> falls back to structure only.
	"""
	highs = [s for s in swings if s["type"] in ("HH", "LH")][-4:]
	lows  = [s for s in swings if s["type"] in ("HL", "LL")][-4:]
	if len(highs) < 2 or len(lows) < 2:
		return "RANGING"

	# Factor 1 — structure: dominant character of the recent swing window
	hh = sum(1 for s in highs if s["type"] == "HH"); lh = len(highs) - hh
	hl = sum(1 for s in lows  if s["type"] == "HL"); ll = len(lows)  - hl
	if hh > lh and hl > ll:
		struct = "BULLISH"
	elif lh > hh and ll > hl:
		struct = "BEARISH"
	else:
		struct = "RANGING"

	# price-break override: a close beyond the recent swing band means the
	# confirmed swings are stale -> trust the break, not the old labels.
	if last_close is not None:
		lo_band = min(s["price"] for s in lows)
		hi_band = max(s["price"] for s in highs)
		if last_close < lo_band * 0.999:
			struct = "BEARISH"
		elif last_close > hi_band * 1.001:
			struct = "BULLISH"

	# Factor 2 — EMA filter (smooth, never stale)
	ema = "RANGING"
	if last_close is not None and ema_50 is not None and ema_200 is not None:
		if last_close > ema_50 > ema_200:
			ema = "BULLISH"
		elif last_close < ema_50 < ema_200:
			ema = "BEARISH"
	elif last_close is not None and ema_50 is not None:
		if last_close > ema_50:
			ema = "BULLISH"
		elif last_close < ema_50:
			ema = "BEARISH"

	# Combine: agreement -> trend; one neutral -> the other; conflict -> RANGING
	if struct == ema:
		return struct
	if struct == "RANGING":
		return ema
	if ema == "RANGING":
		return struct
	return "RANGING"


def _classify_retest(
	df:               pd.DataFrame,
	event_bar_idx:    int,
	event_type:       str,
	level:            float,
	max_lookforward:  int   = 15,
	retest_tol_pct:   float = 0.005,   # 0.5% — covers Turtle Soup deep retest
	recovery_pct:     float = 0.005,   # 0.5% — clear directional confirmation after retest
	fail_pct:         float = 0.005,   # 0.5% — close past the level = broken retest
) -> str:
	"""
	Classifies a BOS/CHoCH retest outcome by inspecting the candles AFTER the event.

	Returns one of:
	    PENDING            — event recent, lookforward window not yet filled OR retest yet to occur
	    NO_RETEST_YET      — full window observed, price never returned to the level
	    RETESTED_RESPECTED — price returned to the level then continued in the BOS direction
	    RETESTED_BROKEN    — price returned to the level then broke through (FAKEOUT)
	"""
	n = len(df)
	if event_bar_idx >= n - 1:
		return "PENDING"

	# FLAW #4 fix: ATR-relative tolerance instead of a fixed 0.5%. A fixed band is
	# too tight for volatile instruments (a single BTC 4h bar can be 1-2%, skipping
	# a 0.5% retest zone → RETESTED_RESPECTED, the strongest C.3 signal, never
	# fires) and needlessly wide in quiet regimes. Adapt per symbol/regime.
	if level > 0:
		_lo  = max(0, event_bar_idx - 14)
		_win = df.iloc[_lo: event_bar_idx + 1]
		_tr  = (_win["high"] - _win["low"]).abs()
		_atr = float(_tr.mean()) if len(_tr) else 0.0
		if _atr > 0:
			_tol = max(0.003, min(0.02, _atr * 0.5 / level))   # clamp 0.3%–2.0%
			retest_tol_pct = recovery_pct = fail_pct = _tol

	after = df.iloc[event_bar_idx + 1: event_bar_idx + 1 + max_lookforward]
	if len(after) == 0:
		return "PENDING"

	highs = after["high"].values
	lows  = after["low"].values
	close = after["close"].values

	is_bull = event_type in ("BOS_BULL", "CHOCH_BULL")

	if is_bull:
		# Confirmed break: the event bar CLOSED above the level (the detector fires
		# on close>level, so a wick-only break that closes back never reaches here).
		# The retest must come on a LATER bar — the breaking bar's own move never
		# counts. The broken level now acts as support: a later bar must pull back
		# DOWN into the level zone, then the first decisive close decides held vs fake.
		tol = retest_tol_pct
		touch_idx = -1
		for j in range(len(after)):
			if lows[j] <= level * (1 + tol):        # pulled back down to the level
				touch_idx = j
				break
		if touch_idx == -1:
			return "PENDING" if len(after) < max_lookforward else "NO_RETEST_YET"
		# INCLUSIVE of the retest bar: a bar that wicks to the level and closes back
		# above is itself a held retest (the common single-bar case). First close wins.
		recovery_target = level * (1 + recovery_pct)   # close back above = RESPECTED
		fail_target     = level * (1 - fail_pct)       # close below the level = FAKEOUT
		for close_val in after["close"].values[touch_idx:]:
			if close_val >= recovery_target:
				return "RETESTED_RESPECTED"
			if close_val <= fail_target:
				return "RETESTED_BROKEN"
		return "PENDING"

	else:  # bear event
		# Mirror: confirmed break closed BELOW the level; the broken level now acts
		# as resistance. A later bar must pull back UP into the level zone, then the
		# first decisive close decides held (continued down) vs fakeout (closed back up).
		tol = retest_tol_pct
		touch_idx = -1
		for j in range(len(after)):
			if highs[j] >= level * (1 - tol):       # pulled back up to the level
				touch_idx = j
				break
		if touch_idx == -1:
			return "PENDING" if len(after) < max_lookforward else "NO_RETEST_YET"
		recovery_target = level * (1 - recovery_pct)   # close back below = RESPECTED
		fail_target     = level * (1 + fail_pct)       # close above the level = FAKEOUT
		for close_val in after["close"].values[touch_idx:]:
			if close_val <= recovery_target:
				return "RETESTED_RESPECTED"
			if close_val >= fail_target:
				return "RETESTED_BROKEN"
		return "PENDING"


def _detect_structure_events(
	df:		   pd.DataFrame,
	swings:	   list[dict],
	trend:	   str,
	scan_last: int = 30,
	vol_sma_period: int = 20,
) -> list[dict]:
	"""
	Detects BOS (Break of Structure) and CHoCH (Change of Character) events
	over the most recent `scan_last` candles.

	Each event is enriched with:
	    bars_since       — bar distance from the latest candle (0 = current bar)
	    is_strong_close  — True if candle body > 50% of total range
	    volume_backed    — True if event-bar volume > 1.2x 20-bar volume SMA (>20% above; fakeout filter)
	    retest_status    — PENDING / NO_RETEST_YET / RETESTED_RESPECTED / RETESTED_BROKEN

	Dedupe rule: keep the FIRST break of each unique (event_type, level) pair within the
	scan window. Subsequent re-breaks of the same level are continuation, not new events.
	"""
	events: list[dict] = []
	if not swings:
		return events

	n = len(df)
	if n == 0:
		return events

	close  = df["close"].values
	open_  = df["open"].values
	high   = df["high"].values
	low    = df["low"].values
	volume = df["volume"].values if "volume" in df.columns else None

	# Pre-compute rolling volume SMA for volume_backed check.
	# IMPORTANT: shift(1) so the bar being measured is NOT in its own benchmark.
	# Without shift, a high-volume breakout candle pulls up the SMA it's being
	# compared against — small bias that can flip borderline volume calls.
	vol_sma = None
	if volume is not None and n >= vol_sma_period:
		vol_sma = pd.Series(volume).shift(1).rolling(vol_sma_period, min_periods=1).mean().values

	start = max(0, n - scan_last)

	def _latest_swing_before(bar_idx: int, types: tuple) -> Optional[dict]:
		"""Find the latest swing of the given types whose index is < bar_idx."""
		for s in reversed(swings):
			if s["index"] < bar_idx and s["type"] in types:
				return s
		return None

	# Build raw event list — every bar that breaks a structurally-relevant swing
	for i in range(start, n):
		c  = float(close[i])
		o  = float(open_[i])
		h  = float(high[i])
		l  = float(low[i])
		ts = str(df.index[i])

		bar_range = h - l
		body      = abs(c - o)
		is_strong = bool(body > 0.5 * bar_range) if bar_range > 1e-9 else False

		if vol_sma is not None and i < len(vol_sma):
			vol_avg = float(vol_sma[i]) if not pd.isna(vol_sma[i]) else 0.0
			volume_backed = bool(volume[i] > 1.2 * vol_avg) if vol_avg > 0 else None  # >20% above 20-bar SMA
		else:
			volume_backed = None

		bars_since = (n - 1) - i

		if trend == "BULLISH":
			hh = _latest_swing_before(i, ("HH", "SH"))
			hl = _latest_swing_before(i, ("HL",))
			if hh and c > hh["price"]:
				events.append({
					"type":            "BOS_BULL",
					"level":           hh["price"],
					"close":           round(c, 6),
					"timestamp":       ts,
					"bar_idx":         i,
					"bars_since":      bars_since,
					"is_strong_close": is_strong,
					"volume_backed":   volume_backed,
					"description":     f"Bullish BOS @ t-{bars_since}: close {c:.4f} > HH {hh['price']:.4f}",
				})
			if hl and c < hl["price"]:
				events.append({
					"type":            "CHOCH_BEAR",
					"level":           hl["price"],
					"close":           round(c, 6),
					"timestamp":       ts,
					"bar_idx":         i,
					"bars_since":      bars_since,
					"is_strong_close": is_strong,
					"volume_backed":   volume_backed,
					"description":     f"Bearish CHoCH @ t-{bars_since}: close {c:.4f} < HL {hl['price']:.4f}",
				})

		elif trend == "BEARISH":
			ll = _latest_swing_before(i, ("LL", "SL"))
			lh = _latest_swing_before(i, ("LH",))
			if ll and c < ll["price"]:
				events.append({
					"type":            "BOS_BEAR",
					"level":           ll["price"],
					"close":           round(c, 6),
					"timestamp":       ts,
					"bar_idx":         i,
					"bars_since":      bars_since,
					"is_strong_close": is_strong,
					"volume_backed":   volume_backed,
					"description":     f"Bearish BOS @ t-{bars_since}: close {c:.4f} < LL {ll['price']:.4f}",
				})
			if lh and c > lh["price"]:
				events.append({
					"type":            "CHOCH_BULL",
					"level":           lh["price"],
					"close":           round(c, 6),
					"timestamp":       ts,
					"bar_idx":         i,
					"bars_since":      bars_since,
					"is_strong_close": is_strong,
					"volume_backed":   volume_backed,
					"description":     f"Bullish CHoCH @ t-{bars_since}: close {c:.4f} > LH {lh['price']:.4f}",
				})

		elif trend == "RANGING":
			# Range bounds = MAX swing high / MIN swing low, with two refinements:
			#  (a) WIDER lookback (50 bars) than BOS/CHoCH detection. Gemini correctly
			#      flagged that a 30-bar window can sit INSIDE a wider 50-bar range and
			#      fire false breakouts off a minor interior swing. A wider window
			#      captures the consolidation's true edges.
			#  (b) DISCARD swings further than 12% from the current close — those are
			#      remnants of a PRIOR trend, not part of the active range. This avoids
			#      the bug that Gemini's "last-12-swings" proposal would introduce: an
			#      old downtrend high (e.g. 2500 while price ranges near 2075) would
			#      become a phantom range top and mask every genuine breakout.
			range_start    = max(0, i - 50)
			max_range_dist = c * 0.12
			window_swings_before = [
				s for s in swings
				if s["index"] < i and s["index"] >= range_start
				and abs(s["price"] - c) <= max_range_dist
			]
			highs_in_window = [s for s in window_swings_before
							   if s["type"] in ("SH", "HH", "LH")]
			lows_in_window  = [s for s in window_swings_before
							   if s["type"] in ("SL", "HL", "LL")]

			range_top = max(highs_in_window, key=lambda s: s["price"]) if highs_in_window else None
			range_bot = min(lows_in_window,  key=lambda s: s["price"]) if lows_in_window  else None

			# Range top break = bullish breakout (potential new trend birth)
			if range_top and c > range_top["price"]:
				events.append({
					"type":            "BOS_BULL",
					"level":           range_top["price"],
					"close":           round(c, 6),
					"timestamp":       ts,
					"bar_idx":         i,
					"bars_since":      bars_since,
					"is_strong_close": is_strong,
					"volume_backed":   volume_backed,
					"description":     f"Range Top Breakout @ t-{bars_since}: close {c:.4f} > Range Top {range_top['price']:.4f}",
				})
			# Range bottom break = bearish breakdown (potential new trend birth)
			if range_bot and c < range_bot["price"]:
				events.append({
					"type":            "BOS_BEAR",
					"level":           range_bot["price"],
					"close":           round(c, 6),
					"timestamp":       ts,
					"bar_idx":         i,
					"bars_since":      bars_since,
					"is_strong_close": is_strong,
					"volume_backed":   volume_backed,
					"description":     f"Range Bottom Breakdown @ t-{bars_since}: close {c:.4f} < Range Bot {range_bot['price']:.4f}",
				})

	# Dedupe by (type, level) — keep the MOST RECENT break.
	# When the same structural level is broken multiple times within the scan
	# window (e.g. broken at t-40, retraced, broken again at t-2), the latest
	# event is the structurally fresh moment. AI's `bars_since <= 5` freshness
	# filter relies on this — keeping the old timestamp would mask fresh signals.
	seen: set = set()
	deduped: list[dict] = []
	for ev in reversed(events):
		key = (ev["type"], round(ev["level"], 6))
		if key in seen:
			continue
		seen.add(key)
		deduped.append(ev)
	deduped.reverse()   # restore chronological order (oldest first, newest last)

	# Classify retest status for each event using post-event candles
	for ev in deduped:
		ev["retest_status"] = _classify_retest(
			df=df,
			event_bar_idx=ev["bar_idx"],
			event_type=ev["type"],
			level=ev["level"],
		)
		# Remove internal bar_idx before returning (not needed downstream)
		ev.pop("bar_idx", None)

	# Sort by recency (most recent last) and cap
	deduped.sort(key=lambda e: -e["bars_since"])   # oldest first, newest last
	return deduped[-5:]


# ═══════════════════════════════════════════════════════════════════
# SECTION 4 — Fair Value Gap (FVG) detection
# ═══════════════════════════════════════════════════════════════════

def _detect_fvgs(df: pd.DataFrame, lookback: int = 100) -> list[dict]:
	"""
	Detects unmitigated 3-candle Fair Value Gaps within the last `lookback` candles.
	Bull FVG: high[i-2] < low[i].
	Bear FVG: low[i-2] > high[i].
	Mitigation: any subsequent candle that closes into the gap mid removes it.
	"""
	fvgs: list[dict] = []
	start  = max(2, len(df) - lookback)
	high   = df["high"].values
	low	   = df["low"].values

	for i in range(start, len(df)):
		h0, l0 = float(high[i - 2]), float(low[i - 2])
		h2, l2 = float(high[i]),	 float(low[i])

		# Bull FVG
		if h0 < l2:
			gap_lo, gap_hi = h0, l2
			mid = (gap_lo + gap_hi) / 2.0
			if not any(float(low[j]) <= mid for j in range(i + 1, len(df))):
				fvgs.append({
					"type":		 "BULL_FVG",
					"upper":	 round(gap_hi, 6),
					"lower":	 round(gap_lo, 6),
					"size_pct":	 round((gap_hi - gap_lo) / gap_lo * 100, 4) if gap_lo else 0.0,
					"timestamp": str(df.index[i]),
				})

		# Bear FVG
		if l0 > h2:
			gap_lo, gap_hi = h2, l0
			mid = (gap_lo + gap_hi) / 2.0
			if not any(float(high[j]) >= mid for j in range(i + 1, len(df))):
				fvgs.append({
					"type":		 "BEAR_FVG",
					"upper":	 round(gap_hi, 6),
					"lower":	 round(gap_lo, 6),
					"size_pct":	 round((gap_hi - gap_lo) / gap_lo * 100, 4) if gap_lo else 0.0,
					"timestamp": str(df.index[i]),
				})

	return fvgs[-6:]


# ═══════════════════════════════════════════════════════════════════
# SECTION 5 — Order Block detection
# ═══════════════════════════════════════════════════════════════════

def _detect_order_blocks(df: pd.DataFrame, lookback: int = 80) -> list[dict]:
	"""
	Institutional Order Block detection (ICT/SMC compliant):

	Conditions for a valid OB:
		1. Anchor candle (i) is the LAST opposite-color candle before an impulse.
		2. Impulse candle (i+1) has a strong body (>= 1.5×ATR or >= 1% notional).
		3. *** FVG VALIDATION ***: The 3-candle window (i, i+1, i+2) MUST form an
		   imbalance (Fair Value Gap). Without an FVG, the impulse is just a
		   correction candle, not institutional displacement.
		     Bull OB: high[i] < low[i+2]    (gap above the anchor)
		     Bear OB: low[i]  > high[i+2]   (gap below the anchor)

	Mitigation (tap-based, ICT compliant):
		Bull OB is mitigated when price returns DOWN to the OB's UPPER edge
		(any subsequent low ≤ ob_high). At that point institutional orders at
		the zone get filled — the OB has "done its job."
		Bear OB mirror: high ≥ ob_low.

	Returns the last 4 unmitigated OBs.
	"""
	obs: list[dict] = []
	n = len(df)
	if n < 25:
		return obs

	start = max(2, n - lookback)
	o = df["open"].values
	h = df["high"].values
	l = df["low"].values
	c = df["close"].values

	atr = _calc_atr(df, 14).values if n >= 16 else None

	# We need bar i+2 to exist for FVG validation, so loop ends at n-2 not n-1.
	for i in range(start, n - 2):
		# Forward impulse candle: i+1
		body_next = abs(c[i + 1] - o[i + 1])
		ref_atr   = atr[i + 1] if atr is not None else (h[i + 1] - l[i + 1])
		if body_next < 1.5 * ref_atr and body_next / max(o[i + 1], 1e-9) < 0.01:
			continue

		# Bull OB: last DOWN candle before strong UP impulse + FVG validation
		if c[i + 1] > o[i + 1] and c[i] < o[i]:
			has_bull_fvg = h[i] < l[i + 2]
			if not has_bull_fvg:
				continue   # not a true institutional OB (no displacement gap)
			ob_low    = float(l[i])
			ob_high   = float(max(o[i], c[i]))
			# Tap-based mitigation: any subsequent LOW reaching the TOP of the OB.
			mitigated = any(float(l[j]) <= ob_high for j in range(i + 3, n))
			if not mitigated:
				obs.append({
					"type":         "BULL_OB",
					"upper":        round(ob_high, 6),
					"lower":        round(ob_low, 6),
					"fvg_gap":      round(l[i + 2] - h[i], 6),
					"impulse_pct":  round(body_next / max(o[i + 1], 1e-9) * 100, 3),
					"timestamp":    str(df.index[i]),
				})

		# Bear OB: last UP candle before strong DOWN impulse + FVG validation
		if c[i + 1] < o[i + 1] and c[i] > o[i]:
			has_bear_fvg = l[i] > h[i + 2]
			if not has_bear_fvg:
				continue
			ob_low    = float(min(o[i], c[i]))
			ob_high   = float(h[i])
			# Tap-based mitigation: any subsequent HIGH reaching the BOTTOM of the OB.
			mitigated = any(float(h[j]) >= ob_low for j in range(i + 3, n))
			if not mitigated:
				obs.append({
					"type":         "BEAR_OB",
					"upper":        round(ob_high, 6),
					"lower":        round(ob_low, 6),
					"fvg_gap":      round(l[i] - h[i + 2], 6),
					"impulse_pct":  round(body_next / max(o[i + 1], 1e-9) * 100, 3),
					"timestamp":    str(df.index[i]),
				})

	return obs[-4:]


# ═══════════════════════════════════════════════════════════════════
# SECTION 6 — Candlestick patterns
# ═══════════════════════════════════════════════════════════════════

def _detect_patterns(df: pd.DataFrame, lookback: int = 30) -> list[dict]:
	"""
	Detects a small set of well-known candlestick reversal/continuation patterns
	over the last `lookback` candles. Patterns recognised:
		- Bullish / Bearish Engulfing
		- Hammer / Inverted Hammer
		- Shooting Star
		- Doji
	"""
	patterns: list[dict] = []
	n = len(df)
	if n < 3:
		return patterns

	start = max(1, n - lookback)
	o = df["open"].values
	h = df["high"].values
	l = df["low"].values
	c = df["close"].values

	for i in range(start, n):
		body_size  = abs(c[i] - o[i])
		upper_wick = h[i] - max(o[i], c[i])
		lower_wick = min(o[i], c[i]) - l[i]
		total_rng  = h[i] - l[i]
		if total_rng <= 0:
			continue

		body_pct = body_size / total_rng

		# Local position of THIS candle within the recent window decides whether a
		# single-wick pattern is a bullish bottom-reversal or a bearish top-reversal.
		# Classic candlestick theory keys off WHERE the shape forms, not its colour:
		#   long LOWER wick at a swing LOW  → Hammer (bullish)
		#   long LOWER wick at a swing HIGH → Hanging Man (bearish)
		#   long UPPER wick at a swing HIGH → Shooting Star (bearish)
		#   long UPPER wick at a swing LOW  → Inverted Hammer (bullish)
		win_lo_idx = max(0, i - 10)
		win_high   = float(h[win_lo_idx:i + 1].max())
		win_low    = float(l[win_lo_idx:i + 1].min())
		win_range  = win_high - win_low
		if win_range > 1e-9:
			pos = (c[i] - win_low) / win_range   # 0.0 = at bottom, 1.0 = at top
		else:
			pos = 0.5
		is_at_top    = pos > 0.66
		is_at_bottom = pos < 0.34

		# Engulfing
		if i >= 1:
			prev_body = abs(c[i - 1] - o[i - 1])
			if (c[i] > o[i] and c[i - 1] < o[i - 1]
					and o[i] <= c[i - 1] and c[i] >= o[i - 1]
					and body_size > prev_body):
				patterns.append({
					"pattern":	 "BULLISH_ENGULFING",
					"direction": "BULLISH",
					"timestamp": str(df.index[i]),
				})
			if (c[i] < o[i] and c[i - 1] > o[i - 1]
					and o[i] >= c[i - 1] and c[i] <= o[i - 1]
					and body_size > prev_body):
				patterns.append({
					"pattern":	 "BEARISH_ENGULFING",
					"direction": "BEARISH",
					"timestamp": str(df.index[i]),
				})

		# Long LOWER wick (hammer shape) — location decides bull vs bear
		if body_pct < 0.35 and lower_wick > 2 * body_size and upper_wick < body_size:
			if is_at_bottom:
				patterns.append({
					"pattern":	 "HAMMER",
					"direction": "BULLISH",
					"timestamp": str(df.index[i]),
				})
			elif is_at_top:
				patterns.append({
					"pattern":	 "HANGING_MAN",
					"direction": "BEARISH",
					"timestamp": str(df.index[i]),
				})
			# mid-range → ambiguous, emit nothing

		# Long UPPER wick (inverted shape) — location decides bull vs bear
		if body_pct < 0.35 and upper_wick > 2 * body_size and lower_wick < body_size:
			if is_at_top:
				patterns.append({
					"pattern":	 "SHOOTING_STAR",
					"direction": "BEARISH",
					"timestamp": str(df.index[i]),
				})
			elif is_at_bottom:
				patterns.append({
					"pattern":	 "INVERTED_HAMMER",
					"direction": "BULLISH",
					"timestamp": str(df.index[i]),
				})
			# mid-range → ambiguous, emit nothing

		# Doji
		if body_pct < 0.10:
			patterns.append({
				"pattern":	 "DOJI",
				"direction": "NEUTRAL",
				"timestamp": str(df.index[i]),
			})

	return patterns[-8:]


# ═══════════════════════════════════════════════════════════════════
# SECTION 7 — Historical Major S/R clustering (1D + 4H)
# ═══════════════════════════════════════════════════════════════════

# S/R bir BOLGE (zone): kume merkezi +/- yari-genislik. Dokunus yayilimini
# kullanir, ama en az %0.25 (0.5% genis) ve en cok %0.6 (1.2% genis) ile sinirlar.
_SR_ZONE_MIN_HALF_PCT = 0.0025
_SR_ZONE_MAX_HALF_PCT = 0.006

# 4H'e ozel ayri S/R havuzu (saf 4H, gun-dedup YOK). Step 1: tespit+teshis.
H4_SWING_ORDER       = 2       # hassas: 4H fitil diplerini yakala (2 bar her yanda)
H4_TOLERANCE_PCT     = 0.02    # 4H fitilleri <=%2 banda kumele (zone genisligi <=%2 otomatik)
H4_ZONE_MAX_HALF_PCT = 0.01    # zone yari-genislik <=%1 -> genislik <=%2
H4_ZIGZAG_PCT        = 0.02    # prominence: pivot icin donus >= %2 (milim-milim chop'u eler)
D1_TOLERANCE_PCT     = 0.02    # gunluk fitilleri <=%2 banda kumele
D1_ZONE_MAX_HALF_PCT = 0.01    # zone yari-genislik <=%1 -> genislik <=%2
D1_ZIGZAG_PCT        = 0.03    # gunluk: pivot icin donus >= %3 (gunluk swing iri; major seviyeler)
D1_MIN_TOUCHES       = 4       # harman: gunluk seviye gecerli icin min dokunus (denesel: 4)
H4_MIN_TOUCHES       = 7       # harman: 4H seviye gecerli icin min dokunus
BLEND_OVERLAP_PCT    = 0.01    # 1D & 4H zon merkezleri <=%1 ise ortusme (zone bandi kesisimi de sayilir)

def _cluster_swing_levels(
	levels:		   list[dict],
	tolerance_pct: float = 0.015,
	dedup_by_date: bool = True,
	zone_max_half_pct: Optional[float] = None,
) -> list[dict]:
	if not levels:
		return []

	sorted_levels = sorted(levels, key=lambda x: x["price"])
	clusters: list[dict] = []

	for lv in sorted_levels:
		placed = False
		for cl in clusters:
			# DÜZELTME 1: Çapa (Anchor) eklendi! Merkez kayması kesin olarak engellendi.
			ref = cl["anchor_price"] 
			
			if abs(lv["price"] - ref) / ref <= tolerance_pct:
				
				# DÜZELTME 2: Çifte Sayım (Double Counting) Engellemesi.
				# Aynı güne ait 1D ve 4H tepeleri tek bir dokunuş (touch) sayılacak.
				lv_date = str(lv["timestamp"])[:10]  # Tarihin YYYY-MM-DD kısmını al
				is_duplicate = False
				if dedup_by_date:
					for ts in cl["raw_timestamps"]:
						if str(ts)[:10] == lv_date:
							is_duplicate = True
							break
				
				if not is_duplicate:
					cl["sh_touches"]  += 1 if lv["kind"] == "SH" else 0
					cl["sl_touches"]  += 1 if lv["kind"] == "SL" else 0
					cl["touch_count"] += 1
				
				cl["prices"].append(lv["price"])
				cl["raw_timestamps"].append(lv["timestamp"])
				
				# Ortalama fiyat görsel amaçlı hesaplanıyor, eşleşmeler ANCHOR fiyata göre devam ediyor.
				cl["price"] = sum(cl["prices"]) / len(cl["prices"])
				placed = True
				break
				
		if not placed:
			clusters.append({
				"anchor_price": lv["price"],  # YENİ: Kümeyi başlatan sabit referans fiyatı
				"price":		lv["price"],
				"prices":		[lv["price"]],
				"sh_touches":	1 if lv["kind"] == "SH" else 0,
				"sl_touches":	1 if lv["kind"] == "SL" else 0,
				"touch_count":	1,
				"raw_timestamps": [lv["timestamp"]],
			})

	result: list[dict] = []
	for cl in clusters:
		lo = min(cl["prices"])
		hi = max(cl["prices"])
		_center = cl["price"]
		_zhalf  = (hi - lo) / 2.0
		_max_half = zone_max_half_pct if zone_max_half_pct is not None else _SR_ZONE_MAX_HALF_PCT
		_zhalf  = max(_center * _SR_ZONE_MIN_HALF_PCT, min(_zhalf, _center * _max_half))
		result.append({
			"price":	   round(cl["price"], 6),
			"zone_low":    round(_center - _zhalf, 6),
			"zone_high":   round(_center + _zhalf, 6),
			"price_range": f"{lo:.4f}-{hi:.4f}",
			"touch_count": cl["touch_count"],
			"sh_touches":  cl["sh_touches"],
			"sl_touches":  cl["sl_touches"],
			"timestamps":  [str(ts) for ts in cl["raw_timestamps"][-5:]],
		})

	return result


def _psychological_levels(
	current_price: float,
	increment:     float,
	n_above:       int   = 3,
	n_below:       int   = 3,
) -> list[float]:
	"""
	Generate psychological round-number price levels around the current price.

	Example:
	    current_price = 76_400, increment = 5000
	    → [75_000, 80_000, 85_000, 90_000, 70_000, 65_000, 60_000]
	"""
	if increment <= 0 or current_price <= 0:
		return []
	# Snap to nearest multiple at or below current price
	base   = math.floor(current_price / increment) * increment
	levels = []
	for k in range(1, n_above + 1):
		levels.append(round(base + k * increment, 6))
	for k in range(0, n_below):
		v = base - k * increment
		if v > 0:
			levels.append(round(v, 6))
	return sorted(set(levels))


def _blend_sr_pools(d1_levels: list[dict], h4_levels: list[dict], current_price: float) -> list[dict]:
	"""
	Blend the 1D and 4H S/R pools into one set (Step 2):
	  - keep 1D levels with touch_count >= D1_MIN_TOUCHES, 4H with >= H4_MIN_TOUCHES
	  - 1D & 4H zone OVERLAP (bands intersect, or centers within BLEND_OVERLAP_PCT)
	    => MAJOR (two-sided confirmation): merged, touch_counts summed, heaviest weight
	  - 1D-only (far/deep, no 4H nearby) => kept, strong weight
	  - 4H-only => kept, lighter weight
	Each entry tagged tf_source / is_major / weight. Sorted by price ascending.
	"""
	d1 = [dict(x) for x in d1_levels if x.get("touch_count", 0) >= D1_MIN_TOUCHES]
	h4 = [dict(x) for x in h4_levels if x.get("touch_count", 0) >= H4_MIN_TOUCHES]

	def _zone(x):
		return (x.get("zone_low", x["price"]), x.get("zone_high", x["price"]))

	def _overlaps(a, b):
		alo, ahi = _zone(a)
		blo, bhi = _zone(b)
		if alo <= bhi and blo <= ahi:
			return True
		ref = max(abs(a["price"]), 1e-9)
		return abs(a["price"] - b["price"]) / ref <= BLEND_OVERLAP_PCT

	blended: list[dict] = []
	used_h4: set = set()
	for a in d1:
		matches = [j for j, b in enumerate(h4) if j not in used_h4 and _overlaps(a, b)]
		entry = dict(a)
		if matches:
			entry["touch_count"] = a.get("touch_count", 0) + sum(h4[j].get("touch_count", 0) for j in matches)
			entry["tf_source"]   = "MAJOR_1D+4H"
			entry["is_major"]    = True
			entry["weight"]      = 1.0
			for j in matches:
				used_h4.add(j)
		else:
			entry["tf_source"] = "D1"
			entry["is_major"]  = False
			entry["weight"]    = 0.85
		blended.append(entry)
	for j, b in enumerate(h4):
		if j in used_h4:
			continue
		entry = dict(b)
		entry["tf_source"] = "H4"
		entry["is_major"]  = False
		entry["weight"]    = 0.70
		blended.append(entry)

	blended.sort(key=lambda x: x["price"])
	return blended


def _find_historical_key_levels_from_ohlcv(
	ohlcv_dict:			 dict,
	current_price:		 float,
	swing_order:		 int   = 5,
	swing_orders_per_tf: Optional[dict] = None,    # PR-4: adaptive per-TF orders
	tolerance_pct:		 float = 0.015,
	min_touches:		 int   = 3,
	sr_flip_min_touches: int   = 3,
	psych_increment:     Optional[float] = None,
	poc_price:           Optional[float] = None,
	intraday_min_touches: int  = 4,
	intraday_tolerance_pct: float = 0.008,
) -> dict:
	"""
	Cluster swing points from 1D and 4H to produce major support / resistance levels
	plus an additional 1h intraday level pool. Inject POC and psychological round
	numbers as labeled levels when provided.

	swing_orders_per_tf — if provided, uses TF-specific swing order (e.g. {1d:5, 4h:5,
	1h:4, 15m:3}). Falls back to `swing_order` for any TF not in the dict.
	"""
	def _order_for(tf: str) -> int:
		if swing_orders_per_tf:
			return swing_orders_per_tf.get(tf, swing_order)
		return swing_order

	# ─── HTF (1D + 4H) clustering ──────────────────────────────────────────
	htf_levels: list[dict] = []
	for tf in ("1d", "4h"):
		df = ohlcv_dict.get(tf)
		order = _order_for(tf)
		if df is None or len(df) < order * 2 + 10:
			continue
		df_c = df.copy()
		df_c.columns = [c.lower() for c in df_c.columns]
		sh_idx = _swing_highs(df_c["high"].values, order)
		sl_idx = _swing_lows(df_c["low"].values,   order)
		for i in sh_idx:
			htf_levels.append({
				"price":     float(df_c["high"].iloc[i]),
				"kind":      "SH",
				"tf":        tf,
				"timestamp": str(df_c.index[i]),
			})
		for i in sl_idx:
			htf_levels.append({
				"price":     float(df_c["low"].iloc[i]),
				"kind":      "SL",
				"tf":        tf,
				"timestamp": str(df_c.index[i]),
			})

	major_resistances: list[dict] = []
	major_supports:    list[dict] = []
	sr_flips:          list[dict] = []

	if htf_levels:
		clusters = _cluster_swing_levels(htf_levels, tolerance_pct)
		valid    = [c for c in clusters if c["touch_count"] >= min_touches]

		# --- TESHIS (gecici): mevcut fiyatin ALTINDAKI aday kumeler + dokunus sayisi ---
		#     Sup=0 sebebini gormek icin: altta seviye var ama eslik mi eliyor, yoksa veri yok mu?
		_below = sorted([c for c in clusters if c["price"] <= current_price],
		                key=lambda c: c["price"], reverse=True)
		if _below:
			logger.info("[PA-DIAG] current=%.4f | altta %d aday kume (min_touches=%d):",
			            current_price, len(_below), min_touches)
			for _c in _below[:8]:
				logger.info("[PA-DIAG]   price=%.4f touches=%d (sl=%d sh=%d) -> %s",
				            _c["price"], _c["touch_count"], _c["sl_touches"], _c["sh_touches"],
				            "GECTI" if _c["touch_count"] >= min_touches else "ELENDI (<min)")
		else:
			logger.info("[PA-DIAG] current=%.4f | altta HIC swing-low kumesi YOK (veri penceresinde dip yok)", current_price)

		for cluster in valid:
			price = cluster["price"]
			sh    = cluster["sh_touches"]
			sl    = cluster["sl_touches"]
			entry = dict(cluster)
			entry["source"] = "HTF_SWING"

			if price > current_price:
				entry["label"] = "RESISTANCE"
				if sl >= sr_flip_min_touches and sl > sh:
					entry["label"] = "SR_FLIP_RESISTANCE"
					entry["flip_desc"] = (
						f"Former support → new resistance: tested as support {sl} time(s). "
						f"Prime SHORT zone."
					)
					sr_flips.append(entry)
				major_resistances.append(entry)
			else:
				entry["label"] = "SUPPORT"
				if sh >= sr_flip_min_touches and sh > sl:
					entry["label"] = "SR_FLIP_SUPPORT"
					entry["flip_desc"] = (
						f"Former resistance → new support: tested as resistance {sh} time(s). "
						f"Prime LONG zone."
					)
					sr_flips.append(entry)
				major_supports.append(entry)

	# --- 4H OZEL HAVUZ (TESHIS - 1D'den ayri, saf 4H) ---
	#     Saf 4H fitil kumeleme - TF-arasi gun-dedup YOK (her 4H pivotu = 1
	#     dokunus), hassas order, <=%2 zone. Step 1: sadece tespit+log;
	#     skorlama hala HTF/major havuzunu kullaniyor (degismedi).
	h4_resistances: list[dict] = []
	h4_supports:    list[dict] = []
	df_4h = ohlcv_dict.get("4h")
	if df_4h is not None and len(df_4h) >= 20:
		df_4h_c = df_4h.copy()
		df_4h_c.columns = [c.lower() for c in df_4h_c.columns]
		_zz_4h = _zigzag_pivots(df_4h_c["high"].values, df_4h_c["low"].values, H4_ZIGZAG_PCT)
		h4_levels: list[dict] = []
		for _p in _zz_4h:
			h4_levels.append({"price": _p["price"], "kind": _p["kind"], "tf": "4h", "timestamp": str(df_4h_c.index[_p["idx"]])})
		if h4_levels:
			h4_clusters = _cluster_swing_levels(h4_levels, H4_TOLERANCE_PCT, dedup_by_date=False, zone_max_half_pct=H4_ZONE_MAX_HALF_PCT)
			_h4_below = sorted([c for c in h4_clusters if c["price"] <= current_price], key=lambda c: c["price"], reverse=True)
			_h4_above = sorted([c for c in h4_clusters if c["price"] > current_price], key=lambda c: c["price"])
			logger.info("[PA-DIAG-H4] current=%.4f | 4H saf havuz (zigzag>=%.1f%% donus, dedup=YOK):", current_price, H4_ZIGZAG_PCT * 100)
			for _c in _h4_below[:8]:
				logger.info("[PA-DIAG-H4]   SUP zone=%.4f [%.4f-%.4f] touches=%d (sl=%d sh=%d)", _c["price"], _c["zone_low"], _c["zone_high"], _c["touch_count"], _c["sl_touches"], _c["sh_touches"])
			for _c in _h4_above[:8]:
				logger.info("[PA-DIAG-H4]   RES zone=%.4f [%.4f-%.4f] touches=%d (sl=%d sh=%d)", _c["price"], _c["zone_low"], _c["zone_high"], _c["touch_count"], _c["sl_touches"], _c["sh_touches"])
			for _c in h4_clusters:
				entry = dict(_c)
				entry["source"] = "H4_SWING"
				if _c["price"] > current_price:
					entry["label"] = "H4_RESISTANCE"
					h4_resistances.append(entry)
				else:
					entry["label"] = "H4_SUPPORT"
					h4_supports.append(entry)
			h4_resistances.sort(key=lambda x: x["price"])
			h4_supports.sort(key=lambda x: x["price"], reverse=True)

	# --- 1D OZEL HAVUZ (Step 1: sadece tespit+log; harman/baglama henuz YOK) ---
	#     4H ile ayni ZigZag+kume pipeline'i, gunluk uzerinde. Gunluk swing'ler iri
	#     oldugu icin esik %3 (4H %2). Buyuk/uzak yapi + eski tabanlar icindir.
	d1_resistances: list[dict] = []
	d1_supports:    list[dict] = []
	df_1d = ohlcv_dict.get("1d")
	if df_1d is not None and len(df_1d) >= 20:
		df_1d_c = df_1d.copy()
		df_1d_c.columns = [c.lower() for c in df_1d_c.columns]
		_zz_1d = _zigzag_pivots(df_1d_c["high"].values, df_1d_c["low"].values, D1_ZIGZAG_PCT)
		d1_levels: list[dict] = []
		for _p in _zz_1d:
			d1_levels.append({"price": _p["price"], "kind": _p["kind"], "tf": "1d", "timestamp": str(df_1d_c.index[_p["idx"]])})
		if d1_levels:
			d1_clusters = _cluster_swing_levels(d1_levels, D1_TOLERANCE_PCT, dedup_by_date=False, zone_max_half_pct=D1_ZONE_MAX_HALF_PCT)
			_d1_below = sorted([c for c in d1_clusters if c["price"] <= current_price], key=lambda c: c["price"], reverse=True)
			_d1_above = sorted([c for c in d1_clusters if c["price"] > current_price], key=lambda c: c["price"])
			logger.info("[PA-DIAG-D1] current=%.4f | 1D saf havuz (zigzag>=%.1f%% donus):", current_price, D1_ZIGZAG_PCT * 100)
			for _c in _d1_below[:8]:
				logger.info("[PA-DIAG-D1]   SUP zone=%.4f [%.4f-%.4f] touches=%d (sl=%d sh=%d)", _c["price"], _c["zone_low"], _c["zone_high"], _c["touch_count"], _c["sl_touches"], _c["sh_touches"])
			for _c in _d1_above[:8]:
				logger.info("[PA-DIAG-D1]   RES zone=%.4f [%.4f-%.4f] touches=%d (sl=%d sh=%d)", _c["price"], _c["zone_low"], _c["zone_high"], _c["touch_count"], _c["sl_touches"], _c["sh_touches"])
			for _c in d1_clusters:
				entry = dict(_c)
				entry["source"] = "D1_SWING"
				if _c["price"] > current_price:
					entry["label"] = "D1_RESISTANCE"
					d1_resistances.append(entry)
				else:
					entry["label"] = "D1_SUPPORT"
					d1_supports.append(entry)
			d1_resistances.sort(key=lambda x: x["price"])
			d1_supports.sort(key=lambda x: x["price"], reverse=True)

	# --- HARMAN (Step 2a: 1D+4H birlestir, GOZLEM; scorer/LLM henuz baglanmadi) ---
	#     Ortusen 1D+4H = MAJOR (iki tarafli teyit). Uzak/derin 4H'siz = sadece 1D.
	#     Floor: gunluk>=5, 4H>=7 (zayif/chop seviyeler elenir).
	_blended_all = _blend_sr_pools(d1_resistances + d1_supports, h4_resistances + h4_supports, current_price)
	blended_resistances = sorted([x for x in _blended_all if x["price"] > current_price], key=lambda x: x["price"])
	blended_supports    = sorted([x for x in _blended_all if x["price"] <= current_price], key=lambda x: x["price"], reverse=True)
	logger.info("[PA-DIAG-BLEND] current=%.4f | HARMAN (1D>=%d, 4H>=%d dokunus; ortusen=MAJOR):", current_price, D1_MIN_TOUCHES, H4_MIN_TOUCHES)
	if not blended_supports:
		logger.info("[PA-DIAG-BLEND]   SUP: yok (fiyatin altinda guclu seviye yok -> gap)")
	for _b in blended_supports[:6]:
		logger.info("[PA-DIAG-BLEND]   SUP %.4f touches=%d [%s w=%.2f]", _b["price"], _b["touch_count"], _b.get("tf_source", "?"), _b.get("weight", 0.0))
	for _b in blended_resistances[:6]:
		logger.info("[PA-DIAG-BLEND]   RES %.4f touches=%d [%s w=%.2f]", _b["price"], _b["touch_count"], _b.get("tf_source", "?"), _b.get("weight", 0.0))

	# ─── 1h intraday clustering (PR-2: separate pool with tighter filter) ──
	intraday_resistances: list[dict] = []
	intraday_supports:    list[dict] = []
	df_1h = ohlcv_dict.get("1h")
	order_1h = _order_for("1h")
	if df_1h is not None and len(df_1h) >= order_1h * 2 + 10:
		df_1h_c = df_1h.copy()
		df_1h_c.columns = [c.lower() for c in df_1h_c.columns]
		sh_1h = _swing_highs(df_1h_c["high"].values, order_1h)
		sl_1h = _swing_lows(df_1h_c["low"].values,   order_1h)

		intra_levels: list[dict] = []
		for i in sh_1h:
			intra_levels.append({
				"price": float(df_1h_c["high"].iloc[i]), "kind": "SH", "tf": "1h",
				"timestamp": str(df_1h_c.index[i]),
			})
		for i in sl_1h:
			intra_levels.append({
				"price": float(df_1h_c["low"].iloc[i]), "kind": "SL", "tf": "1h",
				"timestamp": str(df_1h_c.index[i]),
			})
		if intra_levels:
			intra_clusters = _cluster_swing_levels(intra_levels, intraday_tolerance_pct)
			intra_valid    = [c for c in intra_clusters if c["touch_count"] >= intraday_min_touches]
			for cluster in intra_valid:
				price = cluster["price"]
				entry = dict(cluster)
				entry["source"] = "INTRADAY_1H"
				entry["label"]  = "INTRADAY_RES" if price > current_price else "INTRADAY_SUP"
				if price > current_price:
					intraday_resistances.append(entry)
				else:
					intraday_supports.append(entry)

	# ─── POC injection ─────────────────────────────────────────────────────
	poc_entry = None
	if poc_price is not None and poc_price > 0:
		poc_entry = {
			"price":       round(poc_price, 6),
			"side":        "resistance" if poc_price > current_price else "support",
			"label":       "POC",
			"source":      "VOLUME_PROFILE_1D",
			"touch_count": 0,
			"sh_touches":  0,
			"sl_touches":  0,
		}

	# ─── Psychological levels ──────────────────────────────────────────────
	psych_above: list[float] = []
	psych_below: list[float] = []
	if psych_increment is not None and psych_increment > 0:
		all_psych = _psychological_levels(
			current_price=current_price, increment=psych_increment,
			n_above=3, n_below=3,
		)
		psych_above = sorted([p for p in all_psych if p > current_price])
		psych_below = sorted([p for p in all_psych if p < current_price], reverse=True)

	# ─── Sort & cap HTF lists ──────────────────────────────────────────────
	major_resistances.sort(key=lambda x: x["price"])              # ascending (nearest above first)
	major_supports.sort(key=lambda x: x["price"], reverse=True)   # descending (nearest below first)
	intraday_resistances.sort(key=lambda x: x["price"])
	intraday_supports.sort(key=lambda x: x["price"], reverse=True)

	return {
		"major_resistances":    major_resistances[:6],
		"major_supports":       major_supports[:6],
		"h4_resistances": h4_resistances[:8],
		"h4_supports": h4_supports[:8],
		"d1_resistances": d1_resistances[:8],
		"d1_supports": d1_supports[:8],
		"blended_resistances": blended_resistances[:8],
		"blended_supports": blended_supports[:8],
		"sr_flips":             sr_flips[:5],
		"intraday_resistances": intraday_resistances[:5],
		"intraday_supports":    intraday_supports[:5],
		"poc_level":            poc_entry,
		"psychological_above":  psych_above[:3],
		"psychological_below":  psych_below[:3],
	}


# ═══════════════════════════════════════════════════════════════════
# SECTION 8 — TechnicalAnalyzer (public facing class)
# ═══════════════════════════════════════════════════════════════════

class TechnicalAnalyzer:
	"""
	Combines per-timeframe SMC/ICT analysis with classical indicators and
	historical major S/R clustering from 1D and 4H data.

	Example:
		analyzer = TechnicalAnalyzer(swing_order=5)
		result	 = analyzer.analyze_all(
			{"1d": df_1d, "4h": df_4h, "1h": df_1h, "15m": df_15m},
			current_price=75_500.0,
		)
		result["15m"]["indicators"]["rsi_14"]	-> 42.30
		result["4h"]["order_blocks"]			-> [...]
		result["key_levels"]["major_supports"]	-> [...]
		result["key_levels"]["sr_flips"]		-> [...]
	"""

	# Default per-TF swing orders. Lower order = faster detection (less lag) but
	# more noise on lower timeframes. Higher TFs use larger order for stable
	# pivots. 15m at order=3 cuts swing-detection lag from 75min → 45min.
	DEFAULT_SWING_ORDERS = {
		"1d":  5,
		"4h":  5,
		"1h":  4,
		"15m": 3,
		"5m":  3,
		"1m":  2,
	}

	def __init__(
		self,
		swing_order:         int = 5,
		swing_orders_per_tf: Optional[dict] = None,
	) -> None:
		# `swing_order` is the LEGACY fallback used for any TF not listed in the
		# per-TF dict (and for callers that pre-date adaptive ordering).
		self.swing_order         = swing_order
		self.swing_orders_per_tf = swing_orders_per_tf or dict(self.DEFAULT_SWING_ORDERS)

	def _get_swing_order(self, timeframe: str) -> int:
		return self.swing_orders_per_tf.get(timeframe.lower(), self.swing_order)

	# ── Single timeframe -----------------------------------------------------

	def analyze_timeframe(
		self,
		df:			   Optional[pd.DataFrame],
		timeframe:	   str,
		current_price: float,
	) -> dict:
		order = self._get_swing_order(timeframe)
		if df is None or len(df) < order * 2 + 10:
			logger.warning(
				"[PA][%s] Insufficient data (%s candles, need %d)",
				timeframe, len(df) if df is not None else 0, order * 2 + 10,
			)
			return {"timeframe": timeframe, "error": "Insufficient data"}

		df = df.copy()
		df.columns = [c.lower() for c in df.columns]

		sh_idx	   = _swing_highs(df["high"].values, order)
		sl_idx	   = _swing_lows(df["low"].values,	 order)
		swings	   = _classify_swings(df, sh_idx, sl_idx)
		indicators = _compute_indicators(df)
		_last_close = float(df["close"].iloc[-1])
		trend	   = _determine_trend(swings, _last_close, indicators.get("ema_50"), indicators.get("ema_200"))
		_dh = [(s["type"], round(s["price"], 6)) for s in swings if s["type"] in ("SH", "HH", "LH")][-6:]
		_dl = [(s["type"], round(s["price"], 6)) for s in swings if s["type"] in ("SL", "HL", "LL")][-6:]
		logger.info("[PA-DIAG-TREND] %s trend=%s | close=%.4f EMA50=%s EMA200=%s | tepeler=%s | dipler=%s", timeframe, trend, _last_close, indicators.get("ema_50"), indicators.get("ema_200"), _dh, _dl)
		events	   = _detect_structure_events(df, swings, trend)
		fvgs	   = _detect_fvgs(df)
		obs		   = _detect_order_blocks(df)
		pats	   = _detect_patterns(df)

		last_open_val  = float(df["open"].iloc[-1])
		last_high_val  = float(df["high"].iloc[-1])
		last_low_val   = float(df["low"].iloc[-1])
		last_close_val = float(df["close"].iloc[-1])

		return {
			"timeframe":             timeframe,
			"trend":                 trend,
			"last_open":             round(last_open_val,  6),
			"last_high":             round(last_high_val,  6),
			"last_low":              round(last_low_val,   6),
			"last_close":            round(last_close_val, 6),
			"last_candle_direction": "GREEN" if last_close_val > last_open_val else "RED",
			"last_timestamp":        str(df.index[-1]),
			"swing_points":          swings[-8:],
			"structure_events":      events,
			"unmitigated_fvgs":      fvgs,
			"order_blocks":          obs,
			"candlestick_patterns":  pats,
			"indicators":            indicators,
		}

	# ── Historical Major S/R --------------------------------------------------

	def _find_historical_key_levels(
		self,
		ohlcv_dict:			  dict[str, Optional[pd.DataFrame]],
		current_price:		  float,
		tolerance_pct:		  float = 0.015,
		min_touches:		  int	= 3,
		sr_flip_min_touches:  int	= 3,
		psych_increment:      Optional[float] = None,
		poc_price:            Optional[float] = None,
	) -> dict:
		return _find_historical_key_levels_from_ohlcv(
			ohlcv_dict=ohlcv_dict,
			current_price=current_price,
			swing_order=self.swing_order,
			swing_orders_per_tf=self.swing_orders_per_tf,
			tolerance_pct=tolerance_pct,
			min_touches=min_touches,
			sr_flip_min_touches=sr_flip_min_touches,
			psych_increment=psych_increment,
			poc_price=poc_price,
		)

	# ── All timeframes + Major S/R ------------------------------------------

	def analyze_all(
		self,
		ohlcv_dict:	     dict[str, Optional[pd.DataFrame]],
		current_price:   float,
		psych_increment: Optional[float] = None,
		poc_price:       Optional[float] = None,
		min_touches:     int   = 3,
	) -> dict[str, dict]:
		"""
		Returns a dict shaped like:
			{
				"1d":		  {...},
				"4h":		  {...},
				"1h":		  {...},
				"15m":		  {...},
				"key_levels": {
					"major_resistances":    [...],
					"major_supports":	    [...],
					"sr_flips":			    [...],
					"intraday_resistances": [...],
					"intraday_supports":    [...],
					"poc_level":            {...} or None,
					"psychological_above":  [...],
					"psychological_below":  [...]
				}
			}
		"""
		results: dict[str, dict] = {
			tf: self.analyze_timeframe(df, tf, current_price)
			for tf, df in ohlcv_dict.items()
		}

		results["key_levels"] = self._find_historical_key_levels(
			ohlcv_dict=ohlcv_dict,
			current_price=current_price,
			min_touches=min_touches,
			psych_increment=psych_increment,
			poc_price=poc_price,
		)

		kl = results["key_levels"]
		logger.info(
			"[PA] S/R | HTF Res=%d Sup=%d Flips=%d | Intraday Res=%d Sup=%d | "
			"POC=%s | Psych Above=%d Below=%d",
			len(kl.get("major_resistances", [])),
			len(kl.get("major_supports", [])),
			len(kl.get("sr_flips", [])),
			len(kl.get("intraday_resistances", [])),
			len(kl.get("intraday_supports", [])),
			(kl.get("poc_level") or {}).get("price", "-"),
			len(kl.get("psychological_above", [])),
			len(kl.get("psychological_below", [])),
		)

		return results