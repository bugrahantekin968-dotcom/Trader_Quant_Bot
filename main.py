"""
main.py — FundedNext Swing Trade Bot
=====================================

Architecture
------------
  · MT5 (MetaTrader 5) for both market data (OHLCV) and order execution.
	CCXT and L2 order-book logic have been removed entirely.
  · CoinGlass v4 API for whale sentiment: Open Interest, Funding Rate,
	Long/Short Ratio, and aggregated liquidation history.
  · Historical Major S/R clustering from 1D + 4H wick data (modules.price_action).
  · Dual-sided, granular AI scoring matrix (LONG and SHORT computed in parallel).
  · Two-stage trade management engine:
		Stage 1 (50% progress) — partial TP + breakeven SL.
		Stage 2 (75% progress) — AI consultation; CLOSE_ALL or HOLD_REST.

Public-facing strings (logs, prompts, dict keys, AI outputs) are in English.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import re
import sys
import time

# Windows konsol kodlama kalkani (cp1254 vb.): log/print icindeki Unicode (->, >=,
# matematik sembolleri, emoji) kodlanamayinca CANLI BOTU COKERTMESIN. stdout/stderr'i
# UTF-8'e sabitle (errors='replace' -> asla patlamaz). Zaten UTF-8 ise no-op.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Optional
from anthropic import AsyncAnthropic
try:
    from backtest.scorer import (
        score as _python_score,
        exhaustion_score,
        _trend_conflict_block,   # G1.3/1.4 — deterministic counter-trend brake (shared with scorer)
        _premium_discount_block, # G6.2 — premium/discount konum vetosu (shared)
        _ltf_trigger_ok,         # G6.3 — 15m giris tetigi (shared)
        _over_extended,          # ENTRY-B — over-extension (post-LLM backstop icin)
        _dealing_range,          # G6.2 — LLM brief'ine dealing-range satiri icin
    )
except Exception:
    _python_score = None  # FAIL-SAFE: yoksa/patlarsa eski LLM yoluna duser
    exhaustion_score = None
    _trend_conflict_block = None
    _premium_discount_block = None
    _ltf_trigger_ok = None
    _over_extended = None
    _dealing_range = None

import aiohttp
import pandas as pd
try:
	import ccxt
except ImportError:  # live needs it; sandbox/backtest may not
	ccxt = None

from modules.executor import ExecutionResult, ExecutionStatus, TradeExecutor
from modules.price_action import TechnicalAnalyzer

try:
	import MetaTrader5 as mt5
	_MT5_AVAILABLE = True
except ImportError:
	_MT5_AVAILABLE = False

_MAGIC = 202605


# ─────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────

logging.basicConfig(
	level=logging.INFO,
	format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
	datefmt="%Y-%m-%d %H:%M:%S",
)
logger		 = logging.getLogger("algo_bot")
trade_logger = logging.getLogger("trade")


# ─────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────

@dataclass
class BotConfig:
	# ── Credentials
	mt5_login:		   int = 7943052
	mt5_password:	   str = "uLgT5#8B"
	mt5_server:		   str = "Eightcap-Demo"
	anthropic_api_key: str = ""
	# 2026-08-01 (maliyet minimizasyonu): claude-sonnet-4-6 -> claude-haiku-4-5.
	# Giris karari artik LLM'siz (llm_entry_enabled=False, scorer-only); kalan
	# cagrilar basit yargi isleri (exhaustion hakemi, %75 danisma, post-mortem)
	# — Haiku 4.5 ($1/$5 per MTok) bunlar icin yeterli. llm_entry_enabled tekrar
	# True yapilirsa girisin rubrik hesabi icin sonnet'e donmek mantikli olur.
	llm_model:		   str = "claude-haiku-4-5"
	# G7 — SCORER-ONLY GIRIS (2026-08-01): False -> giris karari %100 deterministik
	# scorer'dan gelir (backtest ile BIREBIR ayni motor; +54R/+%82.5 sim bu motorla
	# olculdu). LLM giris cagrisi tamamen atlanir (~$11/ay tasarruf). True -> eski
	# davranis: scorer on-filtre + LLM ikinci gorus. Scorer patlarsa her durumda
	# LLM yoluna dusulur (fail-safe).
	llm_entry_enabled: bool = False
	coinglass_api_key: str = "29a104fc4b2e44c09d6337c5e0c591b9"

	# ── Hybrid volume: use CoinGlass real USD spot volume on HTF (1h/4h/1d),
	# keep MT5 tick volume on 15m. Set use_coinglass_volume=False to disable
	# entirely (pure MT5 tick volume everywhere — the old behaviour).
	use_coinglass_volume: bool = False  # Binance OHLCV has native full-depth volume; CoinGlass 50-bar merge skewed POC
	coinglass_volume_bars: int = 50   # bars fetched per HTF for volume override

	# ── Symbols (CoinGlass naming ↔ MT5 broker naming — order must match)
	data_symbols: list[str] = field(
		default_factory=lambda: ["BTC/USDT", "ETH/USDT"]
	)
	mt5_symbols: list[str] = field(
		default_factory=lambda: ["BTCUSD", "ETHUSD"]
	)

	# ── Timeframes and historical depths
	# 2026-08-01: BACKTEST ILE BIREBIR ESITLENDI (kullanici karari) — dogrulanmis
	# davranis backtest'inki oldugu icin canli ona uyduruldu (tersi degil).
	# DataFeed._DEPTHS = {1d:200, 4h:600, 1h:400} ile ayni. Eski derin degerler
	# (1d:1000, 4h:2190) S/R dokunus sayilarini backtest'ten farkli sisiriyordu.
	timeframes:		list[str] = field(default_factory=lambda: ["1d", "4h", "1h", "15m"])
	candles_per_tf: dict	  = field(default_factory=lambda: {
		"1d":  200,		  # ~200 gun (backtest slice_at ile ayni pencere)
		"4h":  600,		  # ~100 gun
		"1h":  400,
		"15m": 300,
	})

	# ── Cadence
	ai_interval_seconds:	 int = 900	   # Main analysis at most every 15 min per symbol
	loop_interval_sec:		 int = 60	   # Outer scheduler tick
	history_check_interval:	 int = 300	   # Closed-position scan every 5 min (G3.5: stop→guard state fresh before next 15-min entry)
	trade_mgmt_interval_sec: int = 60	   # Active trade management every 60 s

	# ── Trading hours (UTC). Outside this window, _run_symbol exits early
	# (no LLM call, no new trade). Trade-management loop keeps running 24/7
	# to manage already-open positions.
	# 0 ve 24 = saat kisiti YOK (7/24 analiz). Eski deger end=23 idi ve her gun
	# UTC 23:00-24:00 arasi analiz sessizce atlaniyordu (K8 duzeltmesi).
	trading_hours_start_utc: int = 0
	trading_hours_end_utc:   int = 24

	# ── FundedNext Sizing & SL Rules (FIXED USD MODEL — no percentage risk)
	leverage:				   int	 = 1

	# Fixed position sizes by score tier (USD notional exposure)
	pos_usd_75_80:			   float = 10_000.0	  # Score [58, 80] → $10,000 exposure (lower-confluence band)
	pos_usd_81_100:			   float = 20_000.0	  # Score [81, 100] → $20,000 exposure

	# Hard caps (2026-07-31: risk %1.3'e cikarilinca buyutuldu — eski $30k tavani
	# $1300 riski tasiyamiyordu: SL %1.5'te gereken nominal $87k. Eightcap kripto
	# CFD marjini ~1:5 -> $100k nominal ≈ $20k marjin; iki sembol $150k ≈ $30k marjin.)
	max_position_usd:		   float = 100_000.0  # Per-symbol position ceiling (1x hesap)
	max_total_exposure_usd:	   float = 150_000.0  # Aggregate portfolio ceiling

	# Risk-based sizing (replaces score tiers): notional = risk_$ / SL-distance
	# 2026-07-31: 0.005 -> 0.013 (kullanici karari; d6 verisiyle beklenen: ort yil
	# ~%22, en iyi yil ~%40, max DD ~%8.5, en kotu gun ~-2.8%. Rolling-DD %6 freni
	# 5 ardisik kayipta (~-6.5%) yeni girisi durdurup %10 prop limitini korur.)
	risk_pct_per_trade:		   float = 0.013    # risk 1.3% of balance per trade
	max_risk_usd:			   float = 2_600.0  # hard cap: 2x baslangic riski (eski oranla ayni)

	# SL distance band (anti-stop-hunt floor + max-loss ceiling)
	# 2026-07-31: 0.025 denendi ve GERI ALINDI — scaleout modunda A/B: tavan 0.025
	# toplami dusurdu (+53.0R vs +54.4R) ve BTC'yi -6.4R bozdu (iki-sembol kurali
	# gecemedi). Dar tavan kotu islemi atlamiyor, ayni isleme dar stop koyuyor.
	min_sl_pct:				   float = 0.015	  # SL must sit at least 1.5% from entry
	max_sl_pct:				   float = 0.04		  # SL must sit at most 4.0% from entry

	min_rr_ratio:			   float = 2.0
	max_rr_ratio:			   float = 3.5		  # RR ust siniri FALLBACK (gecerli olan: per-symbol TP_CAP_RR haritasi)
	# -- Trend-exhaustion early-exit (hybrid: deterministic score -> LLM judge)
	exhaustion_enabled:			   bool  = True
	exhaustion_threshold:		   float = 55.0	  # act only above strong confluence
	exhaustion_check_interval_sec: int   = 300	  # recompute per position at most every 5 min
	exhaustion_consult_cooldown_sec: int = 1800	  # after an LLM HOLD, wait before re-asking
	max_daily_drawdown_pct:	   float = 0.02
	min_lot:				   float = 0.01
	lot_step:				   float = 0.01
	max_lot:				   float = 100_000.0
	min_score:				   int	 = 54   # 58->54 (2026-06-21): whale'li (live-esdeger) holdout, eşik 52-55'in 58'i HER IKI zaman-yarisinda ezdigini gosterdi (R ~%75 artti, beklenti de daha iyi). 58 D-destekli iyi islemleri kesiyordu.

	# ── Per-symbol contract size fallback (MT5 trade_contract_size takes priority)
	symbol_contract_sizes: dict = field(default_factory=lambda: {
		"BTCUSD": 0.05,
		"ETHUSD": 0.05,
		"XRPUSD": 5.0,
	})

	# ── Per-symbol psychological round-number increments (PR-2)
	#    Used to inject magnet levels into the key_levels payload.
	psychological_increments: dict = field(default_factory=lambda: {
		"BTCUSD": 5000.0,    # $70k, $75k, $80k, ...
		"ETHUSD": 200.0,     # $2800, $3000, $3200, ...
		"XRPUSD": 0.10,      # $1.20, $1.30, $1.40, ...
	})

	dry_run: bool = False

	# ── Major S/R clustering parameters
	sr_tolerance_pct:	 float = 0.015
	sr_min_touches:		 int   = 3   # PR-2: raised 2 → 3 (quality filter)
	sr_flip_min_touches: int   = 3

	# ── G3: Post-stop koruma katmanı (stop sonrası tekrar-giriş kalkanı)
	post_stop_cooldown_sec:  int   = 6 * 3600   # bir STOP sonrası AYNI YÖN'de yeni giriş yasağı (sembol+yön)
	relevel_lockout_pct:     float = 0.012      # (artık tam-yasak cooldown'un alt kümesi; stopped_levels kaydı log/uyumluluk için duruyor)
	consecutive_stop_limit:  int   = 3          # aynı sembol+yön'de N ardışık stop -> o yönü askıya al
	side_halt_sec:           int   = 12 * 3600  # ardışık-stop askısının süresi (TP gelince erken kalkar)
	rolling_dd_days:         int   = 7          # kayan pencere (gün) — gece-yarısı sıfırlamadan bağımsız hesap koruması
	rolling_dd_pct:          float = 0.06       # kayan pencerede tepe-bakiyeden %6 düşüş -> yeni giriş durur
	guard_state_path:        str   = "guard_state.json"   # cooldown/streak diski (restart kurtarma)
	# ── %50 ilerlemede SL'i breakeven'a çekme.
	# 2026-08-01: once KAPATILDI (e4_be50 toplam R: +37.1 vs +54.4), sonra AYNI GUN
	# GERI ALINDI. Sebep: o karsilastirma HATALIYDI. Motor iki geri-besleme dongusu
	# tasiyor — (1) guard, BE ile kapanani "stop" saymadigi icin cooldown tetiklemiyor,
	# (2) engine.py taramayi `t = exit_index + 1`'e atliyor, yani cikis zamani sonraki
	# TUM islemleri kaydiriyor. Bu yuzden iki kosunun islem KUMESI farkli olur
	# (210 ortak, 50 sadece d6'da, 75 sadece e4'te) ve toplam R farki cogunlukla
	# "hangi islemler acildi" sansidir. ESLESTIRILMIS test (ayni sinyal+sembol+yon):
	# BE +0.018R/islem DAHA IYI (t=+1.02, anlamsiz ama kesinlikle zararli DEGIL).
	# Toplam R'in standart hatasi ±20R iken 17R'lik farka karar baglanamaz.
	# KURAL: cikis katmani kararlari ESLESTIRILMIS test + holdout ister; toplam R yeter degil.
	breakeven_at_half:       bool  = True

	# ── G2.4: short/long giriş tükenme-vetosu eşiği (alt-TF dönüş).
	# 2026-08-01: KAPATILDI (0 = devre dışı) — A/B backtest (e3_exhveto): veto ile
	# +39.4R vs vetosuz +54.4R (-15R, DD ve kayıp serisi de DAHA KÖTÜ). Kova analizi:
	# girişte exh>=18 olan işlemler EN İYİ beklentiye sahipti (+0.298R, iki sembolde
	# pozitif) — "tükenmiş görünen karşı yön" çoğu kez iyi bir pullback girişiydi.
	# Haziran stop-loop patolojisini artık diğer katmanlar (G1 fren, ADX, 1D kapısı,
	# tam-yasak cooldown) çözüyor. >0 yapılırsa veto tekrar devreye girer.
	entry_exhaustion_veto:   float = 0.0

	# ── G-OPT: ADX rejim filtresi (chop'ta işlem ALMA). Backtest matrisinin KAZANANI
	#    (beklenti +0.198 -> +0.241, ETH +12R -> +18R): trendsiz/yatay rejimde counter-trend
	#    stop'ların olduğu düşük-kaliteli işlemleri eler. 4H ADX < eşik -> yeni giriş yok.
	regime_filter_enabled:   bool  = True
	regime_adx_min:          float = 20.0
	regime_adx_tf:           str   = "4h"

	# ── G6.6: Yuksek etkili haber karantinasi (FOMC / CPI / NFP)
	# news_blackout.json'daki gunler/pencereler boyunca YENI giris acilmaz
	# (analiz+LLM tamamen atlanir; acik pozisyonlar 7/24 yonetilmeye devam).
	# Dosya STATIK: tarihler resmi takvimlerden onceden yazilir, bot hicbir
	# API'ye/AI'a sormaz. Kaynaklar (yeni yil geldiginde guncelle):
	#   FOMC: federalreserve.gov/monetarypolicy/fomccalendars.htm
	#   CPI : bls.gov/schedule/news_release/cpi.htm
	#   NFP : bls.gov/schedule/news_release/empsit.htm
	news_blackout_enabled: bool = True
	news_blackout_path:    str  = "news_blackout.json"
	# Pencere baslamadan bu kadar ONCE de yeni emir yok: 4h omurlu limit emri
	# haberin icine dolmasin (tam-gun bloklarda onceki gun 20:00 UTC'den itibaren).
	news_blackout_lead_sec: int = 4 * 3600

	# ── Post-trade feedback files
	journal_path: str = "trade_journal.json"
	trade_state_path: str = "trade_state.json"   # scale-out state diski (restart kurtarma)
	lessons_path: str = "lessons_learned.json"


# ─────────────────────────────────────────────────────────────────
# Risk Manager
# ─────────────────────────────────────────────────────────────────

class RiskManager:
	"""
	FundedNext-aware risk and position-sizing engine.

	  * Lot sizing uses MT5 trade_contract_size (with config fallback).
	  * Hard cap: SL distance <= max_sl_pct.
	  * Hard cap: Risk/Reward >= min_rr_ratio.
	  * Lot rounded DOWN to lot_step (never up — risk-conservative).
	  * Daily drawdown reset (start_balance refreshed at UTC day rollover).
	"""

	def __init__(self, cfg: "BotConfig") -> None:
		self.cfg			  = cfg
		self.max_dd_pct		  = cfg.max_daily_drawdown_pct
		self.max_position_usd = cfg.max_position_usd
		self._start_balance	  = 0.0
		self._current_balance = 0.0
		self._kill_switch	  = False
		self._last_reset_date = None
		# G3.4 — rolling-window balance history for a multi-day drawdown breaker that
		# is INDEPENDENT of the daily kill-switch (which self-resets every UTC midnight
		# and so can never see a slow week-long bleed). In-memory; rebuilds as the bot
		# runs. (price, ts) samples, pruned to the rolling window on each update.
		self._bal_hist: list[tuple[float, float]] = []   # (timestamp, balance)

	def initialize(self, balance: float) -> None:
		self._start_balance	  = balance
		self._current_balance = balance
		self._kill_switch	  = False
		self._last_reset_date = datetime.now(timezone.utc).date()
		logger.info(
			"[RISK] Initialized | Balance: $%.2f | Max Single Pos: $%.0f | "
			"Max Total Exposure: $%.0f | Max SL: %.1f%% | Min RR: %.1f | Max DD: %.1f%%",
			balance,
			self.max_position_usd,
			self.cfg.max_total_exposure_usd,
			self.cfg.max_sl_pct * 100,
			self.cfg.min_rr_ratio,
			self.max_dd_pct * 100,
		)

	def update_balance(self, new_balance: float) -> None:
		today = datetime.now(timezone.utc).date()
		if self._last_reset_date and today != self._last_reset_date:
			logger.info(
				"[RISK] New UTC day (%s) | Previous start: $%.2f → New start: $%.2f",
				today, self._start_balance, new_balance,
			)
			self._start_balance	  = new_balance
			self._kill_switch	  = False
			self._last_reset_date = today

		if self._start_balance <= 0:
			self._start_balance = new_balance

		self._current_balance = new_balance
		drawdown = (self._start_balance - new_balance) / max(self._start_balance, 1)
		if drawdown >= self.max_dd_pct and not self._kill_switch:
			self._kill_switch = True
			logger.critical(
				"[RISK] KILL-SWITCH ENGAGED | Daily DD: %.2f%% >= %.2f%% | "
				"Balance: $%.2f | Start: $%.2f",
				drawdown * 100, self.max_dd_pct * 100,
				new_balance, self._start_balance,
			)

		# G3.4 — record sample + prune to the rolling window
		if new_balance > 0:
			now = time.time()
			self._bal_hist.append((now, new_balance))
			horizon = now - self.cfg.rolling_dd_days * 86400
			self._bal_hist = [(t, b) for (t, b) in self._bal_hist if t >= horizon]

	def rolling_drawdown_pct(self) -> float:
		"""G3.4 — drawdown from the PEAK balance within the rolling window. Survives
		the UTC-midnight reset, so a multi-day losing streak is finally visible."""
		if not self._bal_hist:
			return 0.0
		peak = max(b for _, b in self._bal_hist)
		if peak <= 0:
			return 0.0
		return max(0.0, (peak - self._current_balance) / peak)

	def rolling_dd_breached(self) -> bool:
		"""True when the rolling-window drawdown breaches the configured ceiling —
		new entries should be suppressed (open positions stay managed)."""
		return self.rolling_drawdown_pct() >= self.cfg.rolling_dd_pct

	@property
	def is_active(self) -> bool:
		return not self._kill_switch

	@property
	def balance(self) -> float:
		return self._current_balance

	@property
	def current_drawdown_pct(self) -> float:
		if self._start_balance <= 0:
			return 0.0
		return max(0.0, (self._start_balance - self._current_balance) / self._start_balance)

	def _get_contract_size(self, mt5_sym: str) -> float:
		if _MT5_AVAILABLE:
			try:
				info = mt5.symbol_info(mt5_sym)
				if info and info.trade_contract_size > 0:
					return float(info.trade_contract_size)
			except Exception:
				pass
		return float(self.cfg.symbol_contract_sizes.get(mt5_sym, 1.0))

	def _floor_to_step(self, lot: float) -> float:
		step = self.cfg.lot_step
		return math.floor(lot / step) * step

	def compute_trade_plan(
		self,
		mt5_sym:			  str,
		entry_price:		  float,
		sl_price:			  float,
		tp_price:			  Optional[float],
		target_position_usd:  float,
		current_exposure_usd: float = 0.0,
	) -> dict:
		"""
		Computes the final trading plan from a FIXED USD exposure target.
		Lot is derived directly: lot = target_position_usd / lot_value_usd.
		No percentage-of-account risk model.

		Hard gates (return ok=False with reason):
			* SL distance below cfg.min_sl_pct (1.5% stop-hunt floor)
			* SL distance above cfg.max_sl_pct (4.0% max loss ceiling)
			* RR below cfg.min_rr_ratio
			* Portfolio aggregate cap breached
		"""
		out = {
			"ok": False, "reason": "", "lot": 0.0,
			"sl_pct": 0.0, "rr": 0.0, "lot_value_usd": 0.0,
			"target_position_usd": target_position_usd,
			"actual_position_usd": 0.0,
			"limiting_factor": "-",
		}

		# 0. Sanity
		if entry_price <= 0 or sl_price <= 0:
			out["reason"] = "Entry or SL is non-positive"
			return out
		if target_position_usd <= 0:
			out["reason"] = f"target_position_usd must be > 0 (got {target_position_usd})"
			return out

		sl_distance = abs(entry_price - sl_price)
		if sl_distance < 1e-9:
			out["reason"] = "SL distance near zero"
			return out

		sl_pct = sl_distance / entry_price
		out["sl_pct"] = sl_pct

		# 1. SL band gates
		if sl_pct < self.cfg.min_sl_pct - 1e-4:	# 1e-4 = 2dp fiyat yuvarlama toleransi
			out["reason"] = (
				f"SL distance {sl_pct*100:.2f}% below minimum {self.cfg.min_sl_pct*100:.2f}% "
				f"(stop-hunt protection)"
			)
			return out
		if sl_pct > self.cfg.max_sl_pct:
			out["reason"] = (
				f"SL distance {sl_pct*100:.2f}% above maximum {self.cfg.max_sl_pct*100:.2f}%"
			)
			return out

		# 2. RR gate
		if tp_price and tp_price > 0:
			tp_distance = abs(tp_price - entry_price)
			rr = tp_distance / sl_distance
			out["rr"] = rr
			if rr < self.cfg.min_rr_ratio:
				out["reason"] = f"R/R {rr:.2f} below minimum {self.cfg.min_rr_ratio}"
				return out
		else:
			out["reason"] = "TP missing; cannot validate R/R"
			return out

		# 3. Contract & lot value
		contract_size = self._get_contract_size(mt5_sym)
		if contract_size <= 0:
			out["reason"] = f"Invalid contract size ({contract_size})"
			return out

		lot_value_usd = contract_size * entry_price
		out["lot_value_usd"] = lot_value_usd

		# 4. Fixed-USD lot derivation
		raw_lot = target_position_usd / lot_value_usd

		# 5. Caps
		max_pos_lot = self.max_position_usd / lot_value_usd

		remaining_exposure = self.cfg.max_total_exposure_usd - current_exposure_usd
		if remaining_exposure <= 0:
			out["reason"] = (
				f"Portfolio exposure full: ${current_exposure_usd:,.0f} / "
				f"${self.cfg.max_total_exposure_usd:,.0f}"
			)
			return out
		portfolio_lot = remaining_exposure / lot_value_usd

		candidates = {
			"target":	 raw_lot,
			"single":	 max_pos_lot,
			"portfolio": portfolio_lot,
			"max_lot":	 self.cfg.max_lot,
		}
		chosen_lot = min(candidates.values())
		limiting   = min(candidates, key=candidates.get)

		floored = self._floor_to_step(chosen_lot)

		# 6. Min-lot floor — only allow if it stays within max_position_usd
		if floored < self.cfg.min_lot:
			min_lot_value = self.cfg.min_lot * lot_value_usd
			if min_lot_value > self.max_position_usd:
				out["reason"] = (
					f"Even min lot {self.cfg.min_lot} (${min_lot_value:,.0f}) exceeds "
					f"max position cap ${self.max_position_usd:,.0f}"
				)
				return out
			floored	 = self.cfg.min_lot
			limiting = "min_lot_floor"

		actual_usd = floored * lot_value_usd
		out.update({
			"ok":				   True,
			"reason":			   "OK",
			"lot":				   round(floored, 3),
			"actual_position_usd": round(actual_usd, 2),
			"limiting_factor":	   limiting,
		})

		logger.info(
			"[RISK][%s] Plan OK | entry=%.4f sl=%.4f tp=%.4f | SL%%=%.2f RR=%.2f | "
			"target=$%.0f actual=$%.0f | lot=%.3f (target=%.3f single=%.3f port=%.3f) | "
			"limit=%s | lotUSD=$%.0f",
			mt5_sym, entry_price, sl_price, tp_price,
			sl_pct * 100, out["rr"],
			target_position_usd, actual_usd, floored,
			raw_lot, max_pos_lot, portfolio_lot, limiting, lot_value_usd,
		)
		return out


# ─────────────────────────────────────────────────────────────────
# MT5 Helpers — Balance, exposure, OHLCV
# ─────────────────────────────────────────────────────────────────

# Mapping from string timeframe to MT5 constant
_TF_MAP: dict[str, int] = {}

def _init_tf_map() -> None:
	global _TF_MAP
	if not _MT5_AVAILABLE or _TF_MAP:
		return
	_TF_MAP = {
		"1d":  mt5.TIMEFRAME_D1,
		"4h":  mt5.TIMEFRAME_H4,
		"1h":  mt5.TIMEFRAME_H1,
		"30m": mt5.TIMEFRAME_M30,
		"15m": mt5.TIMEFRAME_M15,
		"5m":  mt5.TIMEFRAME_M5,
		"1m":  mt5.TIMEFRAME_M1,
	}


def _normalize_volume(mt5_sym: str, volume: float) -> float:
	if not _MT5_AVAILABLE:
		return round(volume, 2)
	info = mt5.symbol_info(mt5_sym)
	if info is None:
		return round(volume, 2)
	step	   = info.volume_step
	min_vol	   = info.volume_min
	max_vol	   = info.volume_max
	normalized = round(round(volume / step) * step, 2)
	return max(min_vol, min(normalized, max_vol))


async def _fetch_mt5_balance(dry_run: bool = False) -> float:
	if dry_run or not _MT5_AVAILABLE:
		return 100_000.0
	try:
		info = await asyncio.to_thread(mt5.account_info)
		if info is not None:
			balance = float(info.balance)
			if balance == 0:
				logger.warning("MT5 account_info returned balance=0.")
			return balance
		logger.warning("MT5 account_info returned None.")
		return 100_000.0
	except Exception as exc:
		logger.error("MT5 balance fetch error: %s", exc)
		return 100_000.0


async def _fetch_total_exposure_usd(cfg: "BotConfig", dry_run: bool = False) -> float:
	"""Sum of contract_size * volume * price across all open positions."""
	if dry_run or not _MT5_AVAILABLE:
		return 0.0
	try:
		positions = await asyncio.to_thread(mt5.positions_get)
		if not positions:
			return 0.0
		total = 0.0
		for pos in positions:
			info = await asyncio.to_thread(mt5.symbol_info, pos.symbol)
			if info is None:
				cs = cfg.symbol_contract_sizes.get(pos.symbol, 1.0)
			else:
				cs = info.trade_contract_size
			total += pos.volume * cs * pos.price_current
		return total
	except Exception as exc:
		logger.error("[EXPOSURE] Calculation error: %s", exc)
		return 0.0


async def _fetch_mt5_ohlcv(
	mt5_sym:   str,
	timeframe: str,
	count:	   int,
) -> Optional[pd.DataFrame]:
	"""
	Pulls OHLCV from MT5 for the given symbol and timeframe.
	Returns a UTC-indexed DataFrame with columns [open, high, low, close, volume].
	"""
	if not _MT5_AVAILABLE:
		return None

	_init_tf_map()
	mt5_tf = _TF_MAP.get(timeframe.lower())
	if mt5_tf is None:
		logger.error("[OHLCV] Unsupported timeframe: %s", timeframe)
		return None

	try:
		# CRITICAL: start_pos=1 skips the currently FORMING (incomplete) bar.
		# Forming bar would otherwise give partial volume/close/range and poison
		# every downstream indicator (RSI, MACD state, volume_backed, retest, BOS).
		# We use only CLOSED bars for analysis; live price comes via _get_market_price().
		rates = await asyncio.to_thread(mt5.copy_rates_from_pos, mt5_sym, mt5_tf, 1, count)
		if rates is None or len(rates) == 0:
			err = await asyncio.to_thread(mt5.last_error)
			logger.warning("[OHLCV] No data for %s %s (count=%d) | mt5_err=%s",
						   mt5_sym, timeframe, count, err)
			return None

		df = pd.DataFrame(rates)
		df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
		df = df.set_index("time")

		# MT5 returns tick_volume; some brokers also have real_volume.
		if "tick_volume" in df.columns:
			df = df.rename(columns={"tick_volume": "volume"})
		elif "real_volume" in df.columns:
			df = df.rename(columns={"real_volume": "volume"})

		return df[["open", "high", "low", "close", "volume"]].astype("float64")
	except Exception as exc:
		logger.error("[OHLCV][%s/%s] Fetch error: %s", mt5_sym, timeframe, exc)
		return None


# ── Binance (ccxt) OHLCV for ANALYSIS — deep history, matches the backtest
#    source. EXECUTION, live price, balance & exposure stay on MT5/Eightcap.
_BINANCE_EX = None


def _get_binance_exchange():
	"""Lazily build one public (keyless) ccxt Binance client (rate-limited)."""
	global _BINANCE_EX
	if _BINANCE_EX is None:
		if ccxt is None:
			logger.error("[OHLCV] ccxt not installed — cannot fetch Binance OHLCV (pip install ccxt).")
			return None
		_BINANCE_EX = ccxt.binance({"enableRateLimit": True})
	return _BINANCE_EX


def _binance_paginate(ex, symbol, timeframe, since_ms, batch_limit=1000, max_retries=5):
	"""Walk fetch_ohlcv forward from since_ms to now (SYNC — call via to_thread)."""
	if not ex.markets:
		ex.load_markets()
	tf_ms = ex.parse_timeframe(timeframe) * 1000
	now_ms = ex.milliseconds()
	rows = []
	cursor = since_ms
	while cursor < now_ms:
		batch = None
		for attempt in range(max_retries):
			try:
				batch = ex.fetch_ohlcv(symbol, timeframe, since=cursor, limit=batch_limit)
				break
			except Exception:
				time.sleep(2 ** attempt)
		if not batch:
			break
		rows.extend(batch)
		nxt = batch[-1][0] + tf_ms
		if nxt <= cursor:
			break
		cursor = nxt
		if len(batch) < batch_limit:
			break
	return rows


async def _fetch_binance_ohlcv(data_sym, timeframe, count):
	"""
	Pull `count` CLOSED OHLCV bars from Binance (ccxt) for analysis. Returns a
	UTC-indexed DataFrame [open, high, low, close, volume] — identical shape to
	the MT5 fetcher. The forming (incomplete) last bar is dropped so indicators
	see only closed bars (matches the old MT5 start_pos=1 behaviour).
	"""
	ex = _get_binance_exchange()
	if ex is None:
		return None
	try:
		tf_ms = ex.parse_timeframe(timeframe) * 1000
		since_ms = ex.milliseconds() - (count + 5) * tf_ms
		rows = await asyncio.to_thread(_binance_paginate, ex, data_sym, timeframe, since_ms)
		if not rows:
			logger.warning("[OHLCV-BINANCE] No data for %s %s (count=%d).", data_sym, timeframe, count)
			return None
		now_ms = ex.milliseconds()
		rows = [r for r in rows if r[0] + tf_ms <= now_ms]   # drop the forming bar
		if not rows:
			return None
		rows = rows[-count:]
		df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
		df = df.drop_duplicates(subset="ts").sort_values("ts")
		df["time"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
		df = df.set_index("time")
		return df[["open", "high", "low", "close", "volume"]].astype("float64")
	except Exception as exc:
		logger.error("[OHLCV-BINANCE][%s/%s] Fetch error: %s", data_sym, timeframe, exc)
		return None


async def fetch_all_ohlcv(
	data_sym:		str,
	timeframes:		list[str],
	candles_per_tf: dict,
) -> dict[str, Optional[pd.DataFrame]]:
	"""
	Fetch OHLCV for every timeframe from Binance (ccxt), keyed by lowercase
	timeframe. SEQUENTIAL: one shared ccxt client is not thread-safe for
	concurrent calls, and a few REST calls per 15-min cycle is cheap.
	Execution / live price / balance still come from MT5/Eightcap.
	"""
	out: dict[str, Optional[pd.DataFrame]] = {}
	for tf in timeframes:
		out[tf] = await _fetch_binance_ohlcv(data_sym, tf, candles_per_tf.get(tf, 200))
	return out


def _merge_real_volume(
	df:          pd.DataFrame,
	cg_volumes:  list[dict],
) -> tuple[pd.DataFrame, int]:
	"""
	Replace a DataFrame's tick-volume column with CoinGlass real USD volume.

	Strategy: timestamp-based alignment first (both sides normalised to UTC ms).
	If fewer than 30% of bars match by timestamp (e.g. broker server-time offset
	differs from CoinGlass UTC), fall back to POSITIONAL alignment of the most
	recent N bars — both series are chronologically sorted at the same interval,
	so the trailing bars line up regardless of timezone.

	Returns (possibly-new df, matched_bar_count). On total failure returns the
	original df unchanged with matched=0 (caller keeps tick volume — safe).
	"""
	if df is None or df.empty or not cg_volumes:
		return df, 0
	if "volume" not in df.columns:
		return df, 0

	# Build CoinGlass timestamp(UTC ms) → volume map.
	cg_by_ms: dict[int, float] = {}
	for item in cg_volumes:
		ts_ms = item.get("time")
		vol   = item.get("volume_usd")
		if ts_ms is not None and vol is not None:
			cg_by_ms[int(ts_ms)] = float(vol)
	if not cg_by_ms:
		return df, 0

	new_vol = df["volume"].astype(float).copy()

	# ---- Attempt 1: timestamp match (UTC) ----
	matched = 0
	for idx in df.index:
		try:
			ts = pd.Timestamp(idx)
			if ts.tz is None:
				ts = ts.tz_localize("UTC")
			else:
				ts = ts.tz_convert("UTC")
			ms = int(ts.timestamp() * 1000)
		except Exception:
			continue
		if ms in cg_by_ms:
			new_vol.loc[idx] = cg_by_ms[ms]
			matched += 1

	match_ratio = matched / len(df) if len(df) else 0.0

	# ---- Attempt 2: positional fallback (timezone-agnostic) ----
	# Threshold is deliberately HIGH (0.80): a broker server-time offset can produce
	# a PARTIAL timestamp match (e.g. a +3h offset still aligns a few bars), and a
	# partial match would otherwise write CoinGlass volume onto the WRONG bars. Only
	# trust timestamp alignment when the vast majority of bars line up; otherwise
	# fall back to positional (trailing-bar) alignment which is timezone-agnostic.
	if match_ratio < 0.80:
		cg_clean = sorted(cg_volumes, key=lambda x: x["time"])
		# FLAW #3 fix: df already excludes the forming bar (start_pos=1), but CoinGlass
		# usually INCLUDES the still-forming (partial-volume) bar. Trailing-align without
		# dropping it shifts the whole series by one → every closed bar gets the NEXT
		# bar's volume (future leakage) and the live bar's tiny volume poisons
		# volume_backed. Detect a forming last bar (its timestamp is younger than one
		# full bar interval) and drop it before aligning.
		if len(cg_clean) >= 3:
			last_t = int(cg_clean[-1]["time"])
			prev_t = int(cg_clean[-2]["time"])
			bar_ms = last_t - prev_t
			now_ms = int(time.time() * 1000)
			if bar_ms > 0 and (now_ms - last_t) < bar_ms:
				cg_clean = cg_clean[:-1]   # drop the forming bar
		cg_sorted = [v["volume_usd"] for v in cg_clean]
		n = min(len(df), len(cg_sorted))
		if n == 0:
			return df, 0
		# Align trailing bars: df last n ↔ cg last n (both now closed-bars only).
		df_tail_idx = df.index[-n:]
		cg_tail     = cg_sorted[-n:]
		new_vol = df["volume"].astype(float).copy()
		matched = 0
		for k, idx in enumerate(df_tail_idx):
			new_vol.loc[idx] = float(cg_tail[k])
			matched += 1

	out = df.copy()
	out["volume"] = new_vol
	return out, matched


async def _get_market_price(mt5_sym: str, fallback_df: Optional[pd.DataFrame] = None) -> float:
	"""
	Returns the LIVE market mid-price from the current tick.
	Falls back to the last closed bar's close if MT5 / tick unavailable
	(e.g. dry_run or transient connectivity issue).

	Required because _fetch_mt5_ohlcv now skips the forming bar, so the
	DataFrame's last close is the PREVIOUS closed bar (potentially several
	minutes / hours stale). Entry decisions, SL/TP distance percentages, and
	limit-order pricing must use the live price.
	"""
	if _MT5_AVAILABLE:
		try:
			tick = await asyncio.to_thread(mt5.symbol_info_tick, mt5_sym)
			if tick is not None and tick.bid > 0 and tick.ask > 0:
				return float((tick.bid + tick.ask) / 2.0)
		except Exception as exc:
			logger.warning("[PRICE] Tick fetch error for %s: %s — falling back to last close.",
						   mt5_sym, exc)
	if fallback_df is not None and not fallback_df.empty:
		return float(fallback_df["close"].iloc[-1])
	return 0.0


# ─────────────────────────────────────────────────────────────────
# Point of Control helper (used as supporting context for AI)
# ─────────────────────────────────────────────────────────────────

def _calc_adx(df: Optional[pd.DataFrame], period: int = 14) -> Optional[float]:
	"""Wilder ADX (trend strength) — chop detection for the regime filter (G-OPT).
	Port of backtest engine._adx so live and backtest use the IDENTICAL formula.
	Returns None if insufficient bars."""
	if df is None or len(df) < period * 2 + 2:
		return None
	try:
		h, l, c = df["high"], df["low"], df["close"]
		up, dn = h.diff(), -l.diff()
		plus_dm  = up.where((up > dn) & (up > 0), 0.0)
		minus_dm = dn.where((dn > up) & (dn > 0), 0.0)
		tr  = pd.concat([(h - l), (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
		atr = tr.ewm(alpha=1.0 / period, adjust=False).mean().replace(0, 1e-9)
		pdi = 100.0 * (plus_dm.ewm(alpha=1.0 / period, adjust=False).mean() / atr)
		mdi = 100.0 * (minus_dm.ewm(alpha=1.0 / period, adjust=False).mean() / atr)
		dx  = 100.0 * (pdi - mdi).abs() / (pdi + mdi).replace(0, 1e-9)
		val = dx.ewm(alpha=1.0 / period, adjust=False).mean().iloc[-1]
		return float(val) if pd.notna(val) else None
	except Exception:
		return None


def _calc_poc_from_ohlcv(df: Optional[pd.DataFrame], bins: int = 24) -> Optional[float]:
	"""
	Volume-weighted Point of Control (POC) across the visible 1D range.
	For each bar, distributes its volume evenly across the price bins it spans.
	"""
	if df is None or df.empty or len(df) < 2:
		return None
	try:
		lo = float(df["low"].min())
		hi = float(df["high"].max())
		if lo >= hi:
			return None
		bin_size	= (hi - lo) / bins
		vol_per_bin = [0.0] * bins

		for i in range(len(df)):
			bar_lo = float(df["low"].iloc[i])
			bar_hi = float(df["high"].iloc[i])
			bar_v  = float(df["volume"].iloc[i])
			b_start = max(0, int((bar_lo - lo) / bin_size))
			b_end	= min(bins - 1, int((bar_hi - lo) / bin_size))
			n = b_end - b_start + 1
			if n <= 0:
				continue
			share = bar_v / n
			for b in range(b_start, b_end + 1):
				vol_per_bin[b] += share

		max_b = max(range(bins), key=lambda x: vol_per_bin[x])
		return round(lo + (max_b + 0.5) * bin_size, 4)
	except Exception:
		return None


# ─────────────────────────────────────────────────────────────────
# CoinGlass v4 — Whale Sentiment Manager
# ─────────────────────────────────────────────────────────────────

class CoinglassManager:
	"""
	Fetches OI, Funding Rate, L/S Ratio and aggregated liquidation history
	from the CoinGlass v4 API. No order-book endpoints — those have been removed.
	"""

	BASE_URL = "https://open-api-v4.coinglass.com"

	_SYM_MAP: dict[str, str] = {
		"BTC/USDT": "BTC",
		"ETH/USDT": "ETH",
		"XRP/USDT": "XRP",
		"BNB/USDT": "BNB",
		"SOL/USDT": "SOL",
	}

	def __init__(self, api_key: str) -> None:
		self.api_key  = api_key
		self._session: Optional[aiohttp.ClientSession] = None

	async def __aenter__(self) -> "CoinglassManager":
		self._session = aiohttp.ClientSession(
			headers={
				"CG-API-KEY":	 self.api_key,
				"accept":		 "application/json",
				"Content-Type":	 "application/json",
			},
			timeout=aiohttp.ClientTimeout(total=15),
		)
		return self

	async def __aexit__(self, *_) -> None:
		if self._session:
			await self._session.close()
		self._session = None

	# ── Generic GET --------------------------------------------------------

	async def _get(self, endpoint: str, params: Optional[dict] = None):
		if not self._session:
			return None
		url = f"{self.BASE_URL.rstrip('/')}/{endpoint.lstrip('/')}"
		try:
			async with self._session.get(url, params=params) as resp:
				if resp.status != 200:
					logger.warning("[CG] HTTP %d | %s", resp.status, endpoint)
					return None
				payload = await resp.json(content_type=None)

				success = payload.get("success")
				code	= str(payload.get("code", ""))
				if not (success is True or code in ["0", "000000", "200"]):
					logger.warning(
						"[CG] API failure | %s | msg=%s | code=%s",
						endpoint, payload.get("msg", "?"), code,
					)
					return None
				return payload.get("data")
		except asyncio.TimeoutError:
			logger.warning("[CG] Timeout | %s", endpoint)
			return None
		except Exception as exc:
			logger.error("[CG] Request error | %s: %s", endpoint, exc)
			return None

	# ── Endpoint wrappers --------------------------------------------------

	async def _fetch_open_interest(self, cg_sym: str) -> Optional[list]:
		data = await self._get(
			"/api/futures/open-interest/aggregated-history",
			{"symbol": cg_sym, "interval": "h4", "limit": 5},
		)
		if isinstance(data, dict) and "list" in data:
			return data["list"]
		return data if isinstance(data, (dict, list)) else None

	async def _fetch_funding_rate(self, cg_sym: str) -> Optional[list]:
		data = await self._get(
			"/api/futures/funding-rate/exchange-list",
			{"symbol": cg_sym},
		)
		if isinstance(data, dict) and "list" in data:
			return data["list"]
		return data if isinstance(data, (dict, list)) else None

	async def _fetch_ls_ratio(self, cg_sym: str) -> Optional[list]:
		pair = f"{cg_sym}USDT"
		data = await self._get(
			"/api/futures/top-long-short-account-ratio/history",
			{"exchange": "Binance", "symbol": pair, "interval": "h4", "limit": 1},
		)
		if isinstance(data, dict) and "list" in data:
			return data["list"]
		return data if isinstance(data, (dict, list)) else None

	async def _fetch_liquidation_history(self, cg_sym: str) -> Optional[list]:
		data = await self._get(
			"/api/futures/liquidation/aggregated-history",
			{"symbol": cg_sym, "interval": "h4", "limit": 10,
			 "exchange_list": "Binance,OKX,Bybit,Bitget"},
		)
		if isinstance(data, dict) and "list" in data:
			return data["list"]
		return data if isinstance(data, list) else None

	# CoinGlass interval strings keyed by our internal TF labels.
	# We try the v4 form ("1h") first, then the legacy form ("h1") as fallback,
	# because different CoinGlass endpoints have historically used either.
	_CG_INTERVAL_MAP: dict[str, tuple[str, str]] = {
		"1h": ("1h", "h1"),
		"4h": ("4h", "h4"),
		"1d": ("1d", "d1"),
		# (15m intentionally omitted — we keep MT5 tick volume for the LTF)
	}

	async def fetch_spot_volumes(
		self,
		cg_sym:   str,
		tf:       str,
		limit:    int = 50,
	) -> Optional[list[dict]]:
		"""
		Real USD spot volume per bar from CoinGlass /api/spot/price/history.

		Returns a list of {"time": <ms>, "volume_usd": <float>} dicts (oldest→newest)
		or None on any failure (caller then falls back to MT5 tick volume).

		Endpoint payload sample (per bar):
		  {"time": 1741690800000, "open":.., "high":.., "low":.., "close":..,
		   "volume_usd": 96823535.5724}
		"""
		interval_forms = self._CG_INTERVAL_MAP.get(tf)
		if interval_forms is None:
			return None  # unsupported TF (e.g. 15m) — keep tick volume

		pair = f"{cg_sym}USDT"
		data = None
		for cg_interval in interval_forms:
			# limit+1: the most-recent CoinGlass bar may still be forming; trim later.
			data = await self._get(
				"/api/spot/price/history",
				{"exchange": "Binance", "symbol": pair,
				 "interval": cg_interval, "limit": int(limit) + 1},
			)
			if isinstance(data, dict) and "list" in data:
				data = data["list"]
			if isinstance(data, list) and data:
				break  # this interval form worked
			data = None
		if not isinstance(data, list) or not data:
			return None

		out: list[dict] = []
		for bar in data:
			if not isinstance(bar, dict):
				continue
			ts  = bar.get("time")
			vol = bar.get("volume_usd", bar.get("volume"))
			if ts is None or vol is None:
				continue
			try:
				out.append({"time": int(ts), "volume_usd": float(vol)})
			except (TypeError, ValueError):
				continue
		# Sort oldest→newest so positional/timestamp alignment is deterministic.
		out.sort(key=lambda x: x["time"])
		return out or None

	# ── Parsers ------------------------------------------------------------

	@staticmethod
	def _parse_oi(raw) -> dict:
		if not raw:
			return {"comment": "No data", "change_4h_pct": None, "current_oi_usd": None}
		try:
			if isinstance(raw, list) and len(raw) >= 2:
				cur	 = float(raw[-1].get("openInterest") or raw[-1].get("close")
							 or raw[-1].get("c") or 0)
				prev = float(raw[-2].get("openInterest") or raw[-2].get("close")
							 or raw[-2].get("c") or cur)
			elif isinstance(raw, dict):
				cur	 = float(raw.get("openInterest") or raw.get("openInterestCurrent")
							 or raw.get("close") or 0)
				prev = float(raw.get("openInterestPrev") or raw.get("close") or cur)
			else:
				return {"comment": "Unrecognised format", "change_4h_pct": None}
		except (TypeError, ValueError):
			return {"comment": "Parse error", "change_4h_pct": None}

		change = ((cur - prev) / prev * 100) if prev else 0.0
		if change > 3:
			comment = "OI rising (REAL positioning): new positions opening"
		elif change < -3:
			comment = "OI falling: squeeze / position-closing (fake breakout risk)"
		else:
			comment = "OI flat: no clear directional flow"
		return {
			"current_oi_usd": round(cur, 0),
			"change_4h_pct":  round(change, 2),
			"comment":		  comment,
		}

	@staticmethod
	def _parse_funding_rate(raw, target_symbol: str = "") -> dict:
		if not raw:
			return {"comment": "No data", "rate_pct": None, "rate_raw": 0.0}

		rate		  = 0.0
		exchange_cnt  = 0

		try:
			if isinstance(raw, list) and raw:
				entry = None
				if target_symbol:
					for item in raw:
						if not isinstance(item, dict):
							continue
						sym = str(item.get("symbol", "")).upper()
						if sym == target_symbol.upper():
							entry = item
							break
				if entry is None:
					if target_symbol:
						return {
							"comment": f"FR list does not contain {target_symbol}",
							"rate_pct": None,
							"rate_raw": 0.0,
						}
					entry = raw[0]

				margin_list = entry.get("stablecoin_margin_list", [])
				rates = []
				for item in margin_list:
					val = item.get("funding_rate") or item.get("rate")
					if val is not None:
						try:
							rates.append(float(val))
						except (TypeError, ValueError):
							continue
				if rates:
					rate		 = sum(rates) / len(rates)
					exchange_cnt = len(rates)
			elif isinstance(raw, dict):
				rate		 = float(raw.get("funding_rate") or raw.get("rate") or 0)
				exchange_cnt = 1
		except (TypeError, ValueError):
			rate = 0.0

		# Comment thresholds use rate_decimal (matches D.2 matrix tiers exactly).
		# CoinGlass returns funding rate in percent form (e.g. 0.0054 = 0.0054%);
		# rate_decimal converts to true decimal (e.g. 0.000054) which is what the
		# scoring matrix tiers (-0.0010 / -0.0003 / 0.0003 / 0.0010 / 0.0020) compare against.
		rate_decimal = rate / 100.0
		if rate_decimal > 0.0020:
			comment = "Extreme positive funding — short-squeeze risk"
		elif rate_decimal > 0.0010:
			comment = "Strong positive — long-side pressure"
		elif rate_decimal > 0.0003:
			comment = "Moderate positive — modest long bias"
		elif rate_decimal < -0.0020:
			comment = "Extreme negative funding — long-squeeze opportunity"
		elif rate_decimal < -0.0010:
			comment = "Strong negative — short-side pressure"
		elif rate_decimal < -0.0003:
			comment = "Moderate negative — modest short bias"
		else:
			comment = "Neutral funding"

		extra = f" ({exchange_cnt} exchanges avg)" if exchange_cnt > 1 else ""
		return {
			"rate_raw":	 rate_decimal,		   # 0.0054% için 0.000054 (matrix'in beklediği)
			"rate_pct":	 f"{rate:.4f}%{extra}", # API zaten percent, ×100 yapma
			"comment":	 comment,
		}

	@staticmethod
	def _parse_ls_ratio(raw) -> dict:
		if not raw:
			return {"ratio": None, "comment": "No data"}
		try:
			item = raw[0] if isinstance(raw, list) and raw else raw
			if not isinstance(item, dict):
				return {"ratio": None, "comment": "Unrecognised format"}

			ratio = float(
				item.get("top_account_long_short_ratio")
				or item.get("longShortRatio")
				or item.get("ratio")
				or 0
			)
		except (TypeError, ValueError, ZeroDivisionError):
			return {"ratio": None, "comment": "Parse error"}

		if ratio > 2.5:
			comment = f"Extreme crowd LONG (L/S={ratio:.2f}) — contrarian SHORT signal"
		elif ratio > 1.5:
			comment = f"Long-heavy crowd (L/S={ratio:.2f}) — caution on longs"
		elif 0 < ratio < 0.8:
			comment = f"Extreme crowd SHORT (L/S={ratio:.2f}) — contrarian LONG signal"
		else:
			comment = f"Balanced crowd (L/S={ratio:.2f}) — no contrarian edge"
		return {"ratio": round(ratio, 3), "comment": comment}

	@staticmethod
	def _parse_liquidation_history(raw) -> list[dict]:
		if not raw or not isinstance(raw, list):
			return []
		result: list[dict] = []
		for item in raw[:10]:
			try:
				long_vol = float(
					item.get("aggregated_long_liquidation_usd")
					or item.get("longVolUsd") or item.get("longVol") or 0
				)
				short_vol = float(
					item.get("aggregated_short_liquidation_usd")
					or item.get("shortVolUsd") or item.get("shortVol") or 0
				)
				time_str = str(item.get("createTime") or item.get("time")
							   or item.get("t") or "")[:16]
				price = float(item.get("price") or item.get("liquidationPrice") or 0)

				if long_vol > 0 or short_vol > 0:
					dominant = "LONGS_LIQUIDATED" if long_vol > short_vol else "SHORTS_LIQUIDATED"
					result.append({
						"price":	  0.0,
						"side":		  f"MARKET-WIDE {dominant}",
						"amount_usd": round(max(long_vol, short_vol), 0),
						"time":		  time_str,
					})
				elif price > 0:
					side_raw = str(item.get("side") or item.get("direction") or "").strip().lower()
					side_label = "LONGS_LIQUIDATED" if "sell" in side_raw or "long" in side_raw \
								 else "SHORTS_LIQUIDATED"
					amount = float(item.get("amount") or item.get("usd") or item.get("qty") or 0)
					result.append({
						"price":	  round(price, 4),
						"side":		  side_label,
						"amount_usd": round(amount, 0),
						"time":		  time_str,
					})
			except (TypeError, ValueError):
				continue
		return result

	# ── Aggregate fetch -----------------------------------------------------

	async def get_whale_sentiment(
		self,
		symbol:		   str,
		current_price: float,
		ohlcv_1d:	   Optional[pd.DataFrame] = None,
	) -> dict:
		cg_sym = self._SYM_MAP.get(symbol, symbol.split("/")[0])
		logger.info("[CG][%s] Fetching whale sentiment (sym=%s)...", symbol, cg_sym)

		oi_raw, fr_raw, ls_raw, liq_raw = await asyncio.gather(
			self._fetch_open_interest(cg_sym),
			self._fetch_funding_rate(cg_sym),
			self._fetch_ls_ratio(cg_sym),
			self._fetch_liquidation_history(cg_sym),
			return_exceptions=True,
		)

		for name, val in [("OI", oi_raw), ("FR", fr_raw), ("LS", ls_raw), ("Liq", liq_raw)]:
			if isinstance(val, Exception):
				logger.error("[CG][%s] %s endpoint error: %s", symbol, name, val)

		if isinstance(oi_raw,  Exception): oi_raw  = None
		if isinstance(fr_raw,  Exception): fr_raw  = None
		if isinstance(ls_raw,  Exception): ls_raw  = None
		if isinstance(liq_raw, Exception): liq_raw = None

		oi_sum	 = self._parse_oi(oi_raw)
		fr_sum	 = self._parse_funding_rate(fr_raw, target_symbol=cg_sym)
		ls_sum	 = self._parse_ls_ratio(ls_raw)
		liq_list = self._parse_liquidation_history(liq_raw)
		poc		 = _calc_poc_from_ohlcv(ohlcv_1d)

		logger.info(
			"[CG][%s] OK | OI Δ4h=%s%% | FR=%s | L/S=%s | Liq=%d",
			symbol,
			oi_sum.get("change_4h_pct", "?"),
			fr_sum.get("rate_pct", "?"),
			ls_sum.get("ratio", "?"),
			len(liq_list),
		)

		return {
			"symbol":				symbol,
			"current_price":		current_price,
			"poc_price":			round(poc, 4) if poc else None,
			"open_interest":		oi_sum,
			"funding_rate":			fr_sum,
			"ls_ratio":				ls_sum,
			"liquidation_history":	liq_list,
		}
# ─────────────────────────────────────────────────────────────────
# MT5 Order Operations — Limit Entry, Partial Close, SL Modify
# ─────────────────────────────────────────────────────────────────

def _round_to_tick(mt5_sym: str, price: Optional[float]) -> Optional[float]:
	"""
	Snap a price to the symbol's trade_tick_size and digit precision. MT5 rejects
	pending orders whose price is not an exact multiple of the tick size with
	TRADE_RETCODE_INVALID_PRICE — and the AI-derived entry/SL/TP (current×0.998,
	structural×0.99, raw level) are almost never on-tick. Always snap before sending.
	"""
	if price is None or price <= 0:
		return price
	if not _MT5_AVAILABLE:
		return price
	info = mt5.symbol_info(mt5_sym)
	if info is None:
		return price
	digits = int(getattr(info, "digits", 0) or 0)
	tick   = float(getattr(info, "trade_tick_size", 0) or 0)
	if tick <= 0:
		tick = (10 ** -digits) if digits > 0 else 0.0
	if tick <= 0:
		return round(price, digits) if digits > 0 else price
	snapped = round(price / tick) * tick
	return round(snapped, digits) if digits > 0 else snapped


async def _send_limit_order(
	mt5_sym:	 str,
	decision:	 str,
	entry_price: float,
	volume:		 float,
	sl_price:	 Optional[float],
	tp_price:	 Optional[float],
	comment:	 str,
	dry_run:	 bool = False,
) -> ExecutionResult:
	"""Sends a BUY_LIMIT or SELL_LIMIT pending order to MT5."""
	side = "LONG" if decision.upper() == "LONG" else "SHORT"

	# FLAW #2 fix: snap ALL prices to the symbol tick BEFORE building the request.
	# Off-tick prices → TRADE_RETCODE_INVALID_PRICE. Done before the dry-run branch
	# so simulated logs show the real (snapped) price the live path would send.
	entry_price = _round_to_tick(mt5_sym, entry_price)
	sl_price	= _round_to_tick(mt5_sym, sl_price)
	tp_price	= _round_to_tick(mt5_sym, tp_price)

	if dry_run or not _MT5_AVAILABLE:
		sim_ticket = f"DRY-{int(time.time())}"
		trade_logger.info(
			"[DRY-RUN] %s LIMIT | %s @ %.4f | vol=%.3f | sl=%s | tp=%s | ticket=%s",
			side, mt5_sym, entry_price, volume, sl_price, tp_price, sim_ticket,
		)
		return ExecutionResult(
			status=ExecutionStatus.DRY_RUN,
			order_id=sim_ticket,
			symbol=mt5_sym, side=side,
			price=entry_price, volume=volume,
			sl=sl_price, tp=tp_price,
			message="dry_run=True simulated limit order",
			raw={},
		)

	order_type = mt5.ORDER_TYPE_BUY_LIMIT if side == "LONG" else mt5.ORDER_TYPE_SELL_LIMIT

	# Expiration base = BROKER server time (tick.time), NOT the local clock. MT5
	# compares the expiry to its own server clock; a skewed PC clock or a server
	# epoch offset from UTC can push a local-time expiry into the server past ->
	# "Invalid expiration" / premature cancel. tick.time is the server epoch base.
	_xtick = await asyncio.to_thread(mt5.symbol_info_tick, mt5_sym)
	_broker_now = int(_xtick.time) if (_xtick and getattr(_xtick, "time", 0)) else int(time.time())

	request: dict = {
		"action":		mt5.TRADE_ACTION_PENDING,
		"symbol":		mt5_sym,
		"volume":		float(volume),
		"type":			order_type,
		"price":		float(entry_price),
		"type_time":	mt5.ORDER_TIME_SPECIFIED,  # 4h expiry — prevents zombie orders blocking analysis
		"expiration":   _broker_now + (4 * 3600),  # 4h in the BROKER clock (see above)
		"magic":		_MAGIC,
		"comment":		comment[:31] if comment else "algo-bot",
	}
	if sl_price:
		request["sl"] = float(sl_price)
	if tp_price:
		request["tp"] = float(tp_price)

	result = await asyncio.to_thread(mt5.order_send, request)

	if result is None:
		err = await asyncio.to_thread(mt5.last_error)
		msg = f"order_send returned None | mt5_err={err}"
		logger.error("[LIMIT] %s", msg)
		return ExecutionResult(
			status=ExecutionStatus.FAILED,
			symbol=mt5_sym, side=side,
			price=entry_price, volume=volume,
			sl=sl_price, tp=tp_price,
			message=msg, raw=request,
		)

	if result.retcode == mt5.TRADE_RETCODE_DONE:
		msg = f"Limit order placed | ticket={result.order}"
		logger.info("[LIMIT] %s | %s @ %.4f | vol=%.3f", side, mt5_sym, entry_price, volume)
		return ExecutionResult(
			status=ExecutionStatus.SUCCESS,
			order_id=str(result.order),
			symbol=mt5_sym, side=side,
			price=entry_price, volume=volume,
			sl=sl_price, tp=tp_price,
			message=msg, raw=result._asdict(),
		)

	err = await asyncio.to_thread(mt5.last_error)
	msg = f"Limit order rejected | retcode={result.retcode} | mt5_err={err}"
	logger.error("[LIMIT] %s", msg)
	return ExecutionResult(
		status=ExecutionStatus.FAILED,
		symbol=mt5_sym, side=side,
		price=entry_price, volume=volume,
		sl=sl_price, tp=tp_price,
		message=msg, raw=result._asdict(),
	)


async def _market_close_partial(
	pos,
	close_volume: float,
	comment:	  str,
) -> bool:
	"""
	Sends a market order opposite to the position direction to close
	`close_volume` lots of an open position.
	"""
	if not _MT5_AVAILABLE:
		return False

	tick = await asyncio.to_thread(mt5.symbol_info_tick, pos.symbol)
	if tick is None:
		err = await asyncio.to_thread(mt5.last_error)
		logger.error("[CLOSE] No tick for %s | err=%s", pos.symbol, err)
		return False

	is_long = pos.type == mt5.POSITION_TYPE_BUY
	request = {
		"action":		mt5.TRADE_ACTION_DEAL,
		"symbol":		pos.symbol,
		"volume":		float(close_volume),
		"type":			mt5.ORDER_TYPE_SELL if is_long else mt5.ORDER_TYPE_BUY,
		"position":		pos.ticket,
		"price":		float(tick.bid if is_long else tick.ask),
		"deviation":	20,
		"magic":		_MAGIC,
		"comment":		comment[:31],
		"type_time":	mt5.ORDER_TIME_GTC,
		"type_filling": mt5.ORDER_FILLING_IOC,
	}

	result = await asyncio.to_thread(mt5.order_send, request)
	if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
		rc	= result.retcode if result else "None"
		err = await asyncio.to_thread(mt5.last_error)
		logger.error(
			"[CLOSE] Failed | ticket=%d vol=%.3f retcode=%s err=%s",
			pos.ticket, close_volume, rc, err,
		)
		return False

	logger.info(
		"[CLOSE] OK | ticket=%d %s @ %.5f vol=%.3f (%s)",
		pos.ticket, "SELL" if is_long else "BUY", request["price"], close_volume, comment,
	)
	return True


async def _modify_sl(pos, new_sl: float) -> bool:
	"""Modifies the SL of an open position."""
	if not _MT5_AVAILABLE:
		return False
	# Insurance (claim 3): SL yi sembol tick ine snap et — off-tick bir SL
	# (orn. ileride trailing-stop) TRADE_RETCODE_INVALID_PRICE i onler.
	_snap = _round_to_tick(pos.symbol, new_sl)
	if _snap is not None:
		new_sl = _snap
	request = {
		"action":	mt5.TRADE_ACTION_SLTP,
		"symbol":	pos.symbol,
		"position": pos.ticket,
		"sl":		float(new_sl),
		"tp":		float(pos.tp),
		"magic":	_MAGIC,
	}
	result = await asyncio.to_thread(mt5.order_send, request)
	if result and result.retcode == mt5.TRADE_RETCODE_DONE:
		logger.info("[SL-MODIFY] OK | ticket=%d | SL: %.5f -> %.5f",
					pos.ticket, pos.sl, new_sl)
		return True
	rc	= result.retcode if result else "None"
	err = await asyncio.to_thread(mt5.last_error)
	logger.warning("[SL-MODIFY] Failed | ticket=%d retcode=%s err=%s",
				   pos.ticket, rc, err)
	return False


# ─────────────────────────────────────────────────────────────────
# AI System Prompts (English)
# ─────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """
You are a senior institutional quantitative trading analyst evaluating swing-trade
setups for a FundedNext prop-firm account on crypto CFDs. Your task is to evaluate
each setup using a DUAL-SIDED, LAYERED scoring matrix — compute scores for BOTH
the LONG and the SHORT scenario simultaneously across 10 sub-scores grouped into
4 main categories, then pick the direction with the higher total (provided it
crosses the threshold).

CRITICAL RULES:
  * Do NOT pre-select a bias. Score both sides fairly using the exact tiers below.
  * The matrix is purely mechanical — same input must always produce the same scores.
  * Show EVERY sub-score with a one-line justification inside `calculation_breakdown`
    (10 sub-score lines plus the SL/TP/RR math).
  * If a data field is missing or marked INSUFFICIENT_DATA / "No data", award the
    neutral tier (typically Long +1, Short +1) and note "no data" in the breakdown.
  * FAKEOUT FILTER: Any structural event with `volume_backed=false` OR
    `retest_status="RETESTED_BROKEN"` is treated as INVALID — score 0 for that event.

═══════════════════════════════════════════════════════════════════
SCORING MATRIX — 4 categories × 25 pts each = 100 pts per side
═══════════════════════════════════════════════════════════════════

────────────────────────────────────────────────────────────────────
[A] MACRO TREND (HTF: 1D + 4H) — Max 25 pts per side
────────────────────────────────────────────────────────────────────

    Read the `trend` field from the 1d and 4h timeframe analyses, then apply
    this lookup table. High conviction REQUIRES HTF AGREEMENT: when 1D and 4H
    directly DISAGREE (one bullish, one bearish) the regime is in transition —
    score it NEUTRAL (10/10), do NOT hand full conviction to the slower 1D
    frame. A single-frame disagreement (other frame RANGING) is down-weighted
    toward the trending frame. All 9 combos are explicit:

        1D BULLISH & 4H BULLISH      → Long +25, Short  0
        1D BULLISH & 4H RANGING      → Long +16, Short +4
        1D BULLISH & 4H BEARISH      → Long +10, Short +10  ← HTF conflict (transition) = neutral
        1D RANGING & 4H BULLISH      → Long +16, Short +4
        1D & 4H both RANGING         → Long  +8, Short +8
        1D RANGING & 4H BEARISH      → Long  +4, Short +16
        1D BEARISH & 4H BULLISH      → Long +10, Short +10  ← HTF conflict (transition) = neutral
        1D BEARISH & 4H RANGING      → Long  +4, Short +16
        1D BEARISH & 4H BEARISH      → Long  0, Short +25

────────────────────────────────────────────────────────────────────
[B] SWING MOMENTUM (1H + 4H) — Max 25 pts per side
────────────────────────────────────────────────────────────────────

    Three sub-scores. Sum = max 25 (10 + 10 + 5).

    NOTE: This is a SWING strategy. Momentum is judged on 1H and 4H ONLY.
    The 15m timeframe is NEVER used for directional scoring — it is reserved
    purely for entry timing (see EXECUTION). A 15m wiggle must not move the
    score; only 1H/4H momentum reflects swing-relevant conviction.

    B.1 RSI Multi-band (avg of 1h and 4h rsi_14) — Max 10 pts
        Compute avg = (rsi_1h + rsi_4h) / 2, then apply ASYMMETRIC tiers
        (middle bands lean toward momentum direction, NOT symmetric):

            < 20  (Extreme Oversold)     → Long +10, Short  0
            20-30 (Strong Oversold)      → Long  +8, Short  0
            30-45 (Bearish Momentum)     → Long  +3, Short +5  ← asymmetric
            45-55 (Neutral)              → Long  +2, Short +2
            55-70 (Bullish Momentum)     → Long  +5, Short +3  ← asymmetric
            70-80 (Strong Overbought)    → Long  0, Short  +8
            > 80  (Extreme Overbought)   → Long  0, Short +10

    B.2 EMA Confluence (use 1h ema_confluence; fallback 4h) — Max 10 pts

            BOUNCING_OFF_EMA       → Long +10, Short  0   (prime LONG entry)
            FIRMLY_ABOVE_BOTH      → Long  +7, Short  0
            CHOPPING_BETWEEN_EMAS  → Long  +2, Short +2
            FIRMLY_BELOW_BOTH      → Long  0, Short  +7
            REJECTING_BELOW_EMA    → Long  0, Short +10   (prime SHORT entry)
            INSUFFICIENT_DATA      → Long  +1, Short +1

    B.3 MACD State (1h timeframe) — Max 5 pts
        Read 1h macd_state along with macd_hist trend (you can infer from
        macd_hist sign and magnitude vs prior typical range). Tier:

            BULL_BUILDING   (macd>signal, hist expanding)  → Long +5, Short  0
            BULL_FADING     (macd>signal, hist contracting)→ Long +3, Short +1
            FLAT_NEAR_ZERO  (|macd_line| < 0.1×atr_1h)     → Long +1, Short +1
            BEAR_FADING     (macd<signal, hist contracting)→ Long +1, Short +3
            BEAR_BUILDING   (macd<signal, hist expanding)  → Long  0, Short +5
            INSUFFICIENT_DATA                              → Long +1, Short +1

────────────────────────────────────────────────────────────────────
[C] STRUCTURAL CONTEXT — Max 25 pts per side
────────────────────────────────────────────────────────────────────

    Three sub-scores. Sum = max 25 (15 + 5 + 5).

    C.1 Major S/R Proximity — Max 15 pts (scored INDEPENDENTLY per side)
        LONG column  : distance % from current_price to the nearest major
                       SUPPORT below (0% if price sits inside the level's zone).
        SHORT column : distance % to the nearest major RESISTANCE above.
        Each side gets its own tier from its own level — proximity to a
        support NEVER adds points to the short column, and vice versa.

        Base tier (per side, from that side's distance):
            Inside zone / within 0.3%   → 15
            Within 0.8%                 → 12
            Within 1.5%                 →  9
            Within 2.5%                 →  5
            Farther than 2.5%           →  0

        Then SCALE the base points by that level's touch strength and round
        to the nearest integer:
            touch_count ≥ 6  → ×1.0
            touch_count 4-5  → ×0.8
            touch_count = 3  → ×0.6
            touch_count < 3  → ×0.5

        Example: support 0.5% away with 4 touches → 12 × 0.8 = Long +10;
        nearest resistance 3.1% away → Short +0.

    C.2 S/R Quality Bonus — Max 5 pts (per side, gated on C.1)
        GATE per side: a side scores C.2 ONLY if that side received C.1 points
        (its level was within 2.5%). Evaluate each qualifying side using ITS
        OWN nearest level's metadata (label, touch_count). A side whose C.1
        was 0 gets C.2 = 0.

        Apply the SINGLE highest applicable (no stacking):

            Level is SR_FLIP_SUPPORT (LONG) /
            Level is SR_FLIP_RESISTANCE (SHORT)  → relevant side +5  (premium)
            touch_count ≥ 6                       → relevant side +4
            touch_count 4-5                       → relevant side +3
            touch_count = 3                       → relevant side +2

        Confluence bonus (additive to above, capped at +5 total for the sub):
            If the major level is within 0.3% of the POC level OR within 0.3%
            of a psychological round number → +1 to the relevant side.

    C.3 Structural Events (with FAKEOUT FILTER) — Max 5 pts
        Read structure_events on SWING timeframes — 1h, 4h AND 1d ONLY (events have
        bars_since, is_strong_close, volume_backed, retest_status). DO NOT score
        15m structural events for direction — 15m is entry-timing only; a 15m
        BOS/CHoCH is noise at swing scale. HTF events (4h/1d) include RANGE
        BREAKOUTS (a close beyond the range top/bottom while the TF is RANGING)
        reported as BOS_BULL / BOS_BEAR — treat these as valid structural events
        too. A fresh 4h/1d range breakout is a HIGH-conviction signal (new trend
        birth) — prefer it over a 1h event when both qualify.

        FILTER FIRST — ignore any event where:
            volume_backed == false  OR
            retest_status == "RETESTED_BROKEN"
        These are fakeouts. Skip them entirely.

        Among the remaining (volume-backed AND not-broken) events across all TFs,
        for each side pick the SINGLE highest applicable. "Fresh" = bars_since ≤ 12
        on that event's own timeframe (so a fresh 4h event ≤ 20h old, a fresh 1d
        event ≤ 5 days old — both still structurally relevant for swing trades).

        LONG side:
            Fresh CHoCH_BULL + retest_status="RETESTED_RESPECTED"  → Long +5  (confirmed reversal)
            Fresh CHoCH_BULL + is_strong_close                      → Long +4
            Fresh CHoCH_BULL (default)                              → Long +3
            Fresh BOS_BULL + is_strong_close                        → Long +3
            Fresh BOS_BULL (default)                                → Long +2
            Price inside unmitigated BULL_OB on 1h                  → Long +2
            Price inside unmitigated BULL_FVG on 1h                 → Long +1
            Nothing applicable                                      → Long  0

        SHORT side (mirror):
            Fresh CHoCH_BEAR + RETESTED_RESPECTED                   → Short +5
            Fresh CHoCH_BEAR + is_strong_close                      → Short +4
            Fresh CHoCH_BEAR                                        → Short +3
            Fresh BOS_BEAR + is_strong_close                        → Short +3
            Fresh BOS_BEAR                                          → Short +2
            Price inside unmitigated BEAR_OB on 1h                  → Short +2
            Price inside unmitigated BEAR_FVG on 1h                 → Short +1
            Nothing applicable                                      → Short  0

        CANDLESTICK CONFLUENCE BONUS (+1, applied AFTER tier selection):
            If the most recent bar's `candlestick_patterns` contains an aligned
            reversal/continuation pattern at the same timeframe as the chosen event,
            add +1 to the side that received the C.3 points. Cap C.3 at +5 (do not
            exceed the sub-score ceiling).

            Aligned patterns:
              LONG  : BULLISH_ENGULFING, HAMMER, INVERTED_HAMMER
              SHORT : BEARISH_ENGULFING, SHOOTING_STAR, HANGING_MAN

            Examples:
              Fresh BOS_BULL (default +2) + BULLISH_ENGULFING on same TF → +1 bonus → Long +3
              Fresh CHoCH_BEAR + strong_close (already +4) + SHOOTING_STAR → +1 bonus → Short +5 (capped)
              Fresh CHoCH_BULL + RETESTED_RESPECTED (+5) + HAMMER → +0 (already at cap)

            DOJI is neutral — never triggers the bonus.

        KILLZONE BONUS (+1, applied AFTER candlestick bonus):
            Each event line carries a `utc≈HH:MM` field — this is the event's UTC
            time, already normalised for you (computed from elapsed bars, so it is
            CORRECT regardless of the broker's server-time zone). Use ONLY this field;
            do NOT try to infer UTC from the raw timestamp or from bars_since.
            If the chosen structural event's utc≈ falls in a high-institutional-activity
            window:
                London open killzone : 07:00–11:00 UTC
                New York killzone    : 12:00–16:00 UTC
            add +1 to the same side that received C.3 points. Cap C.3 at +5 total.
            If an event has no utc≈ field, award NO killzone bonus (do not guess).
            (Crypto's institutional flow concentrates in these windows; events here
            are more likely to follow through.)

            Example:
              Fresh BOS_BEAR (+2) + SHOOTING_STAR (+1) + utc≈14:30 → +1 killzone
              → Short +4 total (still under cap).

────────────────────────────────────────────────────────────────────
[D] WHALE & DERIVATIVES (CoinGlass) — Max 25 pts per side
────────────────────────────────────────────────────────────────────

    Three sub-scores. Sum = max 25 (10 + 8 + 7).

    D.1 L/S Ratio (Crowd Fade) — Max 10 pts
        Read whale_data.ls_ratio.ratio. Apply:

            < 0.5     (Extreme Crowd SHORT)   → Long +10, Short  0  (strong contrarian LONG)
            0.5 - 0.7 (Strong Crowd SHORT)    → Long  +8, Short  0
            0.7 - 1.2 (Slight Short Bias)     → Long  +4, Short +1
            1.2 - 2.0 (Slight Long Bias)      → Long  +1, Short +4
            2.0 - 3.0 (Strong Crowd LONG)     → Long  0, Short  +8
            > 3.0     (Extreme Crowd LONG)    → Long  0, Short +10  (strong contrarian SHORT)
            No data                            → Long +1, Short +1

    D.2 Funding Rate — Max 8 pts (Crypto-adjusted thresholds)
        Read whale_data.funding_rate.rate_raw (decimal: 0.0001 = 0.01%).
        In crypto markets sustained positive funding 0.05-0.10% is normal during
        bull phases — true "extreme" starts above 0.15%. Tiers reflect this:

            < -0.0010 (Extreme Negative)               → Long +8, Short  0  (long-squeeze setup)
            -0.0010 to -0.0003 (Strong Negative)       → Long +6, Short +1
            -0.0003 to 0.0003 (Neutral)                → Long +3, Short +3
            0.0003 to 0.0010 (Moderate Positive)       → Long +1, Short +5
            0.0010 to 0.0020 (Strong Positive)         → Long  0, Short +7
            > 0.0020 (Extreme Positive)                → Long  0, Short +8  (short-squeeze setup)
            No data                                    → Long +1, Short +1

    D.3 Open Interest × Price Behavior — Max 7 pts
        Combine TWO signals:
          (a) whale_data.open_interest.change_4h_pct
          (b) Direction of the LAST closed 4h candle: GREEN if 4h close > 4h open, RED otherwise.

        Apply:

            OI rising > +3%  AND 4h candle GREEN → Long +7, Short  0  (real bull positioning)
            OI rising > +3%  AND 4h candle RED   → Long  0, Short +7  (real bear positioning)
            OI falling < -3% AND 4h candle GREEN → Long +1, Short +3  (short squeeze, weak fuel)
            OI falling < -3% AND 4h candle RED   → Long +3, Short +1  (long capitulation, possible bottom)
            OI flat (-3% to +3%)                 → Long +2, Short +2
            No OI data                            → Long +1, Short +1


═══════════════════════════════════════════════════════════════════
DECISION LOGIC
═══════════════════════════════════════════════════════════════════

1. Compute `long_total` = sum of all 10 sub-scores on the long column
   (A + B.1 + B.2 + B.3 + C.1 + C.2 + C.3 + D.1 + D.2 + D.3).
2. Compute `short_total` = sum of all 10 sub-scores on the short column.
3. Determine decision:
       long_total  >= __MIN_SCORE__  AND  long_total  > short_total  → "LONG"
       short_total >= __MIN_SCORE__  AND  short_total > long_total   → "SHORT"
       otherwise                                          → "WAIT"

   COUNTER-TREND PROHIBITION (mandatory — overrides the score):
   Do NOT fade a turning higher-timeframe. A swing strategy must not short a
   market that has bottomed and is recovering, nor long one that has topped.
       • If the 4H trend is BULLISH, or the 4H trend is RANGING while price sits
         ABOVE the 1H and 4H EMA50 (a confirmed recovery off a low) → never "SHORT"
         (force "WAIT"), even if short_total ≥ __MIN_SCORE__ and a resistance/SR-flip is near.
       • Mirror: if the 4H trend is BEARISH, or RANGING while price is BELOW both
         EMA50s (a confirmed breakdown) → never "LONG".
   A mere bounce INTO resistance inside a still-BEARISH 4H (price still below EMA50,
   structure still lower-highs) is NOT a recovery — a continuation SHORT there is
   allowed. The bot enforces this deterministically too; align with it.

   LOCATION PROHIBITION (premium/discount — mandatory, overrides the score):
   "Buy in discount, sell in premium." The brief's DEALING RANGE line gives the
   current 4H swing range and price's position within it (0% = range low,
   100% = range high).
       • Position ≥ 62% (PREMIUM)  → never "LONG"  (do not buy the top of the range)
       • Position ≤ 38% (DISCOUNT) → never "SHORT" (do not sell the bottom)
   Between 38-62% (EQUILIBRIUM) both directions are allowed. The engine enforces
   this deterministically; align with it.

   LTF TRIGGER REQUIREMENT (mandatory): a LONG/SHORT additionally requires a
   FRESH lower-timeframe confirmation visible in the 15m block:
       • a fresh 15m CHoCH/BOS in the trade direction (not volume-refuted,
         retest not broken), OR
       • current price sitting INSIDE an aligned unmitigated 15m OB or FVG, OR
       • an aligned reversal candle among the most recent 15m patterns
         (LONG: BULLISH_ENGULFING/HAMMER/INVERTED_HAMMER;
          SHORT: BEARISH_ENGULFING/SHOOTING_STAR/HANGING_MAN).
   If none is present → "WAIT" ("touched the level" is not enough; we need
   "turned at the level"). The engine enforces this deterministically too.

4. SL / TP construction (only when decision is LONG or SHORT). Follow these
   steps IN ORDER and show every numeric step inside `calculation_breakdown`:

   STEP 4a — Structural SL beyond the NEXT structural level (not the one price is
   testing). Anchoring 1% past the immediate level gets run by ordinary higher-high
   / lower-low noise; place the stop past the level whose break truly invalidates:
       LONG  → structural_sl = (next major support BELOW the nearest one) × 0.99
               (if none/too far → nearest major support × 0.99)
       SHORT → structural_sl = (next major resistance ABOVE the nearest one) × 1.01
               (if none/too far → nearest major resistance × 1.01)
       "Too far" = the resulting distance would exceed the __MAX_SL__% ceiling (STEP 4d).

   STEP 4b — Compute structural SL distance:
       structural_sl_pct = abs(entry_price - structural_sl) / entry_price

   STEP 4c — Apply the stop-hunt floor (minimum 1.5% from entry):
       If structural_sl_pct < 0.015:
           LONG  → sl_price = entry_price × (1 - 0.015)
           SHORT → sl_price = entry_price × (1 + 0.015)
           sl_pct = 0.015
       Else:
           sl_price = structural_sl
           sl_pct   = structural_sl_pct

   STEP 4d — Apply the maximum-distance ceiling (__MAX_SL__%):
       If sl_pct > __MAX_SL_FRAC__ → override decision to "WAIT" and stop here.

   STEP 4e — Construct TP using the NEAREST major level that yields RR ≥ 2.0,
   then CAP the reward at RR __TP_CAP__:
       Build an ordered list of major levels in the trade direction:
         LONG  → every major resistance ABOVE entry_price, sorted ascending
                 (nearest first)
         SHORT → every major support BELOW entry_price, sorted descending
                 (nearest first)
       Walk from nearest to farthest. For each candidate level compute:
         candidate_rr = abs(candidate - entry_price) / abs(entry_price - sl_price)
           • If candidate_rr ≥ 2.0  → set tp_price = candidate, STOP walking.
           • If candidate_rr < 2.0  → TP too close, skip to the next farther level.
       Use the FIRST (nearest) level that reaches RR ≥ 2.0.
       CAP: if that level's RR is ABOVE __TP_CAP__, do NOT chase the far level — place
       tp_price exactly where RR = __TP_CAP__ lands instead:
         LONG  → tp_price = entry_price + __TP_CAP__ * abs(entry_price - sl_price)
         SHORT → tp_price = entry_price - __TP_CAP__ * abs(entry_price - sl_price)
       Rationale: backtests show ~3 RR gives the best expectancy, and the 10-level
       scale-out already harvests profit at each decile of progress (the stop moves
       to breakeven at the halfway mark), so a nearer, reachable TP beats an
       over-ambitious distant one. (The engine also enforces this __TP_CAP__ cap
       deterministically as a safety net.)
       If NO level in the trade direction reaches RR ≥ 2.0 → override to "WAIT".

   STEP 4f — Final Risk/Reward sanity re-check:
       rr_ratio = abs(tp_price - entry_price) / abs(entry_price - sl_price)
       If rr_ratio < 2.0 → override decision to "WAIT".
       Target band: 2.0 ≤ RR ≤ __TP_CAP__ (Step 4e caps the upper side).

5. Position size (only when decision is LONG or SHORT):
       Position SIZE is set by the risk engine (risk-based on the stop distance),
       NOT by you. Output position_usd = 10000.0 for any LONG/SHORT and 0.0 for
       WAIT (placeholder only; the engine recomputes the real notional).
   Recommend LONG/SHORT when your assessed score is >= __MIN_SCORE__ AND the setup is valid;
   below __MIN_SCORE__ -> "WAIT". (`score` = long_total for LONG, short_total for SHORT.)

6. The `entry_price` MUST NOT equal the exact current market price (MT5 would
   reject the limit order with TRADE_RETCODE_INVALID_PRICE). Apply a small
   pullback offset so the limit fills on a minor retracement:

       LONG  → entry_price = current_price × 0.998   (0.2% below current)
       SHORT → entry_price = current_price × 1.002   (0.2% above current)

   This guarantees the limit is accepted by MT5 and captures a small pullback
   edge. Recompute SL/TP percentages from THIS `entry_price`, not from
   current_price (so the 1.5% floor and __MAX_SL__% ceiling apply correctly).
   Do NOT set entry far from current (> 0.5%) — the order may never fill.


═══════════════════════════════════════════════════════════════════
OUTPUT FORMAT — STRICT JSON ONLY
═══════════════════════════════════════════════════════════════════

{
  "calculation_breakdown": "10-line walkthrough PLUS the SL/TP/RR math. Example: 'A 1D BULL & 4H RANG → L+16 S+4 | B.1 RSI avg 27 → L+8 S+0 | B.2 EMA BOUNCING_OFF → L+10 S+0 | B.3 MACD BULL_BUILDING → L+5 S+0 | C.1 sup 74250 dist 0.5% (6 touches ×1.0) → L+12; res 78900 dist 4.3% → S+0 | C.2 SR_FLIP_SUPPORT + POC confluence → L+5 (5 base, capped) | C.3 fresh CHoCH_BULL on 1h vol✓ retest=RESPECTED → L+5 | D.1 L/S 0.62 → L+8 S+0 | D.2 FR -0.012% → L+6 S+0 | D.3 OI +4.2% & 4h GREEN → L+7 S+0 | L_total=84 S_total=3 | LONG SL: nearest_sup 74250×0.99=73507.5 → dist=1.32% < 1.5% → clamp sl=entry×0.985=73512.5 sl_pct=1.50% | TP=next_res 78000 | RR=(78000-74631)/(74631-73512.5)=3.01 ≥2.0 OK | pos_usd=20000 (score 84 in [81,100])'",
  "long_breakdown": {
    "macro_trend": 0,
    "momentum_rsi": 0,
    "momentum_ema": 0,
    "momentum_macd": 0,
    "sr_proximity": 0,
    "sr_quality": 0,
    "structural_events": 0,
    "ls_ratio": 0,
    "funding_rate": 0,
    "oi_behavior": 0
  },
  "short_breakdown": {
    "macro_trend": 0,
    "momentum_rsi": 0,
    "momentum_ema": 0,
    "momentum_macd": 0,
    "sr_proximity": 0,
    "sr_quality": 0,
    "structural_events": 0,
    "ls_ratio": 0,
    "funding_rate": 0,
    "oi_behavior": 0
  },
  "long_total": 0,
  "short_total": 0,
  "decision": "LONG" | "SHORT" | "WAIT",
  "entry_price": 0.0,
  "sl_price": 0.0,
  "tp_price": 0.0,
  "rr_ratio": 0.0,
  "position_usd": 0.0,
  "rationale": "1–2 sentence summary citing the TOP 3 sub-scores driving the decision.",
  "key_levels": ["Major Support: 74000", "Major Resistance: 78000", "POC: 75500"]
}

Output JSON only. No markdown fences. No commentary. No prose outside the JSON.
""".strip()

_POST_MORTEM_SYSTEM_PROMPT = """
You are a senior trade post-mortem analyst. Your task: review a closed trade and
extract one concrete, actionable lesson in 1–2 sentences. Use precise ICT/SMC
and CoinGlass terminology. Avoid platitudes.

Reply ONLY in this exact JSON:
{"lesson": "...", "category": "STOP" or "TP", "importance": "HIGH" or "MEDIUM" or "LOW"}

No markdown, no commentary outside the JSON.
""".strip()


_TRADE_EVALUATION_PROMPT = """
You are an in-flight trade management specialist. An active position is currently
75% of the way toward its take-profit target. Decide whether to close the entire
remaining position immediately or to bank half of the current remaining and let
the final 25% of the original size run to TP.

Decision criteria:
  CLOSE_ALL	 — Strong evidence of a reversal forming against the trade:
			   - LTF RSI extreme exhaustion against trade direction
			   - Hostile crowd-sentiment shift (L/S ratio flipping into our face)
			   - Recent rejection candles at a key major level
			   - Funding rate violently flipping against the trade
  HOLD_REST	 — Momentum still favors the trade direction. Bank half, let the rest run.

Reply ONLY in this exact JSON:
{"action": "CLOSE_ALL" or "HOLD_REST", "reason": "1–2 sentence justification"}

No markdown. No prose outside the JSON.
""".strip()
# ─────────────────────────────────────────────────────────────────
# Journal & Lessons Persistence
# ─────────────────────────────────────────────────────────────────

def _load_json_file(path: str, default):
	try:
		with open(path, "r", encoding="utf-8") as f:
			return json.load(f)
	except (FileNotFoundError, json.JSONDecodeError):
		return default


def _save_json_file(path: str, data) -> None:
	try:
		with open(path, "w", encoding="utf-8") as f:
			json.dump(data, f, ensure_ascii=False, indent=2)
	except Exception as exc:
		logger.error("[JOURNAL] Write error (%s): %s", path, exc)


def _load_news_blackouts(path: str) -> list[tuple[datetime, datetime, str]]:
	"""G6.6 — news_blackout.json'u [(start_utc, end_utc, name), ...] listesine cevir.

	Iki kayit formati desteklenir (karisik kullanilabilir):
	  {"date": "2026-09-16", "name": "FOMC"}                          -> TAM GUN (00:00-24:00 UTC)
	  {"start": "2026-09-16 17:00", "end": "2026-09-16 21:00",
	   "name": "FOMC karar+basin"}                                    -> pencere (UTC)
	Bozuk satirlar loglanir ve atlanir; dosya yoksa bos liste (karantina fiilen kapali)."""
	raw = _load_json_file(path, [])
	if not isinstance(raw, list):
		logger.warning("[NEWS] %s liste degil — karantina bos.", path)
		return []
	out: list[tuple[datetime, datetime, str]] = []
	for item in raw:
		if not isinstance(item, dict):
			continue
		name = str(item.get("name", "?"))
		try:
			if item.get("date"):
				d0 = datetime.fromisoformat(str(item["date"])).replace(tzinfo=timezone.utc)
				out.append((d0, d0 + timedelta(days=1), name))
			elif item.get("start") and item.get("end"):
				s = datetime.fromisoformat(str(item["start"]).replace("Z", "")).replace(tzinfo=timezone.utc)
				e = datetime.fromisoformat(str(item["end"]).replace("Z", "")).replace(tzinfo=timezone.utc)
				if e > s:
					out.append((s, e, name))
				else:
					logger.warning("[NEWS] end <= start, atlandi: %s", item)
			else:
				logger.warning("[NEWS] taninmayan kayit (date ya da start+end gerek): %s", item)
		except (ValueError, TypeError) as exc:
			logger.warning("[NEWS] tarih parse hatasi (%s): %s", item, exc)
	out.sort(key=lambda x: x[0])
	return out


def _journal_add_entry(
	journal_path: str,
	ticket_id:	  str,
	data_sym:	  str,
	mt5_sym:	  str,
	decision:	  dict,
	entry_price:  float,
	sl_price:	  Optional[float],
	tp_price:	  Optional[float],
) -> None:
	journal = _load_json_file(journal_path, [])
	entry = {
		"ticket_id":   ticket_id,
		"data_sym":	   data_sym,
		"mt5_sym":	   mt5_sym,
		"status":	   "PENDING",
		"open_time":   datetime.now(timezone.utc).isoformat(),
		"close_time":  None,
		"entry_price": entry_price,
		"sl_price":	   sl_price,
		"tp_price":	   tp_price,
		"close_price": None,
		"pnl":		   None,
		"decision_snapshot": {
			"decision":				 decision.get("decision"),
			"long_total":			 decision.get("long_total"),
			"short_total":			 decision.get("short_total"),
			"long_breakdown":		 decision.get("long_breakdown"),
			"short_breakdown":		 decision.get("short_breakdown"),
			"calculation_breakdown": decision.get("calculation_breakdown"),
			"rr_ratio":				 decision.get("rr_ratio"),
			"position_usd":			 decision.get("position_usd"),
			"rationale":			 decision.get("rationale"),
			"key_levels":			 decision.get("key_levels"),
		},
	}
	journal.append(entry)
	_save_json_file(journal_path, journal)
	logger.info("[JOURNAL] New entry | ticket=%s | %s @ %.4f", ticket_id, data_sym, entry_price)


def _load_recent_lessons(lessons_path: str, n: int = 5) -> list[str]:
	lessons = _load_json_file(lessons_path, [])
	recent	= lessons[-n:] if len(lessons) > n else lessons
	return [e.get("lesson", "") for e in recent if e.get("lesson")]


# ─────────────────────────────────────────────────────────────────
# Master Payload Builder
# ─────────────────────────────────────────────────────────────────

_TF_ROLES = {
	"1d":  "Primary trend — EMA signals and major structural context",
	"4h":  "Intermediate — OB/FVG, premium/discount zones",
	"1h":  "Liquidity and momentum — RSI divergences",
	"15m": "Entry trigger — CHoCH, OB touch, FVG reaction",
}


PY_GATE = 54   # Python on-filtre esigi = LLM karar esigi (54; whale'li holdout ile dusuruldu, bkz min_score). Python
               # skoru LLM den birkac puan saparsa bile gercek kurulumu KACIRMA.
               # Sadece "bariz WAIT" dongulerini eler (skor << 58). Loglara bakip
               # guvendikce 63-64 e cikar (daha cok tasarruf) ya da dusur.

# ââ Per-symbol TP RR tavani (C-konfig; backtest+OOS dogrulandi: BTC trend kosar, ETH chop) ââ
TP_CAP_RR = {"BTC": 3.5, "ETH": 3.0}	# baz varlik â tavan
TP_CAP_DEFAULT = 3.0					# bilinmeyen sembol â temkinli 3.0

def _tp_cap_for(sym) -> float:
	"""Sembolun TP RR tavani ('BTC/USDT', 'BTCUSD' vb. baz-eslesme)."""
	s = str(sym).upper()
	for _base, _cap in TP_CAP_RR.items():
		if s.startswith(_base):
			return _cap
	return TP_CAP_DEFAULT

def _whale_for_scorer(whale_data):
    """Canli (nested) whale_data -> scorer.py nin bekledigi DUZ format."""
    if not whale_data:
        return None
    ls = (whale_data.get("ls_ratio") or {}).get("ratio")
    fr = (whale_data.get("funding_rate") or {}).get("rate_raw")
    oi = (whale_data.get("open_interest") or {}).get("change_4h_pct")
    return {
        "ls_ratio": ls,
        "funding_raw": fr if fr is not None else 0.0,
        "oi_change_4h_pct": oi,
        # last_4h_green: scorer, tf_analysis["4h"]["last_candle_direction"] den turetir
    }


def build_master_payload(
	symbol:			str,
	current_price:	float,
	tf_analysis:	dict,
	whale_data:		dict,
) -> dict:
	"""
	Builds the consolidated payload sent to the AI. The technical analyzer
	output already includes a `key_levels` sub-dict computed from 1D + 4H.
	"""
	key_levels = tf_analysis.get("key_levels", {})
	tf_only	   = {k: v for k, v in tf_analysis.items() if k != "key_levels"}

	return {
		"meta": {
			"symbol":		  symbol,
			"current_price":  current_price,
			"timestamp_utc":  datetime.now(timezone.utc).isoformat(),
		},
		"top_down_analysis": {
			tf: {"role": _TF_ROLES.get(tf, tf), **data}
			for tf, data in tf_only.items()
		},
		"key_levels":	key_levels,
		"whale_data":	whale_data,
	}


# Bar duration per timeframe (minutes) — used to derive an absolute-time UTC
# stamp for events from bars_since, bypassing the broker's server-time offset.
_TF_MINUTES: dict[str, int] = {"1d": 1440, "4h": 240, "1h": 60, "15m": 15}


def _fmt_events(events: list, tf: Optional[str] = None,
				now_utc: Optional[datetime] = None) -> str:
	if not events:
		return "None detected"
	tf_min = _TF_MINUTES.get(tf or "", 0)
	parts: list[str] = []
	for e in events[-3:]:
		vol = e.get("volume_backed")
		if vol is True:
			vol_tag = "vol✓"
		elif vol is False:
			vol_tag = "vol✗ (likely fakeout)"
		else:
			vol_tag = "vol?"
		strong_tag = "strong-body" if e.get("is_strong_close") else "weak-body"
		retest_tag = e.get("retest_status", "?")
		bars       = e.get("bars_since", "?")
		# KILLZONE TZ fix: derive the event's UTC time from ABSOLUTE elapsed time
		# (now − bars_since × bar_duration), NOT from the broker timestamp (which is
		# in server-time, not UTC). This is the value the AI must use for killzones.
		kz = ""
		if tf_min and now_utc is not None and isinstance(bars, (int, float)):
			ev_utc = now_utc - timedelta(minutes=tf_min * int(bars))
			kz = f", utc≈{ev_utc:%H:%M}"
		parts.append(
			f"{e['type']} @ {e.get('level','?')} (t-{bars}, {strong_tag}, {vol_tag}, "
			f"retest={retest_tag}{kz})"
		)
	return "; ".join(parts)


def _fmt_fvgs(fvgs: list) -> str:
	if not fvgs:
		return "None"
	return "; ".join(
		f"{f['type']} [{f['lower']}-{f['upper']}] size={f.get('size_pct',0):.3f}%"
		for f in fvgs
	)


def _fmt_obs(obs: list) -> str:
	if not obs:
		return "None"
	return "; ".join(
		f"{o['type']} [{o['lower']}-{o['upper']}] impulse={o.get('impulse_pct',0):.1f}%"
		for o in obs
	)


def _fmt_patterns(pats: list) -> str:
	if not pats:
		return "None"
	return "; ".join(
		f"{p['pattern']}({p.get('direction','?')}) @{p.get('timestamp','')[:16]}"
		for p in pats[-4:]
	)


def _fmt_indicators(ind: dict) -> str:
	if not ind:
		return "  No indicators computed"
	lines = []
	e50, e200 = ind.get("ema_50"), ind.get("ema_200")
	if e50:
		lines.append(f"	 EMA-50	 : {e50} | Price: {ind.get('price_vs_ema50','?')}")
	if e200:
		lines.append(
			f"	EMA-200 : {e200} | Price: {ind.get('price_vs_ema200','?')} "
			f"| Signal: {ind.get('ema_signal','?')}"
		)
	conf = ind.get("ema_confluence")
	if conf:
		lines.append(f"	 EMA Confluence : {conf}")
	rsi = ind.get("rsi_14")
	if rsi is not None:
		lines.append(f"	 RSI(14) : {rsi} → zone={ind.get('rsi_zone','?')}")
	_div = ind.get("rsi_divergence")
	if _div and _div != "NONE":
		lines.append(f"	 RSI DIVERGENCE : {_div} ({ind.get('rsi_divergence_detail','')})")
	atr = ind.get("atr_14")
	if atr is not None:
		atr_pct = ind.get("atr_pct") or 0
		lines.append(f"	 ATR(14) : {atr} ({atr_pct:.3f}%)")
	macd_state = ind.get("macd_state")
	if macd_state and macd_state != "INSUFFICIENT_DATA":
		lines.append(
			f"	MACD	: line={ind.get('macd_line')} signal={ind.get('macd_signal')} "
			f"hist={ind.get('macd_hist')} state={macd_state}"
		)
	return "\n".join(lines) if lines else "	 Insufficient data"


def _fmt_key_levels(kl: dict) -> str:
	if not kl:
		return "  No key levels (insufficient 1D/4H data)"
	lines: list[str] = []
	# Prefer the BLENDED 1D+4H pool (Step 2b) — the SAME source the scorer uses, so
	# LLM and pre-filter agree (LLM derives SL/TP from these MAJOR levels). Fall
	# back to the 4H pool, then legacy HTF levels.
	_blr = kl.get("blended_resistances")
	_bls = kl.get("blended_supports")
	if _blr is not None or _bls is not None:
		resistances = _blr or []
		supports	= _bls or []
	else:
		_MIN_T = 3
		_h4r = kl.get("h4_resistances")
		_h4s = kl.get("h4_supports")
		if _h4r is not None or _h4s is not None:
			resistances = [r for r in (_h4r or []) if r.get("touch_count", 0) >= _MIN_T]
			supports	= [s for s in (_h4s or []) if s.get("touch_count", 0) >= _MIN_T]
		else:
			resistances = kl.get("major_resistances", [])
			supports	= kl.get("major_supports", [])
	sr_flips	= kl.get("sr_flips", [])
	intra_res   = kl.get("intraday_resistances", [])
	intra_sup   = kl.get("intraday_supports", [])
	poc         = kl.get("poc_level")
	psych_above = kl.get("psychological_above", [])
	psych_below = kl.get("psychological_below", [])

	if resistances:
		lines.append("  MAJOR RESISTANCES (HTF 1D+4H, nearest first):")
		for r in resistances[:5]:
			flip_tag = "  [SR_FLIP: former support]" if r.get("label") == "SR_FLIP_RESISTANCE" else ""
			lines.append(
				f"    {r['price']:,.4f} | {r['touch_count']} touches "
				f"(SH:{r['sh_touches']} SL:{r['sl_touches']}){flip_tag}"
			)
	else:
		lines.append("  MAJOR RESISTANCES: not enough touches (need ≥3)")

	if supports:
		lines.append("  MAJOR SUPPORTS (HTF 1D+4H, nearest first):")
		for s in supports[:5]:
			flip_tag = "  [SR_FLIP: former resistance]" if s.get("label") == "SR_FLIP_SUPPORT" else ""
			lines.append(
				f"    {s['price']:,.4f} | {s['touch_count']} touches "
				f"(SH:{s['sh_touches']} SL:{s['sl_touches']}){flip_tag}"
			)
	else:
		lines.append("  MAJOR SUPPORTS: not enough touches (need ≥3)")

	if sr_flips:
		lines.append("  S/R FLIP ZONES (prime entry / TP candidates):")
		for flip in sr_flips[:4]:
			lines.append(
				f"    {flip['price']:,.4f} [{flip.get('label','')}] → "
				f"{flip.get('flip_desc', '')}"
			)

	# Intraday 1h levels (PR-2)
	if intra_res or intra_sup:
		lines.append("  INTRADAY LEVELS (1h cluster, tighter — use for SL fine-tuning):")
		for r in intra_res[:3]:
			lines.append(f"    Res {r['price']:,.4f} | {r['touch_count']} touches")
		for s in intra_sup[:3]:
			lines.append(f"    Sup {s['price']:,.4f} | {s['touch_count']} touches")

	# POC injection (PR-2)
	if poc:
		lines.append(
			f"  POC (Volume Profile 1D): {poc['price']:,.4f} → currently {poc['side']}"
		)

	# Value Area (G6.5)
	va = kl.get("value_area")
	if va and va.get("vah") and va.get("val"):
		lines.append(
			f"  VALUE AREA (1D volume profile, ~%{int((va.get('va_pct') or 0.70) * 100)} of volume): "
			f"VAL {va['val']:,.4f} — POC {va['poc']:,.4f} — VAH {va['vah']:,.4f} "
			f"(price magnets; strong TP candidates)"
		)

	# Psychological round numbers (PR-2)
	if psych_above or psych_below:
		above_str = ", ".join(f"{p:,.4f}" for p in psych_above) or "—"
		below_str = ", ".join(f"{p:,.4f}" for p in psych_below) or "—"
		lines.append(f"  PSYCHOLOGICAL ROUND NUMBERS:")
		lines.append(f"    Above: {above_str}")
		lines.append(f"    Below: {below_str}")

	return "\n".join(lines)


def _fmt_whale(cg: dict) -> str:
	if not cg:
		return "  No whale data"
	lines: list[str] = []

	poc = cg.get("poc_price")
	if poc:
		lines.append(f"	 POC (Volume Profile) : {poc:,.4f}")

	oi = cg.get("open_interest", {})
	if oi.get("comment", "No data") != "No data":
		lines.append(
			f"	OPEN INTEREST		: {oi.get('comment','?')} "
			f"| Δ4H={oi.get('change_4h_pct','?')}%"
		)
	else:
		lines.append("	OPEN INTEREST		: No data")

	fr = cg.get("funding_rate", {})
	if fr.get("rate_pct"):
		lines.append(f"	 FUNDING RATE		 : {fr.get('rate_pct','?')} → {fr.get('comment','?')}")
	else:
		lines.append("	FUNDING RATE		: No data")

	ls = cg.get("ls_ratio", {})
	if ls.get("comment", "No data") != "No data":
		lines.append(f"	 L/S RATIO			 : {ls.get('comment','?')}")
	else:
		lines.append("	L/S RATIO			: No data")

	liq_list = cg.get("liquidation_history", [])
	if liq_list:
		lines.append("	RECENT LIQUIDATIONS :")
		for liq in liq_list[:5]:
			price_str = f"${liq['price']:,.4f}" if liq.get("price") else "Aggregated"
			lines.append(
				f"	  {price_str} | {liq['side']} | "
				f"${liq['amount_usd']:,.0f} | {liq['time']}"
			)

	return "\n".join(lines)


def build_llm_prompt(payload: dict, lessons_path: str = "lessons_learned.json") -> str:
	"""
	Composes the user-prompt sent to the LLM (system prompt is injected separately
	via the SDK). Order: Lessons → MTF Analysis → Major S/R → Whale Sentiment → Final ask.
	"""
	meta = payload["meta"]
	tda	 = payload["top_down_analysis"]
	kl	 = payload.get("key_levels", {})
	cg	 = payload.get("whale_data", {})
	cp	 = meta["current_price"]
	sym	 = meta["symbol"]
	ts	 = meta["timestamp_utc"]

	recent_lessons = _load_recent_lessons(lessons_path, n=5)
	if recent_lessons:
		lesson_items = "\n".join(
			f"	{i + 1}. {lesson}" for i, lesson in enumerate(recent_lessons)
		)
		lessons_section = (
			f"\n## LESSONS FROM RECENT CLOSED TRADES (last {len(recent_lessons)})\n"
			"Do not repeat these mistakes.\n\n"
			f"{lesson_items}\n"
		)
	else:
		lessons_section = ""

	_now_utc = datetime.now(timezone.utc)   # for broker-TZ-independent event killzone stamps

	tf_blocks: list[str] = []
	for tf in ["1d", "4h", "1h", "15m"]:
		data = tda.get(tf, {})
		if "error" in data:
			tf_blocks.append(
				f"\n### {tf.upper()} — {_TF_ROLES.get(tf, tf)}\n  ERROR: {data['error']}"
			)
			continue
		block = (
			f"\n### {tf.upper()} — {_TF_ROLES.get(tf, tf)}\n"
			f"- Trend           : {data.get('trend','N/A')}\n"
			f"- Last closed bar : O={data.get('last_open','?')} H={data.get('last_high','?')} "
			f"L={data.get('last_low','?')} C={data.get('last_close','?')} → "
			f"{data.get('last_candle_direction','?')} ({data.get('last_timestamp','')[:16]})\n"
			f"- BOS / CHoCH     : {_fmt_events(data.get('structure_events',[]), tf, _now_utc)}\n"
			f"- Unmitigated FVG : {_fmt_fvgs(data.get('unmitigated_fvgs',[]))}\n"
			f"- Order Blocks    : {_fmt_obs(data.get('order_blocks',[]))}\n"
			f"- Patterns        : {_fmt_patterns(data.get('candlestick_patterns',[]))}\n"
			f"- Indicators      :\n{_fmt_indicators(data.get('indicators',{}))}"
		)
		tf_blocks.append(block)

	# G6.2 — Dealing-range satiri (premium/discount konum yasagi icin veri).
	# Scorer'daki AYNI hesap: LLM ile deterministik veto ayni sayiyi gorur.
	dr_line = ""
	if _dealing_range is not None:
		try:
			_dr = _dealing_range(tda)
			if _dr:
				_lo, _hi, _pos = _dr
				_zone = ("PREMIUM (no LONG)" if _pos >= 0.62
						 else "DISCOUNT (no SHORT)" if _pos <= 0.38 else "EQUILIBRIUM")
				dr_line = (f"DEALING RANGE (4H): low {_lo:,.4f} — high {_hi:,.4f} | "
						   f"position: {_pos*100:.0f}% → {_zone}\n")
		except Exception:
			dr_line = ""

	return (
		f"# {sym} — Swing Trade Analysis Brief\n"
		f"Current price: {cp:,.6f} | UTC: {ts}\n"
		+ dr_line
		+ lessons_section
		+ "\n## 1. TOP-DOWN MULTI-TIMEFRAME ANALYSIS\n"
		+ "".join(tf_blocks)
		+ "\n\n## 2. MAJOR HISTORICAL SUPPORT / RESISTANCE (1D + 4H clustered)\n"
		+ _fmt_key_levels(kl)
		+ "\n\n## 3. WHALE SENTIMENT (CoinGlass — OI / FR / L/S Ratio / Liquidations)\n"
		+ _fmt_whale(cg)
		+ "\n\n## 4. SCORING AND DECISION\n\n"
		+ "Apply the dual-sided scoring matrix exactly as specified in your\n"
		+ "system instructions. Compute long_total and short_total step by step\n"
		+ "and populate `calculation_breakdown` with the full math. Construct\n"
		+ "SL/TP by following STEPS 4a–4f in your system instructions verbatim\n"
		+ "(1% structural buffer, 1.5% floor, max-SL ceiling per STEP 4d, RR ≥ 2.0).\n"
		+ "Set `entry_price` per Step 6 (current_price × 0.998 for LONG, × 1.002\n"
		+ "for SHORT — limit-order requirement).\n\n"
		+ "Return JSON only.\n"
	)


# ─────────────────────────────────────────────────────────────────
# Strict JSON parser (handles markdown fences + nested braces)
# ─────────────────────────────────────────────────────────────────

def _parse_llm_response(text: str, required_key: str = "decision") -> dict:
	"""
	Extracts the first JSON object that contains `required_key`. Falls back to
	the last successfully-parsed JSON if no object had the key.
	"""
	cleaned = re.sub(r"```(?:json)?\s*", "", text).replace("```", "").strip()
	decoder = json.JSONDecoder()
	idx		= 0
	last_valid: Optional[dict] = None

	while idx < len(cleaned):
		brace = cleaned.find("{", idx)
		if brace == -1:
			break
		try:
			obj, end = decoder.raw_decode(cleaned[brace:])
			if isinstance(obj, dict):
				last_valid = obj
				if required_key in obj:
					return obj
			idx = brace + end
		except json.JSONDecodeError:
			idx = brace + 1

	if last_valid is not None:
		return last_valid
	raise ValueError(f"No valid JSON found. Response (first 300 chars): {text[:300]}")


# Per-sub-score ceilings — used to clamp each component before summing, so an
# AI that inflates a single component cannot push the total over threshold.
_SUBSCORE_CAPS: dict[str, int] = {
	"macro_trend":		 25,   # A
	"momentum_rsi":		 10,   # B.1
	"momentum_ema":		 10,   # B.2
	"momentum_macd":	  5,   # B.3
	"sr_proximity":		 15,   # C.1
	"sr_quality":		  5,   # C.2
	"structural_events":  5,   # C.3
	"ls_ratio":			 10,   # D.1
	"funding_rate":		  8,   # D.2
	"oi_behavior":		  7,   # D.3
}   # sum of caps = 100


def _sum_breakdown(bd: dict) -> int:
	"""
	Sum the 10 sub-scores from a breakdown dict, clamping EACH to its ceiling.
	This is the anti-hallucination core: the score the bot acts on is computed
	HERE in Python from the itemised components — never read from the AI's own
	`long_total` / `short_total` (which it can miscompute or inflate).
	"""
	total = 0
	for key, cap in _SUBSCORE_CAPS.items():
		try:
			v = int(round(float(bd.get(key, 0) or 0)))
		except (TypeError, ValueError):
			v = 0
		total += max(0, min(v, cap))   # clamp to [0, cap]
	return total


def _apply_decision_postprocess(
	decision: dict,
	cfg: BotConfig,
	current_price: float = 0.0,
	tp_cap: float | None = None,
) -> dict:
	"""
	Final guardrails applied on top of the AI's output. ALL constraints below
	can force the decision to "WAIT". They never silently auto-fix:

	  * Force WAIT if score < min_score.
	  * Force WAIT if RR < min_rr_ratio.
	  * Force WAIT if SL distance < min_sl_pct (anti-stop-hunt floor).
	  * Force WAIT if SL distance > max_sl_pct.
	  * Hard-clamp position_usd to the correct tier ($10k or $20k) regardless
		of what the AI wrote (defense against hallucination).
	"""
	d = dict(decision)

	# 0. RECOMPUTE TOTALS IN PYTHON (anti-hallucination).
	#    Ignore the AI's long_total/short_total entirely; sum the itemised
	#    sub-scores ourselves with per-component clamping. The score every
	#    downstream gate uses is THIS value, not the model's arithmetic.
	lb = d.get("long_breakdown", {})  or {}
	sb = d.get("short_breakdown", {}) or {}
	py_long  = _sum_breakdown(lb)
	py_short = _sum_breakdown(sb)
	ai_long  = int(d.get("long_total",  0) or 0)
	ai_short = int(d.get("short_total", 0) or 0)
	if ai_long != py_long or ai_short != py_short:
		logger.warning(
			"[POSTPROCESS] AI totals (L=%d S=%d) ≠ Python recompute (L=%d S=%d) "
			"— trusting Python (sub-score sum).",
			ai_long, ai_short, py_long, py_short,
		)
	d["long_total"]  = py_long
	d["short_total"] = py_short

	side	  = str(d.get("decision", "WAIT")).upper()
	long_tot  = py_long
	short_tot = py_short
	score	  = long_tot if side == "LONG" else (short_tot if side == "SHORT" else 0)

	# 1. Threshold gate
	if score < cfg.min_score:
		if side != "WAIT":
			logger.warning(
				"[POSTPROCESS] Score %d < %d → forcing WAIT (AI said %s).",
				score, cfg.min_score, side,
			)
		d["decision"]	  = "WAIT"
		d["position_usd"] = 0.0
		return d

	# 2. RR gate
	try:
		rr = float(d.get("rr_ratio", 0) or 0)
	except (TypeError, ValueError):
		rr = 0.0
	if rr < cfg.min_rr_ratio and side != "WAIT":
		logger.warning("[POSTPROCESS] RR %.2f < %.1f → forcing WAIT", rr, cfg.min_rr_ratio)
		d["decision"]	  = "WAIT"
		d["position_usd"] = 0.0
		return d

	# 3. SL distance band gates
	try:
		entry = float(d.get("entry_price", 0) or 0)
		sl	  = float(d.get("sl_price", 0) or 0)
	except (TypeError, ValueError):
		entry, sl = 0.0, 0.0

	if side in ("LONG", "SHORT"):
		if entry <= 0 or sl <= 0:
			logger.warning("[POSTPROCESS] Missing/invalid entry or SL → forcing WAIT")
			d["decision"]	  = "WAIT"
			d["position_usd"] = 0.0
			return d

		# EDGE #1b: SL/TP sidedness. MT5 zaten yanlis-tarafli stopu reddeder, ama
		# burada yakalayip temiz bir WAIT verelim (order-rejection hatasi yerine).
		# LONG -> stop entry ALTINDA, hedef USTUNDE; SHORT -> tersi.
		try:
			tp_chk = float(d.get("tp_price", 0) or 0)
		except (TypeError, ValueError):
			tp_chk = 0.0
		if tp_chk <= 0:
			logger.warning("[POSTPROCESS] Missing/invalid TP -> forcing WAIT")
			d["decision"] = "WAIT"; d["position_usd"] = 0.0
			return d
		if side == "LONG" and not (sl < entry < tp_chk):
			logger.warning("[POSTPROCESS] LONG sidedness invalid (need SL<entry<TP; "
			               "got SL=%.6f entry=%.6f TP=%.6f) -> forcing WAIT", sl, entry, tp_chk)
			d["decision"] = "WAIT"; d["position_usd"] = 0.0
			return d
		if side == "SHORT" and not (tp_chk < entry < sl):
			logger.warning("[POSTPROCESS] SHORT sidedness invalid (need TP<entry<SL; "
			               "got TP=%.6f entry=%.6f SL=%.6f) -> forcing WAIT", tp_chk, entry, sl)
			d["decision"] = "WAIT"; d["position_usd"] = 0.0
			return d

		# RR UST SINIRI (cap): RR > max_rr ise uzak TP yerine max_rr mesafesine cek.
		# Backtest ~3 RR en iyi getiriyi veriyor; scale-out yol boyu zaten topluyor.
		_maxrr = tp_cap if tp_cap is not None else cfg.max_rr_ratio	# per-symbol tavan (C-konfig)
		_risk = abs(entry - sl)
		if _risk > 0:
			_cur_rr = abs(tp_chk - entry) / _risk
			if _cur_rr > _maxrr:
				_tp_cap = (entry + _maxrr * _risk) if side == "LONG" else (entry - _maxrr * _risk)
				logger.info("[POSTPROCESS] RR %.2f > max %.1f -> TP %.6f -> %.6f (cap)",
				            _cur_rr, _maxrr, tp_chk, _tp_cap)
				d["tp_price"] = _tp_cap
				d["rr_ratio"] = _maxrr
				tp_chk = _tp_cap
		sl_pct = abs(entry - sl) / entry

		if sl_pct < cfg.min_sl_pct - 1e-4:	# 1e-4 = 2dp yuvarlama toleransi (%1.4999 vakasi)
			logger.warning(
				"[POSTPROCESS] SL distance %.3f%% < min %.3f%% → forcing WAIT "
				"(AI failed to apply the 1.5%% stop-hunt floor).",
				sl_pct * 100, cfg.min_sl_pct * 100,
			)
			d["decision"]	  = "WAIT"
			d["position_usd"] = 0.0
			return d

		if sl_pct > cfg.max_sl_pct:
			logger.warning(
				"[POSTPROCESS] SL distance %.3f%% > max %.3f%% → forcing WAIT",
				sl_pct * 100, cfg.max_sl_pct * 100,
			)
			d["decision"]	  = "WAIT"
			d["position_usd"] = 0.0
			return d

		# EDGE #1: entry_price sanity vs live price. The AI is told to place the
		# limit at current×0.998 (LONG) / current×1.002 (SHORT). If it hallucinates
		# an entry on the WRONG side of price (or absurdly far), a BUY_LIMIT above
		# market fills like a market order and defeats the pullback logic. Reject
		# anything not sitting just below (LONG) / just above (SHORT) current, within
		# ~1.5%. Skipped when current_price is unknown (0).
		if current_price > 0:
			if side == "LONG" and not (current_price * 0.985 <= entry <= current_price * 1.0005):
				logger.warning(
					"[POSTPROCESS] LONG entry %.6f invalid vs current %.6f "
					"(must be just below price) → forcing WAIT", entry, current_price,
				)
				d["decision"] = "WAIT"; d["position_usd"] = 0.0
				return d
			if side == "SHORT" and not (current_price * 0.9995 <= entry <= current_price * 1.015):
				logger.warning(
					"[POSTPROCESS] SHORT entry %.6f invalid vs current %.6f "
					"(must be just above price) → forcing WAIT", entry, current_price,
				)
				d["decision"] = "WAIT"; d["position_usd"] = 0.0
				return d

	# EDGE #2: Macro veto. If the 1D/4H macro pillar is strongly one-sided
	# (macro_trend ≥ 22 on the opposite column) yet the trade is counter-trend,
	# the score is almost entirely LTF-driven — exactly the "fade a freight train"
	# setup a swing strategy should refuse. Force WAIT.
	if side in ("LONG", "SHORT"):
		macro_long  = max(0, min(int(round(float(lb.get("macro_trend", 0) or 0))), 25))
		macro_short = max(0, min(int(round(float(sb.get("macro_trend", 0) or 0))), 25))
		if side == "LONG" and macro_short >= 22 and macro_long < 5:
			logger.warning(
				"[POSTPROCESS] Macro strongly BEARISH (A_short=%d, A_long=%d) but "
				"decision=LONG → counter-trend veto → forcing WAIT.",
				macro_short, macro_long,
			)
			d["decision"] = "WAIT"; d["position_usd"] = 0.0
			return d
		if side == "SHORT" and macro_long >= 22 and macro_short < 5:
			logger.warning(
				"[POSTPROCESS] Macro strongly BULLISH (A_long=%d, A_short=%d) but "
				"decision=SHORT → counter-trend veto → forcing WAIT.",
				macro_long, macro_short,
			)
			d["decision"] = "WAIT"; d["position_usd"] = 0.0
			return d

	# 4. Position-size tier clamp (anti-hallucination)
	if side in ("LONG", "SHORT"):
		# Tier by score band, anchored to 80 (NOT to min_score). This stays correct
		# no matter where min_score sits: anything from min_score..80 → small tier,
		# 81..100 → large tier. (If this were "72 <= score <= 80", lowering min_score
		# to 65 would drop 65-71 scores into the else-branch and hand the WEAKEST
		# setups the LARGEST position — the opposite of intended.)
		if score <= 80:
			expected_usd = cfg.pos_usd_75_80
		else:  # 81–100
			expected_usd = cfg.pos_usd_81_100

		try:
			ai_usd = float(d.get("position_usd", 0) or 0)
		except (TypeError, ValueError):
			ai_usd = 0.0

		if abs(ai_usd - expected_usd) > 0.01:
			if ai_usd > 0:
				logger.warning(
					"[POSTPROCESS] AI returned position_usd=$%.0f for score=%d; "
					"overriding to tier value $%.0f.",
					ai_usd, score, expected_usd,
				)
			d["position_usd"] = expected_usd

	return d
# ─────────────────────────────────────────────────────────────────
# AlgoBot — Main Trading Engine
# ─────────────────────────────────────────────────────────────────

class AlgoBot:
	"""
	Orchestrates the swing-trade pipeline:
	  * Periodic per-symbol scoring (15-minute cadence by default)
	  * Continuous active-position management (every 60 seconds)
	  * Hourly closed-position post-mortem
	  * Kill-switch and risk gating
	"""

	def __init__(self, config: BotConfig) -> None:
		self.cfg	  = config
		self.analyzer = TechnicalAnalyzer(swing_order=5)
		self.risk_mgr = RiskManager(cfg=config)

		# Per-ticket trade-management state.
		# Shape: {ticket: {"initial_volume": float, "stage_50_done": bool,
		#				   "stage_75_done": bool, "symbol": str, "data_sym": str}}
		self._trade_state: dict[int, dict] = {}
		self._load_trade_state()   # restart'ta disk varsa EXACT state geri yuklenir

		# G3 — per-symbol re-entry guard state (post-stop cooldown, level lockout,
		# consecutive-loss halt). Persisted to disk so a restart can't wipe the
		# memory that stops the bot re-shorting the zone it just got stopped in.
		self._guard_state: dict[str, dict] = {}
		self._load_guard_state()

		self._last_ai: dict[str, float] = {sym: 0.0 for sym in config.data_symbols}
		self._last_history_check: float = 0.0

		# Symbol mapping mt5 → data, used by trade-management AI consult.
		self._mt5_to_data: dict[str, str] = {
			m: d for m, d in zip(config.mt5_symbols, config.data_symbols)
		}

		# G6.6 — Haber karantinasi: STATIK dosyadan BIR KEZ yukle (API/AI yok).
		self._news_blackouts: list[tuple[datetime, datetime, str]] = []
		if config.news_blackout_enabled:
			self._news_blackouts = _load_news_blackouts(config.news_blackout_path)
			_future = [b for b in self._news_blackouts
			           if b[1] > datetime.now(timezone.utc)]
			logger.info("[NEWS] Karantina yuklendi: %d kayit (%d gelecekte).%s",
			            len(self._news_blackouts), len(_future),
			            f" Siradaki: {_future[0][2]} @ {_future[0][0]:%Y-%m-%d %H:%M} UTC" if _future else "")
			if not self._news_blackouts:
				logger.warning("[NEWS] %s bos/yok — haber karantinasi fiilen devre disi.",
				               config.news_blackout_path)

		# Claude (Anthropic) Setup
		self._ai_client = AsyncAnthropic(api_key=config.anthropic_api_key)

	def _news_blackout_active(self, now: Optional[datetime] = None) -> tuple[bool, str]:
		"""G6.6 — su an (ya da lead penceresi icinde) bir haber karantinasi var mi?
		lead: pencere baslamadan news_blackout_lead_sec once de yeni emir yok —
		4h omurlu limit emri haberin icine dolmasin."""
		if not self.cfg.news_blackout_enabled or not self._news_blackouts:
			return False, ""
		now = now or datetime.now(timezone.utc)
		lead = timedelta(seconds=self.cfg.news_blackout_lead_sec)
		for start, end, name in self._news_blackouts:
			if start - lead <= now < end:
				if now < start:
					return True, f"{name} — {start:%Y-%m-%d %H:%M} UTC'de basliyor (lead {lead})"
				return True, f"{name} karantinasi ({start:%Y-%m-%d %H:%M} → {end:%Y-%m-%d %H:%M} UTC)"
		return False, ""

	# ── Claude Queries ──────────────────────────────────────────────────

	async def _query_llm(self, payload: dict, data_sym: str) -> dict:
		prompt = build_llm_prompt(payload, lessons_path=self.cfg.lessons_path)
		logger.info("[LLM][%s] Request: %d chars", data_sym, len(prompt))

		response = await self._ai_client.messages.create(
			model=self.cfg.llm_model,
			max_tokens=2048,
			temperature=0.0,  # BUZ GİBİ MAKİNE! SIFIR DUYGU!
			system=_SYSTEM_PROMPT
				.replace("__TP_CAP__", f"{_tp_cap_for(data_sym):.1f}")
				.replace("__MIN_SCORE__", str(self.cfg.min_score))
				.replace("__MAX_SL_FRAC__", f"{self.cfg.max_sl_pct:g}")
				.replace("__MAX_SL__", f"{self.cfg.max_sl_pct * 100:.1f}"),
			messages=[{"role": "user", "content": prompt}],
		)

		raw		 = _parse_llm_response(response.content[0].text, required_key="decision")
		_cp		 = float(payload.get("meta", {}).get("current_price", 0) or 0)
		decision = _apply_decision_postprocess(raw, self.cfg, current_price=_cp, tp_cap=_tp_cap_for(data_sym))

		long_tot  = decision.get("long_total",	0)
		short_tot = decision.get("short_total", 0)
		pos_usd	  = float(decision.get("position_usd", 0) or 0)
		logger.info(
			"[LLM][%s] Decision=%-6s | L=%d/100 S=%d/100 | RR=%.2f | pos_usd=$%.0f",
			data_sym,
			decision.get("decision"),
			long_tot, short_tot,
			float(decision.get("rr_ratio", 0) or 0),
			pos_usd,
		)
		if decision.get("calculation_breakdown"):
			logger.info("[LLM][%s] Math: %s", data_sym, decision["calculation_breakdown"])
		return decision

	async def _query_trade_evaluation(self, payload_text: str, mt5_sym: str) -> dict:
		logger.info("[LLM-MGMT][%s] Stage-2 consultation request", mt5_sym)
		response = await self._ai_client.messages.create(
			model=self.cfg.llm_model,
			max_tokens=512,
			temperature=0.0,
			system=_TRADE_EVALUATION_PROMPT,
			messages=[{"role": "user", "content": payload_text}]
		)
		result	 = _parse_llm_response(response.content[0].text, required_key="action")
		action	 = str(result.get("action", "HOLD_REST")).upper()
		if action not in ("CLOSE_ALL", "HOLD_REST"):
			logger.warning("[LLM-MGMT][%s] Unknown action '%s' → defaulting to HOLD_REST",
						   mt5_sym, action)
			action = "HOLD_REST"
		return {"action": action, "reason": result.get("reason", "")}

	# ── Post-mortem ─────────────────────────────────────────────────────

	async def _run_post_mortem(self, entry: dict, outcome: str) -> None:
		snap   = entry.get("decision_snapshot", {})
		result = "STOP (loss)" if outcome == "STOP" else "TP (profit)"

		user_prompt = (
			f"A swing trade has closed. Provide the post-mortem analysis.\n\n"
			f"Symbol		   : {entry.get('data_sym')} ({entry.get('mt5_sym')})\n"
			f"Direction		   : {snap.get('decision', '?')}\n"
			f"Outcome		   : {result}\n"
			f"Long Total	   : {snap.get('long_total', '?')} / 100\n"
			f"Short Total	   : {snap.get('short_total', '?')} / 100\n"
			f"Rationale at entry: {snap.get('rationale', '')}\n"
			f"Calc breakdown   : {snap.get('calculation_breakdown', '')}\n"
			f"Key Levels	   : {snap.get('key_levels', [])}\n"
			f"Entry Price	   : {entry.get('entry_price', '?')}\n"
			f"Stop-Loss		   : {entry.get('sl_price', '?')}\n"
			f"Take-Profit	   : {entry.get('tp_price', '?')}\n"
			f"Close Price	   : {entry.get('close_price', 'unknown')}\n"
			f"PnL			   : {entry.get('pnl', 'unknown')} $\n\n"
			"Distill a single concrete, actionable lesson (1–2 sentences). "
			"Reply JSON only, exactly: "
			'{"lesson": "...", "category": "STOP" or "TP", "importance": "HIGH"|"MEDIUM"|"LOW"}'
		)

		try:
			response = await self._ai_client.messages.create(
				model=self.cfg.llm_model,
				max_tokens=256,
				temperature=0.2, # Ders çıkarırken azıcık yorum katabilir
				system=_POST_MORTEM_SYSTEM_PROMPT,
				messages=[{"role": "user", "content": user_prompt}]
			)
			lesson_data = _parse_llm_response(response.content[0].text, required_key="lesson")

			lessons = _load_json_file(self.cfg.lessons_path, [])
			lessons.append({
				"ticket_id":   entry.get("ticket_id"),
				"data_sym":	   entry.get("data_sym"),
				"outcome":	   outcome,
				"timestamp":   datetime.now(timezone.utc).isoformat(),
				"lesson":	   lesson_data.get("lesson", ""),
				"category":	   lesson_data.get("category", outcome),
				"importance":  lesson_data.get("importance", "MEDIUM"),
				"entry_snapshot": {
					"long_total":  snap.get("long_total"),
					"short_total": snap.get("short_total"),
					"decision":	   snap.get("decision"),
					"entry":	   entry.get("entry_price"),
					"sl":		   entry.get("sl_price"),
					"tp":		   entry.get("tp_price"),
					"close":	   entry.get("close_price"),
					"pnl":		   entry.get("pnl"),
				},
			})
			_save_json_file(self.cfg.lessons_path, lessons)
			logger.info(
				"[POST-MORTEM] Lesson saved | %s %s | importance=%s",
				entry.get("data_sym"), outcome, lesson_data.get("importance", "?"),
			)
			logger.info("[POST-MORTEM] %s", lesson_data.get("lesson", "")[:140])
		except Exception as exc:
			logger.error("[POST-MORTEM] Analysis error: %s", exc)

	# ── Closed-position scan ────────────────────────────────────────────

	async def _check_closed_positions(self) -> None:
		if not _MT5_AVAILABLE or self.cfg.dry_run:
			return

		journal	 = _load_json_file(self.cfg.journal_path, [])
		to_check = [e for e in journal if e.get("status") in ("PENDING", "ACTIVE")]
		if not to_check:
			return

		logger.info("[HISTORY] Reviewing %d open entries...", len(to_check))
		changed = False

		for entry in to_check:
			ticket_str = str(entry.get("ticket_id", "0"))
			if ticket_str.startswith("DRY-"):
				continue
			try:
				ticket = int(ticket_str)
			except (ValueError, TypeError):
				continue

			pending_orders = await asyncio.to_thread(mt5.orders_get, ticket=ticket)
			if pending_orders and len(pending_orders) > 0:
				continue

			hist_orders = await asyncio.to_thread(mt5.history_orders_get, ticket=ticket)
			if not hist_orders or len(hist_orders) == 0:
				continue

			hist_order = hist_orders[0]

			if hist_order.state in (mt5.ORDER_STATE_CANCELED, mt5.ORDER_STATE_EXPIRED):
				entry["status"]		= "CANCELLED"
				entry["close_time"] = datetime.now(timezone.utc).isoformat()
				changed = True
				logger.info("[HISTORY] ticket=%s cancelled.", ticket_str)
				continue

			position_id = hist_order.position_id
			if position_id == 0:
				continue

			if entry["status"] == "PENDING":
				entry["status"] = "ACTIVE"
				changed = True
				logger.info("[HISTORY] ticket=%s now ACTIVE (pos=%d).", ticket_str, position_id)

			open_positions = await asyncio.to_thread(mt5.positions_get, ticket=position_id)
			if open_positions and len(open_positions) > 0:
				continue

			closing_deals = await asyncio.to_thread(
				mt5.history_deals_get, position=position_id
			)
			if not closing_deals:
				continue

			out_deals = [d for d in closing_deals if d.entry == mt5.DEAL_ENTRY_OUT]
			if not out_deals:
				continue

			closing_deal = out_deals[-1]
			# PnL = TUM cikis deal'lerinin toplami. Scale-out'lu pozisyonda tek tek
			# kademeler ayri OUT deal'leridir; eskiden sadece SON deal'in kari
			# yaziliyordu (ornegin %75'i karda kapatilip kalan BE-stop'a gelen islem
			# journal'a ~0$ olarak geciyordu).
			total_pnl = float(sum(d.profit for d in out_deals))
			if closing_deal.reason == mt5.DEAL_REASON_SL:
				outcome = "STOP"
			elif closing_deal.reason == mt5.DEAL_REASON_TP:
				outcome = "TP"
			else:
				outcome = "TP" if total_pnl >= 0 else "STOP"

			entry["status"]		 = outcome
			entry["close_price"] = float(closing_deal.price)
			entry["close_time"]	 = datetime.fromtimestamp(
				closing_deal.time, tz=timezone.utc
			).isoformat()
			entry["pnl"] = total_pnl
			changed = True
			logger.info(
				"[HISTORY] ticket=%s %s | close=%.4f | PnL=$%.2f (%d cikis deal'i)",
				ticket_str, outcome, closing_deal.price, total_pnl, len(out_deals),
			)
			# G3 — feed the re-entry guard: STOP arms cooldown/streak halt for that
			# symbol+side; TP clears that side's penalties.
			#
			# BREAKEVEN-STOP (SL'e degdi ama scale-out sayesinde toplam PnL >= 0):
			# 2026-08-01'de olculdu — bu durumu "cezasiz" (NEUTRAL) birakmak PAHALIYA
			# mal oluyor. e4 backtest'i, tek kosu ICINDE (kaskad kirlenmesi YOK):
			#   BE cikisindan sonra 6 saat icinde AYNI yon + AYNI seviye (±%1.2)
			#   bandina yeniden giris = 14 islem, ort -0.527R, %79 stop
			#   digerleri = 271 islem, ort +0.164R, %49 stop  -> fark -0.69R, t=-2.44
			#   sonumleme: 6h -0.53R | 24h -0.34R | 72h -0.02R (normale doner)
			#   kontrol: d6'da (cooldown normal kuruluyor) ayni desen sadece 1 islem.
			# Cikarim: cooldown bir CEZA degil, "bu seviye su an cekismeli, hemen geri
			# donme" kuralidir — kar/zarar meselesi degil, SEVIYE meselesi.
			# Bu yuzden BE-stop: cooldown + seviye-kilidi KURULUR, ama ardisik-stop
			# sayacina EKLENMEZ (para kaybedilmedi, tez curumus degil).
			try:
				_gside = str((entry.get("decision_snapshot", {}) or {}).get("decision", "")).upper()
				guard_outcome = outcome
				if outcome == "STOP" and total_pnl >= 0:
					guard_outcome = "BE_STOP"
					logger.info("[GUARD] ticket=%s BE-stop (PnL=$%.2f >= 0) -> seviye kilidi "
					            "kuruluyor, ardisik-stop sayaci ARTMIYOR.", ticket_str, total_pnl)
				self._register_outcome(
					data_sym=entry.get("data_sym", entry.get("mt5_sym", "?")),
					side=_gside, outcome=guard_outcome,
					entry_price=float(entry.get("entry_price", 0) or 0),
				)
			except Exception as _exc:
				logger.warning("[GUARD] outcome kaydi basarisiz: %s", _exc)
			await self._run_post_mortem(entry, outcome)

		if changed:
			_save_json_file(self.cfg.journal_path, journal)
			logger.info("[HISTORY] Journal updated.")

	# ── Trade Management Engine ─────────────────────────────────────────

	async def trade_management_loop(self, cg_manager: CoinglassManager) -> None:
		"""
		Background coroutine: every 60 s, scan open positions for stage 1
		(50% profit progress) and stage 2 (75% progress, AI consult) actions.
		"""
		logger.info("[TRADE-MGMT] Loop started (interval=%ds).", self.cfg.trade_mgmt_interval_sec)
		while True:
			try:
				await self._manage_active_positions(cg_manager)
			except asyncio.CancelledError:
				raise
			except Exception as exc:
				logger.exception("[TRADE-MGMT] Iteration error: %s", exc)
			await asyncio.sleep(self.cfg.trade_mgmt_interval_sec)

	def _load_trade_state(self) -> None:
		"""Restart'ta scale-out state'ini diskten geri yukle (RAM silininca EXACT kurtarma)."""
		import os
		path = self.cfg.trade_state_path
		if not os.path.exists(path):
			return
		try:
			with open(path, "r", encoding="utf-8") as f:
				raw = json.load(f)
		except Exception as exc:
			logger.warning("[TRADE-MGMT] State dosyasi okunamadi (%s): %s — sifirdan.", path, exc)
			return
		if not isinstance(raw, dict):
			return
		for k, v in raw.items():
			try:
				self._trade_state[int(k)] = v
			except (ValueError, TypeError):
				continue
		if self._trade_state:
			logger.info("[TRADE-MGMT] Diskten %d pozisyon state'i geri yuklendi: %s",
			            len(self._trade_state), list(self._trade_state.keys()))

	def _save_trade_state(self) -> None:
		"""Scale-out + tukenme state'ini diske ATOMIK yaz (sadece kalici skaler alanlar)."""
		import os
		path = self.cfg.trade_state_path
		_KEEP = ("initial_volume", "levels_done", "be_done", "consult_done",
		         "consult_action", "stage_50_done", "stage_75_done",
		         "exh_last_ts",
		         "exh_cooldown_until", "exh_close_pending", "symbol", "data_sym")
		try:
			out = {}
			for ticket, st in self._trade_state.items():
				out[str(ticket)] = {k: st[k] for k in _KEEP if k in st}
			tmp = path + ".tmp"
			with open(tmp, "w", encoding="utf-8") as f:
				json.dump(out, f, indent=2, default=str)
			os.replace(tmp, path)   # atomik (yarim yazma yok)
		except Exception as exc:
			logger.warning("[TRADE-MGMT] State diske yazilamadi: %s", exc)

	# ── G3: Re-entry guard state (cooldown / level-lockout / consecutive-stop halt) ──
	def _load_guard_state(self) -> None:
		"""Restart'ta cooldown/streak guard state'ini diskten geri yukle."""
		import os
		path = self.cfg.guard_state_path
		if not os.path.exists(path):
			return
		try:
			with open(path, "r", encoding="utf-8") as f:
				raw = json.load(f)
			if isinstance(raw, dict):
				self._guard_state = raw
				logger.info("[GUARD] Diskten %d sembol guard state'i geri yuklendi.", len(raw))
		except Exception as exc:
			logger.warning("[GUARD] State dosyasi okunamadi (%s): %s — sifirdan.", path, exc)

	def _save_guard_state(self) -> None:
		"""Guard state'i diske ATOMIK yaz."""
		import os
		path = self.cfg.guard_state_path
		try:
			tmp = path + ".tmp"
			with open(tmp, "w", encoding="utf-8") as f:
				json.dump(self._guard_state, f, indent=2, default=str)
			os.replace(tmp, path)
		except Exception as exc:
			logger.warning("[GUARD] State diske yazilamadi: %s", exc)

	def _gs_for(self, data_sym: str) -> dict:
		"""Sembol icin guard state kaydini getir/olustur (varsayilan iskelet)."""
		gs = self._guard_state.get(data_sym)
		if gs is None:
			gs = {"cooldown_until": {}, "consec_stops": {}, "halt_until": {},
			      "stopped_levels": [], "be_cooldown_until": {}}
			self._guard_state[data_sym] = gs
		# defensive: eski/eksik anahtarlari tamamla
		gs.setdefault("cooldown_until", {})
		gs.setdefault("consec_stops", {})
		gs.setdefault("halt_until", {})
		gs.setdefault("stopped_levels", [])
		gs.setdefault("be_cooldown_until", {})   # BE-stop: sadece seviye bandi kilidi
		return gs

	def _register_outcome(self, data_sym: str, side: str, outcome: str,
	                      entry_price: float) -> None:
		"""G3 — bir islem kapandiginda guard state'i guncelle.
		  STOP    : cooldown + seviye kaydi + ardisik-stop sayaci (esik asilirsa askiya alma)
		  BE_STOP : cooldown + seviye kaydi, ardisik-stop sayaci ARTMAZ (kayip yok)
		  TP      : o yonun tum cezalarini temizle (kazanc, tezi dogruladi)
		BE_STOP ayrimi 2026-08-01'de olculdu: BE'den sonra 6 saat icinde ayni seviyeye
		donen girisler ort -0.53R / %79 stop (digerleri +0.16R / %49). Bkz. cagiran
		taraftaki uzun not."""
		side = str(side).upper()
		if side not in ("LONG", "SHORT"):
			return
		gs = self._gs_for(data_sym)
		now = time.time()
		if outcome == "BE_STOP":
			# SADECE seviye bandi kilidi (tam yasak DEGIL — gercek stop'tan farki bu).
			# Ardisik-stop sayacina DOKUNULMAZ (ne artar ne sifirlanir).
			gs["be_cooldown_until"][side] = now + self.cfg.post_stop_cooldown_sec
			if entry_price and entry_price > 0:
				gs["stopped_levels"].append({"price": float(entry_price), "side": side, "ts": now})
				gs["stopped_levels"] = gs["stopped_levels"][-12:]
			logger.info("[GUARD] %s %s BE-stop -> %.1fh SEVIYE kilidi (±%.1f%%), tam yasak yok, "
			            "ardisik sayac=%d (degismedi).",
			            data_sym, side, self.cfg.post_stop_cooldown_sec / 3600,
			            self.cfg.relevel_lockout_pct * 100,
			            int(gs["consec_stops"].get(side, 0)))
		elif outcome == "STOP":
			gs["cooldown_until"][side] = now + self.cfg.post_stop_cooldown_sec
			gs["consec_stops"][side] = int(gs["consec_stops"].get(side, 0)) + 1
			if entry_price and entry_price > 0:
				gs["stopped_levels"].append({"price": float(entry_price), "side": side, "ts": now})
				gs["stopped_levels"] = gs["stopped_levels"][-12:]   # son 12 ile sinirli
			n = gs["consec_stops"][side]
			logger.warning("[GUARD] %s %s STOP -> cooldown %.1fh | ardisik=%d/%d",
			               data_sym, side, self.cfg.post_stop_cooldown_sec / 3600, n,
			               self.cfg.consecutive_stop_limit)
			if n >= self.cfg.consecutive_stop_limit:
				gs["halt_until"][side] = now + self.cfg.side_halt_sec
				gs["consec_stops"][side] = 0   # askidan sonra yeniden say
				logger.critical("[GUARD] %s %s ASKIYA ALINDI (%.1fh) — %d ardisik stop.",
				                data_sym, side, self.cfg.side_halt_sec / 3600,
				                self.cfg.consecutive_stop_limit)
		elif outcome == "TP":
			gs["consec_stops"][side] = 0
			gs["halt_until"][side] = 0.0
			gs["cooldown_until"][side] = 0.0
			logger.info("[GUARD] %s %s TP -> o yonun cezalari temizlendi.", data_sym, side)
		self._save_guard_state()

	def _guard_blocks(self, data_sym: str, side: str, ref_price: float) -> tuple[bool, str]:
		"""G3 — bu sembol+yon'de yeni giris engellenmeli mi? (blocked, reason)."""
		side = str(side).upper()
		gs = self._guard_state.get(data_sym)
		if not gs:
			return False, ""
		now = time.time()
		if now < float(gs.get("halt_until", {}).get(side, 0.0)):
			_left = (gs["halt_until"][side] - now) / 3600.0
			return True, f"{side} askida (ardisik-stop) {_left:.1f}h kaldi"
		if now < float(gs.get("cooldown_until", {}).get(side, 0.0)):
			# Cooldown = o SEMBOL+YON'de TAM yasak (config yorumundaki niyet). Eski kod
			# yalnizca stop seviyesinin ±relevel_lockout_pct bandini engelliyordu; farkli
			# bir seviyeden ayni yonde hemen yeniden giris serbest kaliyordu (stop-loop
			# korumasi fiilen devre disiydi). Seviye-kilidi artik tam yasagin alt kumesi.
			_left = (gs["cooldown_until"][side] - now) / 3600.0
			return True, f"{side} post-stop cooldown ({_left:.1f}h kaldi) — ayni yonde yeni giris yasak"
		# BE-stop kilidi: TAM yasak degil — sadece BE olunan seviyenin ±relevel bandi.
		# (Olcum: BE'den sonra 6h icinde ayni seviyeye donus ort -0.53R / %79 stop.)
		if now < float(gs.get("be_cooldown_until", {}).get(side, 0.0)) and ref_price and ref_price > 0:
			_left = (gs["be_cooldown_until"][side] - now) / 3600.0
			for lv in gs.get("stopped_levels", []):
				if str(lv.get("side")).upper() != side:
					continue
				lp = float(lv.get("price", 0) or 0)
				if lp > 0 and abs(ref_price - lp) / ref_price <= self.cfg.relevel_lockout_pct:
					return True, (f"{side} BE-stop seviye kilidi ({_left:.1f}h kaldi): {lp:.4f} "
					              f"bandinda (±{self.cfg.relevel_lockout_pct*100:.1f}%)")
		return False, ""

	async def _manage_active_positions(self, cg_manager: CoinglassManager) -> None:
		if not _MT5_AVAILABLE or self.cfg.dry_run:
			return

		positions = await asyncio.to_thread(mt5.positions_get)
		positions = positions or []

		# Filter to our magic and build active ticket set
		our_positions = [p for p in positions if p.magic == _MAGIC]
		active_tickets: set[int] = set()

		for pos in our_positions:
			active_tickets.add(pos.ticket)

			# Initialise state on first sight
			if pos.ticket not in self._trade_state:
				data_sym = self._mt5_to_data.get(pos.symbol, pos.symbol)

				# Bug #2 fix: Defensive restart-recovery. If SL is at breakeven (within 0.1% of entry),
				# Stage 1 has already executed BEFORE this bot process started. Without this guard,
				# we would treat the half-closed position as a fresh trade and fire Stage 1 again,
				# closing 50% of the remaining 50% → only 25% of original size left.
				sl_at_breakeven = False
				if pos.sl > 0 and pos.price_open > 0:
					sl_at_breakeven = abs(pos.sl - pos.price_open) / pos.price_open < 0.001

				if sl_at_breakeven:
					# pos.volume is the REMAINING volume (~50% of original after Stage 1).
					# Estimate initial = current × 2. Slightly off due to lot_step rounding but
					# Stage 2's "close 50% of CURRENT remaining" math is unaffected.
					estimated_initial = pos.volume * 2
					self._trade_state[pos.ticket] = {
						"initial_volume": estimated_initial,
						"stage_50_done":  True,
						"stage_75_done":  False,
						"symbol":         pos.symbol,
						"data_sym":       data_sym,
					}
					logger.warning(
						"[TRADE-MGMT] Restart-recovery: ticket=%d %s already at breakeven "
						"(SL=%.5f≈entry=%.5f). Assuming stage_50_done=True. Est initial=%.3f, current=%.3f.",
						pos.ticket, pos.symbol, pos.sl, pos.price_open,
						estimated_initial, pos.volume,
					)
				else:
					self._trade_state[pos.ticket] = {
						"initial_volume": pos.volume,
						"stage_50_done":  False,
						"stage_75_done":  False,
						"symbol":         pos.symbol,
						"data_sym":       data_sym,
					}
					logger.info(
						"[TRADE-MGMT] Tracking new position | ticket=%d %s vol=%.3f entry=%.5f tp=%.5f",
						pos.ticket, pos.symbol, pos.volume, pos.price_open, pos.tp,
					)

			st = self._trade_state[pos.ticket]

			if pos.tp == 0:
				continue

			entry		= pos.price_open
			current		= pos.price_current
			is_long		= pos.type == mt5.POSITION_TYPE_BUY

			# TREND-EXHAUSTION EARLY EXIT (hybrid: deterministic score -> LLM judge)
			_now = time.time()
			# (a) retry a pending exhaustion close every tick (cheap, no re-eval/LLM)
			if st.get("exh_close_pending"):
				if await _market_close_partial(pos, pos.volume, "exhaustion_close"):
					st["levels_done"] = 4
					st["exh_close_pending"] = False
				continue
			# (b) periodic exhaustion evaluation (throttled)
			if (exhaustion_score is not None and self.cfg.exhaustion_enabled
					and (_now - st.get("exh_last_ts", 0.0)) >= self.cfg.exhaustion_check_interval_sec):
				st["exh_last_ts"] = _now
				_exh_side = "long" if is_long else "short"
				try:
					_exh = await self._evaluate_exhaustion(pos, st, _exh_side)
				except Exception as _exc:
					logger.warning("[EXHAUST] eval failed ticket=%d: %s", pos.ticket, _exc)
					_exh = None
				if _exh is not None:
					# her kontrolde skoru yaz (gorunurluk: tukenme degerlendirmesi calisti mi + skor kac)
					logger.info("[EXHAUST] ticket=%d %s side=%s score=%.0f (thr=%.0f) breakdown=%s%s",
								pos.ticket, pos.symbol, _exh_side, _exh["total"],
								self.cfg.exhaustion_threshold, _exh["breakdown"],
								" [HARD-OVERRIDE]" if _exh["hard_override"] else "")
					if _exh["hard_override"]:
						logger.warning("[EXHAUST] ticket=%d %s HARD OVERRIDE score=%.0f | %s -> CLOSE_ALL",
									   pos.ticket, pos.symbol, _exh["total"], "; ".join(_exh["reasons"]))
						st["exh_close_pending"] = True
						if await _market_close_partial(pos, pos.volume, "exhaustion_override"):
							st["levels_done"] = 4
							st["exh_close_pending"] = False
						continue
					if (_exh["total"] >= self.cfg.exhaustion_threshold
							and _now >= st.get("exh_cooldown_until", 0.0)):
						logger.info("[EXHAUST] ticket=%d %s score=%.0f>=%.0f -> LLM judge | %s",
									pos.ticket, pos.symbol, _exh["total"], self.cfg.exhaustion_threshold,
									"; ".join(_exh["reasons"]))
						_verdict = await self._exhaustion_consult(pos, st, cg_manager, _exh)
						if _verdict == "CLOSE_ALL":
							st["exh_close_pending"] = True
							if await _market_close_partial(pos, pos.volume, "exhaustion_llm"):
								st["levels_done"] = 4
								st["exh_close_pending"] = False
							continue
						st["exh_cooldown_until"] = _now + self.cfg.exhaustion_consult_cooldown_sec
						logger.info("[EXHAUST] LLM=HOLD -> cooldown %.0fs", self.cfg.exhaustion_consult_cooldown_sec)
			# end exhaustion exit

			tp_distance = abs(pos.tp - entry)
			if tp_distance < 1e-9:
				continue

			if is_long:
				progress = max(0.0, (current - entry) / tp_distance) * 100.0
			else:
				progress = max(0.0, (entry - current) / tp_distance) * 100.0

			# ── 4-KADEME SCALEOUT + 3. kademede LLM danismasi ──────────────
			# entry->TP arasi 4 ceyrege bolunur. Her dilim asilinca initial in
			# %25 i kapanir (1-3. adim; son %25 TP emrine birakilir). %50 de SL
			# breakeven a cekilir (eski Stage-1 den korunan guvenlik — istemezsen
			# be_done blogunu sil). 3. adimda (%75) LLM e sorulur: pozisyon
			# yonumuzde devam mi edebilir yoksa kapatmali mi.
			st.setdefault("levels_done", min(4, int(progress // 25)))
			st.setdefault("consult_done", progress >= 75.0)
			# FIX (K6): be_done'i "progress >= 50" varsayimiyla DEGIL, SL'in GERCEKTEN
			# breakeven'da olup olmadigiyla belirle. State dosyasi kayipken pozisyon ilk
			# kez >= %50 progress'te gorulurse eski kod be_done=True yazar, SL asagida
			# zararda kalirdi (breakeven'a hic cekilmezdi).
			_sl_at_be = (pos.sl > 0 and pos.price_open > 0
			             and abs(pos.sl - pos.price_open) / pos.price_open < 0.001)
			st.setdefault("be_done", st.get("stage_50_done", False) or _sl_at_be)

			crossed = min(4, int(progress // 25))

			# %50 → SL breakeven (cfg.breakeven_at_half ile kapatilabilir; 2026-08-01
			# A/B sonucu default KAPALI — bkz. BotConfig.breakeven_at_half yorumu).
			if self.cfg.breakeven_at_half and not st["be_done"] and progress >= 50.0:
				# FIX (state desync): be_done YALNIZCA MT5 modify onaylarsa True.
				# Aksi halde bot stop u cektim sanir ama stop asagida zararda kalir.
				if await _modify_sl(pos, float(pos.price_open)):
					st["be_done"] = True

			# Bir sonraki dilime ulasildi mi? (tick basina en fazla 1 dilim)
			if st["levels_done"] < crossed:
				nxt = st["levels_done"] + 1

				# 3. adim: LLM danismasi. Sonuc cache lenir (consult_action) -> MT5
				# kapatmayi reddederse LLM YENIDEN CAGRILMADAN sonraki tick te tekrar denenir.
				if nxt == 3 and not st["consult_done"]:
					st["consult_action"] = await self._step3_ai_consultation(pos, st, cg_manager)
					st["consult_done"]   = True

				if nxt == 3 and st.get("consult_action") == "CLOSE_ALL":
					# FIX (state desync): levels_done=4 YALNIZCA kapatma onaylanirsa.
					# Reddedilirse levels_done degismez -> sonraki tick tam kapatmayi tekrar dener.
					if await _market_close_partial(pos, pos.volume, "step3_close_all"):
						st["levels_done"] = 4
					continue
				# CONTINUE → 3. ceyregi de kapatip son %25 ile devam

				if nxt >= 4:
					st["levels_done"] = 4        # son %25 TP emrine birakilir
				else:
					if await self._scaleout_close_quarter(pos, st, nxt):
						st["levels_done"] = nxt

		# Purge state for closed tickets
		closed = set(self._trade_state.keys()) - active_tickets
		for t in closed:
			logger.info("[TRADE-MGMT] Ticket %d no longer active — state purged.", t)
			self._trade_state.pop(t, None)
		# Her tik: state'i diske yaz — restart EXACT geri yuklesin (tahmin etmesin)
		self._save_trade_state()

	async def _scaleout_close_quarter(self, pos, st: dict, level: int) -> bool:
		"""4-kademe scaleout: initial hacmin %25 ini kapatir (tek dilim)."""
		raw  = st["initial_volume"] * 0.25
		step = self.cfg.lot_step
		vol  = round(math.floor(raw / step) * step, 3)
		if vol < self.cfg.min_lot:
			logger.info("[SCALEOUT] L%d ticket=%d: %%25 (%.3f) < min_lot — atlaniyor (pozisyon kucuk).",
			            level, pos.ticket, vol)
			return True
		vol = min(vol, pos.volume - self.cfg.min_lot * 0.5)
		if vol <= 0:
			return True
		logger.info("[SCALEOUT] ticket=%d %s | L%d (~%%%d) | %.3f kapatiliyor (initial %.3f)",
		            pos.ticket, pos.symbol, level, level * 25, vol, st["initial_volume"])
		return await _market_close_partial(pos, vol, f"scaleout_L{level}")

	async def _step3_ai_consultation(self, pos, st: dict, cg_manager) -> str:
		"""%75 progress (3. adim) LLM danismasi. 'CLOSE_ALL' veya 'CONTINUE' doner."""
		data_sym = st["data_sym"]
		logger.info("[STEP-3] ticket=%d %s | ~%%75 progress | LLM danismasi...", pos.ticket, pos.symbol)
		try:
			ctx = await self._build_stage2_context(pos, data_sym, cg_manager)
			ai  = await asyncio.wait_for(self._query_trade_evaluation(ctx, pos.symbol), timeout=30.0)
		except asyncio.TimeoutError:
			logger.warning("[STEP-3] timeout (>30s) — CONTINUE (SL zaten breakeven da).")
			return "CONTINUE"
		except Exception as exc:
			logger.error("[STEP-3] hata: %s — CONTINUE.", exc)
			return "CONTINUE"
		action = str(ai.get("action", "HOLD_REST")).upper()
		logger.info("[STEP-3] ticket=%d AI=%s | reason=%s", pos.ticket, action, ai.get("reason", ""))
		return "CLOSE_ALL" if action == "CLOSE_ALL" else "CONTINUE"

	async def _stage1_partial_tp_breakeven(self, pos, st: dict) -> None:
		"""
		Closes exactly 50% of the INITIAL position volume and moves SL to entry.
		"""
		raw_close = st["initial_volume"] * 0.5
		step	  = self.cfg.lot_step
		close_vol = math.floor(raw_close / step) * step
		close_vol = round(close_vol, 3)

		if close_vol < self.cfg.min_lot:
			logger.warning(
				"[STAGE-1] Computed close volume %.3f < min_lot — marking done to avoid retry.",
				close_vol,
			)
			st["stage_50_done"] = True
			return

		# Cap close volume to currently remaining
		close_vol = min(close_vol, pos.volume - self.cfg.min_lot * 0.5)
		if close_vol <= 0:
			st["stage_50_done"] = True
			return

		logger.info(
			"[STAGE-1] ticket=%d %s | progress >= 50%% | closing %.3f of initial %.3f",
			pos.ticket, pos.symbol, close_vol, st["initial_volume"],
		)

		ok = await _market_close_partial(pos, close_vol, "stage1_partial_tp")
		if not ok:
			return

		# Move SL to breakeven (entry)
		await _modify_sl(pos, float(pos.price_open))
		st["stage_50_done"] = True

	async def _stage2_ai_consultation(
		self,
		pos,
		st:			dict,
		cg_manager: CoinglassManager,
	) -> None:
		"""
		AI consult at 75% progress.
		  CLOSE_ALL → close entire remaining volume immediately.
		  HOLD_REST → close 50% of CURRENT remaining (banks more profit; leaves
					  25% of the original to run to TP).
		"""
		data_sym = st["data_sym"]
		logger.info(
			"[STAGE-2] ticket=%d %s | progress >= 75%% | building consult payload...",
			pos.ticket, pos.symbol,
		)

		# Build a lightweight LTF context for the AI
		ltf_text = await self._build_stage2_context(pos, data_sym, cg_manager)

		# Bug #8: Wrap the AI call in a timeout. Without this, a slow/hanging API call
		# blocks the entire trade-management loop and starves other positions of Stage 1/2 checks.
		try:
			ai_decision = await asyncio.wait_for(
				self._query_trade_evaluation(ltf_text, pos.symbol),
				timeout=30.0,
			)
		except asyncio.TimeoutError:
			logger.warning(
				"[STAGE-2] AI consult timeout (>30s) for ticket=%d — defaulting to HOLD_REST.",
				pos.ticket,
			)
			ai_decision = {"action": "HOLD_REST", "reason": "ai_timeout_fallback"}
		except Exception as exc:
			logger.error("[STAGE-2] AI consult error: %s — defaulting to HOLD_REST", exc)
			ai_decision = {"action": "HOLD_REST", "reason": "ai_error_fallback"}

		action = ai_decision["action"]
		reason = ai_decision.get("reason", "")
		logger.info("[STAGE-2] ticket=%d AI=%s | reason=%s", pos.ticket, action, reason)

		if action == "CLOSE_ALL":
			# Close everything that's left
			await _market_close_partial(pos, pos.volume, "stage2_close_all")
		else:  # HOLD_REST
			remaining	  = pos.volume
			raw_close	  = remaining * 0.5
			step		  = self.cfg.lot_step
			close_vol	  = math.floor(raw_close / step) * step
			close_vol	  = round(close_vol, 3)
			if close_vol < self.cfg.min_lot:
				logger.info(
					"[STAGE-2] HOLD_REST: remaining %.3f too small to split (min %.3f) — leaving to TP.",
					remaining, self.cfg.min_lot,
				)
			else:
				close_vol = min(close_vol, remaining - self.cfg.min_lot * 0.5)
				if close_vol > 0:
					await _market_close_partial(pos, close_vol, "stage2_hold_rest")

		st["stage_75_done"] = True

	async def _evaluate_exhaustion(self, pos, st, side):
		"""Fetch 1H+4H, compute the exhaustion score for an OPEN position.
		Returns the exhaustion_score dict, or None if data/scorer unavailable.
		Stores the latest RSI in `st` so the NEXT call can detect the TURN."""
		if exhaustion_score is None:
			return None
		tasks = {
			"1h": _fetch_mt5_ohlcv(pos.symbol, "1h", 200),
			"4h": _fetch_mt5_ohlcv(pos.symbol, "4h", 200),
		}
		res = dict(zip(tasks.keys(),
					   await asyncio.gather(*tasks.values(), return_exceptions=True)))
		analysis: dict = {}
		for tf, df in res.items():
			if isinstance(df, Exception) or df is None or getattr(df, "empty", True):
				continue
			data = self.analyzer.analyze_timeframe(df, tf, float(pos.price_current))
			if "error" not in data:
				analysis[tf] = data
		if not analysis:
			return None
		# prev degerleri artik ONCEKI BAR'dan (price_action indicators dict) geliyor —
		# st-bazli "onceki kontrol" saklamasi kaldirildi (memory-paradox cozuldu).
		exh = exhaustion_score(analysis, side)
		st["exh_analysis"]    = analysis
		return exh

	async def _exhaustion_consult(self, pos, st, cg_manager, exh) -> str:
		"""LLM judge for trend exhaustion. Returns 'CLOSE_ALL' or 'HOLD'.
		Conservative on any error/timeout (-> HOLD, keep riding)."""
		try:
			ctx = await self._build_exhaustion_context(pos, st, exh, cg_manager)
			ai  = await asyncio.wait_for(
				self._query_trade_evaluation(ctx, pos.symbol), timeout=30.0)
		except asyncio.TimeoutError:
			logger.warning("[EXHAUST] LLM timeout (>30s) -> HOLD (conservative).")
			return "HOLD"
		except Exception as exc:
			logger.warning("[EXHAUST] LLM error: %s -> HOLD.", exc)
			return "HOLD"
		action = str(ai.get("action", "HOLD_REST")).upper()
		logger.info("[EXHAUST] ticket=%d LLM verdict=%s | reason=%s",
					pos.ticket, action, ai.get("reason", ""))
		return "CLOSE_ALL" if action == "CLOSE_ALL" else "HOLD"

	async def _build_exhaustion_context(self, pos, st, exh, cg_manager) -> str:
		"""Compact payload for the exhaustion LLM judge: position, the deterministic
		exhaustion screen + which signals fired, a 4H/1H snapshot, and whale flow."""
		data_sym  = st["data_sym"]
		is_long   = pos.type == mt5.POSITION_TYPE_BUY
		direction = "LONG" if is_long else "SHORT"
		progress  = abs(pos.price_current - pos.price_open) / max(abs(pos.tp - pos.price_open), 1e-9) * 100.0
		analysis  = st.get("exh_analysis", {}) or {}
		snap = []
		for tf in ("4h", "1h"):
			d   = analysis.get(tf) or {}
			ind = d.get("indicators", {}) or {}
			snap.append(
				f"  {tf}: trend={d.get('trend','?')} | "
				f"RSI={ind.get('rsi_14','?')}({ind.get('rsi_zone','?')}) | "
				f"EMA={ind.get('ema_confluence','?')} | MACD={ind.get('macd_state','?')} | "
				f"events={_fmt_events(d.get('structure_events', []))}"
			)
		try:
			whale = await cg_manager.get_whale_sentiment(
				symbol=data_sym, current_price=float(pos.price_current))
		except Exception:
			whale = {}
		factors = "; ".join(exh.get("reasons", [])) or "(none)"
		return (
			f"# Trend-Exhaustion Exit Check\n"
			f"Position     : {direction} {pos.symbol} ({data_sym})\n"
			f"Entry/SL/TP  : {pos.price_open:.5f} / {pos.sl:.5f} / {pos.tp:.5f}\n"
			f"Current      : {pos.price_current:.5f}\n"
			f"Progress     : {progress:.1f}% toward TP | remaining vol {pos.volume:.3f}\n\n"
			f"## EXHAUSTION SCREEN (deterministic, total {exh['total']:.0f})\n"
			f"  breakdown   : {exh['breakdown']}\n"
			f"  signals     : {factors}\n\n"
			f"## MULTI-TF SNAPSHOT (higher TF first)\n" + "\n".join(snap) +
			f"\n\n## WHALE SENTIMENT\n" + _fmt_whale(whale) +
			f"\n\nThe deterministic screen suggests the {direction} trend may be EXHAUSTING. "
			f"Confirm or veto. Is the trend in our direction GENUINELY exhausted / reversing "
			f"(-> CLOSE_ALL, bank the remaining runner), or just a normal pullback / consolidation "
			f"that will resume in our favour (-> HOLD_REST, let it ride)? Be CONSERVATIVE: only "
			f"CLOSE_ALL when multiple independent signals truly confirm exhaustion, not on a single "
			f"stretched indicator. Decide CLOSE_ALL or HOLD_REST. Reply JSON only."
		)

	async def _build_stage2_context(
		self,
		pos,
		data_sym:	str,
		cg_manager: CoinglassManager,
	) -> str:
		"""
		Pulls fresh 15m + 1H OHLCV from MT5, computes indicators, fetches latest
		CoinGlass sentiment, and serialises a compact text payload for the AI.
		"""
		ohlcv_tasks = {
			"1h":  _fetch_mt5_ohlcv(pos.symbol, "1h",  150),
			"15m": _fetch_mt5_ohlcv(pos.symbol, "15m", 120),
		}
		ohlcv_results = dict(zip(
			ohlcv_tasks.keys(),
			await asyncio.gather(*ohlcv_tasks.values(), return_exceptions=False),
		))

		ltf_summary: list[str] = []
		for tf, df in ohlcv_results.items():
			if df is None or df.empty:
				ltf_summary.append(f"  {tf}: no data")
				continue
			data = self.analyzer.analyze_timeframe(df, tf, float(pos.price_current))
			if "error" in data:
				ltf_summary.append(f"  {tf}: {data['error']}")
				continue
			ind	  = data.get("indicators", {})
			ltf_summary.append(
				f"	{tf}: trend={data.get('trend','?')} | "
				f"RSI={ind.get('rsi_14','?')}({ind.get('rsi_zone','?')}) | "
				f"EMA conf={ind.get('ema_confluence','?')} | "
				f"MACD={ind.get('macd_state','?')} | "
				f"events={_fmt_events(data.get('structure_events', []))}"
			)

		# Whale sentiment refresh
		try:
			whale = await cg_manager.get_whale_sentiment(
				symbol=data_sym, current_price=float(pos.price_current),
			)
		except Exception as exc:
			logger.warning("[STAGE-2] Whale refresh failed: %s", exc)
			whale = {}

		is_long = pos.type == mt5.POSITION_TYPE_BUY
		direction = "LONG" if is_long else "SHORT"
		progress = abs(pos.price_current - pos.price_open) / max(abs(pos.tp - pos.price_open), 1e-9) * 100.0

		return (
			f"# Trade Management Consultation\n"
			f"Symbol	   : {pos.symbol} ({data_sym})\n"
			f"Direction	   : {direction}\n"
			f"Entry		   : {pos.price_open:.5f}\n"
			f"Stop-Loss	   : {pos.sl:.5f}\n"
			f"Take-Profit  : {pos.tp:.5f}\n"
			f"Current	   : {pos.price_current:.5f}\n"
			f"Progress	   : {progress:.1f}% toward TP\n"
			f"Remaining vol: {pos.volume:.3f}\n\n"
			f"## LTF SNAPSHOT\n"
			+ "\n".join(ltf_summary)
			+ "\n\n## WHALE SENTIMENT\n"
			+ _fmt_whale(whale)
			+ "\n\nDecide CLOSE_ALL or HOLD_REST. Reply JSON only."
		)

	# ── Per-symbol analysis cycle ───────────────────────────────────────

	async def _run_symbol(
		self,
		data_sym: str,
		mt5_sym:  str,
		cg:		  CoinglassManager,
	) -> None:
		logger.info("---- [%s %s] Analysis cycle ----", data_sym, mt5_sym)

		# 0. Trading-hours guard — outside the configured window we skip the
		# expensive analysis (CG fetch + LLM call) entirely. Trade-management
		# loop runs 24/7 and is NOT affected, so open positions stay managed.
		start_h = self.cfg.trading_hours_start_utc
		end_h   = self.cfg.trading_hours_end_utc
		if not (start_h == 0 and end_h == 24):
			now_utc_h = datetime.now(timezone.utc).hour
			in_window = (start_h <= now_utc_h < end_h) if start_h < end_h \
						else (now_utc_h >= start_h or now_utc_h < end_h)
			if not in_window:
				logger.info(
					"[%s] Off-hours (UTC %02d:00 outside %02d-%02d) — skipping analysis.",
					data_sym, now_utc_h, start_h, end_h,
				)
				return

		# 0b. G6.6 — Haber karantinasi (FOMC/CPI/NFP): analiz+LLM tamamen atlanir,
		# yeni giris yok. Acik pozisyonlar trade-management dongusunde yonetilmeye
		# devam eder (bu guard sadece YENI islem icindir).
		_nblk, _nreason = self._news_blackout_active()
		if _nblk:
			logger.info("[%s] HABER KARANTINASI — yeni giris yok: %s", data_sym, _nreason)
			return

		# 1. Kill-switch
		if not self.risk_mgr.is_active:
			logger.warning("[%s] Kill-switch active — skipping.", data_sym)
			return

		# 2. Order-stacking protector
		if _MT5_AVAILABLE and not self.cfg.dry_run:
			existing_orders = await asyncio.to_thread(mt5.orders_get, symbol=mt5_sym)
			existing_pos	= await asyncio.to_thread(mt5.positions_get, symbol=mt5_sym)
			if (existing_orders and len(existing_orders) > 0) or \
			   (existing_pos and len(existing_pos) > 0):
				logger.info("[%s] Existing order/position — skipping cycle.", data_sym)
				return

		# 3. Balance + exposure
		balance = await _fetch_mt5_balance(self.cfg.dry_run)
		if balance > 0:
			self.risk_mgr.update_balance(balance)
		if not self.risk_mgr.is_active:
			return

		current_exposure = await _fetch_total_exposure_usd(self.cfg, self.cfg.dry_run)
		if current_exposure >= self.cfg.max_total_exposure_usd:
			logger.warning(
				"[%s] Portfolio exposure cap hit: $%.0f / $%.0f — skipping.",
				data_sym, current_exposure, self.cfg.max_total_exposure_usd,
			)
			return

		# 3b. G3.4 — rolling multi-day drawdown breaker (account-level, side-agnostic).
		# Independent of the daily kill-switch (which self-resets at UTC midnight and so
		# never sees a slow week-long bleed). Open positions stay managed; only NEW
		# entries are suppressed until equity recovers.
		if self.risk_mgr.rolling_dd_breached():
			logger.warning(
				"[%s] Rolling %dg drawdown %.2f%% >= %.2f%% — yeni giris DURDU (acik pozisyonlar yonetiliyor).",
				data_sym, self.cfg.rolling_dd_days, self.risk_mgr.rolling_drawdown_pct() * 100,
				self.cfg.rolling_dd_pct * 100,
			)
			return

		# 4. OHLCV from Binance (deep history; matches backtest source). Execution stays on MT5/Eightcap.
		ohlcv_dict = await fetch_all_ohlcv(data_sym, self.cfg.timeframes, self.cfg.candles_per_tf)

		# 4b. HYBRID VOLUME — override MT5 tick volume with CoinGlass real USD volume
		# on the higher timeframes (1h/4h/1d), where structural BOS/CHoCH events are
		# scored and institutional flow confirmation actually matters. 15m keeps MT5
		# tick volume (cheap, good enough for the noisy LTF). On any CoinGlass failure
		# we silently keep tick volume so the bot never stalls on a data-source issue.
		if self.cfg.use_coinglass_volume:
			cg_sym_vol = CoinglassManager._SYM_MAP.get(data_sym, data_sym.split("/")[0])
			for tf in ("4h", "1d"):  # 1h dropped: CoinGlass free plan lacks 1h spot history (403); 1h keeps tick volume
				df_tf = ohlcv_dict.get(tf)
				if df_tf is None or df_tf.empty:
					continue
				try:
					cg_vols = await cg.fetch_spot_volumes(
						cg_sym_vol, tf, limit=self.cfg.coinglass_volume_bars)
					if cg_vols:
						merged, matched = _merge_real_volume(df_tf, cg_vols)
						if matched > 0:
							ohlcv_dict[tf] = merged
							logger.info(
								"[%s] Real volume merged %s: %d/%d bars (CoinGlass USD).",
								data_sym, tf, matched, len(df_tf),
							)
						else:
							logger.warning(
								"[%s] Volume merge %s: 0 bars matched — keeping tick volume.",
								data_sym, tf,
							)
					else:
						logger.warning(
							"[%s] CoinGlass volume %s unavailable — keeping tick volume.",
							data_sym, tf,
						)
				except Exception as exc:
					logger.warning(
						"[%s] Volume merge %s failed (%s) — keeping tick volume.",
						data_sym, tf, exc,
					)

		base_df = ohlcv_dict.get("15m")
		if base_df is None or base_df.empty:
			base_df = ohlcv_dict.get("1h")
		if base_df is None or base_df.empty:
			base_df = ohlcv_dict.get("4h")

		if base_df is None or base_df.empty:
			logger.error("[%s] No price data available — skipping.", data_sym)
			return

		# Bug #7: Surface OHLCV depth shortfalls — silent degradation otherwise hides
		# why scoring keeps returning WAIT (insufficient bars → empty key_levels → C.x=0).
		depth_warnings: list[str] = []
		for tf, df_tf in ohlcv_dict.items():
			expected = self.cfg.candles_per_tf.get(tf, 200)
			actual   = len(df_tf) if df_tf is not None else 0
			if actual < expected * 0.7:
				depth_warnings.append(f"{tf}:{actual}/{expected}")
		if depth_warnings:
			logger.warning(
				"[%s] OHLCV depth shortfall: %s — scoring may degrade (empty S/R, missing EMA-200 etc.)",
				data_sym, ", ".join(depth_warnings),
			)

		# Bug #0: Live market price via tick — the OHLCV last bar is now the LAST CLOSED bar
		# (forming bar excluded for indicator correctness), so we cannot use base_df['close'][-1]
		# as the live entry price anymore.
		current_price = await _get_market_price(mt5_sym, fallback_df=base_df)
		if current_price <= 0:
			logger.error("[%s] Could not determine current price — skipping.", data_sym)
			return
		last_closed_bar_close = float(base_df["close"].iloc[-1])
		drift_pct = (
			(current_price - last_closed_bar_close) / last_closed_bar_close * 100
			if last_closed_bar_close else 0.0
		)
		logger.info(
			"[%s] Live price: %.6f | Last closed bar close: %.6f (drift %.3f%%)",
			data_sym, current_price, last_closed_bar_close, drift_pct,
		)

		# DRIFT GUARD (volatility-aware): a TRULY stale frame (e.g. first cycle after
		# startup, before MT5 streamed the latest bars) shows the live tick diverging
		# wildly from the last CLOSED bar. But a volatile, trending instrument (XRP) has
		# its 15m close legitimately sitting 1-2% behind the live tick — that is normal
		# feed lag + trend, NOT stale data. A fixed 1.5% threshold wrongly froze XRP for
		# hours. Scale the threshold to the symbol's own 15m volatility (ATR%): quiet
		# symbols (BTC) keep a tight ~1.5% gate; volatile ones widen automatically. A
		# hard 3.5% ceiling still catches genuinely stale (hours-old) frames.
		atr_pct = 0.0
		try:
			_tr  = (base_df["high"].astype(float) - base_df["low"].astype(float)).abs()
			_atr = float(_tr.tail(14).mean())
			if last_closed_bar_close > 0 and _atr > 0:
				atr_pct = _atr / last_closed_bar_close * 100
		except Exception:
			atr_pct = 0.0
		drift_threshold = min(3.5, max(1.5, atr_pct * 5.0))
		if abs(drift_pct) > drift_threshold:
			logger.warning(
				"[%s] Stale data guard: drift %.3f%% > %.2f%% threshold "
				"(15m ATR%%=%.3f; live=%.6f vs last-closed=%.6f). Skipping cycle.",
				data_sym, drift_pct, drift_threshold, atr_pct,
				current_price, last_closed_bar_close,
			)
			return

		# 4c. G-OPT — ADX REGIME FILTER (backtest matrisinin kazananı). Trendsiz/yatay
		# (chop) rejimde counter-trend stop'ların olduğu düşük-kaliteli işlemleri eler.
		# 4H ADX < eşik -> skor/CG/LLM'e hiç girme (maliyet de düşer). Açık pozisyonlar
		# yönetilmeye devam eder (bu sadece YENİ giriş geçididir).
		if self.cfg.regime_filter_enabled:
			_adx_df = ohlcv_dict.get(self.cfg.regime_adx_tf)
			_adx_val = _calc_adx(_adx_df)
			if _adx_val is not None and _adx_val < self.cfg.regime_adx_min:
				logger.info("[%s] Rejim filtresi: %s ADX=%.1f < %.1f (chop) — yeni giris ATLANDI.",
				            data_sym, self.cfg.regime_adx_tf, _adx_val, self.cfg.regime_adx_min)
				return

		# 5. Top-down PA analysis + Major S/R (with POC + psychological round numbers)
		poc_1d = _calc_poc_from_ohlcv(ohlcv_dict.get("1d"))
		psych_inc = self.cfg.psychological_increments.get(mt5_sym)
		tf_analysis = self.analyzer.analyze_all(
			ohlcv_dict=ohlcv_dict,
			current_price=current_price,
			psych_increment=psych_inc,
			poc_price=poc_1d,
			min_touches=self.cfg.sr_min_touches,
		)

		# 6. CoinGlass whale sentiment
		try:
			whale_data = await cg.get_whale_sentiment(
				symbol=data_sym, current_price=current_price,
				ohlcv_1d=ohlcv_dict.get("1d"),
			)
		except Exception as exc:
			logger.error("[%s] CoinGlass error: %s", data_sym, exc)
			whale_data = {}

		# 7. Build master payload
		payload = build_master_payload(data_sym, current_price, tf_analysis, whale_data)

		# 7b. PYTHON ON-FILTRE / SCORER-ONLY KARAR — once deterministik skor.
		#      llm_entry_enabled=False (G7, default): karar TAMAMEN scorer'dan gelir,
		#      LLM giris cagrisi hic yapilmaz (backtest ile birebir ayni motor).
		#      llm_entry_enabled=True: eski davranis — skor geciti gecerse LLM cagrilir.
		#      FAIL-SAFE: Python skorlama patlarsa her durumda LLM yoluna duser.
		_scorer_decision = None
		if _python_score is not None:
			try:
				_whale_flat = _whale_for_scorer(whale_data)
				_py = _python_score(tf_analysis, current_price, whale=_whale_flat,
				                    threshold=self.cfg.min_score, enable_reversal=False,
				                    min_sl=self.cfg.min_sl_pct, max_sl=self.cfg.max_sl_pct,
				                    min_rr=self.cfg.min_rr_ratio, max_rr=_tp_cap_for(data_sym))
				_pl, _ps = int(_py["long_total"]), int(_py["short_total"])
				_lv, _sv = bool(_py.get("long_vetoed")), bool(_py.get("short_vetoed"))
				logger.info("[%s] Python on-skor: L=%d%s S=%d%s (esik=%d gecit=%d)",
				            data_sym, _pl, "(veto)" if _lv else "", _ps, "(veto)" if _sv else "",
				            self.cfg.min_score, PY_GATE)
				logger.info("[%s]   L-detay: %s", data_sym, _py["long_breakdown"])
				logger.info("[%s]   S-detay: %s", data_sym, _py["short_breakdown"])
				# G1.3/1.4 — a side blocked by the deterministic counter-trend brake
				# must NOT reach the LLM. A side qualifies only if total>=gate AND not vetoed.
				_long_ok  = (_pl >= PY_GATE) and not _lv
				_short_ok = (_ps >= PY_GATE) and not _sv
				if not _long_ok and not _short_ok:
					_vmsg = _py.get("veto_reason") or ""
					logger.info("[%s] WAIT (Python on-filtre) | L=%d S=%d%s",
					            data_sym, _pl, _ps, f" | veto: {_vmsg}" if _vmsg else "")
					return
				if not self.cfg.llm_entry_enabled:
					# G7 — scorer-only: karar + plan dogrudan scorer'dan. LLM'in cikti
					# formatina cevrilir ki asagidaki veto/sizing/emir yolu AYNEN calissin.
					if _py.get("decision") in ("LONG", "SHORT") and _py.get("plan"):
						_plan = _py["plan"]
						_scorer_decision = {
							"decision":     _py["decision"],
							"long_total":   _pl,
							"short_total":  _ps,
							"long_breakdown":  _py["long_breakdown"],
							"short_breakdown": _py["short_breakdown"],
							"entry_price":  _plan["entry"],
							"sl_price":     _plan["sl"],
							"tp_price":     _plan["tp"],
							"rr_ratio":     _plan["rr"],
							"position_usd": _plan.get("position_usd", 10000.0),
							"rationale":    "deterministic scorer decision (llm_entry_enabled=False)",
							"calculation_breakdown": (
								f"SCORER-ONLY | L={_pl} S={_ps} | entry={_plan['entry']} "
								f"sl={_plan['sl']} ({_plan.get('sl_pct', 0)*100:.2f}%) "
								f"tp={_plan['tp']} rr={_plan['rr']}"),
							"key_levels":   [],
						}
					else:
						logger.info("[%s] WAIT (scorer-only: karar=%s, plan %s) | L=%d S=%d",
						            data_sym, _py.get("decision"),
						            "var" if _py.get("plan") else "yok", _pl, _ps)
						return
			except Exception as _exc:
				logger.warning("[%s] Python on-filtre hatasi (%s) — LLM e dusuluyor.",
				               data_sym, _exc)
				_scorer_decision = None

		## 8. Karar: scorer-only ise postprocess'ten gecir; degilse LLM'e sor.
		if _scorer_decision is not None:
			# Ayni guvenlik kapilari (RR/SL bandi, sidedness, entry-sanity, RR cap)
			# scorer kararina da uygulanir — savunma katmani bedava.
			decision = _apply_decision_postprocess(
				_scorer_decision, self.cfg,
				current_price=current_price, tp_cap=_tp_cap_for(data_sym))
			logger.info("[%s] SCORER-ONLY karar: %s | L=%d S=%d | RR=%.2f (LLM cagrilmadi)",
			            data_sym, decision.get("decision"),
			            int(decision.get("long_total", 0) or 0),
			            int(decision.get("short_total", 0) or 0),
			            float(decision.get("rr_ratio", 0) or 0))
		else:
			try:
				decision = await self._query_llm(payload, data_sym)
			except Exception as e:
				logger.error("[%s] LLM isteği başarısız oldu: %s", data_sym, e)
				return

		side = str(decision.get("decision", "WAIT")).upper()
		if side == "WAIT":
			score = max(int(decision.get("long_total", 0) or 0),
						int(decision.get("short_total", 0) or 0))
			logger.info("[%s] WAIT | max_score=%d", data_sym, score)
			return
		# -- COUNTER-TREND + GUARD VETO (G1.4 / G2.4 / G3) — side artik belli; son kapi.
		#    Scorer'in deterministik counter-trend freni LLM cagrisini zaten geciriyor;
		#    burada (a) state-bazli freni backstop olarak yeniden uygula (LLM ters yon
		#    secebilir), (b) event-bazli exhaustion vetosunu uygula, (c) sembol+yon
		#    yeniden-giris kalkanini (cooldown / seviye-kilidi / ardisik-stop askisi)
		#    zorla. Hicbiri islem EKLEMEZ; yalnizca engeller.
		_vside = side.lower()
		# (a) state-based counter-trend brake (4H frame turned against the trade)
		if _trend_conflict_block is not None:
			try:
				_blk, _why = _trend_conflict_block(tf_analysis, _vside)
				if _blk:
					logger.info("[%s] VETO (counter-trend): %s -> %s ALINMADI", data_sym, _why, side)
					return
			except Exception:
				pass
		# (a2) konum vetolari — over-extension + premium/discount (scorer ile ortak;
		#      LLM on-filtreden gecmeyen TERS yonu secerse buradan yakalanir)
		if _over_extended is not None:
			try:
				_ov, _ovr = _over_extended(tf_analysis, _vside)
				if _ov:
					logger.info("[%s] VETO (over-extension): %s -> %s ALINMADI", data_sym, _ovr, side)
					return
			except Exception:
				pass
		if _premium_discount_block is not None:
			try:
				_pd, _pdr = _premium_discount_block(tf_analysis, _vside)
				if _pd:
					logger.info("[%s] VETO (premium/discount): %s -> %s ALINMADI", data_sym, _pdr, side)
					return
			except Exception:
				pass
		# (a3) G6.3 — LTF (15m) giris tetigi: alt-TF'te somut donus kaniti yoksa girme
		if _ltf_trigger_ok is not None:
			try:
				_lt, _ltr = _ltf_trigger_ok(tf_analysis, _vside, current_price)
				if not _lt:
					logger.info("[%s] VETO (LTF tetik): %s -> %s ALINMADI", data_sym, _ltr, side)
					return
			except Exception:
				pass
		# (b) event-based exhaustion veto (alt-TF donus). esik 0/negatif = DEVRE DISI
		# (0'a "her sey >= 0" diye takilmasin diye acik >0 kontrolu sart).
		if exhaustion_score is not None and self.cfg.entry_exhaustion_veto > 0:
			try:
				_exhv = float(exhaustion_score(tf_analysis, _vside).get("total", 0.0))
				if _exhv >= self.cfg.entry_exhaustion_veto:
					logger.info("[%s] VETO (alt-TF donus): %s exhaustion=%.0f >= %.0f -> ALINMADI",
					            data_sym, side, _exhv, self.cfg.entry_exhaustion_veto)
					return
			except Exception:
				pass
		# (c) G3 re-entry guard — post-stop cooldown / level lockout / consecutive-stop halt
		try:
			_gblk, _greason = self._guard_blocks(data_sym, side, current_price)
			if _gblk:
				logger.warning("[%s] GUARD veto: %s -> %s ALINMADI", data_sym, _greason, side)
				return
		except Exception as _exc:
			logger.warning("[%s] guard kontrolu hatasi: %s", data_sym, _exc)
# 9. Validate / extract SL / TP / entry
		try:
			entry_p	 = float(decision.get("entry_price", current_price))
			sl_price = float(decision.get("sl_price", 0) or 0)
			tp_price = float(decision.get("tp_price", 0) or 0)
		except (TypeError, ValueError):
			logger.error("[%s] Decision number-parsing failed — abort.", data_sym)
			return

		if sl_price <= 0 or tp_price <= 0:
			logger.warning("[%s] LLM returned invalid SL or TP — rejected.", data_sym)
			return

		# 9b. RISK-BASED sizing: notional = risk_$ / SL-distance, so every trade risks a
		# fixed fraction of balance (capped at max_risk_usd), INDEPENDENT of the score
		# (the backtest showed score is not predictive at the high end). Capped by the
		# per-symbol ceiling. The LLM position_usd is only a WAIT flag at this point.
		if float(decision.get("position_usd", 0) or 0) <= 0:
			logger.warning("[%s] position_usd=0 after postprocess — WAIT, rejecting.", data_sym)
			return
		_sl_dist = abs(entry_p - sl_price) / entry_p if entry_p > 0 else 0.0
		if _sl_dist <= 0:
			logger.warning("[%s] invalid SL distance — rejecting.", data_sym)
			return
		_risk_usd = min(self.cfg.risk_pct_per_trade * self.risk_mgr.balance, self.cfg.max_risk_usd)
		position_usd = min(_risk_usd / _sl_dist, self.cfg.max_position_usd)
		logger.info(
			"[%s] Risk-based size: $%.0f risk (%.2f%% of $%.0f) / SL %.2f%% -> notional $%.0f (cap $%.0f)",
			data_sym, _risk_usd, self.cfg.risk_pct_per_trade * 100, self.risk_mgr.balance,
			_sl_dist * 100, position_usd, self.cfg.max_position_usd,
		)

		# 10. Risk plan (fixed-USD sizing)
		plan = self.risk_mgr.compute_trade_plan(
			mt5_sym=mt5_sym,
			entry_price=entry_p,
			sl_price=sl_price,
			tp_price=tp_price,
			target_position_usd=position_usd,
			current_exposure_usd=current_exposure,
		)
		if not plan["ok"]:
			logger.warning("[%s] Trade plan rejected: %s", data_sym, plan["reason"])
			return

		volume = _normalize_volume(mt5_sym, plan["lot"])

		# 11. Place limit order
		score_for_comment = decision.get("long_total" if side == "LONG" else "short_total", 0)
		comment = f"algo s{score_for_comment} rr{plan['rr']:.1f}"

		result = await _send_limit_order(
			mt5_sym=mt5_sym, decision=side,
			entry_price=entry_p, volume=volume,
			sl_price=sl_price, tp_price=tp_price,
			comment=comment,
			dry_run=self.cfg.dry_run,
		)

		# 12. Trade log
		trade_logger.info(
			"[%s->%s] %-12s | LIMIT @ %.4f | vol=%.3f | sl=%.4f | tp=%.4f | "
			"SL%%=%.2f RR=%.2f | pos_usd=$%.0f (target=$%.0f) | L=%d S=%d | limit=%s | order=%s",
			data_sym, mt5_sym, result.status.value,
			entry_p, volume, sl_price, tp_price,
			plan["sl_pct"] * 100, plan["rr"],
			plan["actual_position_usd"], position_usd,
			int(decision.get("long_total",	0) or 0),
			int(decision.get("short_total", 0) or 0),
			plan["limiting_factor"],
			result.order_id or "-",
		)

		# 13. Journal
		if result.status in (ExecutionStatus.SUCCESS, ExecutionStatus.DRY_RUN):
			ticket = result.order_id or f"DRY-{int(time.time())}"
			_journal_add_entry(
				journal_path=self.cfg.journal_path,
				ticket_id=ticket,
				data_sym=data_sym, mt5_sym=mt5_sym,
				decision=decision, entry_price=entry_p,
				sl_price=sl_price, tp_price=tp_price,
			)

	# ── Outer scheduler ─────────────────────────────────────────────────

	async def _cycle(self, cg: CoinglassManager) -> None:
		now = time.monotonic()

		# Hourly closed-position check
		if now - self._last_history_check >= self.cfg.history_check_interval:
			logger.info(
				"[HISTORY] Hourly review (elapsed=%.0fs)...",
				now - self._last_history_check,
			)
			try:
				await self._check_closed_positions()
				self._last_history_check = time.monotonic()
			except Exception as exc:
				logger.error("[HISTORY] Review error: %s", exc)

		# Sequential per-symbol analysis (BTC first → ETH → XRP)
		for data_sym, mt5_sym in zip(self.cfg.data_symbols, self.cfg.mt5_symbols):
			elapsed = now - self._last_ai.get(data_sym, 0.0)
			if elapsed < self.cfg.ai_interval_seconds:
				remaining = self.cfg.ai_interval_seconds - elapsed
				logger.debug("[%s] Next analysis in %.0fs.", data_sym, remaining)
				continue
			try:
				await self._run_symbol(data_sym, mt5_sym, cg)
				self._last_ai[data_sym] = time.monotonic()
			except Exception as exc:
				logger.exception("[%s] Unexpected error: %s", data_sym, exc)
				# Back off ~2 minutes after a failure
				self._last_ai[data_sym] = (
					time.monotonic() - self.cfg.ai_interval_seconds + 120
				)

	# ── Entry point ─────────────────────────────────────────────────────

	async def run(self) -> None:
		cfg = self.cfg
		logger.info(
			"FundedNext Swing Bot starting up\n"
			"  Symbols		: %s\n"
			"  Timeframes	: %s | Depth: %s\n"
			"  Risk			: %dx leverage | Max single: $%.0f | Max total: $%.0f\n"
			"  Min Score	: %d/100 | AI Interval: %ds | Dry-run: %s\n"
			"  Trade Mgmt	: Stage1=50%% partial+BE, Stage2=75%% AI consult (every %ds)\n"
			"  S/R			: %.1f%% tolerance | min=%d touches | flip=%d touches\n"
			"  Journal		: %s | Lessons: %s",
			list(zip(cfg.data_symbols, cfg.mt5_symbols)),
			cfg.timeframes, cfg.candles_per_tf,
			cfg.leverage, cfg.max_position_usd, cfg.max_total_exposure_usd,
			cfg.min_score, cfg.ai_interval_seconds, cfg.dry_run,
			cfg.trade_mgmt_interval_sec,
			cfg.sr_tolerance_pct * 100, cfg.sr_min_touches, cfg.sr_flip_min_touches,
			cfg.journal_path, cfg.lessons_path,
		)

		try:
			async with (
				CoinglassManager(cfg.coinglass_api_key) as cg,
				TradeExecutor(
					login=cfg.mt5_login,
					password=cfg.mt5_password,
					server=cfg.mt5_server,
					dry_run=cfg.dry_run,
					trade_logger=trade_logger,
				),
			):
				init_balance = await _fetch_mt5_balance(cfg.dry_run)
				self.risk_mgr.initialize(init_balance)

				# Initial closed-position scan
				await self._check_closed_positions()
				self._last_history_check = time.monotonic()

				# Spin up the trade-management coroutine
				mgmt_task = asyncio.create_task(self.trade_management_loop(cg))

				try:
					while True:
						try:
							await self._cycle(cg)
						except Exception as exc:
							logger.exception("Main-cycle error (continuing): %s", exc)

						logger.info(
							"Idle %ds | Balance: $%.2f | DD: %.2f%% | Kill-switch: %s",
							cfg.loop_interval_sec,
							self.risk_mgr.balance,
							self.risk_mgr.current_drawdown_pct * 100,
							"ENGAGED" if not self.risk_mgr.is_active else "off",
						)
						await asyncio.sleep(cfg.loop_interval_sec)
				finally:
					mgmt_task.cancel()
					try:
						await mgmt_task
					except asyncio.CancelledError:
						pass
		finally:
			logger.info("Bot shutdown complete.")


# ─────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────

def main() -> None:
	config = BotConfig()
	bot	   = AlgoBot(config)
	asyncio.run(bot.run())


if __name__ == "__main__":
	main()