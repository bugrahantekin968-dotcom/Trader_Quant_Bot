"""
main.py — FundedNext Swing Trade Bot
  · CoinGlass v4 API — Aggregated Orderbook, OI, FR, L/S Ratio, Likidasyon
  · Heatmap tamamen kaldirildi; yerine Aggregated Ask/Bid Orderbook clustering
  · Tarihsel Majör S/R & S/R Flip analizi (price_action.py den)
  · Order Stacking Protector KORUNDU
  · Post-Trade Feedback Loop (journal + lessons) KORUNDU
  · Kill-Switch ve tüm güvenlik önlemleri KORUNDU
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
import math

import aiohttp
import ccxt.async_support as ccxt_async
import google.generativeai as genai
import pandas as pd

from modules.executor import ExecutionResult, ExecutionStatus, TradeExecutor
from modules.price_action import TechnicalAnalyzer

try:
	import MetaTrader5 as mt5
	_MT5_AVAILABLE = True
except ImportError:
	_MT5_AVAILABLE = False

_MAGIC = 202605

# ─────────────────────────────────────────────────────────────────
# Loglama
# ─────────────────────────────────────────────────────────────────

logging.basicConfig(
	level=logging.INFO,
	format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
	datefmt="%Y-%m-%d %H:%M:%S",
)
logger		 = logging.getLogger("algo_bot")
trade_logger = logging.getLogger("trade")


# ─────────────────────────────────────────────────────────────────
# Konfigürasyon
# ─────────────────────────────────────────────────────────────────

@dataclass
class BotConfig:
	# ── Kimlik bilgileri
	mt5_login:		   int = 7943052
	mt5_password:	   str = "uLgT5#8B"
	mt5_server:		   str = "Eightcap-Demo"
	gemini_api_key:	   str = "AIzaSyCH5FVxMjmbFTX-igg2Gdsmt4OOkoVVn7w"
	gemini_model:	   str = "gemini-3.1-pro-preview"
	coinglass_api_key: str = "29a104fc4b2e44c09d6337c5e0c591b9"

	# ── Semboller (Binance ↔ MT5 — sıra birebir eşleşmeli)
	data_symbols: list[str] = field(
		default_factory=lambda: ["BTC/USDT", "ETH/USDT", "XRP/USDT"]
	)
	mt5_symbols: list[str] = field(
		default_factory=lambda: ["BTCUSD", "ETHUSD", "XRPUSD"]
	)

	# ── Zaman dilimleri
	timeframes:	   list[str] = field(default_factory=lambda: ["1d", "4h", "1h", "15m"])
	candles_limit: int		 = 200

	# ── Zamanlama
	ai_interval_seconds:	int = 900
	loop_interval_sec:		int = 60
	history_check_interval: int = 3600

	# ── FundedNext Risk Kuralları (REVİZE)
	leverage:				 int   = 1
	risk_pct_75_84:			 float = 0.005		# 75-84 puan: %0.5
	risk_pct_85_100:		 float = 0.010		# 85-100 puan: %1.0
	max_position_usd:		 float = 20_000.0	# Tek sembolde maks pozisyon değeri ($)
	max_total_exposure_usd:	 float = 60_000.0	# Tüm açık pozisyonların toplam değeri ($)
	max_sl_pct:				 float = 0.03		# Maks SL mesafesi: fiyatın %3'ü
	min_rr_ratio:			 float = 2.0		# Min Risk/Reward
	max_daily_drawdown_pct:	 float = 0.02		# Günlük DD %2
	min_lot:				 float = 0.01
	lot_step:				 float = 0.01
	max_lot:				 float = 5000.0
	min_score:				 int   = 75

# ── Sembol Spec'leri (MT5 trade_contract_size dinamik gelirse onun yerine kullanılır)
	symbol_contract_sizes: dict = field(default_factory=lambda: {
		"BTCUSD": 0.05,	  # 1 lot = 0.05 BTC
		"ETHUSD": 0.05,	  # 1 lot = 0.05 ETH
		"XRPUSD": 5.0,	  # 1 lot = 5 XRP
	})

	# ── Dry-run
	dry_run: bool = False

	# ── CCXT Orderbook clustering parametreleri (LiquidityManager — Binance/OKX/Bybit)
	ob_depth_pct: float = 0.20	 # +-%%20 fiyat araliginda clustering
	ob_tier_pct:  float = 0.01	 # %%1'lik dilimler
	ob_top_n:	  int	= 3		 # En büyük 3 duvar

	# ── Majör S/R parametreleri
	sr_tolerance_pct:	 float = 0.015
	sr_min_touches:		 int   = 2
	sr_flip_min_touches: int   = 3

	# ── Post-trade feedback dosya yolları
	journal_path: str = "trade_journal.json"
	lessons_path: str = "lessons_learned.json"


# ─────────────────────────────────────────────────────────────────
# FundedNext Risk Yöneticisi
# ─────────────────────────────────────────────────────────────────

class RiskManager:
	"""
	FundedNext bilinçli risk yöneticisi.
	
	- Sembol kontrak boyutuna göre lot hesabı
	- Maks SL %3 kontrolü
	- Minimum R/R 2 kontrolü
	- Lot 0.01 step'e FLOOR (asla yukarı yuvarlama yok)
	- Günlük DD reset (gün değişince start_balance güncellenir)
	"""

	def __init__(
		self,
		cfg: "BotConfig",
	) -> None:
		self.cfg				 = cfg
		self.max_dd_pct			 = cfg.max_daily_drawdown_pct
		self.max_position_usd	 = cfg.max_position_usd
		self._start_balance		 = 0.0
		self._current_balance	 = 0.0
		self._kill_switch		 = False
		self._last_reset_date	 = None	  # günlük DD reset için

	# ── Başlatma ─────────────────────────────────────────────────
	def initialize(self, balance: float) -> None:
		from datetime import datetime, timezone
		self._start_balance	  = balance
		self._current_balance = balance
		self._kill_switch	  = False
		self._last_reset_date = datetime.now(timezone.utc).date()
		logger.info(
			"[RISK] Baslatildi | Bakiye: %.2f$ | Max Pos: $%.0f | "
			"Max Total: $%.0f | Max SL: %.1f%% | Min RR: %.1f | Max DD: %.1f%%",
			balance,
			self.max_position_usd,
			self.cfg.max_total_exposure_usd,
			self.cfg.max_sl_pct * 100,
			self.cfg.min_rr_ratio,
			self.max_dd_pct * 100,
		)

	# ── Bakiye güncelle + günlük reset ──────────────────────────
	def update_balance(self, new_balance: float) -> None:
		from datetime import datetime, timezone
		today = datetime.now(timezone.utc).date()
		if self._last_reset_date and today != self._last_reset_date:
			# Yeni gün → DD baseline yenilen
			logger.info("[RISK] Yeni gün (%s) | Önceki start: %.2f -> Yeni start: %.2f",
						today, self._start_balance, new_balance)
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
				"KILL-SWITCH! Gunluk DD: %.2f%% >= %.2f%% | Bakiye: %.2f$ Start: %.2f$",
				drawdown * 100, self.max_dd_pct * 100, new_balance, self._start_balance,
			)

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

	# ── Sembol kontrak boyutu (MT5 öncelikli, fallback config) ─
	def _get_contract_size(self, mt5_sym: str) -> float:
		if _MT5_AVAILABLE:
			try:
				info = mt5.symbol_info(mt5_sym)
				if info and info.trade_contract_size > 0:
					return float(info.trade_contract_size)
			except Exception:
				pass
		return float(self.cfg.symbol_contract_sizes.get(mt5_sym, 1.0))

	# ── Lot step'e FLOOR (risk korumalı yuvarlama) ──────────────
	def _floor_to_step(self, lot: float) -> float:
		step = self.cfg.lot_step
		return math.floor(lot / step) * step

	# ── ANA HESAPLAYICI ─────────────────────────────────────────
	def compute_trade_plan(
		self,
		mt5_sym:	   str,
		entry_price:   float,
		sl_price:	   float,
		tp_price:	   Optional[float],
		risk_pct:	   float,
		current_exposure_usd: float = 0.0,
	) -> dict:
		"""
		Tek bir trade için tam plan hesaplar.
		Dönüş:
		{
			"ok": bool,
			"reason": str,
			"lot": float,
			"sl_pct": float,
			"rr": float,
			"lot_value_usd": float,
			"limiting_factor": str,
		}
		"""
		out = {
			"ok": False, "reason": "", "lot": 0.0,
			"sl_pct": 0.0, "rr": 0.0, "lot_value_usd": 0.0,
			"limiting_factor": "-",
		}

		# 0. Sanity
		if entry_price <= 0 or sl_price <= 0:
			out["reason"] = "entry veya SL <= 0"
			return out

		sl_distance = abs(entry_price - sl_price)
		if sl_distance < 1e-9:
			out["reason"] = "SL mesafesi 0'a yakin"
			return out

		sl_pct = sl_distance / entry_price
		out["sl_pct"] = sl_pct

		# 1. Maks SL %3 KONTROLÜ — sert kapı
		if sl_pct > self.cfg.max_sl_pct:
			out["reason"] = (
				f"SL mesafesi %{sl_pct*100:.2f} > izin verilen maks %{self.cfg.max_sl_pct*100:.1f}"
			)
			return out

		# 2. R/R KONTROLÜ — sert kapı (TP varsa)
		if tp_price and tp_price > 0:
			tp_distance = abs(tp_price - entry_price)
			rr = tp_distance / sl_distance
			out["rr"] = rr
			if rr < self.cfg.min_rr_ratio:
				out["reason"] = f"R/R {rr:.2f} < min {self.cfg.min_rr_ratio}"
				return out
		else:
			out["reason"] = "TP yok, RR hesaplanamadi"
			return out

		# 3. Sembol kontrak boyutu
		contract_size = self._get_contract_size(mt5_sym)
		if contract_size <= 0:
			out["reason"] = f"contract_size geçersiz ({contract_size})"
			return out

		lot_value_usd = contract_size * entry_price
		out["lot_value_usd"] = lot_value_usd

		# 4. Risk-based lot
		risk_amount	  = self._current_balance * risk_pct
		loss_per_lot  = sl_distance * contract_size
		risk_lot	  = risk_amount / loss_per_lot if loss_per_lot > 0 else 0.0

		# 5. Tek sembol pozisyon limiti
		max_pos_lot	  = self.max_position_usd / lot_value_usd

		# 6. Portföy toplam hacim limiti
		remaining_exposure = self.cfg.max_total_exposure_usd - current_exposure_usd
		if remaining_exposure <= 0:
			out["reason"] = (
				f"Portföy maks hacmi dolu: ${current_exposure_usd:,.0f} / "
				f"${self.cfg.max_total_exposure_usd:,.0f}"
			)
			return out
		portfolio_lot = remaining_exposure / lot_value_usd

		# 7. En kısıtlayıcı sınır
		candidates = {
			"risk":		 risk_lot,
			"tek_pos":	 max_pos_lot,
			"portfoy":	 portfolio_lot,
			"max_lot":	 self.cfg.max_lot,
		}
		raw_lot = min(candidates.values())
		limiting = min(candidates, key=candidates.get)

		# 8. Step'e FLOOR (asla yukarı yuvarlama)
		floored = self._floor_to_step(raw_lot)

		# 9. Min lot kontrolü — min lot bile risk_amount'u 1.5x aşarsa REDDET
		if floored < self.cfg.min_lot:
			min_lot_loss = self.cfg.min_lot * loss_per_lot
			if min_lot_loss > risk_amount * 1.5:
				out["reason"] = (
					f"Min lot {self.cfg.min_lot} ile zarar ${min_lot_loss:.2f}, "
					f"izin verilen risk ${risk_amount:.2f}"
				)
				return out
			floored = self.cfg.min_lot
			limiting = "min_lot_floor"

		out.update({
			"ok": True,
			"reason": "OK",
			"lot": round(floored, 3),
			"limiting_factor": limiting,
		})

		logger.info(
			"[RISK][%s] Plan OK | entry=%.4f sl=%.4f tp=%.4f | SL%%=%.2f RR=%.2f | "
			"lot=%.2f (risk=%.2f tek=%.2f port=%.2f) | limit=%s | lotUSD=$%.0f",
			mt5_sym, entry_price, sl_price, tp_price,
			sl_pct*100, out["rr"], floored,
			risk_lot, max_pos_lot, portfolio_lot, limiting, lot_value_usd,
		)
		return out


# ─────────────────────────────────────────────────────────────────
# MT5 Yardımcıları
# ─────────────────────────────────────────────────────────────────

def _normalize_volume(mt5_sym: str, volume: float) -> float:
	if not _MT5_AVAILABLE:
		return round(volume, 2)
	info = mt5.symbol_info(mt5_sym)
	if info is None:
		return round(volume, 2)
	step	= info.volume_step
	min_vol = info.volume_min
	max_vol = info.volume_max
	normalized = round(round(volume / step) * step, 2)
	return max(min_vol, min(normalized, max_vol))


async def _fetch_mt5_balance(dry_run: bool = False) -> float:
	if dry_run or not _MT5_AVAILABLE:
		return 100_000.0
	try:
		if not mt5.initialize():
			logger.error("MT5 baslatilamadi!")
			return 100_000.0
		account_info = mt5.account_info()
		if account_info is not None:
			balance = float(account_info.balance)
			if balance == 0:
				logger.warning("MT5 bakiye 0 döndu.")
			return balance
		logger.warning("MT5 account_info None döndu.")
		return 100_000.0
	except Exception as exc:
		logger.error("MT5 bakiye hatasi: %s", exc)
		return 100_000.0

async def _fetch_total_exposure_usd(cfg: "BotConfig", dry_run: bool = False) -> float:
	"""Tüm açık pozisyonların toplam USD değeri (contract_size × volume × price)."""
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
				# Fallback: config'ten
				cs = cfg.symbol_contract_sizes.get(pos.symbol, 1.0)
			else:
				cs = info.trade_contract_size
			total += pos.volume * cs * pos.price_current
		return total
	except Exception as exc:
		logger.error("[EXPOSURE] Hesaplama hatasi: %s", exc)
		return 0.0

# ─────────────────────────────────────────────────────────────────
# CoinGlass Yöneticisi	(v4 API — Aggregated Orderbook + Balina Verileri)
# ─────────────────────────────────────────────────────────────────

def _calc_poc_from_ohlcv(df: Optional[pd.DataFrame], bins: int = 24) -> Optional[float]:
	if df is None or df.empty or len(df) < 2:
		return None
	try:
		lo = float(df["low"].min())
		hi = float(df["high"].max())
		if lo >= hi:
			return None
		bin_size = (hi - lo) / bins
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


class CoinglassManager:
	"""
	CoinGlass v4 API üzerinden tüm piyasa verilerini çeker.
	Heatmap KALDIRILDI. Bunun yerine Aggregated Orderbook Clustering.
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
				"CG-API-KEY": self.api_key,
				"accept": "application/json",
				"Content-Type": "application/json",
			},
			timeout=aiohttp.ClientTimeout(total=15),
		)
		return self

	async def __aexit__(self, *_) -> None:
		if self._session:
			await self._session.close()
		self._session = None

	# ── HTTP Yardımcısı ─────────────────────────────────────────

	async def _get(self, endpoint: str, params: Optional[dict] = None) -> any:
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
				code = str(payload.get("code", ""))
				
				if not (success is True or code in ["0", "000000", "200"]):
					logger.warning(
						"[CG] API basarisiz | %s | msg=%s | code=%s",
						endpoint, payload.get("msg", "?"), code
					)
					return None
				return payload.get("data")
		except asyncio.TimeoutError:
			logger.warning("[CG] Timeout | %s", endpoint)
			return None
		except Exception as exc:
			logger.error("[CG] Istek hatasi | %s: %s", endpoint, exc)
			return None

	# ── Endpoint Çekiciler ───────────────────────────────────────

	async def _fetch_open_interest(self, cg_sym: str) -> Optional[list]:
		data = await self._get(
			"/api/futures/open-interest/aggregated-history",
			{"symbol": cg_sym, "interval": "h4", "limit": 5},
		)
		if isinstance(data, dict) and "list" in data: return data["list"]
		return data if isinstance(data, (dict, list)) else None

	async def _fetch_funding_rate(self, cg_sym: str) -> Optional[list]:
		data = await self._get(
			"/api/futures/funding-rate/exchange-list",
			{"symbol": cg_sym},
		)
		if isinstance(data, dict) and "list" in data: return data["list"]
		return data if isinstance(data, (dict, list)) else None

	async def _fetch_ls_ratio(self, cg_sym: str) -> Optional[list]:
		pair = f"{cg_sym}USDT"
		data = await self._get(
			"/api/futures/top-long-short-account-ratio/history",
			{"exchange": "Binance", "symbol": pair, "interval": "h4", "limit": 1},
		)
		if isinstance(data, dict) and "list" in data: return data["list"]
		return data if isinstance(data, (dict, list)) else None

	async def _fetch_liquidation_history(self, cg_sym: str) -> Optional[list]:
		# Hata Çözümü: exchange_list eklendi.
		data = await self._get(
			"/api/futures/liquidation/aggregated-history",
			{"symbol": cg_sym, "interval": "h4", "limit": 10, "exchange_list": "Binance,OKX,Bybit,Bitget"},
		)
		if isinstance(data, dict) and "list" in data: return data["list"]
		return data if isinstance(data, list) else None

	# ── Parse Yardımcıları ───────────────────────────────────────

	@staticmethod
	def _parse_oi(raw: Optional[dict | list]) -> dict:
		if not raw:
			return {"yorum": "Veri alinamadi", "degisim_4h_pct": None}
		try:
			if isinstance(raw, list) and len(raw) >= 2:
				# Loglarda gördüğümüz 'close' key'i eklendi
				cur	 = float(raw[-1].get("openInterest") or raw[-1].get("close") or raw[-1].get("c") or 0)
				prev = float(raw[-2].get("openInterest") or raw[-2].get("close") or raw[-2].get("c") or cur)
			elif isinstance(raw, dict):
				cur	 = float(raw.get("openInterest") or raw.get("openInterestCurrent") or raw.get("close") or 0)
				prev = float(raw.get("openInterestPrev") or raw.get("close") or cur)
			else:
				return {"yorum": "Format taninamadi", "degisim_4h_pct": None}
		except (TypeError, ValueError):
			return {"yorum": "Parse hatasi", "degisim_4h_pct": None}
			
		change = ((cur - prev) / prev * 100) if prev else 0.0
		if change > 3:
			yorum = "OI artisi (GERCEK TREND): Pozisyon aciliyor"
		elif change < -3:
			yorum = "OI dususu: Squeeze/kapanis riski — SAHTE KIRILIM"
		else:
			yorum = "OI dengeli: Net yon yok"
		return {"mevcut_oi_usd": round(cur, 0), "degisim_4h_pct": round(change, 2), "yorum": yorum}

	@staticmethod
	def _parse_funding_rate(raw: Optional[dict | list], target_symbol: str = "") -> dict:
		if not raw:
			return {"yorum": "Veri alinamadi", "oran_pct": None}

		rate = 0.0
		borsa_sayisi = 0

		try:
			# Eğer liste döndüyse: hedef sembolü bul, yoksa ilk öğeye düş
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
					# Hedef sembol listede yoksa: hata bildir, eski davranışa düşme
					if target_symbol:
						return {
							"yorum": f"FR listesinde {target_symbol} bulunamadi (API filter calismiyor)",
							"oran_pct": None,
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
					rate = sum(rates) / len(rates)
					borsa_sayisi = len(rates)

			elif isinstance(raw, dict):
				rate = float(raw.get("funding_rate") or raw.get("rate") or 0)
				borsa_sayisi = 1
		except (TypeError, ValueError):
			rate = 0.0

		if rate > 0.01:
			yorum = "YUKSEK pozitif: Short squeeze riski"
		elif rate > 0.003:
			yorum = "Pozitif: Hafif long agirlikli"
		elif rate < -0.01:
			yorum = "YUKSEK negatif: Long squeeze firsati"
		elif rate < -0.003:
			yorum = "Negatif: Hafif short agirlikli"
		else:
			yorum = "Normal: Funding dengeli"

		ek_bilgi = f" ({borsa_sayisi} borsa ort.)" if borsa_sayisi > 1 else ""
		return {"oran_pct": f"{rate * 100:.4f}%{ek_bilgi}", "yorum": yorum}

	@staticmethod
	def _parse_ls_ratio(raw: Optional[dict | list]) -> dict:
		if not raw:
			return {"oran": None, "yorum": "Veri alinamadi"}
		try:
			item = raw[0] if isinstance(raw, list) and raw else raw
			if not isinstance(item, dict):
				return {"oran": None, "yorum": "Format taninamadi"}

			# Loglarda gördüğümüz 'top_account_long_short_ratio' eklendi
			ratio = float(
				item.get("top_account_long_short_ratio")
				or item.get("longShortRatio")
				or item.get("ratio")
				or 0
			)
		except (TypeError, ValueError, ZeroDivisionError):
			return {"oran": None, "yorum": "Parse hatasi"}

		if ratio > 2.5:
			yorum = f"ASIRI LONG (L/S={ratio:.2f}): Market Maker karsi taraf acar. SHORT sinyali!"
		elif ratio > 0 and ratio < 0.8:
			yorum = f"ASIRI SHORT (L/S={ratio:.2f}): Market Maker karsi taraf acar. LONG sinyali!"
		elif ratio > 1.5:
			yorum = f"Long agirlikli (L/S={ratio:.2f}): Dikkatli ol"
		else:
			yorum = f"Dengeli (L/S={ratio:.2f}): Net Market Maker sinyali yok"

		return {"oran": round(ratio, 3), "yorum": yorum}

	@staticmethod
	def _parse_liquidation_history(raw: Optional[list]) -> list[dict]:
		if not raw or not isinstance(raw, list):
			return []
		result = []
		for item in raw[:10]:
			try:
				# RAW loglarda gördüğümüz yeni V4 kelimeleri eklendi
				long_vol = float(
					item.get("aggregated_long_liquidation_usd") 
					or item.get("longVolUsd") or item.get("longVol") or 0
				)
				short_vol = float(
					item.get("aggregated_short_liquidation_usd") 
					or item.get("shortVolUsd") or item.get("shortVol") or 0
				)
				time_str = str(item.get("createTime") or item.get("time") or item.get("t") or "")[:16]
				price = float(item.get("price") or item.get("liquidationPrice") or 0)
				
				if long_vol > 0 or short_vol > 0:
					dom_side = "LONG PATLADI" if long_vol > short_vol else "SHORT PATLADI"
					result.append({
						"price": 0.0,
						"side": f"TÜM PİYASA {dom_side}",
						"amount_usd": round(max(long_vol, short_vol), 0),
						"time": time_str
					})
				elif price > 0:
					side_raw = str(item.get("side") or item.get("direction") or "").strip().lower()
					side_label = "LONG PATLADI" if "sell" in side_raw or "long" in side_raw else "SHORT PATLADI"
					amount = float(item.get("amount") or item.get("usd") or item.get("qty") or 0)
					result.append({
						"price": round(price, 4),
						"side": side_label,
						"amount_usd": round(amount, 0),
						"time": time_str,
					})
			except (TypeError, ValueError):
				continue
		return result

	# ── Ana Veri Toplama Metodu — Sadece Balina Verileri ────────

	async def get_coinglass_data(
		self,
		symbol:		   str,
		current_price: float,
		ohlcv_1d:	   Optional[pd.DataFrame] = None,
	) -> dict:
		"""
		Sadece balina verilerini ceker (OI, FR, L/S Ratio, Likidasyon).
		Orderbook artik LiquidityManager (CCXT) tarafından saglanir.
		"""
		cg_sym = self._SYM_MAP.get(symbol, symbol.split("/")[0])
		logger.info("[CG][%s] Balina verileri cekiliyor (v4, sym=%s)...", symbol, cg_sym)

		oi_raw, fr_raw, ls_raw, liq_raw = await asyncio.gather(
			self._fetch_open_interest(cg_sym),
			self._fetch_funding_rate(cg_sym),
			self._fetch_ls_ratio(cg_sym),
			self._fetch_liquidation_history(cg_sym),
			return_exceptions=True,
		)

		logger.info(">>> RAW OI	 : %s", str(oi_raw)[:250])
		logger.info(">>> RAW FR	 : %s", str(fr_raw)[:250])
		logger.info(">>> RAW LS	 : %s", str(ls_raw)[:250])
		logger.info(">>> RAW LIQ : %s", str(liq_raw)[:250])

		for name, val in [("OI", oi_raw), ("FR", fr_raw), ("LS", ls_raw), ("Liq", liq_raw)]:
			if isinstance(val, Exception):
				logger.error("[CG][%s] %s hatasi: %s", symbol, name, val)

		if isinstance(oi_raw,  Exception): oi_raw  = None
		if isinstance(fr_raw,  Exception): fr_raw  = None
		if isinstance(ls_raw,  Exception): ls_raw  = None
		if isinstance(liq_raw, Exception): liq_raw = None

		oi_sum	 = self._parse_oi(oi_raw)
		fr_sum	= self._parse_funding_rate(fr_raw, target_symbol=cg_sym)
		ls_sum	 = self._parse_ls_ratio(ls_raw)
		liq_list = self._parse_liquidation_history(liq_raw)
		poc		 = _calc_poc_from_ohlcv(ohlcv_1d)

		logger.info(
			"[CG][%s] Tamamlandi | OI=%s%% | FR=%s | L/S=%s | Liq=%d",
			symbol,
			oi_sum.get("degisim_4h_pct", "?"),
			fr_sum.get("oran_pct", "?"),
			ls_sum.get("oran", "?"),
			len(liq_list),
		)

		return {
			"symbol":			   symbol,
			"current_price":	   current_price,
			"poc_fiyati":		   round(poc, 4) if poc else None,
			"open_interest":	   oi_sum,
			"funding_rate":		   fr_sum,
			"ls_ratio":			   ls_sum,
			"liquidation_history": liq_list,
		}

# ─────────────────────────────────────────────────────────────────
# Limit Emir Gönderici
# ─────────────────────────────────────────────────────────────────

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
	side_str   = "LONG" if decision.upper() == "LONG" else "SHORT"
	order_type = (
		mt5.ORDER_TYPE_BUY_LIMIT if side_str == "LONG"
		else mt5.ORDER_TYPE_SELL_LIMIT
	) if _MT5_AVAILABLE else None

	request: dict = {
		"action":	 "TRADE_ACTION_PENDING",
		"symbol":	 mt5_sym,
		"volume":	 float(volume),
		"type":		 f"{'BUY' if side_str == 'LONG' else 'SELL'}_LIMIT",
		"price":	 float(entry_price),
		"type_time": "ORDER_TIME_GTC",
		"magic":	 _MAGIC,
		"comment":	 comment,
	}
	if sl_price:
		request["sl"] = float(sl_price)
	if tp_price:
		request["tp"] = float(tp_price)

	if dry_run or not _MT5_AVAILABLE:
		sim_ticket = f"DRY-{int(time.time())}"
		trade_logger.info(
			"[DRY-RUN] %s Limit | %s @ %.4f | vol=%.3f | sl=%s | tp=%s | ticket=%s",
			side_str, mt5_sym, entry_price, volume, sl_price, tp_price, sim_ticket,
		)
		return ExecutionResult(
			status=ExecutionStatus.DRY_RUN,
			order_id=sim_ticket,
			symbol=mt5_sym,
			side=side_str,
			price=entry_price,
			volume=volume,
			sl=sl_price,
			tp=tp_price,
			message="dry_run=True Limit emir simule edildi.",
			raw=request,
		)

	mt5_request = {
		"action":	 mt5.TRADE_ACTION_PENDING,
		"symbol":	 mt5_sym,
		"volume":	 float(volume),
		"type":		 order_type,
		"price":	 float(entry_price),
		"type_time": mt5.ORDER_TIME_GTC,
		"magic":	 _MAGIC,
		"comment":	 comment,
	}
	if sl_price:
		mt5_request["sl"] = float(sl_price)
	if tp_price:
		mt5_request["tp"] = float(tp_price)

	result = await asyncio.to_thread(mt5.order_send, mt5_request)

	if result is None:
		err = await asyncio.to_thread(mt5.last_error)
		msg = f"order_send None | mt5_err={err}"
		logger.error("[LIMIT] %s", msg)
		return ExecutionResult(
			status=ExecutionStatus.FAILED,
			symbol=mt5_sym, side=side_str,
			price=entry_price, volume=volume,
			sl=sl_price, tp=tp_price,
			message=msg, raw=mt5_request,
		)

	if result.retcode == mt5.TRADE_RETCODE_DONE:
		msg = f"Limit emir yerlestirildi | ticket={result.order}"
		logger.info("[LIMIT] %s | %s @ %.4f | vol=%.3f", side_str, mt5_sym, entry_price, volume)
		return ExecutionResult(
			status=ExecutionStatus.SUCCESS,
			order_id=str(result.order),
			symbol=mt5_sym, side=side_str,
			price=entry_price, volume=volume,
			sl=sl_price, tp=tp_price,
			message=msg, raw=result._asdict(),
		)

	err = await asyncio.to_thread(mt5.last_error)
	msg = f"Limit emir reddedildi | retcode={result.retcode} | mt5_err={err}"
	logger.error("[LIMIT] %s", msg)
	return ExecutionResult(
		status=ExecutionStatus.FAILED,
		symbol=mt5_sym, side=side_str,
		price=entry_price, volume=volume,
		sl=sl_price, tp=tp_price,
		message=msg, raw=result._asdict(),
	)


# ─────────────────────────────────────────────────────────────────
# OHLCV Çekiciler
# ─────────────────────────────────────────────────────────────────

async def _fetch_ohlcv(
	exchange,
	symbol:	   str,
	timeframe: str,
	limit:	   int,
) -> Optional[pd.DataFrame]:
	try:
		raw = await exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
		if not raw:
			return None
		df = pd.DataFrame(
			raw, columns=["timestamp", "open", "high", "low", "close", "volume"]
		)
		df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
		return df.set_index("timestamp")
	except Exception as exc:
		logger.error("[OHLCV][%s/%s] Hata: %s", symbol, timeframe, exc)
		return None


async def fetch_ohlcv_all(
	exchange,
	symbol:		str,
	timeframes: list[str],
	limit:		int,
) -> dict[str, Optional[pd.DataFrame]]:
	results = await asyncio.gather(
		*(_fetch_ohlcv(exchange, symbol, tf, limit) for tf in timeframes),
		return_exceptions=False,
	)
	return dict(zip(timeframes, results))


# ─────────────────────────────────────────────────────────────────
# Liquidity Manager (CCXT — Binance, OKX, Bybit)
# ─────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────
# Liquidity Manager (CCXT — 5 Borsa Kümülatif Derinlik)
# ─────────────────────────────────────────────────────────────────

class LiquidityManager:
	"""
	CCXT kullanarak 5 dev borsadan eş zamanlı Order Book (L2) verisi çeker.
	"""
	
	_EXCHANGE_OPTS: dict = {
		"enableRateLimit": True, 
		"timeout": 15_000,
		"options": {"defaultType": "swap"}	# ÇOK KRİTİK: Spot değil, Vadeli(Futures) tahtasını çeker
	}

	def __init__(self) -> None:
		self._exchanges: dict = {}

	async def __aenter__(self) -> "LiquidityManager":
		# 5 Borsa Eklendi
		self._exchanges = {
			"binance": ccxt_async.binance(self._EXCHANGE_OPTS),
			"okx":	   ccxt_async.okx(self._EXCHANGE_OPTS),
			"bybit":   ccxt_async.bybit(self._EXCHANGE_OPTS),
			"gate":	   ccxt_async.gate(self._EXCHANGE_OPTS),
			"mexc":	   ccxt_async.mexc(self._EXCHANGE_OPTS),
		}
		return self

	async def __aexit__(self, *_) -> None:
		await asyncio.gather(
			*(ex.close() for ex in self._exchanges.values()),
			return_exceptions=True,
		)
		self._exchanges.clear()

	async def _fetch_one(self, name: str, exchange, symbol: str) -> Optional[dict]:
		# Borsaların maksimum limit sınırları
		limit_map = {"binance": 1000, "okx": 400, "bybit": 500, "gate": 1000, "mexc": 1000}
		limit = limit_map.get(name, 500)
		try:
			ob = await exchange.fetch_order_book(symbol, limit=limit)
			return ob
		except Exception as exc:
			logger.warning("[LM][%s] Order book hatasi (%s): %s", name, symbol, exc)
			return None

	async def get_orderbook_walls(
		self,
		symbol:		   str,
		current_price: float,
		depth_pct:	   float = 0.20,
		tier_pct:	   float = 0.01,
		top_n:		   int	 = 3,
	) -> dict:
		results = await asyncio.gather(
			*(self._fetch_one(name, ex, symbol) for name, ex in self._exchanges.items()),
			return_exceptions=True,
		)

		all_bids: list = []
		all_asks: list = []
		fetched = 0

		for name, result in zip(self._exchanges.keys(), results):
			if isinstance(result, Exception) or result is None:
				continue
			all_bids.extend(result.get("bids", []))
			all_asks.extend(result.get("asks", []))
			fetched += 1

		logger.info(
			"[LM][%s] %d/%d borsa OK | Toplam Çekilen Kademeler: Bid=%d | Ask=%d",
			symbol, fetched, len(self._exchanges), len(all_bids), len(all_asks),
		)

		return self._cluster(symbol, all_bids, all_asks, current_price, depth_pct, tier_pct, top_n, fetched)

	@staticmethod
	def _cluster(
		symbol:			str,
		all_bids:		list,
		all_asks:		list,
		current_price:	float,
		depth_pct:		float,
		tier_pct:		float,
		top_n:			int,
		exchanges_count: int,
	) -> dict:
		lower	= current_price * (1 - depth_pct)
		upper	= current_price * (1 + depth_pct)
		n_tiers = max(1, int(round(depth_pct / tier_pct)))

		bid_buckets: dict[int, float] = {}
		ask_buckets: dict[int, float] = {}

		for level in all_bids:
			try:
				price, qty = float(level[0]), float(level[1])
			except (TypeError, ValueError, IndexError):
				continue
			if lower <= price < current_price:
				k = min(int((current_price - price) / (current_price * tier_pct)), n_tiers - 1)
				bid_buckets[k] = bid_buckets.get(k, 0.0) + price * qty

		for level in all_asks:
			try:
				price, qty = float(level[0]), float(level[1])
			except (TypeError, ValueError, IndexError):
				continue
			if current_price <= price <= upper:
				k = min(int((price - current_price) / (current_price * tier_pct)), n_tiers - 1)
				ask_buckets[k] = ask_buckets.get(k, 0.0) + price * qty

		def _build_walls(buckets: dict, direction: str) -> list[dict]:
			walls = []
			for k, vol in buckets.items():
				if k == 0:	# KANKA İŞTE BURASI: İlk %1'lik Market Maker gürültüsünü atlıyoruz!
					continue
				if direction == "bid":
					hi	= current_price * (1 - k * tier_pct)
					lo	= current_price * (1 - (k + 1) * tier_pct)
					pct = -(k + 0.5) * tier_pct * 100
				else:
					lo	= current_price * (1 + k * tier_pct)
					hi	= current_price * (1 + (k + 1) * tier_pct)
					pct = (k + 0.5) * tier_pct * 100
				walls.append({
					"aralik":			f"{lo:,.2f}-{hi:,.2f}",
					"pct_from_current": round(pct, 1),
					"hacim_usd":		round(vol, 0),
				})
			# Hacme göre KESMİYORUZ, sadece fiyata olan mesafeye göre sıralayıp TÜM listeyi veriyoruz
			return sorted(walls, key=lambda x: abs(x["pct_from_current"]))

		ask_walls = _build_walls(ask_buckets, "ask")
		bid_walls = _build_walls(bid_buckets, "bid")

		toplam_ask = sum(ask_buckets.values())
		toplam_bid = sum(bid_buckets.values())

		# ================= GÖRSEL ANALİZ LOGU (Terminalde Görmek İçin) =================
		logger.info("\n" + "="*50)
		logger.info("🎯 [%s] 5 BORSA KÜMÜLATİF LİKİDİTE HARİTASI", symbol)
		logger.info("Toplam Ask Hacmi: $%s | Toplam Bid Hacmi: $%s", f"{toplam_ask:,.0f}", f"{toplam_bid:,.0f}")
		
		logger.info("--- 🔴 DİRENÇ DUVARLARI (En Büyük %d Ask Dilimi) ---", len(ask_walls))
		if not ask_walls: logger.info("	 >> Duvar tespit edilemedi (Limit yetersiz).")
		for w in ask_walls:
			logger.info("  >> Fiyat: %s | Mesafe: %+1.1f%% | Hacim: $%s", w["aralik"], w["pct_from_current"], f"{w['hacim_usd']:,.0f}")
			
		logger.info("--- 🟢 DESTEK DUVARLARI (En Büyük %d Bid Dilimi) ---", len(bid_walls))
		if not bid_walls: logger.info("	 >> Duvar tespit edilemedi (Limit yetersiz).")
		for w in bid_walls:
			logger.info("  >> Fiyat: %s | Mesafe: %+1.1f%% | Hacim: $%s", w["aralik"], w["pct_from_current"], f"{w['hacim_usd']:,.0f}")
		logger.info("="*50 + "\n")
		# ===============================================================================

		return {
			"exchanges_fetched": exchanges_count,
			"ask_walls":		 ask_walls,
			"bid_walls":		 bid_walls,
			"toplam_ask_usd":	 round(toplam_ask, 0),
			"toplam_bid_usd":	 round(toplam_bid, 0),
		}


# ─────────────────────────────────────────────────────────────────
# Gemini Sistem Promptları	[YENI RUBRIC — Orderbook + Balina]
# ─────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """
Sen, FundedNext prop firması kurallarına, ICT/SMC metodolojisine ve
CoinGlass Balina Verilerine göre Swing Trade fırsatlarını değerlendiren
kurumsal seviyede YORUM YAPMAYAN, KESİN (DETERMİNİSTİK) bir yapay zeka analistisin.

DİKKAT: EMİR DEFTERİ (ORDERBOOK) VERİSİ SİSTEMDEN TAMAMEN KALDIRILMIŞTIR. Destek, direnç ve hedef (TP/SL) belirlemek için SADECE "Majör S/R" seviyelerini kullanacaksın.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ADIM 1: YÖN (BIAS) BELİRLEME (SADECE TEK BİR YÖN SEÇ)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Puanlamaya geçmeden önce işlemin yönüne KESİN olarak karar ver.
- Eğer 1D Trend BULLISH ise ve CoinGlass L/S Ratio < 1.5 (Sürü Short'ta) ise yönün LONG olmalıdır.
- Eğer 1D Trend BEARISH ise ve CoinGlass L/S Ratio > 1.5 (Sürü Long'da) ise yönün SHORT olmalıdır.
- Eğer veriler çelişiyorsa veya 1D Trend RANGING ise doğrudan BEKLE kararı ver.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ADIM 2: DETERMİNİSTİK PUANLAMA RUBRİĞİ (Toplam: 100 Puan)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Seçtiğin YÖNE GÖRE aşağıdaki kuralları test et. Şart sağlanıyorsa TAM PUAN, sağlanmıyorsa 0 puan ver! Ara puan (kısmi puan) KESİNLİKLE YASAKTIR! (Ya 10 puan, ya 0 puan).

[ A ] PRICE ACTION & MAJÖR S/R (Max: 45 Puan)
LONG YÖNÜ İÇİN:
  [+10] 1D ve 4H Trendin İKİSİ BİRDEN "BULLISH" mi? (Evet:10, Hayır:0)
  [+10] Son 30 mumda yukarı yönlü BOS_BULL veya CHOCH_BULL var mı? (Evet:10, Hayır:0)
  [+15] Fiyat şu an bir "MAJÖR DESTEK" veya "SR_FLIP_SUPPORT" seviyesine temas ediyor/çok yakın mı? (Evet:15, Hayır:0)
  [+10] Fiyatın altında destekleyici bir BULL_FVG veya BULL_OB var mı? (Evet:10, Hayır:0)

SHORT YÖNÜ İÇİN:
  [+10] 1D ve 4H Trendin İKİSİ BİRDEN "BEARISH" mi? (Evet:10, Hayır:0)
  [+10] Son 30 mumda aşağı yönlü BOS_BEAR veya CHOCH_BEAR var mı? (Evet:10, Hayır:0)
  [+15] Fiyat şu an bir "MAJÖR DİRENÇ" veya "SR_FLIP_RESISTANCE" seviyesine temas ediyor/çok yakın mı? (Evet:15, Hayır:0)
  [+10] Fiyatın üstünde direnç gösteren bir BEAR_FVG veya BEAR_OB var mı? (Evet:10, Hayır:0)

[ B ] İNDİKATÖRLER (MULTI-BAND) (Max: 35 Puan)
LONG YÖNÜ İÇİN:
  [+10] Fiyat EMA-50 ve EMA-200'ün ÜSTÜNDE mi? (veya Güçlü Bullish EMA Bandı) (Evet:10, Hayır:0)
  [+10] RSI "OVERSOLD" veya "AŞIRI DİP" bandında mı? (Evet:10, Hayır:0)
  [+10] MACD Line > 0 ve Sinyalin Üstünde mi (BULLISH mi)? (Evet:10, Hayır:0)
  [+5] ATR Volatilitesi Sağlıklı veya Yüksek mi? (Ölü/Haber değilse 5, Haber ise 0)

SHORT YÖNÜ İÇİN:
  [+10] Fiyat EMA-50 ve EMA-200'ün ALTINDA mi? (veya Güçlü Bearish EMA Bandı) (Evet:10, Hayır:0)
  [+10] RSI "OVERBOUGHT" veya "AŞIRI TEPE" bandında mı? (Evet:10, Hayır:0)
  [+10] MACD Line < 0 ve Sinyalin Altında mı (BEARISH mi)? (Evet:10, Hayır:0)
  [+5] ATR Volatilitesi Sağlıklı veya Yüksek mi? (Ölü/Haber değilse 5, Haber ise 0)

[ C ] COINGLASS BALİNA VERİLERİ (Max: 20 Puan)
LONG YÖNÜ İÇİN:
  [+8] L/S Ratio Sürünün TERSİNE mi? (Ratio < 0.9 ise Sürü Short'tadır, LONG uygundur). (Evet:8, Hayır:0)
  [+7] Open Interest (OI) 4H değişimi POZİTİF mi? (Evet:7, Hayır:0)
  [+5] Funding Rate (FR) < %0.01 mi? (Aşırı pozitif değilse 5)

SHORT YÖNÜ İÇİN:
  [+8] L/S Ratio Sürünün TERSİNE mi? (Ratio > 2.0 ise Sürü Long'tadır, SHORT uygundur). (Evet:8, Hayır:0)
  [+7] Open Interest (OI) 4H değişimi POZİTİF mi? (Evet:7, Hayır:0)
  [+5] Funding Rate (FR) > -%0.01 mi? (Aşırı negatif değilse 5)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ADIM 3: RİSK, SL/TP HESAPLAMA VE KARAR
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Toplam Puan < 75 : KESİNLİKLE "BEKLE"
- Toplam Puan 75-84: LONG/SHORT, risk_yuzdesi=0.005
- Toplam Puan 85-100: LONG/SHORT, risk_yuzdesi=0.010

STOP-LOSS (SL) FORMÜLÜ (ÇOK KATI):
- SHORT için: SL = En yakın Majör DİRENÇ seviyesi * 1.002
- LONG için : SL = En yakın Majör DESTEK seviyesi * 0.998

TAKE-PROFIT (TP) FORMÜLÜ:
TP seviyesi bir sonraki Majör S/R seviyesi olmalıdır. RR (Risk/Ödül) >= 2.0 ŞARTTIR. Değilse BEKLE ver.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ADIM 4: ZORUNLU ÇIKTI FORMATI (YALNIZCA JSON)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"hesaplama_dokumu" adlı bir liste oluştur. Adım adım hangi maddeden kaç puan aldığını ve nedenini yaz! Bu sayede senin mantığını denetleyebileceğiz.

{
  "hesaplama_dokumu": [
	"YÖN SEÇİMİ: 1D Trend Bullish ve L/S Ratio 0.8. Sürü Short'ta olduğu için LONG yönü seçildi.",
	"PA_1: 1D ve 4H Trend BULLISH olduğu için 10 puan verildi.",
	"PA_2: Son 30 mumda BOS_BULL bulunmadığı için 0 puan verildi.",
	"IND_1: RSI 'OVERSOLD' olmadığı için 0 puan verildi."
  ],
  "karar": "LONG" | "SHORT" | "BEKLE",
  "gerekce": "YÖN: LONG | PA[X/45] | IND[X/35] | CG[X/20] | SL=fiyat | TP=fiyat",
  "giris_fiyati": 0.0,
  "sl": 0.0,
  "tp": 0.0,
  "guvven": 0.0,
  "risk_yuzdesi": 0.0,
  "toplam_puan": 0,
  "puan_detayi": {
	"price_action_majör_sr": 0,
	"indikatorler": 0,
	"coinglass_balina": 0
  },
  "kilit_seviyeler": ["Giriş için Baz Alınan Destek: 60000", "Hedef Direnç: 64000"]
}
"""

_POST_MORTEM_SYSTEM_PROMPT = """
Sen, bir algoritmik trading botunun kapanan islemlerini analiz eden,
ICT/SMC, CoinGlass Orderbook ve Majör S/R metodolojisine göre
ozele stiri yapan teknik bir trading post-mortem analistisin.

Görevin: Verilen islem detaylarini inceleyip, gelecekte ayni hatanin
tekrar yapilmamasi icin somut, uygulanabilir bir ders cikarmak.

Kurallar:
- Teknik ICT/SMC, Orderbook ve Majör S/R terminolojisini kullan.
- Maksimum 2-3 cümle, özlü ve net ol.
- Türkce yanit ver.
- YALNIZCA gecerli JSON döndür, baska metin ekleme.
""".strip()


# ─────────────────────────────────────────────────────────────────
# Post-Trade Feedback Loop — Dosya Yardımcıları
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
		logger.error("[JOURNAL] Dosya yazma hatasi (%s): %s", path, exc)


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
			"karar":		   decision.get("karar"),
			"gerekce":		   decision.get("gerekce"),
			"toplam_puan":	   decision.get("toplam_puan"),
			"puan_detayi":	   decision.get("puan_detayi"),
			"guvven":		   decision.get("guvven"),
			"risk_yuzdesi":	   decision.get("risk_yuzdesi"),
			"kilit_seviyeler": decision.get("kilit_seviyeler"),
		},
	}
	journal.append(entry)
	_save_json_file(journal_path, journal)
	logger.info("[JOURNAL] Yeni giris | ticket=%s | %s @ %.4f", ticket_id, data_sym, entry_price)


def _load_recent_lessons(lessons_path: str, n: int = 5) -> list[str]:
	lessons = _load_json_file(lessons_path, [])
	recent	= lessons[-n:] if len(lessons) > n else lessons
	return [e.get("ders", "") for e in recent if e.get("ders")]


# ─────────────────────────────────────────────────────────────────
# Puan Bazlı Kural Uygulayıcı
# ─────────────────────────────────────────────────────────────────

def _algorithmic_pre_gate(
	tf_analysis:	dict,
	current_price:	float,
	coinglass_data: dict,
	cfg:			"BotConfig",
) -> tuple[bool, str]:
	"""
	LLM çağrısından ÖNCE çalışan sert kapılar.
	Geçemezse BEKLE, Gemini'ye hiç gitmez.
	"""
	reasons: list[str] = []

	# A. ATR volatilite filtresi (15m)
	ind_15m = tf_analysis.get("15m", {}).get("indicators", {}) or {}
	atr_pct = ind_15m.get("atr_pct") or 0.0
	if 0 < atr_pct < 0.15:
		reasons.append(f"15m ATR%%={atr_pct:.3f} cok dusuk (piyasa olu)")
	if atr_pct > 5.0:
		reasons.append(f"15m ATR%%={atr_pct:.3f} cok yuksek (haber/manipulasyon riski)")

	# B. Funding rate ekstrem mi?
	fr = coinglass_data.get("funding_rate", {}) or {}
	fr_str = str(fr.get("oran_pct", "0%"))
	try:
		# "0.0123% (3 borsa ort.)" gibi formattan sayıyı çek
		fr_clean = fr_str.split("%")[0].strip()
		fr_val = float(fr_clean) / 100	# decimal'e çevir
		# 4H funding için %0.075 üstü = aşırı (yıllık ~%82 implied)
		if abs(fr_val) > 0.00075:
			reasons.append(f"FR ekstrem ({fr_val*100:.4f}%%): suru ayni yonde, kontra riski yuksek")
	except (ValueError, AttributeError):
		pass

	# C. HTF trend uyumsuzluğu (1D vs 4H çelişiyorsa swing girmiyoruz)
	trend_1d = tf_analysis.get("1d", {}).get("trend", "")
	trend_4h = tf_analysis.get("4h", {}).get("trend", "")
	if trend_1d and trend_4h:
		# Tam zıt trendler swing trade için ölümcül
		if (trend_1d == "BULLISH" and trend_4h == "BEARISH") or \
		   (trend_1d == "BEARISH" and trend_4h == "BULLISH"):
			reasons.append(f"HTF zit: 1D={trend_1d} | 4H={trend_4h}")

	# D. RSI extrem kırılım (15m'de RSI > 85 veya < 15 ise momentum tamamlanmış)
	rsi_15m = ind_15m.get("rsi_14")
	if rsi_15m is not None:
		if rsi_15m >= 85:
			reasons.append(f"15m RSI={rsi_15m} asiri yorgun (tepe yakini)")
		elif rsi_15m <= 15:
			reasons.append(f"15m RSI={rsi_15m} asiri yorgun (dip yakini)")

	if reasons:
		return False, " | ".join(reasons)
	return True, "OK"

def _build_market_warnings(
	tf_analysis:	dict,
	coinglass_data: dict,
	btc_regime:		Optional[dict] = None,
) -> list[str]:
	"""
	Sert kapı değil. AI'a bilgi olarak verilecek uyarılar listesi.
	Bunlar puanlama girdisi, kesin RED değil.
	"""
	warnings: list[str] = []

	# Funding ekstrem
	fr = coinglass_data.get("funding_rate", {}) or {}
	fr_str = str(fr.get("oran_pct", "0%"))
	try:
		fr_clean = fr_str.split("%")[0].strip()
		fr_val = float(fr_clean) / 100
		if abs(fr_val) > 0.00075:
			yon = "LONG" if fr_val > 0 else "SHORT"
			warnings.append(
				f"FUNDING EKSTREM ({fr_val*100:.4f}%): {yon} surusu kalabalik, "
				f"ters yon (kontra) firsati ara veya {yon} icin daha sert filtre uygula."
			)
	except (ValueError, AttributeError):
		pass

	# ATR volatilite
	ind_15m = tf_analysis.get("15m", {}).get("indicators", {}) or {}
	atr_pct = ind_15m.get("atr_pct") or 0.0
	if 0 < atr_pct < 0.15:
		warnings.append(
			f"DUSUK VOLATILITE (15m ATR%={atr_pct:.3f}): Piyasa olu, "
			f"swing trade icin yetersiz hareket; sadece cok yuksek confluence ile gir."
		)
	elif atr_pct > 5.0:
		warnings.append(
			f"YUKSEK VOLATILITE (15m ATR%={atr_pct:.3f}): Haber/manipulasyon riski; "
			f"SL'i genis tutma ihtiyaci olabilir, ama %3 maks sinirina dikkat."
		)

	# HTF uyumsuzluğu
	trend_1d = tf_analysis.get("1d", {}).get("trend", "")
	trend_4h = tf_analysis.get("4h", {}).get("trend", "")
	if (trend_1d == "BULLISH" and trend_4h == "BEARISH"):
		warnings.append(
			"HTF UYUMSUZLUK: 1D BULLISH, 4H BEARISH — 4H pull-back BULLISH'e donerse "
			"LONG firsati, devam ederse SHORT ama dikkatli."
		)
	elif (trend_1d == "BEARISH" and trend_4h == "BULLISH"):
		warnings.append(
			"HTF UYUMSUZLUK: 1D BEARISH, 4H BULLISH — 4H ralli BULLISH bitirse SHORT, "
			"1D direnclerinde reddedilirse SHORT firsati."
		)

	# RSI extrem (15m)
	rsi_15m = ind_15m.get("rsi_14")
	if rsi_15m is not None:
		if rsi_15m >= 85:
			warnings.append(f"15m RSI={rsi_15m} asiri alis — tepe yakini olabilir.")
		elif rsi_15m <= 15:
			warnings.append(f"15m RSI={rsi_15m} asiri satis — dip yakini olabilir.")

	# BTC bias (3e bölümünde dolduracağız)
	if btc_regime and btc_regime.get("strong_bias"):
		warnings.append(
			f"BTC BIAS: BTC {btc_regime['strong_bias']} (1D+4H uyumlu). "
			f"Altcoin'lerde sadece ayni yonde islem aramayi tercih et, "
			f"ters yonde sadece ASIRI gucu confluence varsa gir."
		)

	return warnings

def _apply_score_rules(decision: dict, min_score: int = 75) -> dict:
	d	  = dict(decision)
	puan  = int(d.get("toplam_puan", 0))
	karar = str(d.get("karar", "BEKLE")).upper()

	if puan < min_score:
		if karar != "BEKLE":
			logger.warning("[SCORE] Puan %d < %d BEKLE zorlandi (LLM: %s).", puan, min_score, karar)
		d["karar"]		  = "BEKLE"
		d["risk_yuzdesi"] = 0.0
		return d

	if 75 <= puan <= 84:
		raw = float(d.get("risk_yuzdesi", 0.005))
		if raw > 0.005:
			logger.info("[SCORE] Puan %d (75-84) risk_yuzdesi 0.005.", puan)
		d["risk_yuzdesi"] = 0.005
		return d

	raw = float(d.get("risk_yuzdesi", 0.010))
	if raw > 0.010:
		logger.info("[SCORE] Puan %d (85-100) risk_yuzdesi 0.010.", puan)
	d["risk_yuzdesi"] = min(raw, 0.010)
	return d


# ─────────────────────────────────────────────────────────────────
# Payload ve Prompt İnşacıları
# ─────────────────────────────────────────────────────────────────

_TF_ROLES = {
	"1d":  "Ana Trend — EMA sinyali ve büyük yapisal baglam",
	"4h":  "Orta Vade — OB/FVG, Premium/Discount bölgeleri",
	"1h":  "Likidite ve RSI Uyumsuzlugu — Anlik momentum",
	"15m": "Giris Tetikleyicisi — CHoCH, OB temas, FVG dönüs",
}


def build_master_payload(
	symbol:			  str,
	current_price:	  float,
	tf_analysis:	  dict,
	coinglass_data:	  dict,
	orderbook_walls:  dict,
	market_warnings:  Optional[list[str]] = None,
) -> dict:
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
		"key_levels":	   key_levels,
		"orderbook_walls": orderbook_walls,
		"coinglass_data":  coinglass_data,
		"market_warnings": market_warnings or [],
	}


def _fmt_events(events: list) -> str:
	if not events:
		return "Tespit edilmedi"
	return "; ".join(
		f"{e['type']} @ {e.get('level','?')} [{e.get('timestamp','')[:16]}]"
		for e in events[-3:]
	)


def _fmt_fvgs(fvgs: list) -> str:
	if not fvgs:
		return "Yok"
	return "; ".join(
		f"{f['type']} [{f['lower']}-{f['upper']}] {f.get('size_pct',0):.3f}%%"
		for f in fvgs
	)


def _fmt_obs(obs: list) -> str:
	if not obs:
		return "Yok"
	return "; ".join(
		f"{o['type']} [{o['lower']}-{o['upper']}] imp={o.get('impulse_pct',0):.1f}%%"
		for o in obs
	)


def _fmt_patterns(pats: list) -> str:
	if not pats:
		return "Yok"
	return "; ".join(
		f"{p['pattern']}({p.get('direction','?')}) @{p.get('timestamp','')[:16]}"
		for p in pats[-4:]
	)


def _fmt_indicators(ind: dict) -> str:
	if not ind:
		return "  Hesaplanamadi"
	lines = []
	e50, e200 = ind.get("ema_50"), ind.get("ema_200")
	if e50:
		lines.append(f"	 EMA-50	 : {e50} | Fiyat: {ind.get('price_vs_ema50','?')}")
	if e200:
		lines.append(
			f"	EMA-200 : {e200} | Fiyat: {ind.get('price_vs_ema200','?')} "
			f"| Sinyal: {ind.get('ema_signal','?')}"
		)
	rsi = ind.get("rsi_14")
	if rsi is not None:
		lines.append(f"	 RSI(14) : {rsi} → {ind.get('rsi_zone','?')}")
	atr = ind.get("atr_14")
	if atr is not None:
		atr_pct = ind.get("atr_pct") or 0
		lines.append(f"	 ATR(14) : {atr} ({atr_pct:.3f}%% | önerilen min SL: {atr_pct * 1.5:.3f}%%)")
	return "\n".join(lines) if lines else "	 Veri yetersiz"


def _fmt_key_levels(kl: dict) -> str:
	if not kl:
		return "  Majör seviye verisi yok (1D/4H verisi yetersiz)"
	lines: list[str] = []
	resistances = kl.get("major_resistances", [])
	supports	= kl.get("major_supports", [])
	sr_flips	= kl.get("sr_flips", [])

	if resistances:
		lines.append("	MAJÖR DIRENCLER (en yakin once):")
		for r in resistances[:4]:
			flip_tag = " [SR_FLIP: Eski Destek]" if r.get("label") == "SR_FLIP_RESISTANCE" else ""
			lines.append(
				f"	  {r['price']:,.4f} | {r['touch_count']} temas "
				f"(SH:{r['sh_touches']} SL:{r['sl_touches']}){flip_tag}"
			)
	else:
		lines.append("	MAJÖR DIRENC: Yeterli temas yok")

	if supports:
		lines.append("	MAJÖR DESTEKLER (en yakin once):")
		for s in supports[:4]:
			flip_tag = " [SR_FLIP: Eski Direnc Retest Bölgesi]" if s.get("label") == "SR_FLIP_SUPPORT" else ""
			lines.append(
				f"	  {s['price']:,.4f} | {s['touch_count']} temas "
				f"(SH:{s['sh_touches']} SL:{s['sl_touches']}){flip_tag}"
			)
	else:
		lines.append("	MAJÖR DESTEK: Yeterli temas yok")

	if sr_flips:
		lines.append("	S/R FLIP SEVİYELERİ (kritik giriş/TP bölgeleri):")
		for flip in sr_flips[:3]:
			lines.append(
				f"	  {flip['price']:,.4f} [{flip.get('label','')}] → "
				f"{flip.get('flip_desc', '')}"
			)
	return "\n".join(lines)


def _fmt_orderbook(ob: dict) -> str:
	"""
	LiquidityManager'dan gelen CCXT Orderbook kademelerini formatlar.
	Yönlendirme yapmaz, saf veriyi sunar.
	"""
	if not ob:
		return "  Orderbook verisi alinamadi (CCXT)."

	lines: list[str] = []
	ask_walls = ob.get("ask_walls", [])
	bid_walls = ob.get("bid_walls", [])
	ex_count  = ob.get("exchanges_fetched", 0)

	lines.append(f"	 Borsa: {ex_count}/5 Kümülatif Toplam (NOT: İlk %1'lik Market Maker gürültüsü filtrelenmiştir!)")

	if ask_walls:
		lines.append("\n  SATIS KADEMELERI (ASK / Mevcut Fiyatın Üzerindeki Emirler):")
		for w in ask_walls:
			lines.append(f"	   Fiyat: {w['aralik']} ({w['pct_from_current']:+.1f}%) -> Hacim: ${w['hacim_usd']:,.0f} USD")
	else:
		lines.append("\n  SATIS KADEMELERI (ASK): Veri yok")

	if bid_walls:
		lines.append("\n  ALIS KADEMELERI (BID / Mevcut Fiyatın Altındaki Emirler):")
		for w in bid_walls:
			lines.append(f"	   Fiyat: {w['aralik']} ({w['pct_from_current']:+.1f}%) -> Hacim: ${w['hacim_usd']:,.0f} USD")
	else:
		lines.append("\n  ALIS KADEMELERI (BID): Veri yok")

	return "\n".join(lines)


def _fmt_coinglass(cg: dict) -> str:
	"""
	CoinGlass v4 balina verilerini LLM'e gönderilecek formata çevirir.
	Sıra: POC → OI → FR → L/S Ratio → Likidasyonlar
	(Orderbook artik ayri bir bolumde — LiquidityManager CCXT)
	"""
	if not cg:
		return "  CoinGlass verisi alinamadi."

	lines: list[str] = []

	# POC
	poc = cg.get("poc_fiyati")
	if poc:
		lines.append(f"	 POC (Hacim Agirlik Merkezi / OHLCV): {poc:,.4f}")

	# Open Interest
	oi = cg.get("open_interest", {})
	if oi.get("yorum", "Veri alinamadi") != "Veri alinamadi":
		lines.append(
			f"\n  OPEN INTEREST: {oi.get('yorum','?')} "
			f"| Degisim(4H): {oi.get('degisim_4h_pct','?')}%%"
		)
	else:
		lines.append("\n  OPEN INTEREST: Veri alinamadi")

	# Funding Rate
	fr = cg.get("funding_rate", {})
	if fr.get("oran_pct"):
		lines.append(f"	 FUNDING RATE: {fr.get('oran_pct','?')} -> {fr.get('yorum','?')}")
	else:
		lines.append("	FUNDING RATE: Veri alinamadi")

	# L/S Ratio
	ls = cg.get("ls_ratio", {})
	if ls.get("yorum", "Veri alinamadi") != "Veri alinamadi":
		lines.append(f"	 L/S RATIO (Sürü Göstergesi): {ls.get('yorum','?')}")
	else:
		lines.append("	L/S RATIO: Veri alinamadi")

	# Son Likidasyonlar
	liq_list = cg.get("liquidation_history", [])
	if liq_list:
		lines.append("\n  SON LIKIDASYONLAR:")
		for liq in liq_list[:5]:
			price_str = f"${liq['price']:,.4f}" if liq.get("price") else "Birlesik"
			lines.append(
				f"	  {price_str} | {liq['side']} | "
				f"${liq['amount_usd']:,.0f} | {liq['time']}"
			)

	return "\n".join(lines)


def build_llm_prompt(payload: dict, lessons_path: str = "lessons_learned.json") -> str:
	"""
	Ana analiz promptu. Deterministik (Kesin) Checklist versiyonu.
	Orderbook kaldirildi.
	"""
	meta = payload["meta"]
	tda  = payload["top_down_analysis"]
	kl   = payload.get("key_levels", {})
	cg   = payload.get("coinglass_data", {})
	cp   = meta["current_price"]
	sym  = meta["symbol"]
	ts   = meta["timestamp_utc"]
	
	# İŞTE HATAYA SEBEP OLAN EKSİK SATIR BURASIYDI:
	sys_warn = meta.get("system_warnings", "Yok")

	# Hafıza: Geçmiş dersler
	recent_lessons = _load_recent_lessons(lessons_path, n=5)
	if recent_lessons:
		lesson_items = "\n".join(
			f"  {i + 1}. {lesson}" for i, lesson in enumerate(recent_lessons)
		)
		lessons_section = (
			"\n## GECMIS HATALARDAN CIKARILAN DERSLER"
			f" (Son {len(recent_lessons)} ders)\n"
			"Bu dersler gecmis islemlerden elde edilmistir. "
			"Ayni hatalari TEKRAR YAPMA!\n\n"
			f"{lesson_items}\n"
		)
	else:
		lessons_section = ""

	# TF Blokları
	tf_blocks = []
	for tf in ["1d", "4h", "1h", "15m"]:
		data = tda.get(tf, {})
		if "error" in data:
			tf_blocks.append(
				f"\n### {tf.upper()} — {_TF_ROLES.get(tf, tf)}\n  HATA: {data['error']}"
			)
			continue
		block = (
			f"\n### {tf.upper()} — {_TF_ROLES.get(tf, tf)}\n"
			f"- Trend/Yapi    : {data.get('trend','N/A')}\n"
			f"- Son Kapanis   : {data.get('last_close','?')} "
			f"({data.get('last_timestamp','')[:16]})\n"
			f"- BOS / CHoCH   : {_fmt_events(data.get('structure_events',[]))}\n"
			f"- Unmitigated FVG: {_fmt_fvgs(data.get('unmitigated_fvgs',[]))}\n"
			f"- Order Block   : {_fmt_obs(data.get('order_blocks',[]))}\n"
			f"- Formasyon     : {_fmt_patterns(data.get('candlestick_patterns',[]))}\n"
			f"- Indikatorler  :\n{_fmt_indicators(data.get('indicators',{}))}"
		)
		tf_blocks.append(block)

	return (
		f"# {sym} — FundedNext Swing Trade Analiz Raporu\n"
		f"Fiyat: {cp:,.6f} | UTC: {ts}\n"
		f"\n⚠️ SISTEM UYARILARI (KATI KURALLAR): {sys_warn}\n"
		+ lessons_section
		+ "\n## 1. TOP-DOWN HIBRIT ANALIZ\n"
		+ "".join(tf_blocks)
		+ "\n\n## 2. MAJÖR YEREL DESTEK/DİRENÇ SEVİYELERİ (1D+4H Historical S/R)\n"
		+ _fmt_key_levels(kl)
		+ "\n\n## 3. COINGLASS BALİNA VERİLERİ (OI / FR / L/S Ratio / Likidasyon)\n"
		+ _fmt_coinglass(cg)
		+ "\n\n## 4. PUANLAMA VE KARAR\n\n"
		+ "DIKKAT: Orderbook (Emir Defteri) YOKTUR! TP/SL için sadece Majör S/R seviyelerini kullan!\n"
		+ "DIKKAT: Eger BTC Ana Trendi ile zit islem acman (Korelasyon) isteniyorsa KESINLIKLE REDDET ve BEKLE ver!\n"
		+ "Her kategoriyi Deterministik Checklist'e gore KESIN puanla (ara puan yasak).\n"
		+ "SL seviyesini KESINLIKLE 1.002 veya 0.998 formuluyle hesapla.\n"
		+ "Puanlama mantigini 'hesaplama_dokumu' listesinde adim adim acikla.\n"
		+ "YALNIZCA JSON formatinda yanit ver:"
	)


# ─────────────────────────────────────────────────────────────────
# LLM Yanıt Ayrıştırıcı
# ─────────────────────────────────────────────────────────────────

def _parse_llm_response(text: str, required_key: str = "karar") -> dict:
	"""
	LLM yanıtından JSON çıkarır.
	required_key: bu anahtarı içeren ilk JSON'u öncelikle döndürür.
				  Hiç bulunamazsa son geçerli JSON'u döndürür.
	"""
	cleaned = re.sub(r"```(?:json)?\s*", "", text).replace("```", "").strip()
	decoder = json.JSONDecoder()
	idx = 0
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
					# Karar için ek default
					if required_key == "karar" and "toplam_puan" not in obj:
						obj["toplam_puan"] = 0
					return obj
			idx = brace + end
		except json.JSONDecodeError:
			idx = brace + 1

	if last_valid is not None:
		return last_valid
	raise ValueError(f"JSON bulunamadi. Yanit (ilk 300 kr): {text[:300]}")


# ─────────────────────────────────────────────────────────────────
# Ana Bot Sınıfı
# ─────────────────────────────────────────────────────────────────

class AlgoBot:
	def __init__(self, config: BotConfig) -> None:
		self.cfg	  = config
		self.analyzer = TechnicalAnalyzer(swing_order=5)
		self.risk_mgr = RiskManager(cfg=config)
		self._btc_regime: dict = {}
		self._last_ai: dict[str, float] = {
			sym: 0.0 for sym in config.data_symbols
		}
		self._last_history_check: float = 0.0

		genai.configure(api_key=config.gemini_api_key)

		# Deterministik puanlama icin dusuk temperature + JSON mode zorla
		generation_config = genai.GenerationConfig(
			temperature=0.2,
			top_p=0.8,
			top_k=40,
			response_mime_type="application/json",
		)

		self._model = genai.GenerativeModel(
			model_name=config.gemini_model,
			system_instruction=_SYSTEM_PROMPT,
			generation_config=generation_config,
		)
		self._post_mortem_model = genai.GenerativeModel(
			model_name=config.gemini_model,
			system_instruction=_POST_MORTEM_SYSTEM_PROMPT,
			generation_config=generation_config,
		)

	# ── Açık pozisyon yönetimi (trailing SL + breakeven) ─────────

	async def _manage_open_positions(self) -> None:
		"""
		Her açık pozisyon için:
		  - Fiyat TP yolunda %50'yi geçtiyse SL'yi entry'e taşı (breakeven)
		  - %75 ilerlemişse, SL'yi entry+%25 progress'e çek (kâr kilidi)
		"""
		if not _MT5_AVAILABLE or self.cfg.dry_run:
			return

		positions = await asyncio.to_thread(mt5.positions_get)
		if not positions:
			return

		for pos in positions:
			if pos.magic != _MAGIC:
				continue  # botun kendi açtığı değilse karışma
			if pos.tp == 0 or pos.sl == 0:
				continue

			entry	= pos.price_open
			cur		= pos.price_current
			tp_dist = abs(pos.tp - entry)
			if tp_dist < 1e-9:
				continue

			is_long = pos.type == mt5.POSITION_TYPE_BUY
			progress = (
				(cur - entry) / tp_dist if is_long
				else (entry - cur) / tp_dist
			)

			new_sl = None
			# 1. %50 ilerleme → breakeven (SL'yi entry'e taşı)
			if progress >= 0.5:
				sl_at_entry_or_better = (
					pos.sl >= entry if is_long else pos.sl <= entry
				)
				if not sl_at_entry_or_better:
					new_sl = entry

			# 2. %75 ilerleme → SL'yi entry + %25 tp_dist'e çek (kâr kilidi)
			if progress >= 0.75:
				lock_distance = tp_dist * 0.25
				candidate = entry + lock_distance if is_long else entry - lock_distance
				improves = (
					candidate > pos.sl if is_long else candidate < pos.sl
				)
				if improves:
					new_sl = candidate

			if new_sl is None:
				continue

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
				logger.info(
					"[TRAIL] ticket=%d %s | progress=%.1f%% | SL: %.4f -> %.4f",
					pos.ticket, pos.symbol, progress*100, pos.sl, new_sl,
				)
			else:
				err = await asyncio.to_thread(mt5.last_error)
				rc = result.retcode if result else "None"
				logger.warning(
					"[TRAIL] ticket=%d SL guncellenemedi | retcode=%s err=%s",
					pos.ticket, rc, err,
				)

	async def _query_gemini(self, payload: dict, data_sym: str) -> dict:
		prompt	 = build_llm_prompt(payload, lessons_path=self.cfg.lessons_path)
		logger.info("[LLM][%s] Istek %d karakter", data_sym, len(prompt))
		response = await self._model.generate_content_async(prompt)
		raw		 = _parse_llm_response(response.text)
		decision = _apply_score_rules(raw, self.cfg.min_score)

		pd_ = decision.get("puan_detayi", {})
		logger.info(
			"[LLM][%s] Karar=%-6s | Puan=%d/100 "
			"[PA=%s OB=%s CG=%s Ind=%s] | Risk=%.1f%%",
			data_sym,
			decision.get("karar"),
			decision.get("toplam_puan", 0),
			pd_.get("price_action_majör_sr", "?"),
			pd_.get("orderbook_emir_duvari", "?"),
			pd_.get("coinglass_balina", "?"),
			pd_.get("indikatorler", "?"),
			float(decision.get("risk_yuzdesi", 0)) * 100,
		)
		return decision

	# ── Post-Mortem ──────────────────────────────────────────────

	async def _run_post_mortem(self, entry: dict, outcome: str) -> None:
		snap	   = entry.get("decision_snapshot", {})
		outcome_tr = "STOP (Zarar)" if outcome == "STOP" else "TP (Kar)"

		prompt = (
			f"Bir swing trade islemi kapandi. Post-mortem analiz yap:\n\n"
			f"Sembol		 : {entry.get('data_sym')} ({entry.get('mt5_sym')})\n"
			f"Yon			 : {snap.get('karar', '?')}\n"
			f"Sonuc			 : {outcome_tr}\n"
			f"Acilis Gerekce : {snap.get('gerekce', 'Bilinmiyor')}\n"
			f"Confluence Puan: {snap.get('toplam_puan', '?')}/100\n"
			f"Kilit Seviyeler: {snap.get('kilit_seviyeler', [])}\n"
			f"Giris Fiyati	 : {entry.get('entry_price', '?')}\n"
			f"Stop-Loss		 : {entry.get('sl_price', '?')}\n"
			f"Take-Profit	 : {entry.get('tp_price', '?')}\n"
			f"Kapanis Fiyati : {entry.get('close_price', 'Bilinmiyor')}\n"
			f"PnL			 : {entry.get('pnl', 'Bilinmiyor')} $\n\n"
			"ICT/SMC, CoinGlass Orderbook ve Majör S/R cercevesinde 1-2 cumlelik, "
			"somut ve uygulanabilir bir ders cikar.\n\n"
			"Yanitini YALNIZCA su JSON formatinda ver:\n"
			'{"ders": "Ders metni", "kategori": "STOP" veya "TP", "onem": "YUKSEK" veya "ORTA"}'
		)

		try:
			response	= await self._post_mortem_model.generate_content_async(prompt)
			lesson_data = _parse_llm_response(response.text, required_key="ders")

			lessons = _load_json_file(self.cfg.lessons_path, [])
			lessons.append({
				"ticket_id":  entry.get("ticket_id"),
				"data_sym":	  entry.get("data_sym"),
				"outcome":	  outcome,
				"timestamp":  datetime.now(timezone.utc).isoformat(),
				"ders":		  lesson_data.get("ders", ""),
				"kategori":	  lesson_data.get("kategori", outcome),
				"onem":		  lesson_data.get("onem", "ORTA"),
				"entry_snapshot": {
					"puan":	 snap.get("toplam_puan"),
					"karar": snap.get("karar"),
					"entry": entry.get("entry_price"),
					"sl":	 entry.get("sl_price"),
					"tp":	 entry.get("tp_price"),
					"close": entry.get("close_price"),
					"pnl":	 entry.get("pnl"),
				},
			})
			_save_json_file(self.cfg.lessons_path, lessons)
			logger.info(
				"[POST-MORTEM] Ders kaydedildi | %s %s | Onem: %s",
				entry.get("data_sym"), outcome, lesson_data.get("onem", "?"),
			)
			logger.info("[POST-MORTEM] %s", lesson_data.get("ders", "")[:120])
		except Exception as exc:
			logger.error("[POST-MORTEM] Analiz hatasi: %s", exc)

	# ── History Checker (Saatte Bir) ─────────────────────────────

	async def _check_closed_positions(self) -> None:
		if not _MT5_AVAILABLE or self.cfg.dry_run:
			logger.debug("[HISTORY] MT5 yok veya dry_run atlandi.")
			return

		journal	 = _load_json_file(self.cfg.journal_path, [])
		to_check = [e for e in journal if e.get("status") in ("PENDING", "ACTIVE")]

		if not to_check:
			logger.debug("[HISTORY] Acik islem yok.")
			return

		logger.info("[HISTORY] %d giris kontrol ediliyor...", len(to_check))
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
				logger.info("[HISTORY] ticket=%s iptal.", ticket_str)
				continue

			position_id = hist_order.position_id
			if position_id == 0:
				continue

			if entry["status"] == "PENDING":
				entry["status"] = "ACTIVE"
				changed = True
				logger.info("[HISTORY] ticket=%s ACTIVE (pos=%d).", ticket_str, position_id)

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
			if closing_deal.reason == mt5.DEAL_REASON_SL:
				outcome = "STOP"
			elif closing_deal.reason == mt5.DEAL_REASON_TP:
				outcome = "TP"
			else:
				outcome = "TP" if closing_deal.profit >= 0 else "STOP"

			entry["status"]		 = outcome
			entry["close_price"] = float(closing_deal.price)
			entry["close_time"]	 = datetime.fromtimestamp(
				closing_deal.time, tz=timezone.utc
			).isoformat()
			entry["pnl"] = float(closing_deal.profit)
			changed = True
			logger.info(
				"[HISTORY] ticket=%s %s | Kapanis: %.4f | PnL: %.2f$",
				ticket_str, outcome, closing_deal.price, closing_deal.profit,
			)
			await self._run_post_mortem(entry, outcome)

		if changed:
			_save_json_file(self.cfg.journal_path, journal)
			logger.info("[HISTORY] Journal guncellendi.")

	# ── Tek sembol analiz ve emir döngüsü ─────────────────────

	async def _run_symbol(
		self,
		data_sym: str,
		mt5_sym:  str,
		exchange,
		lm:		  "LiquidityManager",
		cg:		  "CoinglassManager",
		executor: "TradeExecutor",
	) -> None:
		logger.info("---- [%s %s] Swing analizi basliyor ----", data_sym, mt5_sym)

		# 1. Kill-switch
		if not self.risk_mgr.is_active:
			logger.warning("[%s] Kill-switch aktif - geciliyor.", data_sym)
			return

		# 2. Order Stacking Protector (aynı sembolde aktif işlem/emir varsa pas)
		if _MT5_AVAILABLE and not self.cfg.dry_run:
			ex_orders = await asyncio.to_thread(mt5.orders_get, symbol=mt5_sym)
			ex_pos	  = await asyncio.to_thread(mt5.positions_get, symbol=mt5_sym)
			if (ex_orders and len(ex_orders) > 0) or (ex_pos and len(ex_pos) > 0):
				logger.info("[%s] Aktif islem/emir var - pas.", data_sym)
				return

		# 3. Bakiye & toplam exposure
		balance = await _fetch_mt5_balance(self.cfg.dry_run)
		if balance > 0:
			self.risk_mgr.update_balance(balance)
		if not self.risk_mgr.is_active:
			return

		current_exposure = await _fetch_total_exposure_usd(self.cfg, self.cfg.dry_run)
		if current_exposure >= self.cfg.max_total_exposure_usd:
			logger.warning(
				"[%s] Portfoy maks hacim dolu: $%.0f / $%.0f - pas.",
				data_sym, current_exposure, self.cfg.max_total_exposure_usd,
			)
			return

		# 4. OHLCV
		ohlcv_dict = await fetch_ohlcv_all(
			exchange, data_sym, self.cfg.timeframes, self.cfg.candles_limit
		)
		
		# Hata Çözümü: Pandas Dataframe'leri "or" ile kullanılamaz, açıkça .empty sorulmalı.
		base_df = ohlcv_dict.get("15m")
		if base_df is None or base_df.empty:
			base_df = ohlcv_dict.get("1h")
			
		if base_df is None or base_df.empty:
			logger.error("[%s] Fiyat verisi yok.", data_sym)
			return
			
		current_price = float(base_df["close"].iloc[-1])
		logger.info("[%s] Guncel fiyat: %.6f", data_sym, current_price)

		# 5. Price Action
		tf_analysis = self.analyzer.analyze_all(ohlcv_dict, current_price)

# 5b. BTC bias kaydet (sadece BTC analiz edilirken)
		if data_sym == "BTC/USDT":
			t_1d = tf_analysis.get("1d", {}).get("trend", "RANGING")
			t_4h = tf_analysis.get("4h", {}).get("trend", "RANGING")
			strong_bias = None
			if t_1d == "BULLISH" and t_4h == "BULLISH":
				strong_bias = "BULLISH"
			elif t_1d == "BEARISH" and t_4h == "BEARISH":
				strong_bias = "BEARISH"
			self._btc_regime = {
				"trend_1d":	   t_1d,
				"trend_4h":	   t_4h,
				"strong_bias": strong_bias,
			}
			logger.info(
				"[BTC BIAS] 1D=%s 4H=%s | strong_bias=%s",
				t_1d, t_4h, strong_bias or "yok (karisik)",
			)

		# 6. CoinGlass + Orderbook (paralel)
		cg_task = cg.get_coinglass_data(
			symbol=data_sym, current_price=current_price,
			ohlcv_1d=ohlcv_dict.get("1d"),
		)
		ob_task = lm.get_orderbook_walls(
			symbol=data_sym, current_price=current_price,
			depth_pct=self.cfg.ob_depth_pct,
			tier_pct=self.cfg.ob_tier_pct,
			top_n=self.cfg.ob_top_n,
		)
		cg_res, ob_res = await asyncio.gather(cg_task, ob_task, return_exceptions=True)

		coinglass_data	= {} if isinstance(cg_res, Exception) else cg_res
		orderbook_walls = {} if isinstance(ob_res, Exception) else ob_res

		warnings = _build_market_warnings(
			tf_analysis, coinglass_data,
			btc_regime=self._btc_regime if data_sym != "BTC/USDT" else None,
		)
		payload = build_master_payload(
			data_sym, current_price, tf_analysis, coinglass_data, orderbook_walls,
			market_warnings=warnings,
		)
		if warnings:
			logger.info("[%s] Piyasa uyarilari (%d adet):", data_sym, len(warnings))
			for w in warnings:
				logger.info("  - %s", w)
		try:
			decision = await self._query_gemini(payload, data_sym)
		except Exception as exc:
			logger.error("[%s] LLM hatasi: %s", data_sym, exc)
			return

		karar = str(decision.get("karar", "BEKLE")).upper()
		if karar == "BEKLE":
			puan = int(decision.get("toplam_puan", 0))
			logger.info("[%s] BEKLE | Puan=%d", data_sym, puan)
			return

		# 9. LLM çıktısı VALIDASYON (hesap planını çıkar)
		try:
			entry_p	 = float(decision.get("giris_fiyati", current_price))
			sl_price = float(decision.get("sl", 0) or 0)
			tp_price = float(decision.get("tp", 0) or 0)
		except (TypeError, ValueError):
			logger.error("[%s] LLM cikti format hatasi.", data_sym)
			return

		if sl_price <= 0 or tp_price <= 0:
			logger.warning("[%s] LLM SL veya TP vermedi - REDDEDILDI.", data_sym)
			return

		risk_pct = float(decision.get("risk_yuzdesi", self.cfg.risk_pct_75_84))

		plan = self.risk_mgr.compute_trade_plan(
			mt5_sym=mt5_sym,
			entry_price=entry_p,
			sl_price=sl_price,
			tp_price=tp_price,
			risk_pct=risk_pct,
			current_exposure_usd=current_exposure,
		)

		if not plan["ok"]:
			logger.warning("[%s] TRADE PLAN RED: %s", data_sym, plan["reason"])
			return

		volume = plan["lot"]

		# 10. Limit emir
		result = await _send_limit_order(
			mt5_sym=mt5_sym, decision=karar,
			entry_price=entry_p, volume=volume,
			sl_price=sl_price, tp_price=tp_price,
			comment=f"algo p{decision.get('toplam_puan',0)} rr{plan['rr']:.1f}",
			dry_run=self.cfg.dry_run,
		)

		# 11. Trade log
		trade_logger.info(
			"[%s->%s] %-12s | LIMIT @ %.4f | vol=%.2f | sl=%.4f | tp=%.4f | "
			"SL%%=%.2f RR=%.2f | puan=%d/100 | limit=%s | order=%s",
			data_sym, mt5_sym, result.status.value,
			entry_p, volume, sl_price, tp_price,
			plan["sl_pct"]*100, plan["rr"],
			int(decision.get("toplam_puan", 0)),
			plan["limiting_factor"],
			result.order_id or "-",
		)

		# 12. Journaling
		if result.status in (ExecutionStatus.SUCCESS, ExecutionStatus.DRY_RUN):
			ticket = result.order_id or f"DRY-{int(time.time())}"
			_journal_add_entry(
				journal_path=self.cfg.journal_path,
				ticket_id=ticket, data_sym=data_sym, mt5_sym=mt5_sym,
				decision=decision, entry_price=entry_p,
				sl_price=sl_price, tp_price=tp_price,
			)

	# ── Ana kontrol döngüsü ──────────────────────────────────────

	async def _cycle(
		self,
		exchange,
		lm:		  "LiquidityManager",
		cg:		  "CoinglassManager",
		executor: "TradeExecutor",
	) -> None:
		now = time.monotonic()

		# 1. Saatlik kapalı pozisyon kontrolü
		if now - self._last_history_check >= self.cfg.history_check_interval:
			logger.info(
				"[HISTORY] Saatlik kontrol (son: %.0f sn once)...",
				now - self._last_history_check,
			)
			try:
				await self._check_closed_positions()
				self._last_history_check = time.monotonic()
			except Exception as exc:
				logger.error("[HISTORY] Kontrol hatasi: %s", exc)

		# 2. Açık pozisyonları trail et
		try:
			await self._manage_open_positions()
		except Exception as exc:
			logger.error("[TRAIL] Hata: %s", exc)

		# 3. SIRALI sembol analizi (BTC -> ETH -> XRP)
		#	 BTC bias için BTC trend'ini diğerlerinden ÖNCE bilmemiz gerek.
		for data_sym, mt5_sym in zip(self.cfg.data_symbols, self.cfg.mt5_symbols):
			elapsed = now - self._last_ai.get(data_sym, 0.0)
			if elapsed < self.cfg.ai_interval_seconds:
				remaining = self.cfg.ai_interval_seconds - elapsed
				logger.debug("[%s] Sonraki analiz %.0f sn sonra.", data_sym, remaining)
				continue
			try:
				await self._run_symbol(data_sym, mt5_sym, exchange, lm, cg, executor)
				self._last_ai[data_sym] = time.monotonic()
			except Exception as exc:
				logger.exception("[%s] Beklenmedik hata: %s", data_sym, exc)
				self._last_ai[data_sym] = (
					time.monotonic() - self.cfg.ai_interval_seconds + 120
				)

	# ── Başlangıç ───────────────────────────────────────────────

	async def run(self) -> None:
		cfg = self.cfg
		logger.info(
			"FundedNext Swing Bot (CCXT Orderbook + CoinGlass Balina) baslatiliyor\n"
			"  Semboller   : %s\n"
			"  TF		   : %s\n"
			"  Leverage	   : %dx | Max Pos USD: $%.0f | Risk (75-100): %s\n"
			"  Min Puan	   : %d/100 | AI Interval: %ds | DryRun: %s\n"
			"  OB Cluster  : +-%d%% | %%%d dilim | top-%d duvar (Binance+OKX+Bybit+Gate+MEXC)\n"
			"  S/R		   : %.1f%% tolerans | min=%d temas | flip=%d temas\n"
			"  Journal	   : %s | Lessons: %s",
			list(zip(cfg.data_symbols, cfg.mt5_symbols)),
			cfg.timeframes,
			cfg.leverage, cfg.max_position_usd, f"%{cfg.risk_pct_75_84*100}-%{cfg.risk_pct_85_100*100}",
			cfg.min_score, cfg.ai_interval_seconds, cfg.dry_run,
			int(cfg.ob_depth_pct * 100), int(cfg.ob_tier_pct * 100), cfg.ob_top_n,
			cfg.sr_tolerance_pct * 100, cfg.sr_min_touches, cfg.sr_flip_min_touches,
			cfg.journal_path, cfg.lessons_path,
		)

		exchange = ccxt_async.binance({"enableRateLimit": True})

		try:
			async with (
				LiquidityManager() as lm,
				CoinglassManager(cfg.coinglass_api_key) as cg,
				TradeExecutor(
					login=cfg.mt5_login,
					password=cfg.mt5_password,
					server=cfg.mt5_server,
					dry_run=cfg.dry_run,
					trade_logger=trade_logger,
				) as executor,
			):
				init_balance = await _fetch_mt5_balance(cfg.dry_run)
				self.risk_mgr.initialize(init_balance)

				await self._check_closed_positions()
				self._last_history_check = time.monotonic()

				while True:
					try:
						await self._cycle(exchange, lm, cg, executor)
					except Exception as exc:
						logger.exception("Ana dongu hatasi (devam): %s", exc)

					logger.info(
						"Bekleniyor: %ds | Bakiye: %.2f$ | DD: %.2f%% | Kill-switch: %s",
						cfg.loop_interval_sec,
						self.risk_mgr.balance,
						self.risk_mgr.current_drawdown_pct * 100,
						"AKTIF" if not self.risk_mgr.is_active else "KAPALI",
					)
					await asyncio.sleep(cfg.loop_interval_sec)

		finally:
			await exchange.close()
			logger.info("Exchange baglantisi kapatildi. Bot durdu.")


# ─────────────────────────────────────────────────────────────────
# Giriş Noktası
# ─────────────────────────────────────────────────────────────────

def main() -> None:
	config = BotConfig()
	bot	   = AlgoBot(config)
	asyncio.run(bot.run())


if __name__ == "__main__":
	main()