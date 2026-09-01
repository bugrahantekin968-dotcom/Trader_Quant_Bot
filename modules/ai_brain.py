"""
ai_brain.py
============
Quant Trading Bot - AI Karar Mekanizması ve Trade Loglama

Bu modül, `main.py`'nin ürettiği Master AI Payload (JSON) verisini alıp
bir LLM'e (OpenAI GPT veya Anthropic Claude) gönderir, dönen kararı strict
şekilde parse eder ve uygun log dosyalarına kaydeder.

İçerdiği bileşenler:
	* SYSTEM_PROMPT	 - ICT/SMC uzmanı kurumsal kantitatif trader rol tanımı.
	* TradeDecision	 - LLM çıktısının validated dataclass representation'ı.
	* TradeLogger	 - 'trade_syslog.log' (icra) ve 'trade_syserr.log' (graveyard).
	* AIBrain		 - Asenkron LLM client + payload-to-decision pipeline.

Tasarım Notları
---------------
* Hem OpenAI (chat completions) hem Anthropic (messages) API formatları
  native destekleniyor. Provider değişikliği tek satırlık.
* JSON response mode kullanılıyor (`response_format` veya prompt enforcement).
* Layered JSON extraction: direkt parse → markdown fence regex → bracket span;
  LLM markdown'da veya prosa içinde dönerse otomatik kurtarır.
* Kill-switch respect: `trade_permission=False` ise LLM hiç çağrılmaz.
* Tüm hatalar (LLM call fail, parse fail, validation fail) syserr'e gidiyor —
  bu dosya hem debugging hem AI fine-tuning data source'u olarak kullanılır.

Bağımlılıklar: aiohttp	(standart kütüphane dışı yalnız bu)
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional
import google.generativeai as genai
import aiohttp


# ---------------------------------------------------------------------------
# Modül seviyesinde logger.
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


# ===========================================================================
# SYSTEM PROMPT — LLM'in karakterini, kurallarını ve çıktı formatını tanımlar
# ===========================================================================

SYSTEM_PROMPT = """Sen, kurumsal bir hedge fonu yöneten, deneyimli bir kantitatif tradersın. Uzmanlık alanların:
- Inner Circle Trader (ICT) metodolojisi
- Smart Money Concepts (SMC)
- Likidite analizi ve tasfiye (liquidation) dinamikleri
- Prop firm odaklı katı risk yönetimi

KARAKTERİN VE FELSEFEN:
- Duyguların yok. FOMO yok, intikam trade'i yok, doubling-down yok.
- Yalnızca somut data + ICT/SMC kuralları çerçevesinde karar verirsin.
- Belirsizlikte mutlaka BEKLE'rsin. "Forced trade" felakettir.
- Açgözlülük yok: %1 risk üst sınırın, asla aşma. Tipik trade %0.5.

VERİ FORMATIN (her mesajda sana iletilen master_payload yapısı):
	meta			  -> symbol, timeframe, generated_at
	current_price	  -> anlık fiyat
	price_action:
		indicators		 -> EMA_50, EMA_200, RSI_14, ATR_14, trend
		market_structure -> swing_highs[], swing_lows[], last_*
		fvgs			 -> bullish[] / bearish[] (sadece unmitigated)
		order_blocks	 -> bullish[] / bearish[] (mitigated flag'leriyle)
	liquidity:
		likidite_haritasi_asagi_yonlu  -> 4 tier, durum: NORMAL/YOGUN/ANA_DUVAR
		likidite_haritasi_yukari_yonlu -> 4 tier
	risk_status:
		current_balance, daily_pnl, kill_switch_active, trade_permission,
		daily_dd_used_pct, daily_dd_remaining_usd

KARAR KRİTERLERİN (CONFLUENCE-BASED):
1. TREND ALIGNMENT
   - EMA_50 > EMA_200 ve last_swing_high > previous swing_high -> HTF bullish
   - EMA_50 < EMA_200 ve last_swing_low	 < previous swing_low  -> HTF bearish
2. BULLISH SETUP (LONG için tüm bunlar lazım):
   - HTF trend = bullish
   - En az 1 unmitigated bullish FVG veya unmitigated bullish OB current_price'a yakın
   - Aşağı yönlü likidite haritasında YOGUN/ANA_DUVAR tier var (stop hunt potansiyeli) — fakat oraya KADAR girmeyiz
   - Yukarı yönlü likidite haritasında bir ANA_DUVAR tier hedeflenebilir (TP)
3. BEARISH SETUP (SHORT için ayna görüntü)
4. STOP LOSS YERLEŞTİRMESİ
   - Bullish OB'nin alt sınırının veya last_swing_low'un biraz altı (~0.3 ATR buffer)
   - Bearish için: bearish OB üst sınırı veya last_swing_high üstü
5. TAKE PROFIT
   - Mutlaka YOGUN veya ANA_DUVAR durumundaki bir likidite tier'ına denk gelsin
   - Minimum R:R ratio = 1.5 (genelde 2.0+)
6. RİSK BÜYÜKLÜĞÜ
   - Default 0.5 (yani %0.5)
   - Sinyal çok güçlü (üç+ confluence) ise 1.0'a çıkar — asla aşma
   - daily_dd_used_pct > 50 ise risk_yuzdesi <= 0.25
   - daily_dd_used_pct > 75 ise risk_yuzdesi <= 0.1

ZORUNLU GÜVENLİK KAPILARI:
- risk_status.trade_permission == false			  -> ZORUNLU BEKLE
- risk_status.kill_switch_active == true		  -> ZORUNLU BEKLE
- HTF trend ile setup yönü uyumsuz				  -> BEKLE
- En az bir unmitigated FVG veya OB yoksa		  -> BEKLE
- TP'de gerçek bir likidite hedefi yoksa		  -> BEKLE

ÇIKTI FORMATI (ZORUNLU, KESİNLİKLE):
Sadece tek bir geçerli JSON döndür. Markdown YOK, kod fence YOK, açıklama YOK.
İlk karakter '{' olmalı, son karakter '}'.

{
  "karar": "LONG" | "SHORT" | "BEKLE",
  "giris_fiyati": number | null,
  "stop_loss":	  number | null,
  "take_profit":  number | null,
  "risk_yuzdesi": number,
  "ai_yorumu":	  "Türkçe gerekçelendirme — hangi FVG, hangi OB, hangi likidite hedefi, neden bu R:R."
}

KURALLAR:
- BEKLE durumunda giris_fiyati / stop_loss / take_profit alanları null olur.
- risk_yuzdesi her durumda 0.0 - 1.0 arası bir sayıdır (0 = no risk; 1 = %1 max).
- LONG için stop_loss < giris_fiyati < take_profit zorunlu.
- SHORT için take_profit < giris_fiyati < stop_loss zorunlu.
- ai_yorumu kısa (2-4 cümle), Türkçe, somut: "Hangi seviye?", "Neden bu hedef?"
"""


# ===========================================================================
# TRADE DECISION — LLM çıktısının validated dataclass'ı
# ===========================================================================

class DecisionType(str, Enum):
	LONG  = "LONG"
	SHORT = "SHORT"
	BEKLE = "BEKLE"


@dataclass
class TradeDecision:
	"""
	LLM'den dönen kararın strict-validated representation'ı.

	`from_dict` factory metodu strict validation yapar; geçersiz input
	ValueError fırlatır.
	"""

	karar:		   str
	risk_yuzdesi:  float
	ai_yorumu:	   str
	giris_fiyati:  Optional[float] = None
	stop_loss:	   Optional[float] = None
	take_profit:   Optional[float] = None

	VALID_DECISIONS = ('LONG', 'SHORT', 'BEKLE')

	@classmethod
	def from_dict(cls, data: Dict[str, Any]) -> "TradeDecision":
		"""LLM JSON çıktısını parse + validate eder."""
		if not isinstance(data, dict):
			raise ValueError(f"Beklenen dict, alınan: {type(data).__name__}")

		# --- Zorunlu alanlar ---
		required = {'karar', 'risk_yuzdesi', 'ai_yorumu'}
		missing = required - set(data.keys())
		if missing:
			raise ValueError(f"Eksik alan(lar): {sorted(missing)}")

		# --- karar validation ---
		karar = str(data['karar']).upper().strip()
		if karar not in cls.VALID_DECISIONS:
			raise ValueError(
				f"Geçersiz karar: '{karar}'. Beklenen: {cls.VALID_DECISIONS}"
			)

		# --- risk_yuzdesi validation ---
		try:
			risk_yuzdesi = float(data['risk_yuzdesi'])
		except (TypeError, ValueError):
			raise ValueError(
				f"risk_yuzdesi numerik olmalı, alınan: {data['risk_yuzdesi']!r}"
			)
		if not (0.0 <= risk_yuzdesi <= 1.0):
			raise ValueError(
				f"risk_yuzdesi 0.0-1.0 arası olmalı (max %1), alınan: {risk_yuzdesi}"
			)

		# --- ai_yorumu validation ---
		ai_yorumu = str(data['ai_yorumu']).strip()
		if not ai_yorumu:
			raise ValueError("ai_yorumu boş olamaz")

		# --- Opsiyonel fiyat alanları ---
		def _opt_float(key: str) -> Optional[float]:
			v = data.get(key)
			if v is None:
				return None
			try:
				return float(v)
			except (TypeError, ValueError):
				raise ValueError(f"{key} numerik veya null olmalı, alınan: {v!r}")

		giris = _opt_float('giris_fiyati')
		sl	  = _opt_float('stop_loss')
		tp	  = _opt_float('take_profit')

		# --- karar - fiyat consistency ---
		if karar == 'BEKLE':
			# BEKLE'de fiyatlar null olmalı. LLM yine doldurursa düzeltiriz.
			if giris is not None or sl is not None or tp is not None:
				logger.warning(
					"BEKLE kararında fiyat alanları null olmalıydı; LLM hatası düzeltildi."
				)
				giris = sl = tp = None
		else:
			# LONG / SHORT için fiyatlar zorunlu
			if giris is None or sl is None or tp is None:
				raise ValueError(
					f"{karar} kararında giris_fiyati/stop_loss/take_profit null olamaz"
				)

			# Yön sanity check
			if karar == 'LONG':
				if not (sl < giris < tp):
					raise ValueError(
						f"LONG için sl < giris < tp olmalı; "
						f"sl={sl}, giris={giris}, tp={tp}"
					)
			elif karar == 'SHORT':
				if not (tp < giris < sl):
					raise ValueError(
						f"SHORT için tp < giris < sl olmalı; "
						f"sl={sl}, giris={giris}, tp={tp}"
					)

		return cls(
			karar=karar,
			giris_fiyati=giris,
			stop_loss=sl,
			take_profit=tp,
			risk_yuzdesi=risk_yuzdesi,
			ai_yorumu=ai_yorumu,
		)

	def to_dict(self) -> Dict[str, Any]:
		return asdict(self)

	@property
	def is_actionable(self) -> bool:
		"""LONG/SHORT için True, BEKLE için False."""
		return self.karar in ('LONG', 'SHORT')

	@property
	def risk_reward_ratio(self) -> Optional[float]:
		"""R:R ratio hesaplar (BEKLE için None)."""
		if not self.is_actionable:
			return None
		if self.giris_fiyati is None or self.stop_loss is None or self.take_profit is None:
			return None
		risk   = abs(self.giris_fiyati - self.stop_loss)
		reward = abs(self.take_profit - self.giris_fiyati)
		return reward / risk if risk > 0 else None


# ===========================================================================
# TRADE LOGGER — syslog (icra) ve syserr (graveyard) dosyalarına yazar
# ===========================================================================

class TradeLogger:
	"""
	İki ayrı dosyaya yazan log mekanizması:

		trade_syslog.log  - LONG/SHORT kararları (icra edilen)
		trade_syserr.log  - Hatalar, parse fail'ler, kill-switch trip'ler
							(debugging + AI fine-tuning data source)
	"""

	DEFAULT_SYSLOG_PATH = 'logs/trade_syslog.log'
	DEFAULT_SYSERR_PATH = 'logs/trade_syserr.log'

	DIVIDER		= '=' * 78
	SUB_DIVIDER = '-' * 78

	def __init__(
		self,
		syslog_path: str = DEFAULT_SYSLOG_PATH,
		syserr_path: str = DEFAULT_SYSERR_PATH,
	) -> None:
		self.syslog_path = Path(syslog_path)
		self.syserr_path = Path(syserr_path)

		# Log klasörü garantisi
		self.syslog_path.parent.mkdir(parents=True, exist_ok=True)
		self.syserr_path.parent.mkdir(parents=True, exist_ok=True)

	# ----- helpers -----

	@staticmethod
	def _utc_now_str() -> str:
		return datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')

	@staticmethod
	def _safe_money(val: Optional[float]) -> str:
		if val is None:
			return 'N/A'
		try:
			return f"${val:,.2f}"
		except (TypeError, ValueError):
			return str(val)

	def _write(self, path: Path, lines: list) -> None:
		try:
			with open(path, 'a', encoding='utf-8') as f:
				f.write('\n'.join(lines) + '\n')
		except OSError as exc:
			logger.error("Log dosyası yazma hatası (%s): %s", path, exc)

	# ----- public API -----

	def log_trade_action(
		self,
		decision: TradeDecision,
		symbol: str,
	) -> None:
		"""LONG veya SHORT kararını trade_syslog.log'a yazar. BEKLE atlanır."""
		if not decision.is_actionable:
			return

		rr = decision.risk_reward_ratio
		rr_str = f"{rr:.2f}" if rr is not None else 'N/A'

		lines = [
			'',
			self.DIVIDER,
			f"[{self._utc_now_str()}] TRADE DECISION",
			self.DIVIDER,
			f"Symbol		 : {symbol}",
			f"Decision		 : {decision.karar}",
			f"Entry			 : {self._safe_money(decision.giris_fiyati)}",
			f"Stop Loss		 : {self._safe_money(decision.stop_loss)}",
			f"Take Profit	 : {self._safe_money(decision.take_profit)}",
			f"Risk %		 : {decision.risk_yuzdesi}",
			f"R:R Ratio		 : {rr_str}",
			f"AI Comment	 : {decision.ai_yorumu}",
			self.SUB_DIVIDER,
			f"[JSON]: {json.dumps(decision.to_dict(), ensure_ascii=False)}",
			self.DIVIDER,
		]
		self._write(self.syslog_path, lines)
		logger.info("Trade kararı syslog'a yazıldı: %s %s @ %s",
					symbol, decision.karar, self._safe_money(decision.giris_fiyati))

	def log_trade_error(
		self,
		error_msg: str,
		payload_status: Optional[Dict[str, Any]] = None,
		error_type: str = 'UNCATEGORIZED',
		raw_response: Optional[str] = None,
		symbol: Optional[str] = None,
	) -> None:
		"""
		Hata, exception, parse fail veya stop-out durumlarını
		trade_syserr.log (graveyard) dosyasına yazar.

		Bu dosya hem debugging için hem de AI fine-tuning için kullanılır:
		her hata kaydı bir "failure case" datapoint'idir.
		"""
		lines = [
			'',
			self.DIVIDER,
			f"[{self._utc_now_str()}] TRADE ERROR / GRAVEYARD ENTRY",
			self.DIVIDER,
			f"Error Type	 : {error_type}",
			f"Symbol		 : {symbol or 'UNKNOWN'}",
			f"Error Message	 : {error_msg}",
		]
		if payload_status:
			lines.append(self.SUB_DIVIDER)
			lines.append("[Payload Status]:")
			lines.append(json.dumps(payload_status, indent=2, ensure_ascii=False, default=str))
		if raw_response:
			lines.append(self.SUB_DIVIDER)
			lines.append("[Raw LLM Response]:")
			# Çok uzun response'u kırp
			preview = raw_response if len(raw_response) <= 2000 else raw_response[:2000] + '... [TRUNCATED]'
			lines.append(preview)
		lines.append(self.DIVIDER)

		self._write(self.syserr_path, lines)
		logger.warning("Hata syserr'e yazıldı: [%s] %s", error_type, error_msg)


# ===========================================================================
# AI BRAIN — LLM client + decision pipeline
# ===========================================================================

class AIBrain:
	"""
	LLM API client + master_payload-to-decision pipeline.

	Parametreler
	------------
	api_key : Optional[str]
		LLM provider API anahtarı. None ise otomatik mock_mode'a düşer.
	provider : str
		'openai' veya 'anthropic'.
	model : Optional[str]
		Model adı (None ise provider default'u kullanılır).
	base_url : Optional[str]
		API base URL'i (None ise provider default'u). OpenRouter veya
		self-hosted LLM için override edilebilir.
	timeout : float
		HTTP istek timeout (saniye).
	max_retries : int
		429 / network hataları için retry sayısı.
	temperature : float
		LLM sampling temperature (düşük = deterministik kararlar).
	mock_mode : bool
		True ise gerçek API çağrısı yapmaz; deterministik mock yanıt üretir.
		api_key=None verilince otomatik True olur.
	trade_logger : Optional[TradeLogger]
		Loglama backend'i (None ise default oluşturulur).
	"""

	PROVIDER_CONFIG = {
		'openai': {
			'base_url':       'https://api.openai.com/v1',
			'endpoint':       '/chat/completions',
			'default_model':  'gpt-4o',
		},
		'anthropic': {
			'base_url':       'https://api.anthropic.com/v1',
			'endpoint':       '/messages',
			'default_model':  'claude-3-5-sonnet-20241022',
		},
        # BUNU EKLE:
		'google': {
			'base_url':       '', 
			'endpoint':       '',
			'default_model':  'gemini-2.5-flash',
		},
	}

	def __init__(
		self,
		api_key: Optional[str] = None,
		provider: str = 'openai',
		model: Optional[str] = None,
		base_url: Optional[str] = None,
		timeout: float = 60.0,
		max_retries: int = 3,
		temperature: float = 0.2,
		mock_mode: bool = False,
		system_prompt: str = SYSTEM_PROMPT,
		trade_logger: Optional[TradeLogger] = None,
	) -> None:
		if provider not in self.PROVIDER_CONFIG:
			raise ValueError(
				f"Desteklenmeyen provider: '{provider}'. "
				f"Beklenen: {list(self.PROVIDER_CONFIG.keys())}"
			)

		cfg = self.PROVIDER_CONFIG[provider]
		self.api_key	   = api_key
		self.provider	   = provider
		self.model		   = model or cfg['default_model']
		self.base_url	   = (base_url or cfg['base_url']).rstrip('/')
		self.endpoint	   = cfg['endpoint']
		self.timeout	   = timeout
		self.max_retries   = max_retries
		self.temperature   = temperature
		self.system_prompt = system_prompt
		self.trade_logger  = trade_logger or TradeLogger()

		# API key yoksa otomatik mock
		self.mock_mode = mock_mode or (api_key is None)

		self._session: Optional[aiohttp.ClientSession] = None
		self._owns_session: bool = False

        # BUNU EKLE:
		if self.provider == 'google' and not self.mock_mode:
			genai.configure(api_key=self.api_key)
			self._gemini_model = genai.GenerativeModel(
				model_name=self.model,
				system_instruction=self.system_prompt
			)

	# ----- async context -----

	async def __aenter__(self) -> "AIBrain":
		if not self.mock_mode:
			timeout = aiohttp.ClientTimeout(total=self.timeout)
			self._session = aiohttp.ClientSession(timeout=timeout)
			self._owns_session = True
			logger.info("AIBrain hazır: provider=%s, model=%s", self.provider, self.model)
		else:
			logger.info("AIBrain MOCK MODE'da (API key yok veya mock_mode=True).")
		return self

	async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
		if self._owns_session and self._session is not None:
			await self._session.close()
			self._session = None

	# =======================================================================
	# ANA METOD: evaluate_market
	# =======================================================================

	async def evaluate_market(
		self,
		master_payload: Dict[str, Any],
	) -> TradeDecision:
		"""
		Master payload'ı LLM'e gönderir, dönen kararı validate eder
		ve uygun log dosyasına yazar.

		Pipeline:
			0) Kill-switch / trade_permission kontrolü -> erken BEKLE
			1) LLM çağrısı (mock veya gerçek)
			2) JSON extraction (markdown fence, bracket span fallback'leri)
			3) TradeDecision validation
			4) Aksiyon log'u (LONG/SHORT için)

		Her aşamadaki hata syserr'e yazılır ve re-raise edilir;
		caller (main.py orchestrator) bot'u ayakta tutmak için handle eder.
		"""
		symbol = master_payload.get('meta', {}).get('symbol', 'UNKNOWN')

		# ---- Aşama 0: Kill-switch erken çıkış ----
		risk_status = master_payload.get('risk_status') or {}
		if risk_status.get('trade_permission') is False:
			logger.info("trade_permission=False; LLM bypass edilip BEKLE döndürülüyor.")
			decision = TradeDecision(
				karar='BEKLE',
				giris_fiyati=None,
				stop_loss=None,
				take_profit=None,
				risk_yuzdesi=0.0,
				ai_yorumu=(
					"Kill-switch aktif veya trade_permission=False. "
					"Risk yönetimi kuralı gereği trade alınmadı."
				),
			)
			return decision

		# ---- Aşama 1: LLM çağrısı ----
		raw_response: Optional[str] = None
		try:
			if self.mock_mode:
				raw_response = self._mock_llm_response(master_payload)
			else:
				raw_response = await self._call_llm(master_payload)
		except Exception as exc:
			self.trade_logger.log_trade_error(
				error_type='LLM_CALL_ERROR',
				error_msg=f"{type(exc).__name__}: {exc}",
				payload_status=self._payload_summary(master_payload),
				symbol=symbol,
			)
			raise

		# ---- Aşama 2 & 3: Parse + Validate ----
		try:
			parsed_dict = self._extract_json(raw_response)
			decision = TradeDecision.from_dict(parsed_dict)
		except (ValueError, json.JSONDecodeError) as exc:
			self.trade_logger.log_trade_error(
				error_type='LLM_PARSE_ERROR',
				error_msg=f"{type(exc).__name__}: {exc}",
				payload_status=self._payload_summary(master_payload),
				raw_response=raw_response,
				symbol=symbol,
			)
			raise

		# ---- Aşama 4: Aksiyon log'u ----
		if decision.is_actionable:
			self.trade_logger.log_trade_action(decision, symbol=symbol)
		else:
			logger.info("Karar: BEKLE (symbol=%s). Yorum: %s",
						symbol, decision.ai_yorumu)

		return decision

	# =======================================================================
	# PRIVATE — LLM HTTP call (provider-agnostic)
	# =======================================================================

	async def _call_llm(self, master_payload: Dict[str, Any]) -> str:
		"""OpenAI, Anthropic veya Google Gemini API'sine async POST atar."""
		
		# --- GEMINI (GOOGLE) ÖZEL BLOK ---
		if self.provider == 'google':
			payload_str = json.dumps(master_payload, ensure_ascii=False, default=str)
			for attempt in range(1, self.max_retries + 1):
				try:
					# Google SDK kendi HTTP bağlantısını yönetir
					response = await self._gemini_model.generate_content_async(payload_str)
					return response.text
				except Exception as exc:
					logger.warning("Gemini bağlantı hatası deneme %d/%d: %s", attempt, self.max_retries, exc)
					if attempt == self.max_retries:
						raise RuntimeError(f"Gemini API çağrısı başarısız: {exc}")
					await asyncio.sleep(1.0 * attempt)
					continue
		
		# --- OPENAI & ANTHROPIC BLOK ---
		if self._session is None:
			raise RuntimeError("AIBrain 'async with' içinde kullanılmalı.")
		if not self.api_key:
			raise RuntimeError("API key tanımlanmamış.")

		url, headers, body = self._build_request(master_payload)
		backoff: float = 1.0
		last_exc: Optional[Exception] = None

		for attempt in range(1, self.max_retries + 1):
			try:
				async with self._session.post(url, headers=headers, json=body) as resp:
					if resp.status == 429:
						retry_after = resp.headers.get('Retry-After')
						wait_s = float(retry_after) if retry_after else backoff
						logger.warning("LLM rate limit; deneme %d/%d", attempt, self.max_retries)
						await asyncio.sleep(wait_s)
						backoff *= 2
						continue
					resp.raise_for_status()
					response_json = await resp.json()
					return self._extract_content(response_json)
			except Exception as exc:
				last_exc = exc
				await asyncio.sleep(backoff)
				backoff *= 2

		raise RuntimeError(f"LLM çağrısı başarısız: {last_exc}")
	def _build_request(
		self,
		master_payload: Dict[str, Any],
	) -> tuple[str, Dict[str, str], Dict[str, Any]]:
		"""Provider'a göre URL/headers/body üretir."""
		url = f"{self.base_url}{self.endpoint}"
		payload_str = json.dumps(master_payload, ensure_ascii=False, default=str)

		if self.provider == 'openai':
			headers = {
				'Authorization': f'Bearer {self.api_key}',
				'Content-Type':	 'application/json',
			}
			body = {
				'model': self.model,
				'messages': [
					{'role': 'system', 'content': self.system_prompt},
					{'role': 'user',   'content': payload_str},
				],
				'temperature':	   self.temperature,
				'response_format': {'type': 'json_object'},
			}
		elif self.provider == 'anthropic':
			headers = {
				'x-api-key':		 self.api_key,
				'anthropic-version': '2023-06-01',
				'content-type':		 'application/json',
			}
			body = {
				'model':	   self.model,
				'max_tokens':  1024,
				'temperature': self.temperature,
				'system':	   self.system_prompt,
				'messages': [
					{'role': 'user', 'content': payload_str},
				],
			}
		else:
			raise ValueError(f"Bilinmeyen provider: {self.provider}")

		return url, headers, body

	def _extract_content(self, response_json: Dict[str, Any]) -> str:
		"""Provider response JSON'undan asıl text content'i çıkarır."""
		try:
			if self.provider == 'openai':
				return response_json['choices'][0]['message']['content']
			elif self.provider == 'anthropic':
				return response_json['content'][0]['text']
		except (KeyError, IndexError, TypeError) as exc:
			raise ValueError(
				f"{self.provider} response parse edilemedi: {exc}. "
				f"Response: {json.dumps(response_json)[:300]}"
			)
		raise ValueError(f"Bilinmeyen provider: {self.provider}")

	# =======================================================================
	# PRIVATE — JSON extraction (LLM çıktı şişmanlığına karşı)
	# =======================================================================

	@staticmethod
	def _extract_json(raw: str) -> Dict[str, Any]:
		"""
		LLM çıktısından JSON'ı güvenle çıkarır. 3 katmanlı fallback:
			1) Direkt parse
			2) Markdown code fence regex (```json ... ```)
			3) İlk '{' ile son '}' arası slice
		"""
		if not isinstance(raw, str):
			raise ValueError(f"LLM cevabı string değil: {type(raw).__name__}")

		raw_stripped = raw.strip()
		if not raw_stripped:
			raise ValueError("LLM cevabı boş")

		# 1) Direkt parse
		try:
			return json.loads(raw_stripped)
		except json.JSONDecodeError:
			pass

		# 2) Markdown code fence
		fence_match = re.search(
			r'```(?:json)?\s*\n(.+?)\n```',
			raw_stripped,
			re.DOTALL,
		)
		if fence_match:
			try:
				return json.loads(fence_match.group(1).strip())
			except json.JSONDecodeError:
				pass

		# 3) Bracket span: ilk '{' -> son '}'
		first = raw_stripped.find('{')
		last  = raw_stripped.rfind('}')
		if first >= 0 and last > first:
			candidate = raw_stripped[first:last + 1]
			try:
				return json.loads(candidate)
			except json.JSONDecodeError as exc:
				raise ValueError(
					f"JSON parse edilemedi (bracket span). Hata: {exc}. "
					f"Raw: {raw_stripped[:300]}"
				)

		raise ValueError(f"Cevapta JSON bulunamadı. Raw: {raw_stripped[:300]}")

	# =======================================================================
	# PRIVATE — payload summary (hata loglarında full payload yerine özet)
	# =======================================================================

	@staticmethod
	def _payload_summary(payload: Dict[str, Any]) -> Dict[str, Any]:
		"""Hata loglarında full payload yerine özet bilgi."""
		meta = payload.get('meta', {}) or {}
		pa	 = payload.get('price_action') or {}
		ind	 = pa.get('indicators') or {}
		rs	 = payload.get('risk_status') or {}

		return {
			'symbol':	 meta.get('symbol'),
			'timeframe': meta.get('timeframe'),
			'iteration': meta.get('iteration'),
			'current_price':	   payload.get('current_price'),
			'trend':			   ind.get('trend'),
			'rsi':				   ind.get('RSI_14'),
			'trade_permission':	   rs.get('trade_permission'),
			'kill_switch_active':  rs.get('kill_switch_active'),
			'daily_dd_used_pct':   rs.get('daily_dd_used_pct'),
		}

	# =======================================================================
	# PRIVATE — mock LLM response (test/demo amaçlı deterministik karar)
	# =======================================================================

	def _mock_llm_response(self, payload: Dict[str, Any]) -> str:
		"""
		Gerçek LLM'i taklit eden deterministik mock.

		Payload'daki signal'lere bakar:
			* trend=bullish + bullish FVG/OB var		  -> LONG
			* trend=bearish + bearish FVG/OB var		  -> SHORT
			* Diğer her durum							   -> BEKLE

		Fiyatlar ATR-based: SL = entry ± 1.5*ATR, TP = en yakın likidite duvarı.
		"""
		current_price = float(payload.get('current_price') or 0)
		pa = payload.get('price_action') or {}
		ind = pa.get('indicators') or {}
		trend = ind.get('trend', 'undetermined')
		atr = float(ind.get('ATR_14') or current_price * 0.005 or 100)

		bull_fvgs = (pa.get('fvgs') or {}).get('count', {}).get('bullish', 0)
		bear_fvgs = (pa.get('fvgs') or {}).get('count', {}).get('bearish', 0)
		bull_obs  = (pa.get('order_blocks') or {}).get('count', {}).get('bullish', 0)
		bear_obs  = (pa.get('order_blocks') or {}).get('count', {}).get('bearish', 0)

		liq = payload.get('liquidity') or {}
		up_tiers   = liq.get('likidite_haritasi_yukari_yonlu', []) or []
		down_tiers = liq.get('likidite_haritasi_asagi_yonlu', []) or []

		def _first_wall(tiers):
			"""En yakın YOGUN/ANA_DUVAR tier'ın merkez fiyatını döndürür."""
			for t in tiers:
				if t.get('durum') in ('YOGUN', 'ANA_DUVAR'):
					try:
						lo, hi = t['aralik'].split('-')
						return (float(lo) + float(hi)) / 2
					except (KeyError, ValueError):
						continue
			return None

		# ---- LONG setup ----
		if trend == 'bullish' and (bull_fvgs + bull_obs) >= 1 and current_price > 0:
			entry = round(current_price, 2)
			sl	  = round(entry - 1.5 * atr, 2)
			target = _first_wall(up_tiers)
			tp	  = round(target if target and target > entry else entry + 3 * atr, 2)
			return json.dumps({
				'karar':		 'LONG',
				'giris_fiyati':	 entry,
				'stop_loss':	 sl,
				'take_profit':	 tp,
				'risk_yuzdesi':	 0.5,
				'ai_yorumu': (
					f"[MOCK] HTF trend bullish ({bull_fvgs} unmitigated FVG, "
					f"{bull_obs} bullish OB). Yukarıda likidite duvarı hedefleniyor. "
					f"SL son swing low altında, R:R ≈ {(tp-entry)/(entry-sl):.2f}."
				),
			})

		# ---- SHORT setup ----
		if trend == 'bearish' and (bear_fvgs + bear_obs) >= 1 and current_price > 0:
			entry = round(current_price, 2)
			sl	  = round(entry + 1.5 * atr, 2)
			target = _first_wall(down_tiers)
			tp	  = round(target if target and target < entry else entry - 3 * atr, 2)
			return json.dumps({
				'karar':		 'SHORT',
				'giris_fiyati':	 entry,
				'stop_loss':	 sl,
				'take_profit':	 tp,
				'risk_yuzdesi':	 0.5,
				'ai_yorumu': (
					f"[MOCK] HTF trend bearish ({bear_fvgs} unmitigated FVG, "
					f"{bear_obs} bearish OB). Aşağıda likidite duvarı hedefleniyor. "
					f"SL son swing high üstünde, R:R ≈ {(entry-tp)/(sl-entry):.2f}."
				),
			})

		# ---- BEKLE ----
		return json.dumps({
			'karar':		 'BEKLE',
			'giris_fiyati':	 None,
			'stop_loss':	 None,
			'take_profit':	 None,
			'risk_yuzdesi':	 0.0,
			'ai_yorumu': (
				f"[MOCK] Confluence yetersiz: trend={trend}, "
				f"bullish FVG+OB={bull_fvgs + bull_obs}, "
				f"bearish FVG+OB={bear_fvgs + bear_obs}. Bekleniyor."
			),
		})


# ===========================================================================
# DEMO BÖLÜMÜ
# ===========================================================================

def _build_sample_payload() -> Dict[str, Any]:
	"""Demo için bullish setup içeren sahte Master Payload."""
	return {
		'meta': {
			'symbol':		'BTC/USDT',
			'timeframe':	'15m',
			'iteration':	1,
			'generated_at': datetime.now(timezone.utc).isoformat(),
			'bot_version':	'0.4.0',
		},
		'current_price': 77000.0,
		'price_action': {
			'indicators': {
				'EMA_50':  76500.0,
				'EMA_200': 75800.0,
				'RSI_14':  58.4,
				'ATR_14':  320.0,
				'trend':   'bullish',
			},
			'market_structure': {
				'swing_highs':	   [{'time': '2024-...', 'price': 77800}],
				'swing_lows':	   [{'time': '2024-...', 'price': 76200}],
				'last_swing_high': {'time': '2024-...', 'price': 77800},
				'last_swing_low':  {'time': '2024-...', 'price': 76200},
				'count': {'highs': 8, 'lows': 7},
			},
			'fvgs': {
				'bullish': [{'time': '2024-...', 'lower': 76800, 'upper': 76950}],
				'bearish': [],
				'count':   {'bullish': 1, 'bearish': 0},
			},
			'order_blocks': {
				'bullish': [{'time': '2024-...', 'high': 76900, 'low': 76700, 'mitigated': False}],
				'bearish': [],
				'count':   {'bullish': 1, 'bearish': 0},
			},
		},
		'liquidity': {
			'likidite_haritasi_asagi_yonlu': [
				{'kademe': 1, 'aralik': '76600-76800', 'hacim_milyon_usd': 25,	'durum': 'NORMAL'},
				{'kademe': 2, 'aralik': '75800-76000', 'hacim_milyon_usd': 95,	'durum': 'YOGUN'},
				{'kademe': 3, 'aralik': '74600-74900', 'hacim_milyon_usd': 180, 'durum': 'ANA_DUVAR'},
				{'kademe': 4, 'aralik': '73000-73400', 'hacim_milyon_usd': 240, 'durum': 'ANA_DUVAR'},
			],
			'likidite_haritasi_yukari_yonlu': [
				{'kademe': 1, 'aralik': '77400-77600', 'hacim_milyon_usd': 30,	'durum': 'NORMAL'},
				{'kademe': 2, 'aralik': '78200-78400', 'hacim_milyon_usd': 110, 'durum': 'YOGUN'},
				{'kademe': 3, 'aralik': '79200-79500', 'hacim_milyon_usd': 195, 'durum': 'ANA_DUVAR'},
				{'kademe': 4, 'aralik': '80500-81000', 'hacim_milyon_usd': 220, 'durum': 'ANA_DUVAR'},
			],
		},
		'risk_status': {
			'initial_balance':			100000.0,
			'current_balance':			100500.0,
			'daily_pnl':				500.0,
			'daily_drawdown_limit_usd': 2000.0,
			'daily_dd_used_usd':		0.0,
			'daily_dd_used_pct':		0.0,
			'daily_dd_remaining_usd':	2000.0,
			'kill_switch_active':		False,
			'trade_permission':			True,
		},
	}


async def _demo() -> None:
	"""ai_brain.py modülünün hızlı self-test'i."""
	logging.basicConfig(
		level=logging.INFO,
		format='%(asctime)s | %(levelname)-7s | %(name)s | %(message)s',
	)

	# 1) Sahte payload üret
	payload = _build_sample_payload()
	print("\n" + "=" * 78)
	print("AI BRAIN DEMO — BTC/USDT bullish setup, mock LLM")
	print("=" * 78)
	print(f"\n[Payload özeti]")
	print(f"  Symbol		 : {payload['meta']['symbol']}")
	print(f"  Current Price	 : ${payload['current_price']:,.2f}")
	print(f"  Trend			 : {payload['price_action']['indicators']['trend']}")
	print(f"  Bullish FVGs	 : {payload['price_action']['fvgs']['count']['bullish']}")
	print(f"  Bullish OBs	 : {payload['price_action']['order_blocks']['count']['bullish']}")
	print(f"  Permission	 : {payload['risk_status']['trade_permission']}")

	# 2) Mock mode AIBrain
	# api_key=None -> otomatik mock_mode
	logger_path = 'logs/'
	async with AIBrain(api_key=None) as brain:

		# ---- TEST 1: Bullish payload -> LONG bekleniyor ----
		print(f"\n[Test 1] evaluate_market(bullish payload)...")
		decision = await brain.evaluate_market(payload)

		print(f"\n	Decision	 : {decision.karar}")
		print(f"  Entry		   : {decision.giris_fiyati}")
		print(f"  Stop Loss	   : {decision.stop_loss}")
		print(f"  Take Profit  : {decision.take_profit}")
		print(f"  Risk %	   : {decision.risk_yuzdesi}")
		print(f"  R:R Ratio	   : {decision.risk_reward_ratio:.2f}"
			  if decision.risk_reward_ratio else "	R:R Ratio	 : N/A")
		print(f"  AI Comment   : {decision.ai_yorumu}")

		# ---- TEST 2: Kill-switch payload -> BEKLE bekleniyor ----
		print(f"\n[Test 2] evaluate_market(trade_permission=False)...")
		kill_payload = _build_sample_payload()
		kill_payload['risk_status']['trade_permission']	  = False
		kill_payload['risk_status']['kill_switch_active'] = True
		decision_killswitch = await brain.evaluate_market(kill_payload)
		print(f"  Decision	   : {decision_killswitch.karar}")
		print(f"  AI Comment   : {decision_killswitch.ai_yorumu}")

		# ---- TEST 3: Hatalı LLM cevabı simülasyonu -> syserr'e yazılmalı ----
		print(f"\n[Test 3] log_trade_error(simüle edilmiş parse hatası)...")
		brain.trade_logger.log_trade_error(
			error_type='LLM_PARSE_ERROR',
			error_msg='Geçersiz karar: "BELKI". Beklenen: LONG/SHORT/BEKLE',
			symbol='BTC/USDT',
			payload_status=brain._payload_summary(payload),
			raw_response='{"karar": "BELKI", "ai_yorumu": "hmm"}',
		)

	# 3) Sonuç dosyalarını göster
	print(f"\n[Log dosyaları]")
	syslog = Path('logs/trade_syslog.log')
	syserr = Path('logs/trade_syserr.log')
	print(f"  syslog: {syslog.resolve()} ({syslog.stat().st_size if syslog.exists() else 0} bytes)")
	print(f"  syserr: {syserr.resolve()} ({syserr.stat().st_size if syserr.exists() else 0} bytes)")

	print("\n[trade_syslog.log içeriği — son trade]")
	if syslog.exists():
		print(syslog.read_text(encoding='utf-8'))

	print("\n[trade_syserr.log içeriği — son hata]")
	if syserr.exists():
		print(syserr.read_text(encoding='utf-8'))

	print("=" * 78)


if __name__ == '__main__':
	asyncio.run(_demo())