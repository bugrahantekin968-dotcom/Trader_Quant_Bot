"""
executor.py
MetaTrader5 tabanlı emir yürütücü.
main.py asenkron çalıştığından TradeExecutor async context manager
arayüzünü korur; MT5 çağrıları senkrondur.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import MetaTrader5 as mt5

logger = logging.getLogger(__name__)

MAGIC_NUMBER = 202605


# ---------------------------------------------------------------------------
# Veri modelleri (değiştirilmeden korundu)
# ---------------------------------------------------------------------------

class ExecutionStatus(Enum):
	SUCCESS		  = "SUCCESS"
	FAILED		  = "FAILED"
	SKIPPED_WAIT  = "SKIPPED_WAIT"
	DRY_RUN		  = "DRY_RUN"


@dataclass
class ExecutionResult:
	status:		ExecutionStatus
	order_id:	Optional[str]  = None
	symbol:		Optional[str]  = None
	side:		Optional[str]  = None
	price:		Optional[float] = None
	volume:		Optional[float] = None
	sl:			Optional[float] = None
	tp:			Optional[float] = None
	message:	str				= ""
	raw:		dict			= field(default_factory=dict)


# ---------------------------------------------------------------------------
# Yardımcı fonksiyonlar
# ---------------------------------------------------------------------------

def _normalize_symbol(symbol: str) -> str:
	"""
	ccxt/standart formatını MT5 formatına çevirir.

	Örnekler:
		"BTC/USDT"		 → "BTCUSDT"
		"BTC/USDT:USDT"	 → "BTCUSDT"
		"ETH/USD"		 → "ETHUSD"
		"BTCUSD"		 → "BTCUSD"	  (zaten MT5 formatı)
	"""
	# Leverage suffix (":USDT", ":USD" gibi) varsa at
	if ":" in symbol:
		symbol = symbol.split(":")[0]
	# "/" ayracını kaldır
	symbol = symbol.replace("/", "")
	return symbol.upper()


def _get_order_type(decision: str):
	"""Karar stringini MT5 emir tipine çevirir."""
	return {
		"LONG":	 mt5.ORDER_TYPE_BUY,
		"SHORT": mt5.ORDER_TYPE_SELL,
	}.get(decision.upper())


# ---------------------------------------------------------------------------
# Ana sınıf
# ---------------------------------------------------------------------------

class TradeExecutor:
	"""
	Kullanım:
		async with TradeExecutor(login=..., password=..., server=...) as ex:
			result = await ex.execute_decision(decision, symbol, volume, sl, tp)
	"""

	def __init__(
		self,
		login:		  int,
		password:	  str,
		server:		  str,
		dry_run:	  bool = False,
		trade_logger: Optional[logging.Logger] = None,
	) -> None:
		self.login		  = login
		self.password	  = password
		self.server		  = server
		self.dry_run	  = dry_run
		self.log		  = trade_logger or logger
		self._connected	  = False

	# ------------------------------------------------------------------
	# Context manager
	# ------------------------------------------------------------------

	async def __aenter__(self) -> "TradeExecutor":
		if not mt5.initialize():
			err = mt5.last_error()
			self.log.error("MT5 initialize() başarısız: %s", err)
			return self

		if not mt5.login(self.login, self.password, self.server):
			err = mt5.last_error()
			self.log.error(
				"MT5 login() başarısız | login=%s server=%s hata=%s",
				self.login, self.server, err,
			)
			mt5.shutdown()
			return self

		self._connected = True
		info = mt5.account_info()
		self.log.info(
			"MT5 bağlantısı kuruldu | hesap=%s broker=%s",
			info.login if info else self.login,
			info.company if info else self.server,
		)
		return self

	async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
		mt5.shutdown()
		self._connected = False
		self.log.info("MT5 bağlantısı kapatıldı.")

	# ------------------------------------------------------------------
	# Emir yürütme
	# ------------------------------------------------------------------

	async def execute_decision(
		self,
		decision:	str,
		symbol:		str,
		volume:		float,
		sl_price:	Optional[float] = None,
		tp_price:	Optional[float] = None,
		comment:	str = "algo-bot",
	) -> ExecutionResult:
		"""
		Tek bir çağrıyla bracket order (SL + TP dahil) gönderir.

		Parameters
		----------
		decision  : "LONG" | "SHORT" | "BEKLE"
		symbol	  : "BTC/USDT", "BTC/USDT:USDT" veya "BTCUSDT" formatı
		volume	  : Lot büyüklüğü
		sl_price  : Stop-loss fiyatı (None ise request'e eklenmez)
		tp_price  : Take-profit fiyatı (None ise request'e eklenmez)
		comment	  : MT5 emir yorumu
		"""
		# 1. BEKLE kararı
		if decision.upper() == "BEKLE":
			self.log.info("Karar=BEKLE, emir atlanıyor.")
			return ExecutionResult(status=ExecutionStatus.SKIPPED_WAIT)

		# 2. Bağlantı kontrolü
		if not self._connected:
			msg = "MT5 bağlantısı yok, emir gönderilemedi."
			self.log.error(msg)
			return ExecutionResult(
				status=ExecutionStatus.FAILED,
				message=msg,
			)

		# 3. Sembol normalizasyonu
		mt5_symbol = _normalize_symbol(symbol)

		# 4. Emir tipi
		order_type = _get_order_type(decision)
		if order_type is None:
			msg = f"Bilinmeyen karar: '{decision}'"
			self.log.error(msg)
			return ExecutionResult(
				status=ExecutionStatus.FAILED,
				symbol=mt5_symbol,
				message=msg,
			)

		# 5. Güncel fiyat
		tick = mt5.symbol_info_tick(mt5_symbol)
		if tick is None:
			err = mt5.last_error()
			msg = f"Tick verisi alınamadı ({mt5_symbol}): {err}"
			self.log.error(msg)
			return ExecutionResult(
				status=ExecutionStatus.FAILED,
				symbol=mt5_symbol,
				message=msg,
			)

		entry_price = tick.ask if order_type == mt5.ORDER_TYPE_BUY else tick.bid

		# 6. Request sözlüğü (bracket order)
		request: dict = {
			"action":	   mt5.TRADE_ACTION_DEAL,
			"symbol":	   mt5_symbol,
			"volume":	   float(volume),
			"type":		   order_type,
			"price":	   entry_price,
			"type_time":   mt5.ORDER_TIME_GTC,
			"type_filling": mt5.ORDER_FILLING_IOC,
			"magic":	   MAGIC_NUMBER,
			"comment":	   comment,
			"deviation":   10,			# fiyat sapma toleransı (puan)
		}

		if sl_price is not None:
			request["sl"] = float(sl_price)
		if tp_price is not None:
			request["tp"] = float(tp_price)

		side_str = "LONG" if order_type == mt5.ORDER_TYPE_BUY else "SHORT"

		# 7. Dry-run kontrolü
		if self.dry_run:
			self.log.info(
				"[DRY-RUN] Emir gönderilmedi | %s %s vol=%.4f "
				"fiyat=%.5f sl=%s tp=%s",
				side_str, mt5_symbol, volume,
				entry_price, sl_price, tp_price,
			)
			return ExecutionResult(
				status=ExecutionStatus.DRY_RUN,
				symbol=mt5_symbol,
				side=side_str,
				price=entry_price,
				volume=volume,
				sl=sl_price,
				tp=tp_price,
				message="dry_run=True, emir gönderilmedi.",
				raw=request,
			)

		# 8. Emri gönder
		result = mt5.order_send(request)

		if result is None:
			err = mt5.last_error()
			msg = f"order_send None döndü: {err}"
			self.log.error(msg)
			return ExecutionResult(
				status=ExecutionStatus.FAILED,
				symbol=mt5_symbol,
				side=side_str,
				price=entry_price,
				volume=volume,
				sl=sl_price,
				tp=tp_price,
				message=msg,
				raw=request,
			)

		# 9. Sonuç değerlendirme
		# mt5.TRADE_RETCODE_DONE == 10009
		if result.retcode == mt5.TRADE_RETCODE_DONE:
			self.log.info(
				"Emir başarılı | ticket=%s %s %s vol=%.4f "
				"fiyat=%.5f sl=%s tp=%s",
				result.order, side_str, mt5_symbol,
				volume, result.price, sl_price, tp_price,
			)
			return ExecutionResult(
				status=ExecutionStatus.SUCCESS,
				order_id=str(result.order),
				symbol=mt5_symbol,
				side=side_str,
				price=result.price,
				volume=volume,
				sl=sl_price,
				tp=tp_price,
				message=f"retcode={result.retcode}",
				raw=result._asdict(),
			)
		else:
			err = mt5.last_error()
			msg = (
				f"Emir reddedildi | retcode={result.retcode} "
				f"comment='{result.comment}' mt5_err={err}"
			)
			self.log.error(msg)
			return ExecutionResult(
				status=ExecutionStatus.FAILED,
				symbol=mt5_symbol,
				side=side_str,
				price=entry_price,
				volume=volume,
				sl=sl_price,
				tp=tp_price,
				message=msg,
				raw=result._asdict(),
			)