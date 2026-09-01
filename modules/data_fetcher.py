"""
data_fetcher.py
================
Quant Trading Bot - Asenkron Veri Çekme Modülü

Bu modül, Binance borsasından çoklu sembol ve çoklu zaman dilimi (timeframe)
için OHLCV (Open, High, Low, Close, Volume) verisini asenkron olarak çekmek
üzere tasarlanmıştır.

Öne çıkan özellikler
--------------------
* Asenkron yapı (asyncio + ccxt.async_support) -> yüksek I/O verimi.
* Üstel geri çekilmeli (exponential backoff) yeniden deneme mekanizması.
* RateLimitExceeded, NetworkError, ExchangeError için ayrı hata yakalama
  ve yapılandırılmış logging.
* Binance native olarak `45m` zaman dilimini desteklemediği için 15m
  verisinin otomatik olarak pandas.resample ile agregelenmesi.
* 1000 mumdan fazla veri istendiğinde `since` parametresiyle otomatik
  sayfalama (pagination).
* `async with` ile kullanıma uygun context manager arayüzü (resource leak yok).

Kullanım
--------
>>> import asyncio
>>> from data_fetcher import DataFetcher
>>> async def main():
...		async with DataFetcher() as fetcher:
...			data = await fetcher.fetch_all()
...		return data
>>> result = asyncio.run(main())

Gerekli paketler:  ccxt >= 4.0, pandas >= 2.0
"""

from __future__ import annotations

import asyncio
import logging
from typing import Dict, List, Optional, Tuple

import ccxt
import ccxt.async_support as ccxt_async
import pandas as pd


# ---------------------------------------------------------------------------
# Modül seviyesinde logger.
# Uygulamanın giriş noktasında logging.basicConfig(...) veya logging.config
# yapılandırması ile bu logger'ı global politikaya bağlayabilirsiniz.
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


class DataFetcher:
	"""
	Binance üzerinden çoklu sembol/timeframe OHLCV verisi çeken asenkron sınıf.

	Parametreler
	------------
	symbols : List[str], opsiyonel
		Çekilecek pariteler (örn: ['BTC/USDT', 'ETH/USDT']).
	timeframes : List[str], opsiyonel
		Zaman dilimleri. Binance native (15m, 1h, 4h, 1d...) destekleniyor;
		`45m` için 15m verisinden otomatik resample uygulanır.
	candle_limit : int
		Her sembol/timeframe için hedeflenen mum sayısı (varsayılan 1000).
	max_retries : int
		Hata durumunda maksimum yeniden deneme sayısı.
	initial_backoff : float
		İlk geri çekilme süresi (saniye). Her başarısız denemede 2 katına çıkar.
	exchange_options : Optional[Dict]
		ccxt Exchange yapılandırma sözlüğü (api key, secret, sandbox vb.).
	"""

	# -------------------------------------------------------------------
	# Sınıf seviyesinde sabitler.
	# -------------------------------------------------------------------

	# Binance'in native olarak desteklediği zaman dilimleri.
	# Bunların dışındakiler için resample/aggregation gerekir.
	NATIVE_TIMEFRAMES: frozenset = frozenset({
		'1m', '3m', '5m', '15m', '30m',
		'1h', '2h', '4h', '6h', '8h', '12h',
		'1d', '3d', '1w', '1M',
	})

	# Native olmayan özel timeframe'lerin tanımı.
	# `base`	 : Çekilecek native timeframe
	# `rule`	 : pandas.resample kuralı
	# `multiplier`: 1 hedef mum için kaç base mum gerekiyor
	CUSTOM_TIMEFRAME_MAP: Dict[str, Dict[str, object]] = {
		'45m': {'base': '15m', 'rule': '45min', 'multiplier': 3},
	}

	# pandas.resample sırasında uygulanacak OHLCV agregasyon kuralları.
	_OHLCV_AGG: Dict[str, str] = {
		'open':	  'first',
		'high':	  'max',
		'low':	  'min',
		'close':  'last',
		'volume': 'sum',
	}

	# ccxt'nin tek seferde Binance'ten döndürebileceği maksimum mum sayısı.
	_MAX_CANDLES_PER_REQUEST: int = 1000

	# ---------------------------------------------------------------- __init__
	def __init__(
		self,
		symbols: Optional[List[str]] = None,
		timeframes: Optional[List[str]] = None,
		candle_limit: int = 1000,
		max_retries: int = 5,
		initial_backoff: float = 1.0,
		exchange_options: Optional[Dict] = None,
	) -> None:
		self.symbols: List[str] = symbols or [
			'BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'XRP/USDT',
		]
		self.timeframes: List[str] = timeframes or [
			'15m', '45m', '4h', '1d',
		]
		self.candle_limit: int = candle_limit
		self.max_retries: int = max_retries
		self.initial_backoff: float = initial_backoff

		# Exchange örneği `async with` bloğunda oluşturulur.
		# Böylece bağlantı kaynakları yaşam döngüsüne sıkı sıkıya bağlanır.
		self._exchange_options: Dict = exchange_options or {}
		self.exchange: Optional[ccxt_async.binance] = None

	# ---------------------------------------------------------------- Context
	async def __aenter__(self) -> "DataFetcher":
		"""`async with` girişi: exchange'i oluşturur ve marketleri yükler."""
		# enableRateLimit=True -> ccxt kendi içinde isteklere otomatik throttle uygular.
		options: Dict = {
			'enableRateLimit': True,
			'options': {'defaultType': 'spot'},
		}
		options.update(self._exchange_options)

		self.exchange = ccxt_async.binance(options)
		logger.info("Binance exchange oluşturuldu; marketler yükleniyor...")

		try:
			await self.exchange.load_markets()
			logger.info("Marketler başarıyla yüklendi (%d sembol).", len(self.exchange.markets))
		except Exception as exc:
			# Market yüklemesi başarısız olursa kaynakları temiz kapatıyoruz.
			logger.exception("Market yükleme hatası: %s", exc)
			await self.exchange.close()
			raise
		return self

	async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
		"""`async with` çıkışı: exchange bağlantısını her durumda kapatır."""
		if self.exchange is not None:
			try:
				await self.exchange.close()
				logger.info("Binance exchange bağlantısı kapatıldı.")
			except Exception as exc:
				# Kapatma hatasının uygulamayı çökertmesine izin vermiyoruz.
				logger.warning("Exchange kapatılırken uyarı: %s", exc)

	# ---------------------------------------------------------------- Retry'lı tek istek
	async def _fetch_ohlcv_once(
		self,
		symbol: str,
		timeframe: str,
		limit: int,
		since: Optional[int] = None,
	) -> Optional[List[List[float]]]:
		"""
		Tek bir `fetch_ohlcv` çağrısını retry + üstel geri çekilme ile yapar.

		Dönüş
		-----
		Başarılı: ham OHLCV listesi (her eleman: [ts_ms, o, h, l, c, v])
		Başarısız: None (max_retries tükendi)
		"""
		assert self.exchange is not None, "Exchange başlatılmamış. async with kullanın."

		backoff: float = self.initial_backoff

		for attempt in range(1, self.max_retries + 1):
			try:
				ohlcv = await self.exchange.fetch_ohlcv(
					symbol=symbol,
					timeframe=timeframe,
					since=since,
					limit=limit,
				)
				logger.debug(
					"Veri alındı: %s %s, %d mum (deneme %d)",
					symbol, timeframe, len(ohlcv), attempt,
				)
				return ohlcv

			except ccxt.RateLimitExceeded as exc:
				# Borsa "yavaşla" diyor -> beklemek zorundayız.
				logger.warning(
					"Rate limit aşıldı (%s %s) deneme %d/%d. %.1fs bekleniyor. Hata: %s",
					symbol, timeframe, attempt, self.max_retries, backoff, exc,
				)

			except ccxt.NetworkError as exc:
				# Geçici ağ hatası (timeout, dns, connection reset vs.)
				logger.warning(
					"Ağ hatası (%s %s) deneme %d/%d. %.1fs bekleniyor. Hata: %s",
					symbol, timeframe, attempt, self.max_retries, backoff, exc,
				)

			except ccxt.ExchangeError as exc:
				# ExchangeError çoğunlukla kalıcıdır (yanlış sembol, izin sorunu)
				# ama bazı 5xx durumlarda retry işe yarayabilir, o yüzden döngüde kalıyoruz.
				logger.error(
					"Borsa hatası (%s %s) deneme %d/%d: %s",
					symbol, timeframe, attempt, self.max_retries, exc,
				)

			except Exception as exc:
				# Beklenmedik hata; stack trace ile detaylı logla.
				logger.exception(
					"Beklenmedik hata (%s %s) deneme %d/%d: %s",
					symbol, timeframe, attempt, self.max_retries, exc,
				)

			# Son denemeyi de yaptıysak uyumaya gerek yok, çıkıyoruz.
			if attempt >= self.max_retries:
				break

			await asyncio.sleep(backoff)
			backoff *= 2  # Üstel geri çekilme: 1s -> 2s -> 4s -> 8s -> 16s

		logger.error(
			"Veri çekme başarısız (%s %s). Toplam deneme: %d.",
			symbol, timeframe, self.max_retries,
		)
		return None

	# ---------------------------------------------------------------- Sayfalama
	async def _fetch_ohlcv_paginated(
		self,
		symbol: str,
		timeframe: str,
		total_limit: int,
	) -> Optional[List[List[float]]]:
		"""
		ccxt tek seferde max 1000 mum verir. `total_limit` 1000'i aşıyorsa
		`since` parametresiyle geriye dönük olarak sayfalama yapar.

		Strateji
		--------
		1) En güncel `min(1000, total_limit)` mum çekilir (since=None).
		2) En eski mumun timestamp'inden geriye doğru bloklar alınır.
		3) Çakışan / tekrarlayan timestamp'ler filtrelenir.
		4) Son `total_limit` adet mum döndürülür.
		"""
		assert self.exchange is not None

		# ---------- Adım 1: En güncel blok ----------
		first_limit = min(self._MAX_CANDLES_PER_REQUEST, total_limit)
		first_block = await self._fetch_ohlcv_once(symbol, timeframe, limit=first_limit)
		if not first_block:
			return None

		all_candles: List[List[float]] = list(first_block)

		# Tek istek yetiyorsa erken dön.
		if total_limit <= self._MAX_CANDLES_PER_REQUEST:
			return all_candles

		# ---------- Adım 2: Geriye doğru sayfalama ----------
		# parse_timeframe -> saniye; *1000 -> ms.
		tf_ms: int = self.exchange.parse_timeframe(timeframe) * 1000
		earliest_ts: int = first_block[0][0]

		while len(all_candles) < total_limit:
			needed = total_limit - len(all_candles)
			request_limit = min(self._MAX_CANDLES_PER_REQUEST, needed)

			# En eski timestamp'imizden `request_limit * tf_ms` kadar geriye git.
			since = earliest_ts - request_limit * tf_ms
			if since < 0:
				since = 0  # Borsa açılışından önce veri yok

			older_block = await self._fetch_ohlcv_once(
				symbol, timeframe, limit=request_limit, since=since,
			)
			if not older_block:
				logger.warning(
					"Sayfalama erken durdu: %s %s, mevcut %d/%d mum.",
					symbol, timeframe, len(all_candles), total_limit,
				)
				break

			# Çakışan timestamp'leri at (yeni mumlardan eski olanları sakla).
			older_filtered = [c for c in older_block if c[0] < earliest_ts]
			if not older_filtered:
				# Daha eski veri kalmamış olabilir.
				logger.info(
					"Daha eski veri bulunamadı: %s %s. %d mumla devam.",
					symbol, timeframe, len(all_candles),
				)
				break

			all_candles = older_filtered + all_candles
			earliest_ts = all_candles[0][0]

		# Aşırı veri çektiysek sondan kırp.
		return all_candles[-total_limit:]

	# ---------------------------------------------------------------- DataFrame
	@staticmethod
	def _ohlcv_to_dataframe(ohlcv: List[List[float]]) -> pd.DataFrame:
		"""
		Ham ccxt OHLCV listesini standart bir pandas DataFrame'e dönüştürür.

		Sonuç:
			Index: tz-aware UTC DatetimeIndex
			Sütunlar: open, high, low, close, volume (float64)
		"""
		df = pd.DataFrame(
			ohlcv,
			columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'],
		)
		# ms epoch -> UTC datetime
		df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
		df = df.set_index('timestamp')
		# Tip garantisi: tüm OHLCV float64
		df = df.astype('float64')
		# Olası duplike timestamp'leri temizle (sayfalama kenar durumu)
		df = df[~df.index.duplicated(keep='last')]
		df.sort_index(inplace=True)
		return df

	# ---------------------------------------------------------------- Resample
	@classmethod
	def _resample(cls, df_base: pd.DataFrame, rule: str) -> pd.DataFrame:
		"""Verilen base DataFrame'i pandas resample kuralıyla OHLCV olarak agregeler."""
		resampled = (
			df_base
			.resample(rule, label='left', closed='left')
			.agg(cls._OHLCV_AGG)
			.dropna(how='any')	 # Eksik (kısmi) bucket'ları at
		)
		return resampled

	# ---------------------------------------------------------------- Tek sembol-TF
	async def fetch_single(
		self,
		symbol: str,
		timeframe: str,
	) -> Optional[pd.DataFrame]:
		"""
		Belirtilen (symbol, timeframe) için DataFrame döndürür.
		Hata durumunda None döner; istisna fırlatmaz (parallel gather için ideal).
		"""
		# --- Native timeframe: doğrudan çek ---
		if timeframe in self.NATIVE_TIMEFRAMES:
			ohlcv = await self._fetch_ohlcv_paginated(
				symbol, timeframe, total_limit=self.candle_limit,
			)
			if not ohlcv:
				return None
			return self._ohlcv_to_dataframe(ohlcv)

		# --- Özel (resample) timeframe: base TF'den agregele ---
		if timeframe in self.CUSTOM_TIMEFRAME_MAP:
			cfg = self.CUSTOM_TIMEFRAME_MAP[timeframe]
			base_tf: str = cfg['base']			  # 'rule' ve 'multiplier' int/str
			rule: str = cfg['rule']
			multiplier: int = cfg['multiplier']

			# `candle_limit` adet 45m için en az `candle_limit * 3` adet 15m gerekir.
			# Resample sınırlarında veri kaybı olabileceği için ufak bir tampon ekliyoruz.
			base_needed = self.candle_limit * multiplier + multiplier

			base_ohlcv = await self._fetch_ohlcv_paginated(
				symbol, base_tf, total_limit=base_needed,
			)
			if not base_ohlcv:
				return None

			df_base = self._ohlcv_to_dataframe(base_ohlcv)
			df_resampled = self._resample(df_base, rule)
			return df_resampled.tail(self.candle_limit)

		# --- Tanımsız timeframe ---
		logger.error("Desteklenmeyen timeframe: %s", timeframe)
		return None

	# ---------------------------------------------------------------- Tümü
	async def fetch_all(self) -> Dict[str, Dict[str, pd.DataFrame]]:
		"""
		Tüm sembol x timeframe kombinasyonları için paralel veri çeker.

		Dönüş
		-----
		{
			'BTC/USDT': {'15m': DataFrame, '45m': DataFrame, ...},
			'ETH/USDT': {...},
			...
		}
		Hatalı kombinasyonlar sonucunda eksik olabilir.
		"""
		if self.exchange is None:
			raise RuntimeError("DataFetcher 'async with' bloğu içinde kullanılmalı.")

		tasks: List[asyncio.Task] = []
		keys: List[Tuple[str, str]] = []

		for symbol in self.symbols:
			for tf in self.timeframes:
				tasks.append(asyncio.create_task(self.fetch_single(symbol, tf)))
				keys.append((symbol, tf))

		logger.info(
			"Toplam %d görev paralel olarak çalıştırılıyor (%d sembol x %d timeframe).",
			len(tasks), len(self.symbols), len(self.timeframes),
		)

		# return_exceptions=True -> bir görev patlasa bile diğerleri devam etsin.
		results = await asyncio.gather(*tasks, return_exceptions=True)

		out: Dict[str, Dict[str, pd.DataFrame]] = {}
		success_count = 0

		for (symbol, tf), result in zip(keys, results):
			if isinstance(result, Exception):
				logger.error("Görev hatası (%s %s): %s", symbol, tf, result)
				continue
			if result is None or result.empty:
				logger.warning("Boş veri (%s %s).", symbol, tf)
				continue
			out.setdefault(symbol, {})[tf] = result
			success_count += 1

		logger.info(
			"Veri çekme tamamlandı: %d/%d başarılı.",
			success_count, len(tasks),
		)
		return out


# ---------------------------------------------------------------------------
# Modül doğrudan çalıştırıldığında çalışacak demo bölümü.
#	python data_fetcher.py
# ---------------------------------------------------------------------------
async def _demo() -> None:
	logging.basicConfig(
		level=logging.INFO,
		format='%(asctime)s | %(levelname)-7s | %(name)s | %(message)s',
	)

	async with DataFetcher() as fetcher:
		data = await fetcher.fetch_all()

	# Basit özet yazdır.
	for symbol, tf_map in data.items():
		for tf, df in tf_map.items():
			print(f"\n=== {symbol} | {tf} | {len(df)} mum ===")
			print(df.head(2))
			print('...')
			print(df.tail(2))


if __name__ == '__main__':
	asyncio.run(_demo())