"""
liquidity.py
Asenkron L2 Order Book toplayıcı — Binance, Bybit, OKX, Gate.io, MEXC
main.py ile uyumlu get_dynamic_distribution arayüzü.
"""

import asyncio
import logging
from typing import Optional
import ccxt.async_support as ccxt_async

logger = logging.getLogger(__name__)


def _build_exchange_instances() -> dict:
	opts = {"enableRateLimit": True, "timeout": 10_000}
	return {
		"binance": ccxt_async.binance(opts),
		"bybit":   ccxt_async.bybit(opts),
		"okx":	   ccxt_async.okx(opts),
		"gateio":  ccxt_async.gateio(opts),
		"mexc":	   ccxt_async.mexc(opts),
	}


class LiquidityManager:
	"""
	Kullanım:
		async with LiquidityManager() as lm:
			dist = await lm.get_dynamic_distribution(
				current_price=60000.0,
				symbol="BTC/USDT",
				tier_pct=0.01,
				depth_pct=0.10,
			)
	"""

	def __init__(self):
		self._exchanges: dict = {}

	async def __aenter__(self):
		self._exchanges = _build_exchange_instances()
		return self

	async def __aexit__(self, exc_type, exc_val, exc_tb):
		await asyncio.gather(
			*(ex.close() for ex in self._exchanges.values()),
			return_exceptions=True,
		)
		self._exchanges.clear()

	async def _fetch_one(
		self,
		name: str,
		exchange,
		symbol: str,
		limit: int = 1000,
	) -> Optional[dict]:
		try:
			ob = await exchange.fetch_order_book(symbol, limit=limit)
			return ob
		except Exception as exc:
			logger.warning("[%s] order book hatası (%s): %s", name, symbol, exc)
			return None

	async def get_dynamic_distribution(
		self,
		current_price: float,
		symbol: str,
		tier_pct: float = 0.01,
		depth_pct: float = 0.10,
	) -> dict:
		"""
		5 borsadan L2 verisi çeker, ±depth_pct aralığını tier_pct'lik
		kademelere bölerek her kademedeki hacmin toplam içindeki
		yüzdesini hesaplar.

		Dönen yapı:
		{
			"current_price":		float,
			"depth_range_pct":		int,
			"tier_size_pct":		float,
			"bids_distribution":	[{"tier", "range_usd", "pct_from_current",
									  "volume_usd", "volume_share_pct"}, ...],
			"asks_distribution":	[...],
			"total_bid_volume_usd": float,
			"total_ask_volume_usd": float,
			"bid_ask_ratio":		float,
		}
		"""
		# ── 1. Paralel order book çekimi ────────────────────────
		tasks = [
			self._fetch_one(name, ex, symbol)
			for name, ex in self._exchanges.items()
		]
		results = await asyncio.gather(*tasks, return_exceptions=False)

		all_bids: list[list[float]] = []
		all_asks: list[list[float]] = []
		# ── ESKİ (Hatalı) KISMI SİL VE YERİNE BUNU YAPIŞTIR ──
		for ob in results:
			if ob is None:
				continue
			# Sadece ilk 2 değeri (fiyat ve miktar) alıyoruz, 3. değer (timestamp vb.) varsa eliyoruz.
			all_bids.extend([[float(item[0]), float(item[1])] for item in ob.get("bids", [])])
			all_asks.extend([[float(item[0]), float(item[1])] for item in ob.get("asks", [])])

		# ── 2. Fiyat aralığı sınırları ───────────────────────────
		lower_bound = current_price * (1.0 - depth_pct)
		upper_bound = current_price * (1.0 + depth_pct)

		bids_in_range = [
			[p, q] for p, q in all_bids
			if lower_bound <= p < current_price
		]
		asks_in_range = [
			[p, q] for p, q in all_asks
			if current_price <= p <= upper_bound
		]

		num_tiers = max(1, int(round(depth_pct / tier_pct)))

		# ── 3. Kademe dağılımı ───────────────────────────────────
		bids_dist = self._build_distribution(
			levels=bids_in_range,
			ref_price=current_price,
			num_tiers=num_tiers,
			tier_pct=tier_pct,
			direction="bid",
		)
		asks_dist = self._build_distribution(
			levels=asks_in_range,
			ref_price=current_price,
			num_tiers=num_tiers,
			tier_pct=tier_pct,
			direction="ask",
		)

		# ── 4. Toplamlar ve yüzde payları ────────────────────────
		total_bid = sum(t["volume_usd"] for t in bids_dist)
		total_ask = sum(t["volume_usd"] for t in asks_dist)

		for t in bids_dist:
			t["volume_share_pct"] = (
				round(t["volume_usd"] / total_bid * 100, 2) if total_bid else 0.0
			)
		for t in asks_dist:
			t["volume_share_pct"] = (
				round(t["volume_usd"] / total_ask * 100, 2) if total_ask else 0.0
			)

		bid_ask_ratio = round(total_bid / total_ask, 4) if total_ask else 0.0

		return {
			"current_price":		round(current_price, 6),
			"depth_range_pct":		int(depth_pct * 100),
			"tier_size_pct":		round(tier_pct * 100, 2),
			"bids_distribution":	bids_dist,
			"asks_distribution":	asks_dist,
			"total_bid_volume_usd": round(total_bid, 2),
			"total_ask_volume_usd": round(total_ask, 2),
			"bid_ask_ratio":		bid_ask_ratio,
		}

	@staticmethod
	def _build_distribution(
		levels:	   list[list[float]],
		ref_price: float,
		num_tiers: int,
		tier_pct:  float,
		direction: str,	  # "bid" | "ask"
	) -> list[dict]:
		"""
		Seviyeleri num_tiers kademeye böler.
		volume_share_pct üst katmanda (get_dynamic_distribution) eklenir.
		"""
		tiers: list[dict] = []

		for k in range(1, num_tiers + 1):
			if direction == "bid":
				upper	= ref_price * (1.0 - (k - 1) * tier_pct)
				lower	= ref_price * (1.0 - k * tier_pct)
				mid_pct = -round((k - 0.5) * tier_pct * 100, 2)
			else:
				lower	= ref_price * (1.0 + (k - 1) * tier_pct)
				upper	= ref_price * (1.0 + k * tier_pct)
				mid_pct = round((k - 0.5) * tier_pct * 100, 2)

			vol_usd = sum(
				float(p) * float(q)
				for p, q in levels
				if lower <= float(p) <= upper
			)

			tiers.append({
				"tier":				k,
				"range_usd":		f"{lower:,.4f}-{upper:,.4f}",
				"pct_from_current": mid_pct,
				"volume_usd":		round(vol_usd, 2),
				# volume_share_pct: get_dynamic_distribution içinde eklenir
			})

		return tiers