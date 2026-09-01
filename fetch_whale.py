"""
fetch_whale.py — CoinGlass'tan tarihsel whale (L/S + OI) çekip cache'ler.
Proje kökünde (main.py'nin yanında) BİR KEZ çalıştır:

    python fetch_whale.py

Sonra tam-sistem backtest:
    python -m backtest.run_backtest --csv-dir ./data \
           --whale-cache ./data/whale_cache.json --threshold 65

Not: L/S 1d granülaritede (sınırsız geçmiş) → tüm backtest aralığını kaplar.
     OI 4h granülaritede (planında 180 gün) → canlıdaki "OI change 4h"e birebir.
     Funding nötr varsayılıyor (canlıda ~hep nötrdü, exchange-list'in history'si yok).
"""
import os
from backtest import whale_history

# Botla aynı CoinGlass anahtarı (demo değil — fırsatın olunca rotate et)
COINGLASS_API_KEY = "29a104fc4b2e44c09d6337c5e0c591b9"

SYMBOLS = ["BTCUSD", "ETHUSD", "XRPUSD"]
OUT     = os.path.join("data", "whale_cache.json")

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    whale_history.fetch_and_cache(
        COINGLASS_API_KEY, SYMBOLS, OUT,
        ls_interval="1d",   # sınırsız geçmiş (D.1 contrarian — en önemli pillar)
        oi_interval="4h",   # 180 gün, gerçek 4h OI değişimi (D.3)
        limit=4500,
    )
