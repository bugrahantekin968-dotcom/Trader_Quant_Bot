"""
export_mt5.py — MT5'ten 1h OHLCV çekip backtest için data/ klasörüne yazar.
Bunu proje kökünde (main.py'nin yanında) BİR KEZ çalıştır:

    python export_mt5.py

Sonra backtest:
    python -m backtest.run_backtest --csv-dir ./data --threshold 49
"""
import os
import MetaTrader5 as mt5
import pandas as pd

# Botundaki MT5 sembolleri (BTC/USDT→BTCUSD eşlemesi)
SYMBOLS = ["BTCUSD", "ETHUSD", "XRPUSD"]
BARS = 6000          # ~250 gün 1h → 1d/4h resample için bolca geçmiş
OUT_DIR = "data"

# --- Broker saat dilimi → UTC kaydırma ---------------------------------
# Eightcap sunucu saati UTC DEĞİL. MT5 copy_rates "time" alanını broker-yerel
# saati olarak verir. Killzone (+1 C.3 bonusu) UTC varsayar; küçük bir etki ama
# tam doğru olsun istersen broker'ının UTC farkını buraya yaz (örn UTC+3 → -3).
# Bilmiyorsan 0 bırak; trend/RSI/EMA/S-R hepsi zaman-bağımsız, sadece killzone
# ±1 puan kayar.
BROKER_UTC_OFFSET_HOURS = 0

def main():
    if not mt5.initialize():
        raise SystemExit(f"MT5 initialize başarısız: {mt5.last_error()}")
    os.makedirs(OUT_DIR, exist_ok=True)
    try:
        for sym in SYMBOLS:
            rates = mt5.copy_rates_from_pos(sym, mt5.TIMEFRAME_H1, 0, BARS)
            if rates is None or len(rates) == 0:
                print(f"[atla] {sym}: veri gelmedi ({mt5.last_error()})")
                continue
            df = pd.DataFrame(rates)
            df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
            if BROKER_UTC_OFFSET_HOURS:
                df["time"] = df["time"] - pd.Timedelta(hours=BROKER_UTC_OFFSET_HOURS)
            df = df.rename(columns={"tick_volume": "volume"})
            cols = ["time", "open", "high", "low", "close", "volume"]
            path = os.path.join(OUT_DIR, f"{sym}.csv")
            df[cols].to_csv(path, index=False)
            print(f"[yaz] {sym}: {len(df)} bar → {path}")
    finally:
        mt5.shutdown()
    print("\nBitti. Şimdi: python -m backtest.run_backtest --csv-dir ./data --threshold 49")

if __name__ == "__main__":
    main()
