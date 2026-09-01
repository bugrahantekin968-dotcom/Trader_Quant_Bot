# Liquidation Zone Engine

Multi-symbol × multi-timeframe × multi-exchange likidasyon **bölge** raporu.
CoinGlass tarzı sıcaklık haritası yerine: **aksiyon-edilebilir mıknatıs bölgeleri** + dolar miktarı.

---

## Felsefe

Sıcaklık haritası "şurada renkli bir alan var" der. Trader'ın aslında ihtiyaç duyduğu bilgi:

> "Şu fiyat aralığında $X milyon likide olur, şu kadar uzaktayız, R:R şu"

Bu sistem her (coin, timeframe) için bunu üretiyor:

```
BTC — 3d görünüm
Şu anki fiyat: $66,248    |    Toplam izlenen OI: $14.00B

↓ LONG LİKİDASYONLARI (fiyat aşağı çekilirse)
   Bölge 1:  $59,767 – $65,053   →  $4.71B   ████████████████████
             merkez $59,800  |  -9.73% spot'tan uzak
   Bölge 2:  $58,462 – $59,180   →  $809.2M  ███
   ...

↑ SHORT LİKİDASYONLARI (fiyat yukarı çekilirse)
   Bölge 1:  $62,377 – $68,511   →  $3.46B   ███████████████
   ...

EN YAKIN MIKNATISLAR
   ↑  $70,044   ($503.6M, +5.73%)
   ↓  $59,800   ($4.71B, -9.73%)
```

---

## Mimari

```
   ┌────────────┐       ┌─────────────────────────────────────┐
   │ Exchange   │       │  HeatmapEngine                       │
   │ WS streams │──────▶│  ├─ project_candle (density bins)    │
   │  + REST    │       │  ├─ project_candle (DOLLAR bins) ←★  │
   └────────────┘       │  ├─ project_exact_dex (HL)           │
                        │  └─ percent_bins per (coin, tf)      │
                        └──────────────┬───────────────────────┘
                                       │
                        ┌──────────────▼───────────────────────┐
                        │  Zone Extractor                       │
                        │  ├─ peak detection                    │
                        │  ├─ boundary expansion                │
                        │  ├─ overlap merging                   │
                        │  └─ OI-anchored $ calibration  ←★    │
                        └──────────────┬───────────────────────┘
                                       │
              ┌────────────────────────┼─────────────────────────┐
              ▼                        ▼                          ▼
       Text reports             JSON snapshots          Multi-TF signals
       (12 .txt files)         (bot consumption)        (zone-anchored)
                                                        + R:R estimate
```

★ = Yeni; bölge çıkarımının dayandığı temel.

---

## Tasarım Kararları

### 1. Çift matris yaklaşımı
- **Density bins**: `log(1+vol) × decay × leverage × side_ratio` — log-tamed, görsel kontrast iyi
- **Dollar bins**: `vol × decay × leverage × side_ratio` — linear, dolar kalibrasyonu için
- Aynı projection, iki paralel yazım. Çıktıda dollar matris OI'ye anchor edilir.

### 2. OI-anchored kalibrasyon
```
long_target  = total_OI × long_ratio
scale_factor = long_target / Σ(long_matris)
zone_$       = raw_zone_mass × scale_factor
```
Sonuç: tüm uzun bölgelerin toplamı tam olarak `long_OI`'a eşit olur.

### 3. Bölge tespiti (scipy-free)
1. Her fiyat bin'i için zaman boyunca toplam kütle (`column_mass`)
2. Manuel peak detection: `arr[i] > arr[i±1]` + min %8 prominence
3. Her peak'ten dışa doğru genişlet: kütle peak × %30'un altına düşene kadar
4. Üst üste binen bölgeleri merge et
5. Toplam %3'ten küçük bölgeleri at (gürültü filtresi)
6. Yöne göre top 5

### 4. Sinyal mantığı (bölge-anchored)
Eski: bin yoğunluğunun yön bandı şekline bakardı.
Yeni: en yakın yukarı vs aşağı bölgenin **dolar büyüklüğü** ve **uzaklık** ile ağırlıklı kararı verir. Multi-TF confluence (3d:0.40, 1w:0.30) ağırlıklı.

### 5. R:R tahmini
Hedef: konsensüs yönündeki en yakın büyük bölge.
Stop anchor: ters yöndeki en yakın büyük bölge.
`R:R = |target − spot| / |stop − spot|`. Trader'a doğrudan çıktıda söyler.

---

## Klasör Yapısı

```
liq_heatmap/
├── config.py
├── core/
│   ├── math_models.py
│   ├── exact_liq.py
│   ├── percent_bins.py
│   ├── timeframe.py
│   └── zone_extractor.py            ★ YENİ
├── exchanges/
│   ├── base.py
│   ├── binance_client.py
│   ├── bybit_client.py
│   ├── okx_client.py
│   ├── bitget_client.py
│   ├── hyperliquid_client.py
│   └── registry.py
├── engine/
│   ├── candle_aggregator.py
│   ├── backfill.py
│   ├── heatmap_engine.py            ★ paralel dollar_bins eklendi
│   └── persistence.py
├── viz/
│   └── zone_report.py               ★ YENİ (eski plotly_render.py kaldırıldı)
├── strategy/
│   └── signals.py                   ★ tamamen yeniden yazıldı (zone-anchored)
├── tests/
│   └── test_all.py                  ★ 23 test (4 yeni)
├── demo_offline.py                  ★ mean-reverting price sim
├── main.py                          ★ zone-report loop
└── README.md
```

---

## Hızlı Başlangıç

```bash
pip install numpy plotly aiohttp

python tests/test_all.py            # 23/23 passed
python demo_offline.py              # 12 .txt + 12 .json + 3 .html üretir
python main.py                      # canlı modda 5 borsa + 3 coin
```

---

## Bot Entegrasyonu

JSON formatı:

```json
{
  "coin": "BTC",
  "tf": "3d",
  "current_price": 66248.44,
  "total_oi_usd": 14000000000,
  "zones": {
    "long": [
      {
        "rank": 1,
        "price_low":  59767.0,
        "price_high": 65053.0,
        "price_center": 59800.0,
        "dollars": 4710000000,
        "pct_from_spot": -9.73
      }
    ],
    "short": [...]
  },
  "nearest_magnets": {
    "above": {...},
    "below": {...}
  }
}
```

Bot mantığı:
```python
import json
with open("_out/zones/btc_3d.json") as f:
    data = json.load(f)
near_up   = data["nearest_magnets"]["above"]
near_down = data["nearest_magnets"]["below"]
if near_up and near_up["dollars"] > 1e9:    # >$1B mıknatıs
    bot.set_target(near_up["price_center"])
```

---

## CoinGlass'a Göre Konumumuz

CoinGlass: heatmap görseli + zoom + alttan yatay bar profili.

Biz: bölge raporu + dolar kalibrasyonu + R:R + multi-TF konsensüs.

İkimiz aynı veriyi (per-exchange OI + L/S ratio + price action + leverage dist) işliyoruz. Farkımız: çıktı formatı **daha actionable** — trader doğrudan trade kararına input alıyor.

---

## Üretime Geçerken

- [ ] Tier-based MMR ladder (Binance pozisyon büyüklüğüne göre değişen MMR)
- [ ] 7-14 gün canlı kalibrasyon: `LEVERAGE_DISTRIBUTION` ince ayarı
- [ ] dYdX v4 + GMX (2. ve 3. DEX kapsamı)
- [ ] Bölge stabilitesi takibi: aynı bölge N gün ardışık görünüyorsa "güvenilir mıknatıs" işareti
- [ ] Telegram alert: `consensus_confidence > 0.6 AND R:R > 2.0` olduğunda
