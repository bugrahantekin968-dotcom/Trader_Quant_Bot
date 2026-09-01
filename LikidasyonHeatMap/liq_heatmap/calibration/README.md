# Gerçek-Likidasyon Kalibrasyon Döngüsü

Modeli, **gerçekte olan likidasyonlara** göre kendi kendine ayarlar. Varsayılan
`LEVERAGE_DISTRIBUTION` yerine, piyasanın gerçek kaldıraç davranışını ölçüp
onu kullanır.

## Mantık

Bir likidasyon, pozisyonun kaldıracını ele verir:

```
long  P fiyatında patlar, son tepe H'den geldi   →  L ≈ H / (H − P)
short P fiyatında patlar, son dip Lo'dan geldi    →  L ≈ Lo / (P − Lo)
```

Binlerce likidasyonun ima ettiği kaldıraçları USD-ağırlıklı toplarsak, gerçek
kaldıraç dağılımını elde ederiz. Sentetik testte bilinen bir dağılımı
(%40/30/20/10 @ 10x/25x/50x/100x) neredeyse tam geri kazandı.

## 3 Adım

### 1. Veri topla (otomatik, canlı çalışırken)
`main.py` başladığında `engine.enable_recorder()` otomatik açık. Her gerçek
likidasyon (`forceOrder` / `allLiquidation` WS olayları) referans tepe/dip ile
birlikte `_calib/liquidations.jsonl`'e yazılır.

```bash
python main.py        # canlı çalıştır, likidasyonları biriktirir
```

**Ne kadar süre?** Coin başına ≥1000 likidasyon olayı önerilir. BTC'de
volatil bir haftada bu birkaç günde dolar; sakin piyasada 1-2 hafta.

### 2. Fit et (offline, veri yeterince birikince)
```bash
python -m calibration.fit_leverage --min 1000
```
Bu, `_calib/liquidations.jsonl`'i okur, coin başına ampirik kaldıraç
dağılımını çıkarır ve `_calib/calibrated_leverage.json`'a yazar. Yeterli
olayı olmayan coin'ler atlanır (config default'u korunur).

### 3. Yükle (otomatik, sonraki başlatmada)
Engine her başlangıçta `_calib/calibrated_leverage.json` varsa onu yükler,
yoksa config default'unu kullanır. Hiçbir kod değişikliği gerekmez —
dosya varsa devreye girer.

## Sürekli iyileştirme

Bu bir **döngü**: çalıştır → biriktir → fit et → yükle → tekrar. Her fit,
modeli piyasanın güncel kaldıraç davranışına yaklaştırır. Ayda bir yeniden
fit etmek, kaldıraç rejimi değişimlerini (boğa/ayı) yakalar.

## Doğrulama (opsiyonel ama önerilir)

Fit edilen dağılımın eski varsayımdan daha iyi olduğunu görmek için: fit
öncesi ve sonrası, öngörülen mıknatıs bölgelerini bir sonraki haftanın
gerçek likidasyonlarıyla karşılaştır (out-of-sample). Öngörülen yoğun bölge
ile gerçekleşen likidasyon fiyatları örtüşüyorsa kalibrasyon iyi.

## Notlar

- Referans tepe/dip 24h mum geçmişinden alınır. Lookback'i değiştirmek için
  `_recent_ref` (engine) düzenlenebilir.
- DEX (Hyperliquid) zaten gerçek kaldıraç veriyor; kalibrasyon esas olarak
  CEX projeksiyonunu düzeltir.
- Kaydedici maliyeti ihmal edilebilir (likidasyon başına bir JSONL satırı).
