"""
backtest/run_backtest.py
══════════════════════════════════════════════════════════════════════
CLI entry point for the deterministic walk-forward backtest.

USAGE
    # Demo on synthetic data (no files needed — validates the pipeline):
    python -m backtest.run_backtest --synthetic --days 120 --threshold 49

    # Real data — drop 1h OHLCV CSVs (UTC 'time' col) into a folder:
    #   ./data/BTCUSD.csv  ./data/ETHUSD.csv  ./data/XRPUSD.csv
    python -m backtest.run_backtest --csv-dir ./data --threshold 49

    # Sweep the threshold to find the best expectancy:
    python -m backtest.run_backtest --csv-dir ./data --sweep 40 55 5

HOW TO EXPORT REAL 1h DATA FROM MT5 (run once in your live env):
    import MetaTrader5 as mt5, pandas as pd
    mt5.initialize()
    for mt5_sym in ("BTCUSD", "ETHUSD"):
        r = mt5.copy_rates_from_pos(mt5_sym, mt5.TIMEFRAME_H1, 0, 6000)
        df = pd.DataFrame(r)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df[["time","open","high","low","close","tick_volume"]]\
          .rename(columns={"tick_volume":"volume"})\
          .to_csv(f"data/{mt5_sym}.csv", index=False)
    # 6000 1h bars ≈ 250 days → enough to resample 1d/4h with history.
    # NOTE: MT5 server time may be non-UTC; the killzone (+1 on C.3) assumes
    # UTC. Small effect; for exactness shift 'time' to UTC before saving.
"""
from __future__ import annotations
import argparse, os, sys
from . import datafeed, engine

# Windows konsol kodlama kalkani (cp1254): ozet/print icindeki Unicode (->, >=, ...)
# kodlanamayinca backtest'i COKERTMESIN. stdout/stderr UTF-8'e sabitlenir (no-op if UTF-8).
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


DEFAULT_SYMBOLS = ["BTCUSD", "ETHUSD"]
_START_PRICE = {"BTCUSD": 73000.0, "ETHUSD": 2000.0, "XRPUSD": 1.30}


def build_feeds(args):
    feeds = {}
    if args.csv_dir:
        for sym in args.symbols:
            path = os.path.join(args.csv_dir, f"{sym}.csv")
            if not os.path.exists(path):
                print(f"[skip] {path} yok"); continue
            feeds[sym] = datafeed.DataFeed(datafeed.load_csv(path))
            print(f"[load] {sym}: {len(feeds[sym])} adet 1h bar")
    else:
        for i, sym in enumerate(args.symbols):
            df = datafeed.generate_synthetic(sym, n_hours=24 * args.days, seed=i,
                                             start_price=_START_PRICE.get(sym, 1000.0))
            feeds[sym] = datafeed.DataFeed(df)
            print(f"[synthetic] {sym}: {len(df)} adet 1h bar (~{args.days} gün)")
    return feeds


def main(argv=None):
    p = argparse.ArgumentParser(description="Deterministic walk-forward backtest")
    p.add_argument("--synthetic", action="store_true", help="sentetik veri üret")
    p.add_argument("--csv-dir", type=str, default=None, help="gerçek 1h CSV klasörü")
    p.add_argument("--symbols", type=str, default=",".join(DEFAULT_SYMBOLS))
    p.add_argument("--days", type=int, default=120, help="sentetik gün sayısı")
    p.add_argument("--threshold", type=float, default=49.0, help="karar eşiği (A/B/C-only ≈49)")
    p.add_argument("--warmup", type=int, default=300, help="ısınma 1h barı")
    p.add_argument("--fill-window", type=int, default=4, help="limit emir dolum penceresi (bar)")
    p.add_argument("--max-hold", type=int, default=240, help="maks tutuş (1h bar)")
    p.add_argument("--eval-every", type=int, default=4, help="kaç 1h barda bir değerlendir (1=tam sadık/yavaş, 4=her 4h)")
    p.add_argument("--out-dir", type=str, default="backtest_out")
    p.add_argument("--exit-mode", choices=["raw", "halfway_be", "scaleout", "trail", "ladder"],
                   default="halfway_be",
                   help="çıkış sistemi: raw | halfway_be (canlı Stage-1) | scaleout | trail | ladder (özel merdiven --ladder ile)")
    p.add_argument("--n-parts", type=int, default=10, help="scaleout için parça sayısı")
    p.add_argument("--half-close", type=float, default=0.5,
                   help="exit-mode halfway_be: yarı-yolda kapatılacak oran (0.5=%%50 varsayılan; 0.75=%%75'i yarı-yolda al, %%25 runner)")
    p.add_argument("--ladder", type=str, default="",
                   help='exit-mode ladder: özel merdiven "ilerleme:oran,..." '
                        '(ör. "0.25:0.25,0.5:0.5,1.0:0.25" = yolun %%25inde %%25 al, %%50sinde %%50 al '
                        '(cum %%75), kalan %%25 TPye; %%50yi geçince SL→BE)')
    p.add_argument("--trail", action="store_true",
                   help="scaleout: %%50 satıldıktan sonra stop'u BE'ye çek")
    p.add_argument("--max-rr", type=float, default=None,
                   help="TP üst sınırı: RR bunu aşarsa TP cap'lenir (ör. 3.5) — uzak/ulaşılamaz hedefleri keser")
    p.add_argument("--cost-pct", type=float, default=0.0,
                   help="fill başına işlem maliyeti (fiyat oranı, ör. 0.0003 = %%0.03 spread)")
    p.add_argument("--no-stage1", action="store_true",
                   help="(alias) ham TP/SL → --exit-mode raw ile aynı")
    p.add_argument("--whale-cache", type=str, default=None,
                   help="tarihsel whale cache (data/whale_cache.json) — D sütununu gerçek değerlerle doldurur")
    p.add_argument("--no-reversal", action="store_true",
                   help="CHoCH reversal playbook'unu KAPAT (deneysel; varsayılan açık)")
    p.add_argument("--short-exh-veto", type=float, default=0.0,
                   help="alt-TF dönüş vetosu: short'ta exhaustion >= bu eşik ise işlemi ALMA (ör. 20; 0=kapalı)")
    p.add_argument("--long-exh-veto", type=float, default=0.0,
                   help="alt-TF dönüş vetosu (LONG): long'ta exhaustion >= eşik ise işlemi ALMA (ör. 20; 0=kapalı)")
    p.add_argument("--tp-cap-rr", type=float, default=3.5,
                   help="TP RR tavanı: plan TP'si bu RR'de kesilir (varsayılan 3.5; ETH testi için 3.0)")
    p.add_argument("--tp-cap-map", type=str, default="",
                   help='sembol bazlı TP tavanı, ör: "BTCUSD:3.5,ETHUSD:3.0" (canlı C-konfig ile birebir)')
    p.add_argument("--regime-filter", action="store_true",
                   help="rejim filtresi: ADX < eşik ise (chop) işlem ALMA")
    p.add_argument("--regime-adx-min", type=float, default=20.0,
                   help="rejim ADX eşiği (varsayılan 20; altı=chop=atla)")
    p.add_argument("--regime-tf", type=str, default="4h",
                   help="rejim ADX zaman dilimi (1d/4h/1h; varsayılan 4h)")
    p.add_argument("--require-confirmation", action="store_true",
                   help="GİRİŞ TEYİDİ: seviyeye yakınlık yetmez — 4H kırılım-kapanış / dönüş mumu + 1H momentum şart")
    p.add_argument("--be-at-r", type=float, default=0.0,
                   help="KORUMA: +R bu eşiğe ulaşınca SL'i başabaşa çek (ör. 0.75; 0=kapalı)")
    # G3 — re-entry guard (canlı main.py guard'ının backtest portu; bar=1h)
    p.add_argument("--no-guard", action="store_true",
                   help="G3 yeniden-giriş kalkanını KAPAT (cooldown/ardışık-stop/seviye-kilidi)")
    p.add_argument("--guard-cooldown-bars", type=int, default=6,
                   help="stop sonrası aynı-yön/seviye cooldown (1h bar; canlı 6h=6)")
    p.add_argument("--guard-consec-limit", type=int, default=3,
                   help="aynı yönde N ardışık stop -> o yönü askıya al")
    p.add_argument("--guard-halt-bars", type=int, default=12,
                   help="ardışık-stop askı süresi (1h bar; canlı 12h=12)")
    p.add_argument("--guard-relevel-pct", type=float, default=0.012,
                   help="cooldown icinde stop-seviyesi bandi (oran, varsayilan 0.012 = %%1.2)")
    p.add_argument("--color-mode", choices=["off", "exhaustion", "rejection"], default="off",
                   help="G-B S/R mum-rengi modu: off | exhaustion(yesil-tepe/kirmizi-dip) | rejection(kirmizi-tepe/yesil-dip)")
    p.add_argument("--no-1d-gate", action="store_true",
                   help="G-A 1D yon kapisini KAPAT (1D bullish->no short / bearish->no long)")
    p.add_argument("--sr-min-sep", type=float, default=0.0,
                   help="G-SR: tutulan iki S/R arasi min mesafe orani (0=kapali; 0.03=%%3) — ~%%2 spam'i eler")
    p.add_argument("--sr-major-only", action="store_true",
                   help="G-SR: sadece MAJOR (1D+4H ortusen) S/R seviyelerini kullan")
    p.add_argument("--sr-max-per-side", type=int, default=0,
                   help="G-SR: taraf basina en fazla N S/R seviyesi (0=sinirsiz)")
    p.add_argument("--entry-max-ext", type=float, default=None,
                   help="ENTRY-B over-extension: 4H EMA200'den >bu oran uzakta o yonde girme (0=kapali; default 0.15)")
    p.add_argument("--entry-retest", action="store_true",
                   help="ENTRY-D retest girisi: limiti en yakin S/R seviyesine koy, pullback'te dol (chase eler)")
    p.add_argument("--entry-retest-max", type=float, default=0.03,
                   help="ENTRY-D: retest seviyesi fiyatin bu orani icinde olmali (default 0.03=%%3)")
    p.add_argument("--sweep", nargs=3, type=float, metavar=("LO", "HI", "STEP"),
                   default=None, help="eşik taraması: LO HI STEP")
    p.add_argument("--pd-filter", action="store_true",
                   help="G6.2 premium/discount vetosunu AÇ (default OFF — veri reddetti)")
    p.add_argument("--ltf-trigger", action="store_true",
                   help="G6.3 LTF giriş tetiğini AÇ (default OFF — veri reddetti)")
    p.add_argument("--va-tp", action="store_true",
                   help="G6.5 VAH/VAL/POC TP havuzunu AÇ (default OFF — veri reddetti)")
    p.add_argument("--no-pd-filter", action="store_true",
                   help="G6.2 premium/discount konum vetosunu KAPAT (A/B)")
    p.add_argument("--pd-premium", type=float, default=None,
                   help="premium eşiği (default 0.62)")
    p.add_argument("--pd-discount", type=float, default=None,
                   help="discount eşiği (default 0.38)")
    p.add_argument("--no-ltf-trigger", action="store_true",
                   help="G6.3 LTF giriş tetiğini KAPAT (backtest'te 1h fallback ile çalışır)")
    p.add_argument("--no-va-tp", action="store_true",
                   help="G6.5 VAH/VAL/POC'u TP havuzundan ÇIKAR (eski TP seçimi)")
    p.add_argument("--max-sl", type=float, default=None,
                   help="SL tavanı (oran, örn. 0.025). Verilmezse scorer default (0.04)")
    p.add_argument("--min-sl", type=float, default=None,
                   help="SL tabanı (oran, örn. 0.015). Verilmezse scorer default")
    p.add_argument("--long-only", action="store_true",
                   help="SHORT tarafını komple kapat (sadece LONG işlemler)")
    # S/R kampanyasi (Agu 2026) — S-1..S-5 A/B bayraklari (verilmezse mevcut davranis)
    p.add_argument("--sr-cluster-tol", type=float, default=None,
                   help="S-1/S-2: H4+D1 kumeleme toleransi (oran; mevcut 0.02; test 0.012 / 0.010)")
    p.add_argument("--sr-impulse-4h", type=float, default=None,
                   help="S-3: 4H pivot itki-mumu esigi — pivot..pivot+2 icinde tek-mum range >= oran sart (or. 0.02)")
    p.add_argument("--sr-impulse-1d", type=float, default=None,
                   help="S-3: 1D pivot itki-mumu esigi (or. 0.05; gunluk mumlar iri)")
    p.add_argument("--sr-edge-role", action="store_true",
                   help="S-4: SUP/RES rolu zone KENARINA gore (merkez-flip hatasi duzeltmesi; zone icinde sh/sl karar verir)")
    p.add_argument("--prox-inside-tier", type=int, default=None,
                   help="S-5: 'zone icinde/dibinde' (d<=0.3%%) proximity puani (mevcut 15; test 12)")
    args = p.parse_args(argv)
    # G-A / G-B module-flag overrides (so the matrix can A/B color & 1D from the CLI)
    from modules import price_action as _pa
    from . import scorer as _sc
    if args.color_mode == "off":
        _pa.SR_COLOR_FILTER = False
    else:
        _pa.SR_COLOR_FILTER = True
        _pa.SR_COLOR_MODE = args.color_mode
    if args.no_1d_gate:
        _sc.TREND_GATE_USE_1D = False
    _pa.SR_MIN_SEPARATION_PCT = args.sr_min_sep
    _pa.SR_MAJOR_ONLY = args.sr_major_only
    _pa.SR_MAX_PER_SIDE = args.sr_max_per_side
    if args.entry_max_ext is not None:        # None -> modul default (0.15) korunur
        _sc.ENTRY_MAX_EXT_PCT = args.entry_max_ext
    _sc.ENTRY_RETEST = args.entry_retest
    _sc.ENTRY_RETEST_MAX_PCT = args.entry_retest_max
    # G6.2 / G6.3 / G6.5 — once AC bayraklari (default'lar OFF), sonra KAPAT bayraklari
    if args.pd_filter:
        _sc.PD_FILTER = True
    if args.ltf_trigger:
        _sc.LTF_TRIGGER = True
    if args.va_tp:
        _sc.VA_TP_POOL = True
    if args.no_pd_filter:
        _sc.PD_FILTER = False
    if args.pd_premium is not None:
        _sc.PD_PREMIUM = args.pd_premium
    if args.pd_discount is not None:
        _sc.PD_DISCOUNT = args.pd_discount
    if args.no_ltf_trigger:
        _sc.LTF_TRIGGER = False
    if args.no_va_tp:
        _sc.VA_TP_POOL = False
    # SL bandi + long-only (score() cagrisi engine'den parametresiz gelir -> modul default'unu ez)
    if args.max_sl is not None:
        _sc.DEFAULT_MAX_SL = args.max_sl
    if args.min_sl is not None:
        _sc.DEFAULT_MIN_SL = args.min_sl
    if args.long_only:
        _sc.LONG_ONLY = True
    # S/R kampanyasi — modul bayragi ezmeleri
    if args.sr_cluster_tol is not None:
        _pa.H4_TOLERANCE_PCT = args.sr_cluster_tol
        _pa.D1_TOLERANCE_PCT = args.sr_cluster_tol
    if args.sr_impulse_4h is not None:
        _pa.SR_IMPULSE_H4_PCT = args.sr_impulse_4h
    if args.sr_impulse_1d is not None:
        _pa.SR_IMPULSE_D1_PCT = args.sr_impulse_1d
    if args.sr_edge_role:
        _pa.SR_EDGE_ROLE = True
    if args.prox_inside_tier is not None:
        _sc.PROX_INSIDE_TIER = args.prox_inside_tier
    print(f"[cfg-SR] cluster_tol={_pa.H4_TOLERANCE_PCT:g} impulse_4h={_pa.SR_IMPULSE_H4_PCT:g} "
          f"impulse_1d={_pa.SR_IMPULSE_D1_PCT:g} edge_role={'ON' if _pa.SR_EDGE_ROLE else 'OFF'} "
          f"prox_inside_tier={_sc.PROX_INSIDE_TIER}")
    print(f"[cfg] color_mode={args.color_mode} 1d_gate={'OFF' if args.no_1d_gate else 'ON'} "
          f"sr_min_sep={args.sr_min_sep} sr_major_only={args.sr_major_only} sr_max={args.sr_max_per_side} "
          f"pd_filter={'ON' if _sc.PD_FILTER else 'OFF'} ltf_trigger={'ON' if _sc.LTF_TRIGGER else 'OFF'} "
          f"va_tp={'ON' if _sc.VA_TP_POOL else 'OFF'} sl_band={_sc.DEFAULT_MIN_SL:g}-{_sc.DEFAULT_MAX_SL:g} "
          f"long_only={'ON' if _sc.LONG_ONLY else 'OFF'}")
    _cap_map = {}
    if args.tp_cap_map:
        for _part in args.tp_cap_map.split(","):
            _k, _v = _part.split(":")
            _cap_map[_k.strip().upper()] = float(_v)
    _ladder = None
    if args.ladder:
        _ladder = [(float(_p), float(_f)) for _p, _f in (x.split(":") for x in args.ladder.split(","))]
        _ssum = sum(_f for _, _f in _ladder)
        if abs(_ssum - 1.0) > 1e-6:
            print(f"[ladder] UYARI: oranlar toplamı {_ssum:.2f} (1.0 değil) — kalan TP'de kapanır/eksik kalır")
    args.symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    # tp-cap-map anahtar normalizasyonu: motor TAM sembol adiyla arar (BTCUSD),
    # "BTC:3.5" gibi kisa anahtarlar sessizce ES geciliyordu (d6'da ETH 3.0 tavani
    # hic uygulanmadi — 72/141 ETH islemi rr>3.0). Kisa anahtari tam ada genislet.
    if _cap_map:
        for _sym in args.symbols:
            if _sym not in _cap_map:
                for _k, _v in list(_cap_map.items()):
                    if _sym.upper().startswith(_k):
                        _cap_map[_sym] = _v
                        break
        print(f"[cfg-TP] cap_map={_cap_map}")
    if args.no_stage1:
        args.exit_mode = "raw"
    enable_reversal = not args.no_reversal
    if not enable_reversal:
        print("[reversal] KAPALI — sadece trend-devamı (de-redundant) skoru")

    # DUZEN: kosu kunyesi — tam komut satiri + veri damgasi cikti klasorune yazilir,
    # boylece her klasor neyle kosuldugunu KENDISI belgeler. Damgasi farkli iki
    # kosu kiyaslanamaz (taze veri = yeni pencere = yeni referans sart).
    from datetime import datetime, timezone
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "run_config.txt"), "w", encoding="utf-8") as _f:
        _f.write("cmd: " + " ".join(sys.argv) + "\n")
        _f.write(f"run_at: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC\n")
        _stamp_src = os.path.join(args.csv_dir, "DATA_VERSION.txt") if args.csv_dir else None
        if _stamp_src and os.path.exists(_stamp_src):
            _f.write("--- DATA_VERSION ---\n")
            with open(_stamp_src, encoding="utf-8") as _s:
                _f.write(_s.read())
        elif args.synthetic:
            _f.write("data: SYNTHETIC\n")
        else:
            _f.write("data: DATA_VERSION.txt YOK (damgasiz eski cekim) — kiyaslarken dikkat\n")

    feeds = build_feeds(args)
    if not feeds:
        print("Veri yok — çıkılıyor."); sys.exit(1)

    whale_provider = None
    if args.whale_cache:
        from . import whale_history
        whale_provider = whale_history.load_provider(args.whale_cache)
        if args.threshold == 49.0:          # A/B/C default → tam-sistem 65'e çıkar
            args.threshold = 65.0
            print("[whale] aktif → eşik 65'e çıkarıldı (tam 100 puan ölçeği)")
        else:
            print(f"[whale] aktif → eşik {args.threshold}")

    if args.sweep:
        lo, hi, step = args.sweep
        print(f"\n=== EŞİK TARAMASI {lo}→{hi} adım {step} ===")
        thr = lo
        while thr <= hi + 1e-9:
            trades = engine.run(feeds, threshold=thr, whale_provider=whale_provider,
                                warmup=args.warmup, fill_window=args.fill_window,
                                max_hold=args.max_hold, eval_every=args.eval_every,
                                exit_mode=args.exit_mode, n_parts=args.n_parts,
                                trail=args.trail, cost_pct=args.cost_pct, max_rr=args.max_rr,
                                enable_reversal=enable_reversal,
                                short_exh_veto=args.short_exh_veto,
                                long_exh_veto=args.long_exh_veto, tp_cap_rr=args.tp_cap_rr, tp_cap_map=_cap_map, regime_filter=args.regime_filter,
                                regime_adx_min=args.regime_adx_min, regime_tf=args.regime_tf,
                                require_confirmation=args.require_confirmation, be_at_R=args.be_at_r,
                                half_close=args.half_close, ladder_spec=_ladder,
                                guard_enabled=not args.no_guard,
                                guard_cooldown_bars=args.guard_cooldown_bars,
                                guard_consec_limit=args.guard_consec_limit,
                                guard_halt_bars=args.guard_halt_bars,
                                guard_relevel_pct=args.guard_relevel_pct,
                                out_dir=os.path.join(args.out_dir, f"thr_{thr:g}"))
            n = len(trades)
            if n:
                tot = sum(t["R"] for t in trades)
                wr = sum(1 for t in trades if t["R"] >= 0) / n * 100
                print(f"  eşik {thr:5g}: {n:4d} işlem | WR {wr:5.1f}% | toplam {tot:+7.2f}R | "
                      f"beklenen {tot/n:+.3f}R")
            else:
                print(f"  eşik {thr:5g}:    0 işlem")
            thr += step
    else:
        engine.run(feeds, threshold=args.threshold, whale_provider=whale_provider,
                   warmup=args.warmup, fill_window=args.fill_window,
                   max_hold=args.max_hold, eval_every=args.eval_every,
                   exit_mode=args.exit_mode, n_parts=args.n_parts,
                   trail=args.trail, cost_pct=args.cost_pct, max_rr=args.max_rr,
                   enable_reversal=enable_reversal, short_exh_veto=args.short_exh_veto,
                   long_exh_veto=args.long_exh_veto, tp_cap_rr=args.tp_cap_rr, tp_cap_map=_cap_map, regime_filter=args.regime_filter,
                   regime_adx_min=args.regime_adx_min, regime_tf=args.regime_tf,
                   require_confirmation=args.require_confirmation, be_at_R=args.be_at_r,
                   half_close=args.half_close, ladder_spec=_ladder,
                   guard_enabled=not args.no_guard,
                   guard_cooldown_bars=args.guard_cooldown_bars,
                   guard_consec_limit=args.guard_consec_limit,
                   guard_halt_bars=args.guard_halt_bars,
                   guard_relevel_pct=args.guard_relevel_pct,
                   out_dir=args.out_dir)


if __name__ == "__main__":
    main()
