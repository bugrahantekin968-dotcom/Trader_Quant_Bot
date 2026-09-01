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
    p.add_argument("--exit-mode", choices=["raw", "halfway_be", "scaleout", "trail"],
                   default="halfway_be",
                   help="çıkış sistemi: raw | halfway_be (canlı Stage-1) | scaleout | trail")
    p.add_argument("--n-parts", type=int, default=10, help="scaleout için parça sayısı")
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
    p.add_argument("--sweep", nargs=3, type=float, metavar=("LO", "HI", "STEP"),
                   default=None, help="eşik taraması: LO HI STEP")
    args = p.parse_args(argv)
    _cap_map = {}
    if args.tp_cap_map:
        for _part in args.tp_cap_map.split(","):
            _k, _v = _part.split(":")
            _cap_map[_k.strip().upper()] = float(_v)
    args.symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    if args.no_stage1:
        args.exit_mode = "raw"
    enable_reversal = not args.no_reversal
    if not enable_reversal:
        print("[reversal] KAPALI — sadece trend-devamı (de-redundant) skoru")

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
                   guard_enabled=not args.no_guard,
                   guard_cooldown_bars=args.guard_cooldown_bars,
                   guard_consec_limit=args.guard_consec_limit,
                   guard_halt_bars=args.guard_halt_bars,
                   guard_relevel_pct=args.guard_relevel_pct,
                   out_dir=args.out_dir)


if __name__ == "__main__":
    main()
