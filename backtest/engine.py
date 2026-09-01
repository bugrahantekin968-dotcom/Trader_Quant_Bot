"""
backtest/engine.py
══════════════════════════════════════════════════════════════════════
Walk-forward replay + trade simulation. $0 cost (no LLM), deterministic.

For each symbol it steps T over 1h bars. At each closed 1h bar it builds a
LEAK-FREE multi-TF snapshot (DataFeed.slice_at — only bars closed by T),
runs the real TechnicalAnalyzer, then the deterministic scorer. On a
LONG/SHORT it simulates a limit-order fill + forward SL/TP walk and records
the full reasoning + outcome to systp.txt (winners) / syssl.txt (losers)
plus a flat CSV for per-factor analysis.

One position per symbol at a time (mirrors the live order-stacking guard):
after a trade closes, scanning resumes from the bar AFTER its exit.

Trade model
    • Entry = limit at current×0.998 (long) / ×1.002 (short). Filled if price
      touches it within `fill_window` bars, else cancelled (NOFILL).
    • Exit = first of SL / TP (SL wins same-bar ties — conservative) within
      `max_hold` bars, else TIMEOUT at last close.
    • Records MAE/MFE (in R) and whether the halfway-to-TP point was reached
      (so Stage-1 partial+BE management can be evaluated in analysis WITHOUT
      baking it into the sim — keeps the raw signal-quality stat clean).
"""
from __future__ import annotations
import csv, os
import pandas as pd
from . import scorer

try:                                   # works both in their repo and here
    from modules.price_action import TechnicalAnalyzer
except Exception:                      # fallback: flat layout (rev3/)
    from price_action import TechnicalAnalyzer


_SUBS = ["macro_trend", "momentum_rsi", "momentum_ema", "momentum_macd",
         "sr_proximity", "sr_quality", "structural_events",
         "ls_ratio", "funding_rate", "oi_behavior"]


def _simulate(feed, t_signal, side, plan, fill_window, max_hold,
              exit_mode="halfway_be", n_parts=10, trail=False, cost_pct=0.0, be_at_R=0.0, half_close=0.5,
              ladder_spec=None):
    """
    Unified ladder-based exit simulator. R = realized P&L / initial risk.

    exit_mode:
      "raw"        full position to first of SL/TP (pure signal quality)
      "halfway_be" live Stage-1: 50% at halfway-to-TP, move stop to BE, runner to TP
      "scaleout"   sell 1/n_parts at each i/n_parts of the way to TP.
                   trail=True → move stop to BE once ≥50% is sold.
      "trail"      full position; once +1R in profit, trail the stop 1R behind
                   the running high (locks gains progressively), exit at TP or trail.

    cost_pct: per-fill transaction cost as a fraction of price (e.g. 0.0003 =
              0.03% spread/commission). Charged on the entry and every (partial)
              exit, converted to R via cost_pct / sl_pct. 0.0 = frictionless.

    Intrabar convention: adverse (stop) checked first — SL wins same-bar ties.
    """
    entry, sl, tp = plan["entry"], plan["sl"], plan["tp"]
    n = len(feed)
    fill_idx = None
    for f in range(t_signal + 1, min(t_signal + 1 + fill_window, n)):
        _, h, l, _ = feed.bar(f)
        if side == "LONG" and l <= entry:  fill_idx = f; break
        if side == "SHORT" and h >= entry: fill_idx = f; break
    if fill_idx is None:
        return None                                   # NOFILL — cancelled

    risk = abs(entry - sl)
    if risk <= 0:
        return None
    sl_pct = risk / entry
    cost_R = (cost_pct / sl_pct) if (cost_pct and sl_pct > 0) else 0.0

    def px(progress):                                 # 0..1 from entry toward TP
        return entry + progress * (tp - entry) if side == "LONG" else entry - progress * (entry - tp)

    half_px = px(0.5)
    if exit_mode == "raw" or exit_mode == "trail":
        ladder = [(tp, 1.0)]
    elif exit_mode == "halfway_be":
        _hc = min(max(half_close, 0.01), 1.0)                    # yarı-yolda kapatılacak oran
        ladder = [(half_px, _hc)] if _hc >= 1.0 - 1e-9 else [(half_px, _hc), (tp, 1.0 - _hc)]
    elif exit_mode == "scaleout":
        n_parts = max(1, int(n_parts)); frac = 1.0 / n_parts
        ladder = [(px(i / n_parts), frac) for i in range(1, n_parts + 1)]
    elif exit_mode == "ladder" and ladder_spec:
        ladder = [(px(p), fr) for (p, fr) in ladder_spec]          # özel merdiven: (ilerleme 0..1, oran)
    else:
        ladder = [(tp, 1.0)]

    remaining = 1.0; stop = sl; realized = 0.0
    mae_R = 0.0; mfe_R = 0.0; reached_half = False
    n_fills = 1; rung = 0; cum = 0.0
    exit_idx = exit_price = outcome = None

    for k in range(fill_idx + 1, min(fill_idx + 1 + max_hold, n)):
        _, h, l, c = feed.bar(k)
        if side == "LONG":
            mfe_R = max(mfe_R, (h - entry) / risk)
            mae_R = min(mae_R, (l - entry) / risk)
            if h >= half_px: reached_half = True
            if be_at_R and mfe_R >= be_at_R:                        # +be_at_R → SL'i basabasa cek
                stop = max(stop, entry)
            if exit_mode == "trail" and (h - entry) >= risk:        # +1R → trail 1R behind high
                stop = max(stop, h - risk)
            if l <= stop:                                           # adverse first
                realized += remaining * ((stop - entry) / risk); n_fills += 1
                outcome = "TP" if stop >= tp - 1e-9 else ("BE" if stop >= entry - 1e-9 else "SL")
                exit_idx, exit_price = k, stop; break
            while rung < len(ladder) and h >= ladder[rung][0]:      # favorable fills
                pr, fr = ladder[rung]
                realized += fr * ((pr - entry) / risk); n_fills += 1
                cum += fr; remaining -= fr; rung += 1
                if cum >= 0.5 and (exit_mode in ("halfway_be", "ladder") or (exit_mode == "scaleout" and trail)):
                    stop = entry                                    # move to BE
            if remaining <= 1e-9:
                exit_idx, exit_price, outcome = k, tp, "TP"; break
        else:  # SHORT
            mfe_R = max(mfe_R, (entry - l) / risk)
            mae_R = min(mae_R, (entry - h) / risk)
            if l <= half_px: reached_half = True
            if be_at_R and mfe_R >= be_at_R:                        # +be_at_R → SL'i basabasa cek
                stop = min(stop, entry)
            if exit_mode == "trail" and (entry - l) >= risk:
                stop = min(stop, l + risk)
            if h >= stop:
                realized += remaining * ((entry - stop) / risk); n_fills += 1
                outcome = "TP" if stop <= tp + 1e-9 else ("BE" if stop <= entry + 1e-9 else "SL")
                exit_idx, exit_price = k, stop; break
            while rung < len(ladder) and l <= ladder[rung][0]:
                pr, fr = ladder[rung]
                realized += fr * ((entry - pr) / risk); n_fills += 1
                cum += fr; remaining -= fr; rung += 1
                if cum >= 0.5 and (exit_mode in ("halfway_be", "ladder") or (exit_mode == "scaleout" and trail)):
                    stop = entry
            if remaining <= 1e-9:
                exit_idx, exit_price, outcome = k, tp, "TP"; break

    if exit_idx is None:                                            # TIMEOUT
        exit_idx = min(fill_idx + max_hold, n - 1)
        _, _, _, cpx = feed.bar(exit_idx)
        exit_price = cpx
        realized += remaining * (((cpx - entry) if side == "LONG" else (entry - cpx)) / risk)
        n_fills += 1; outcome = "TIMEOUT"

    # Transaction cost is VOLUME-proportional (spread/%commission): entry crosses
    # the spread on the full position, all exits sum to the full position too, so a
    # round trip costs exactly 2×cost_R regardless of how many partials — splitting
    # into N parts does NOT increase spread cost. (n_fills is reported separately
    # for anyone whose broker also charges a flat per-order fee.)
    realized -= 2.0 * cost_R

    return {
        "fill_index": fill_idx, "exit_index": exit_idx,
        "fill_time": feed.timestamp_at(fill_idx), "exit_time": feed.timestamp_at(exit_idx),
        "exit_price": round(exit_price, 6), "outcome": outcome,
        "R": round(realized, 3), "mae_R": round(mae_R, 3), "mfe_R": round(mfe_R, 3),
        "reached_half": reached_half, "bars_held": exit_idx - fill_idx, "n_fills": n_fills,
    }


def _adx(df, period=14):
    """Wilder ADX (trend gucu). Yetersiz bar -> None. Chop tespiti icin."""
    if df is None or len(df) < period * 2 + 2:
        return None
    h, l, c = df["high"], df["low"], df["close"]
    up, dn = h.diff(), -l.diff()
    plus_dm  = up.where((up > dn) & (up > 0), 0.0)
    minus_dm = dn.where((dn > up) & (dn > 0), 0.0)
    tr  = pd.concat([(h - l), (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0 / period, adjust=False).mean().replace(0, 1e-9)
    pdi = 100.0 * (plus_dm.ewm(alpha=1.0 / period, adjust=False).mean() / atr)
    mdi = 100.0 * (minus_dm.ewm(alpha=1.0 / period, adjust=False).mean() / atr)
    dx  = 100.0 * (pdi - mdi).abs() / (pdi + mdi).replace(0, 1e-9)
    val = dx.ewm(alpha=1.0 / period, adjust=False).mean().iloc[-1]
    return float(val) if pd.notna(val) else None


# ──────────────────────────────────────────────────────────────────────
# G3 — re-entry guard (port of the LIVE main.py guard into backtest-time).
# Bars are 1h, so seconds→bars = sec/3600. Mirrors AlgoBot._guard_blocks /
# _register_outcome EXACTLY: a STOP arms a side cooldown (level-lockout inside it)
# + consecutive-stop counter; N consecutive stops halt the side; a WIN clears it.
# ──────────────────────────────────────────────────────────────────────
def _guard_block(gstate, side, price, bar, relevel_pct):
    """Engellenmeli mi? halt -> kosulsuz; cooldown -> sadece ayni-yon stop seviyesi
    bandinda (canlidaki ile birebir)."""
    if bar < gstate["halt"].get(side, -1):
        return True
    if bar < gstate["cooldown"].get(side, -1) and price > 0:
        for lv in gstate["levels"]:
            if lv["side"] == side and abs(price - lv["price"]) / price <= relevel_pct:
                return True
    return False


def _guard_register(gstate, side, R, entry_price, exit_bar,
                    consec_limit, cooldown_bars, halt_bars):
    """Islem kapaninca guard state'i guncelle. R<0 = STOP (cooldown+sayac),
    R>0 = WIN (o yonu temizle)."""
    if R < -1e-9:
        gstate["cooldown"][side] = exit_bar + cooldown_bars
        gstate["consec"][side] = gstate["consec"].get(side, 0) + 1
        if entry_price and entry_price > 0:
            gstate["levels"].append({"price": float(entry_price), "side": side, "bar": exit_bar})
            gstate["levels"] = gstate["levels"][-12:]
        if gstate["consec"][side] >= consec_limit:
            gstate["halt"][side] = exit_bar + halt_bars
            gstate["consec"][side] = 0
    elif R > 1e-9:
        gstate["consec"][side] = 0
        gstate["halt"][side] = -1
        gstate["cooldown"][side] = -1


def run(feeds, threshold=49.0, whale_provider=None, warmup=300,
        fill_window=4, max_hold=240, eval_every=4,
        exit_mode="halfway_be", n_parts=10, trail=False, cost_pct=0.0,
        be_at_R=0.0, require_confirmation=False, half_close=0.5, ladder_spec=None,
        max_rr=None, enable_reversal=True, short_exh_veto=0.0,
        long_exh_veto=0.0, tp_cap_rr=3.5, tp_cap_map=None, regime_filter=False, regime_adx_min=20.0, regime_tf="4h",
        guard_enabled=True, guard_cooldown_bars=6, guard_consec_limit=3,
        guard_halt_bars=12, guard_relevel_pct=0.012, out_dir="."):
    """
    eval_every : evaluate the score every N 1h bars. 1 = fully faithful to the
                 live cadence but slow; 4 = once per 4h close, ~4x faster.
                 SL/TP is always simulated at 1h granularity.
    exit_mode  : raw | halfway_be (live Stage-1) | scaleout | trail  (see _simulate)
    n_parts    : number of scale-out chunks for exit_mode="scaleout"
    trail      : scaleout → move stop to BE once ≥50% sold
    cost_pct   : per-fill transaction cost as fraction of price (0.0003 = 0.03%)
    """
    analyzer = TechnicalAnalyzer(swing_order=5)
    trades = []

    for symbol, feed in feeds.items():
        _sym_cap = (tp_cap_map or {}).get(symbol, tp_cap_rr)   # per-symbol TP tavani
        gstate = {"cooldown": {}, "consec": {}, "halt": {}, "levels": []}   # G3 re-entry guard (per-symbol)
        n = len(feed); t = max(warmup, 1)
        while t < n - 1:
            ohlcv = feed.slice_at(t)
            if (len(ohlcv["1d"]) < 30 or len(ohlcv["4h"]) < 40 or len(ohlcv["1h"]) < 60):
                t += eval_every; continue
            price = feed.price_at(t)
            analysis = analyzer.analyze_all(ohlcv, price, min_touches=3)
            whale = whale_provider(symbol, feed.timestamp_at(t)) if whale_provider else None
            res = scorer.score(analysis, price, whale=whale, threshold=threshold,
                               enable_reversal=enable_reversal, short_exh_veto=short_exh_veto,
                               long_exh_veto=long_exh_veto, max_rr=_sym_cap,
                               require_confirmation=require_confirmation)

            if res["decision"] in ("LONG", "SHORT") and res["plan"]:
                # G3 — re-entry guard: stop sonrasi ayni-yon/seviye cooldown, ardisik-stop askisi.
                if guard_enabled and _guard_block(gstate, res["decision"], price, t, guard_relevel_pct):
                    t += eval_every; continue
                _adx_val = _adx(ohlcv.get(regime_tf), 14)
                if regime_filter and _adx_val is not None and _adx_val < regime_adx_min:
                    t += eval_every; continue   # rejim filtresi: chop (ADX dusuk) -> atla
                if max_rr and res["plan"]["rr"] > max_rr:        # uzak TP'leri cap'le
                    pl = res["plan"]; e = pl["entry"]; rk = abs(e - pl["sl"])
                    pl["tp"] = e + max_rr * rk if res["decision"] == "LONG" else e - max_rr * rk
                    pl["rr"] = round(max_rr, 3)
                sim = _simulate(feed, t, res["decision"], res["plan"], fill_window, max_hold,
                                exit_mode=exit_mode, n_parts=n_parts, trail=trail, cost_pct=cost_pct,
                                be_at_R=be_at_R, half_close=half_close, ladder_spec=ladder_spec)
                if sim:
                    side = res["decision"]
                    # --- R&D girdi-bağlamı: extension (4H EMA200'e uzaklık) + alt-TF
                    #     dönüş sinyali (mevcut exhaustion_score'u giriş anında çağırıyoruz;
                    #     short için yüksek = düşüş trendi tükeniyor = "bounce'a short" riski) ---
                    _ind4h = (analysis.get("4h") or {}).get("indicators", {}) or {}
                    _ema200 = _ind4h.get("ema_200")
                    _ext = round((price - _ema200) / _ema200, 4) if _ema200 else None
                    _exh = scorer.exhaustion_score(analysis, side.lower())
                    _ltf = round(float(_exh.get("total", 0.0)), 1)
                    _ltf_hard = bool(_exh.get("hard_override", False))
                    bd = res["long_breakdown"] if side == "LONG" else res["short_breakdown"]
                    sc = res["long_total"] if side == "LONG" else res["short_total"]
                    trades.append({
                        "symbol": symbol, "signal_time": feed.timestamp_at(t),
                        "side": side, "score": sc,
                        "playbook": res.get("playbook", "continuation"),
                        "nearest_level": res["nearest_level"], "nearest_label": res["nearest_label"],
                        "entry": res["plan"]["entry"], "sl": res["plan"]["sl"],
                        "tp": res["plan"]["tp"], "rr": res["plan"]["rr"],
                        "sl_pct": res["plan"]["sl_pct"], "position_usd": res["plan"]["position_usd"],
                        "breakdown": bd,
                        "ext_4h_ema200": _ext, "ltf_exh": _ltf, "ltf_hard": _ltf_hard, "adx": (round(_adx_val, 1) if _adx_val is not None else None),
                        **sim,
                    })
                    # G3 — bu sonuca gore guard state'i guncelle (STOP -> cooldown/sayac, WIN -> temizle)
                    if guard_enabled:
                        _guard_register(gstate, side, sim["R"], res["plan"]["entry"],
                                        sim["exit_index"], guard_consec_limit,
                                        guard_cooldown_bars, guard_halt_bars)
                    t = sim["exit_index"] + 1
                    continue
            t += eval_every

    label = exit_mode
    if exit_mode == "scaleout":
        label += f" n={n_parts}" + (" +BE@50%" if trail else " (orijinal SL)")
    if exit_mode == "halfway_be" and abs(half_close - 0.5) > 1e-9:
        label += f" (yarı-yolda %{half_close*100:.0f} kapat)"
    if exit_mode == "ladder" and ladder_spec:
        label += " [" + ", ".join(f"%{fr*100:.0f}@yol{p*100:.0f}%" for p, fr in ladder_spec) + "]"
    if be_at_R:
        label += f"  |  BE@+{be_at_R:g}R"
    if require_confirmation:
        label += "  |  giriş-teyidi"
    if max_rr:
        label += f"  |  max RR={max_rr:g}"
    if cost_pct:
        label += f"  |  maliyet {cost_pct*100:.3f}%/taraf"
    _write_outputs(trades, out_dir, label)
    return trades


# ──────────────────────────────────────────────────────────────────────
# Output: systp.txt / syssl.txt (human-readable) + flat CSV + summary
# ──────────────────────────────────────────────────────────────────────
def _fmt_pct(x):
    return f"{x*100:+.1f}%" if isinstance(x, (int, float)) else "?"


def _reason_block(tr):
    bd = tr["breakdown"]
    parts = " | ".join(f"{k}={bd[k]}" for k in _SUBS if bd[k] != 0)
    pb = tr.get("playbook", "continuation")
    pb_tag = "  [REVERSAL playbook — 4H CHoCH]" if pb == "reversal" else ""
    return (
        f"────────────────────────────────────────────────────────\n"
        f"{tr['signal_time']}  {tr['symbol']}  {tr['side']}  (score {tr['score']}){pb_tag}\n"
        f"  Karar gerekçesi (puan dökümü): {parts}\n"
        f"  En yakın majör seviye / tetik: {tr['nearest_label']} @ {tr['nearest_level']}\n"
        f"  Plan: entry={tr['entry']}  SL={tr['sl']} ({tr['sl_pct']*100:.2f}%)  "
        f"TP={tr['tp']}  RR={tr['rr']}  pos=${tr['position_usd']:.0f}\n"
        f"  SONUÇ: {tr['outcome']}  |  R={tr['R']:+.2f}  |  tutuş={tr['bars_held']}h  "
        f"|  MFE={tr['mfe_R']:+.2f}R MAE={tr['mae_R']:+.2f}R  |  yarı-yola ulaştı={tr['reached_half']}\n"
        f"  Giriş-bağlamı (R&D): 4H EMA200'e uzaklık={_fmt_pct(tr.get('ext_4h_ema200'))}  |  "
        f"alt-TF dönüş (exhaustion)={tr.get('ltf_exh', '?')}/100"
        f"{' !!HARD-FLIP' if tr.get('ltf_hard') else ''}\n"
        f"  Giriş-Çıkış: {tr['fill_time']} -> {tr['exit_time']}\n"
    )


def _write_outputs(trades, out_dir, mode_label=""):
    os.makedirs(out_dir, exist_ok=True)
    tp_path  = os.path.join(out_dir, "systp.txt")
    sl_path  = os.path.join(out_dir, "syssl.txt")
    csv_path = os.path.join(out_dir, "backtest_results.csv")

    wins  = [t for t in trades if t["R"] >= 0]
    loses = [t for t in trades if t["R"] < 0]

    with open(tp_path, "w", encoding="utf-8") as f:
        f.write(f"# systp.txt — KÂRLA kapanan işlemler ({len(wins)} adet)\n")
        f.write("# Her blok: bot hangi puanlarla, nereyi destek/direnç görerek girdi + sonuç\n\n")
        for t in wins:
            f.write(_reason_block(t) + "\n")

    with open(sl_path, "w", encoding="utf-8") as f:
        f.write(f"# syssl.txt — ZARARLA kapanan işlemler ({len(loses)} adet)\n")
        f.write("# Her blok: bot hangi puanlarla, nereyi destek/direnç görerek girdi + sonuç\n\n")
        for t in loses:
            f.write(_reason_block(t) + "\n")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["signal_time", "symbol", "side", "playbook", "score"] + _SUBS +
                   ["nearest_level", "nearest_label", "entry", "sl", "tp", "rr",
                    "sl_pct", "outcome", "R", "mae_R", "mfe_R", "reached_half",
                    "bars_held", "n_fills", "fill_time", "exit_time",
                    "ext_4h_ema200", "ltf_exh", "ltf_hard", "adx"])
        for t in trades:
            bd = t["breakdown"]
            w.writerow([t["signal_time"], t["symbol"], t["side"],
                        t.get("playbook", "continuation"), t["score"]] +
                       [bd[k] for k in _SUBS] +
                       [t["nearest_level"], t["nearest_label"], t["entry"], t["sl"],
                        t["tp"], t["rr"], t["sl_pct"], t["outcome"], t["R"],
                        t["mae_R"], t["mfe_R"], t["reached_half"], t["bars_held"],
                        t.get("n_fills", ""), t["fill_time"], t["exit_time"],
                        t.get("ext_4h_ema200", ""), t.get("ltf_exh", ""),
                        t.get("ltf_hard", ""), t.get("adx", "")])

    _print_summary(trades, csv_path, tp_path, sl_path, mode_label)


def _print_summary(trades, csv_path, tp_path, sl_path, mode_label=""):
    n = len(trades)
    print("\n" + "=" * 64)
    print(f"  BACKTEST ÖZETİ — {n} işlem")
    if mode_label:
        print(f"  Çıkış modu       : {mode_label}")
    print("=" * 64)
    if n == 0:
        print("  Hiç işlem açılmadı (eşik çok yüksek ya da veri yetersiz).")
        return
    import statistics
    wins = [t for t in trades if t["R"] >= 0]
    tp_n = sum(1 for t in trades if t["outcome"] == "TP")
    sl_n = sum(1 for t in trades if t["outcome"] == "SL")
    be_n = sum(1 for t in trades if t["outcome"] == "BE")
    to_n = sum(1 for t in trades if t["outcome"] == "TIMEOUT")
    Rs = [t["R"] for t in trades]
    tot_R = sum(Rs); avg_R = tot_R / n
    std_R = statistics.pstdev(Rs) if n > 1 else 0.0
    wr = len(wins) / n * 100
    half = sum(1 for t in trades if t["reached_half"]) / n * 100
    avg_fills = sum(t.get("n_fills", 0) for t in trades) / n
    print(f"  Win rate (R>=0)  : {wr:5.1f}%   ({len(wins)}/{n})")
    print(f"  Sonuç dağılımı   : TP={tp_n}  BE={be_n}  SL={sl_n}  TIMEOUT={to_n}")
    print(f"  Toplam R         : {tot_R:+.2f}")
    print(f"  Beklenen değer   : {avg_R:+.3f} R / işlem")
    print(f"  Std (varyans)    : {std_R:.2f} R   <- prop firma DD riski için kritik")
    print(f"  Yarı-yola ulaşan : {half:.1f}%")
    if avg_fills:
        print(f"  Ort. fill/işlem  : {avg_fills:.1f}  (giriş + kısmi çıkışlar -> işlem maliyeti)")
    print("  ── Sembol bazında ──")
    for sym in sorted({t["symbol"] for t in trades}):
        s = [t for t in trades if t["symbol"] == sym]
        sR = sum(t["R"] for t in s)
        sw = sum(1 for t in s if t["R"] >= 0) / len(s) * 100
        print(f"    {sym:10s}: {len(s):3d} işlem | WR {sw:5.1f}% | toplam {sR:+.2f}R")
    pbs = sorted({t.get("playbook", "continuation") for t in trades})
    if "reversal" in pbs:
        print("  ── Playbook bazında ──")
        for pb in pbs:
            s = [t for t in trades if t.get("playbook", "continuation") == pb]
            sR = sum(t["R"] for t in s)
            sw = sum(1 for t in s if t["R"] >= 0) / len(s) * 100
            tag = "  ← DENEYSEL, doğrulanmamış" if pb == "reversal" else ""
            print(f"    {pb:12s}: {len(s):3d} işlem | WR {sw:5.1f}% | toplam {sR:+.2f}R{tag}")
    print(f"\n  Çıktılar:\n    {tp_path}\n    {sl_path}\n    {csv_path}")
    print("=" * 64)
