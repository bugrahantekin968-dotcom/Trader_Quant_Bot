# -*- coding: utf-8 -*-
"""05_persym backtest çıktısı analizi — neden stop oluyoruz?
Yalnızca stdlib (csv). Çalıştır: python analyze_stops.py
"""
import csv, os, statistics as st
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "backtest_results.csv")

def f(x, d=0.0):
    try: return float(x)
    except: return d
def b(x): return str(x).strip().lower() in ("true","1","yes")

rows = []
with open(CSV, encoding="utf-8") as fh:
    for r in csv.DictReader(fh):
        r["R"]=f(r["R"]); r["mae_R"]=f(r["mae_R"]); r["mfe_R"]=f(r["mfe_R"])
        r["rr"]=f(r["rr"]); r["sl_pct"]=f(r["sl_pct"]); r["score"]=f(r["score"])
        r["sr_quality"]=f(r["sr_quality"]); r["sr_proximity"]=f(r["sr_proximity"])
        r["bars_held"]=f(r["bars_held"]); r["adx"]=f(r["adx"])
        r["ltf_exh"]=f(r["ltf_exh"]); r["ext"]=f(r.get("ext_4h_ema200"))
        r["reached_half"]=b(r["reached_half"]); r["ltf_hard"]=b(r["ltf_hard"])
        rows.append(r)

N=len(rows)
def s(rs): return sum(x["R"] for x in rs)
def wr(rs): return (100*sum(1 for x in rs if x["R"]>0)/len(rs)) if rs else 0
def avg(vals): return (sum(vals)/len(vals)) if vals else 0
def block(t): print("\n"+"="*64+"\n"+t+"\n"+"="*64)

netR=s(rows); wins=[x for x in rows if x["R"]>0]; losses=[x for x in rows if x["R"]<=0]
gp=sum(x["R"] for x in wins); gl=sum(x["R"] for x in losses)

block("1) GENEL")
print(f"İşlem: {N} | Net R: {netR:+.1f} | Beklenti(R/işlem): {netR/N:+.3f}")
print(f"Kazanan: {len(wins)} ({100*len(wins)/N:.1f}%)  Kaybeden/BE-altı: {len(losses)}")
print(f"Brüt kâr: {gp:+.1f}R | Brüt zarar: {gl:+.1f}R | Profit factor: {gp/abs(gl):.2f}")
print(f"Ort. kazanç: {avg([x['R'] for x in wins]):+.2f}R | Ort. kayıp: {avg([x['R'] for x in losses]):+.2f}R")

block("2) SONUÇ (outcome) DAĞILIMI")
oc=defaultdict(list)
for x in rows: oc[x["outcome"]].append(x)
for k in sorted(oc, key=lambda k:-len(oc[k])):
    g=oc[k]; print(f"{k:8} n={len(g):3} ({100*len(g)/N:4.1f}%)  netR={s(g):+7.1f}  ortR={avg([y['R'] for y in g]):+.2f}  ortMFE={avg([y['mfe_R'] for y in g]):+.2f}  ortMAE={avg([y['mae_R'] for y in g]):+.2f}")

block("3) YÖN (side)")
for side in ("LONG","SHORT"):
    g=[x for x in rows if x["side"]==side]
    print(f"{side:6} n={len(g):3}  winrate={wr(g):4.1f}%  netR={s(g):+7.1f}  ortR={avg([y['R'] for y in g]):+.2f}")

block("4) SEVİYE TİPİ (nearest_label) — S/R KALİTESİ")
lab=defaultdict(list)
for x in rows: lab[x["nearest_label"]].append(x)
print(f"{'label':16} {'n':>3} {'winrate':>8} {'netR':>8} {'ortR':>6} {'ortMFE':>7} {'ortMAE':>7} {'aninda_fail%':>12}")
for k in sorted(lab, key=lambda k:-len(lab[k])):
    g=lab[k]; imm=[x for x in g if x["mfe_R"]<=0.15 and x["R"]<0]
    print(f"{k:16} {len(g):3d} {wr(g):7.1f}% {s(g):+8.1f} {avg([y['R'] for y in g]):+6.2f} {avg([y['mfe_R'] for y in g]):+7.2f} {avg([y['mae_R'] for y in g]):+7.2f} {100*len(imm)/len(g):11.1f}%")

block("5) PLAYBOOK")
pb=defaultdict(list)
for x in rows: pb[x["playbook"]].append(x)
for k in sorted(pb, key=lambda k:-len(pb[k])):
    g=pb[k]; print(f"{k:14} n={len(g):3}  winrate={wr(g):4.1f}%  netR={s(g):+7.1f}  ortR={avg([y['R'] for y in g]):+.2f}")

block("6) 'KORUMA' ETKİSİ — yarı-yola ulaşıp ne oldu?")
rh=[x for x in rows if x["reached_half"]]
nrh=[x for x in rows if not x["reached_half"]]
print(f"Yarı-yola ULAŞTI: n={len(rh)} netR={s(rh):+.1f} winrate={wr(rh):.1f}% ortR={avg([y['R'] for y in rh]):+.2f} ortMFE={avg([y['mfe_R'] for y in rh]):+.2f}")
print(f"  -> bunların outcome dağılımı:", {k:sum(1 for x in rh if x['outcome']==k) for k in oc})
print(f"Yarı-yola ULAŞMADI: n={len(nrh)} netR={s(nrh):+.1f} winrate={wr(nrh):.1f}% ortR={avg([y['R'] for y in nrh]):+.2f}")
be=[x for x in rows if x["outcome"]=="BE"]
print(f"\nBE (başabaş/erken kilit) çıkışlar: n={len(be)} netR={s(be):+.1f} ortR={avg([y['R'] for y in be]):+.2f} ortMFE={avg([y['mfe_R'] for y in be]):+.2f}")
print(f"  -> BE işlemlerde MASADA BIRAKILAN ort: MFE {avg([y['mfe_R'] for y in be]):+.2f}R vs gerçekleşen {avg([y['R'] for y in be]):+.2f}R")

block("7) 'ÖNDEYKEN KAYBETTİK' — koruma boşluğu")
up1_lost=[x for x in rows if x["mfe_R"]>=1.0 and x["R"]<0]
up075_lost=[x for x in rows if x["mfe_R"]>=0.75 and x["R"]<0]
print(f"MFE>=+1.0R görüp sonra ZARARLA kapanan: n={len(up1_lost)}  topladıkları R={s(up1_lost):+.1f}  (ort MFE {avg([y['mfe_R'] for y in up1_lost]):+.2f} -> ort R {avg([y['R'] for y in up1_lost]):+.2f})")
print(f"MFE>=+0.75R görüp sonra ZARARLA kapanan: n={len(up075_lost)}  topladıkları R={s(up075_lost):+.1f}")

block("8) 'ANINDA TERS' — yanlış seviye sinyali (long'u düşen desteğe açma)")
imm=[x for x in rows if x["mfe_R"]<=0.15 and x["R"]<0]
print(f"MFE<=+0.15R (hiç lehe gitmeden) zararla kapanan: n={len(imm)} ({100*len(imm)/N:.1f}%)  netR={s(imm):+.1f}  ort tutuş={avg([y['bars_held'] for y in imm]):.0f}h")
side_imm=defaultdict(int)
for x in imm: side_imm[(x["side"],x["nearest_label"])]+=1
for k in sorted(side_imm, key=lambda k:-side_imm[k])[:10]:
    print(f"   {k[0]:5} @ {k[1]:16} : {side_imm[k]} adet")

block("9) SKOR EŞİĞİ İŞE YARIYOR MU?")
def bucket(v,edges):
    for e in edges:
        if v<e: return f"<{e}"
    return f">={edges[-1]}"
sb=defaultdict(list)
for x in rows: sb[bucket(x["score"],[55,60,65,70])].append(x)
for k in ["<55","<60","<65","<70",">=70"]:
    g=sb.get(k,[]);
    if g: print(f"score {k:5} n={len(g):3} winrate={wr(g):4.1f}% netR={s(g):+7.1f} ortR={avg([y['R'] for y in g]):+.2f}")

block("10) sr_quality / sr_proximity AYIRT EDİYOR MU?")
for col in ("sr_quality","sr_proximity"):
    qb=defaultdict(list)
    for x in rows: qb[x[col]].append(x)
    print(f"-- {col} --")
    for k in sorted(qb):
        g=qb[k]; print(f"   {col}={k:<5} n={len(g):3} winrate={wr(g):4.1f}% netR={s(g):+7.1f}")

block("11) ADX (trend gücü) / EMA200 uzaklığı / exhaustion / HARD-FLIP")
for lo,hi in [(0,20),(20,25),(25,30),(30,100)]:
    g=[x for x in rows if lo<=x["adx"]<hi]
    if g: print(f"ADX {lo:>2}-{hi:<3} n={len(g):3} winrate={wr(g):4.1f}% netR={s(g):+7.1f}")
print()
for lo,hi in [(-100,-5),(-5,-2),(-2,2),(2,5),(5,100)]:
    g=[x for x in rows if lo<=x["ext"]*100<hi]
    if g: print(f"EMA200 uzaklık {lo:>4}..{hi:<4}% n={len(g):3} winrate={wr(g):4.1f}% netR={s(g):+7.1f}")
print()
hard=[x for x in rows if x["ltf_hard"]]
print(f"ltf_hard (HARD-FLIP) işaretliler: n={len(hard)} winrate={wr(hard):.1f}% netR={s(hard):+.1f}  (bu sinyaller filtrelenirse netR {netR-s(hard):+.1f} olur)")
for lo,hi in [(0,10),(10,20),(20,30),(30,101)]:
    g=[x for x in rows if lo<=x["ltf_exh"]<hi]
    if g: print(f"   exhaustion {lo:>2}-{hi:<3} n={len(g):3} winrate={wr(g):4.1f}% netR={s(g):+7.1f}")

block("12) KARŞI-OLGU (what-if) — kabaca, slipaj yok")
# A) +1R'de başabaşa çek: mfe>=1.0 ise kayıpları ~0 yap
def whatif_be(thr):
    tot=0
    for x in rows:
        if x["R"]<0 and x["mfe_R"]>=thr: tot+=0.0     # BE'ye çekildi varsay
        else: tot+=x["R"]
    return tot
for thr in (1.0,0.75,0.5):
    print(f"+{thr:.2f}R'de SL->BE kuralı (mfe o seviyeyi gördüyse kayıp=0): tahmini netR = {whatif_be(thr):+.1f}  (şu an {netR:+.1f})")
# B) Zayıf seviyeleri / aşırı uzak girişleri ele
def excl(pred,name):
    g=[x for x in rows if not pred(x)]; print(f"{name}: kalan n={len(g)} netR={s(g):+.1f} winrate={wr(g):.1f}%")
excl(lambda x:x["nearest_label"].startswith("H4"), "H4 seviyelerini ELE (sadece D1/CHOCH kalır)")
excl(lambda x:x["score"]<60, "score<60'ı ELE")
excl(lambda x:abs(x["ext"])>0.05, "EMA200'den |%5|+ uzak girişleri ELE")
excl(lambda x:x["ltf_hard"], "HARD-FLIP'leri ELE")
excl(lambda x:(x["mfe_R"]<=0.15 and x["R"]<0), "[teorik tavan] anında-ters olanları ELE")

block("13) MAE PROFİLİ — stop'lar fazla mı dar?")
print(f"Kazananların MAE'si (en kötü gidişi): ort={avg([x['mae_R'] for x in wins]):+.2f}R  medyan={st.median([x['mae_R'] for x in wins]):+.2f}R  en kötü={min(x['mae_R'] for x in wins):+.2f}R")
print(f"Kaybedenlerin MAE'si: ort={avg([x['mae_R'] for x in losses]):+.2f}R")
deep=[x for x in wins if x["mae_R"]<=-0.8]
print(f"Kazananlardan MAE<=-0.8R olan (stop'a yakın gidip dönen): n={len(deep)} -> stop biraz daha genişse bunlar korunur, ama anında-ters olanlar yine gider")
print(f"sl_pct dağılımı: ort={avg([x['sl_pct'] for x in rows])*100:.2f}%  min={min(x['sl_pct'] for x in rows)*100:.2f}%  max={max(x['sl_pct'] for x in rows)*100:.2f}%")
