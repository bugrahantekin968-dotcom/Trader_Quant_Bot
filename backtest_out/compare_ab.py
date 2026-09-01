# -*- coding: utf-8 -*-
"""A/B karşılaştırma: her varyantın CSV'sini okuyup tek tabloda özetler.
Çalıştır: python backtest_out/compare_ab.py"""
import csv, os, statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
VARIANTS = [
    ("V0 baseline (mevcut)",                 "ab_v0_base"),
    ("V1 +teyit",                            "ab_v1_confirm"),
    ("V2 +teyit +ADX(rejim)",                "ab_v2_confreg"),
    ("V3a +teyit +ADX +trail +BE0.75",       "ab_v3a"),
    ("V3b +teyit +ADX +%50@yarı +BE0.75",    "ab_v3b"),
]

def f(x, d=0.0):
    try: return float(x)
    except: return d
def b(x): return str(x).strip().lower() in ("true","1","yes")

def load(path):
    rows=[]
    if not os.path.exists(path): return None
    with open(path, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            r["R"]=f(r["R"]); r["mfe_R"]=f(r["mfe_R"]); r["mae_R"]=f(r["mae_R"])
            r["reached_half"]=b(r["reached_half"])
            rows.append(r)
    return rows

print(f"{'Varyant':40} {'n':>4} {'WR%':>6} {'netR':>8} {'bekl':>7} {'std':>5} {'TP/BE/SL/TO':>14} {'anlk-ters%':>10} {'>=.75R kayıp':>13}")
print("-"*120)
base_net=None
for label, d in VARIANTS:
    rows = load(os.path.join(HERE, d, "backtest_results.csv"))
    if not rows:
        print(f"{label:40} (çıktı yok: {d})"); continue
    n=len(rows); Rs=[x["R"] for x in rows]; net=sum(Rs)
    wins=sum(1 for x in rows if x["R"]>0); wr=100*wins/n
    exp=net/n; std=st.pstdev(Rs) if n>1 else 0
    oc={k:sum(1 for x in rows if x["outcome"]==k) for k in ("TP","BE","SL","TIMEOUT")}
    imm=[x for x in rows if x["mfe_R"]<=0.15 and x["R"]<0]
    immp=100*len(imm)/n
    up=[x for x in rows if x["mfe_R"]>=0.75 and x["R"]<0]
    ocs=f"{oc['TP']}/{oc['BE']}/{oc['SL']}/{oc['TIMEOUT']}"
    if base_net is None: base_net=net
    print(f"{label:40} {n:4d} {wr:6.1f} {net:+8.1f} {exp:+7.3f} {std:5.2f} {ocs:>14} {immp:9.1f}% {len(up):3d}/{sum(x['R'] for x in up):+6.1f}R")

print("-"*120)
print("anlk-ters% = MFE<=+0.15R görüp zararla kapanan (yanlış-seviye girişi göstergesi) — DÜŞMESİ iyi.")
print(">=.75R kayıp = +0.75R görüp sonra kaybeden işlem sayısı/R — koruma boşluğu (BE0.75 bunu kapatmalı).")
