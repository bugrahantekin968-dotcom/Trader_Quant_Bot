# -*- coding: utf-8 -*-
"""Ladder deneyi karşılaştırması (CSV okur, backtest KOŞMAZ)."""
import csv, os, statistics as st
HERE = os.path.dirname(os.path.abspath(__file__))
V = [
    ("v0_base      BTC      halfway_be",  "v0_base"),
    ("ladder_btc   BTC      ladder75",    "ladder_btc"),
    ("v0_btceth    BTC+ETH  halfway_be",  "v0_btceth"),
    ("ladder_btceth BTC+ETH ladder75",    "ladder_btceth"),
]
def f(x,d=0.0):
    try: return float(x)
    except: return d
def b(x): return str(x).strip().lower() in ("true","1","yes")
def load(p):
    if not os.path.exists(p): return None
    out=[]
    with open(p,encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            for k in ("R","mfe_R","mae_R"): r[k]=f(r[k])
            r["reached_half"]=b(r["reached_half"]); out.append(r)
    return out
print(f"{'Varyant':36}{'n':>5}{'WR(R>0)%':>9}{'netR':>8}{'bekl':>7}{'std':>6}{'TP/BE/SL/TO':>15}{'kazanORT':>9}")
print("-"*100)
for label,d in V:
    rows=load(os.path.join(HERE,d,"backtest_results.csv"))
    if not rows: print(f"{label:36} (henüz yok: {d})"); continue
    n=len(rows); Rs=[x["R"] for x in rows]; net=sum(Rs)
    wins=[x for x in rows if x["R"]>0]
    wr=100*len(wins)/n; exp=net/n; std=st.pstdev(Rs) if n>1 else 0
    oc={k:sum(1 for x in rows if x["outcome"]==k) for k in ("TP","BE","SL","TIMEOUT")}
    wavg=sum(x["R"] for x in wins)/len(wins) if wins else 0
    ocs=f"{oc['TP']}/{oc['BE']}/{oc['SL']}/{oc['TIMEOUT']}"
    print(f"{label:36}{n:5d}{wr:9.1f}{net:+8.1f}{exp:+7.3f}{std:6.2f}{ocs:>15}{wavg:+9.2f}")
print("-"*100)
print("ladder75 = yolun %25inde %25 al, %50sinde %50 al (cum %75), kalan %25 TP; %50yi geçince SL->BE.")
