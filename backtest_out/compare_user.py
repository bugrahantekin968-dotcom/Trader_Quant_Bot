# -*- coding: utf-8 -*-
"""Kullanıcının koştuğu varyantların CSV'lerini analiz eder (backtest KOŞMAZ, sadece okur)."""
import csv, os, statistics as st
HERE = os.path.dirname(os.path.abspath(__file__))
V = [
    ("v0_base  (baseline halfway_be)",        "v0_base"),
    ("v1_teyit (+giriş teyidi)",              "v1_teyit"),
    ("v2_teyit_adx (+ADX rejim)",             "v2_teyit_adx"),
    ("v3a_protA (+BE0.75 +trail)",            "v3a_protA"),
    ("v3b_btc_eth (+BE0.75, BTC+ETH)",        "v3b_btc_eth"),
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

print(f"{'Varyant':34}{'n':>5}{'WR%':>6}{'netR':>8}{'bekl':>7}{'std':>6}{'TP/BE/SL/TO':>15}{'anlk-ters%':>11}{'kazanORT':>9}{'kayıpORT':>9}")
print("-"*124)
rowsmap={}
for label,d in V:
    rows=load(os.path.join(HERE,d,"backtest_results.csv"))
    if not rows: print(f"{label:34} (yok: {d})"); continue
    rowsmap[d]=rows
    n=len(rows); Rs=[x["R"] for x in rows]; net=sum(Rs)
    wins=[x for x in rows if x["R"]>0]; losses=[x for x in rows if x["R"]<=0]
    wr=100*len(wins)/n; exp=net/n; std=st.pstdev(Rs) if n>1 else 0
    oc={k:sum(1 for x in rows if x["outcome"]==k) for k in ("TP","BE","SL","TIMEOUT")}
    imm=[x for x in rows if x["mfe_R"]<=0.15 and x["R"]<0]; immp=100*len(imm)/n
    wavg=sum(x["R"] for x in wins)/len(wins) if wins else 0
    lavg=sum(x["R"] for x in losses)/len(losses) if losses else 0
    ocs=f"{oc['TP']}/{oc['BE']}/{oc['SL']}/{oc['TIMEOUT']}"
    print(f"{label:34}{n:5d}{wr:6.1f}{net:+8.1f}{exp:+7.3f}{std:6.2f}{ocs:>15}{immp:10.1f}%{wavg:+9.2f}{lavg:+9.2f}")

print("-"*124)
# Derin: teyit anlık-ters'i düşürdü mü? + BE kapanışlarda masada bırakılan
def detail(d):
    rows=rowsmap.get(d)
    if not rows: return
    be=[x for x in rows if x["outcome"]=="BE"]
    tp=[x for x in rows if x["outcome"]=="TP"]
    up=[x for x in rows if x["mfe_R"]>=0.75 and x["R"]<0]
    be_mfe=sum(x["mfe_R"] for x in be)/len(be) if be else 0
    be_r=sum(x["R"] for x in be)/len(be) if be else 0
    print(f"  [{d}] TP={len(tp)} (ortR {sum(x['R'] for x in tp)/len(tp) if tp else 0:+.2f}) | "
          f"BE={len(be)} (ortMFE {be_mfe:+.2f} -> gerçekleşen {be_r:+.2f} = MASADA {be_mfe-be_r:+.2f}R) | "
          f">=.75R görüp kaybeden: {len(up)} ({sum(x['R'] for x in up):+.1f}R)")
for _,d in V: detail(d)
