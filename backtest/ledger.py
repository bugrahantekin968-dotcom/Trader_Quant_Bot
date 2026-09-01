"""
backtest/ledger.py
══════════════════════════════════════════════════════════════════════
Deney defteri: backtest_out/ altindaki TUM kosu klasorlerini tarar ve tek bir
LEDGER.md uretir — hangi kosu hangi veriyle yapildi, kac islem, kac R, hangi
cagda (era). Amac: "gecen sefer 54'tu simdi neden 30" karmasasini bitirmek.

KURAL: farkli veri damgasina (DATA_VERSION) sahip kosular ASLA kiyaslanamaz.
Ledger bunlari ayri bolumlerde gosterir.

Kullanim:
    python -m backtest.ledger              # backtest_out/LEDGER.md uret + ekrana bas
    python -m backtest.ledger --out-root backtest_out

Notlar: backtest_out/notes.json  {"klasor_adi": "kisa not/karar"} — tabloya eklenir.
"""
from __future__ import annotations
import argparse
import csv
import json
import os
import sys
from datetime import datetime

try:
    sys.stdout.reconfigure(encoding="utf-8")   # tr-Windows cp1254 kalkani
except Exception:
    pass


def _read_run(dirpath: str) -> dict | None:
    csv_path = os.path.join(dirpath, "backtest_results.csv")
    if not os.path.exists(csv_path):
        return None
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    info: dict = {
        "name": os.path.basename(dirpath),
        "mtime": datetime.fromtimestamp(os.path.getmtime(csv_path)),
        "n": len(rows),
    }
    if rows:
        R = [float(r["R"]) for r in rows]
        times = sorted(r["signal_time"] for r in rows)
        mid = times[len(times) // 2]
        info.update(
            tot=sum(R),
            exp=sum(R) / len(R),
            wr=sum(1 for x in R if x > 0) / len(R) * 100,
            btc=sum(float(r["R"]) for r in rows if r["symbol"] == "BTCUSD"),
            eth=sum(float(r["R"]) for r in rows if r["symbol"] == "ETHUSD"),
            h1=sum(float(r["R"]) for r in rows if r["signal_time"] < mid),
            h2=sum(float(r["R"]) for r in rows if r["signal_time"] >= mid),
            first=times[0][:10], last=times[-1][:10],
        )
    else:
        info.update(tot=0.0, exp=0.0, wr=0.0, btc=0.0, eth=0.0, h1=0.0, h2=0.0,
                    first="-", last="-")
    # kunye (run_config.txt): komut + veri damgasi
    info["cmd"] = ""
    info["era"] = "DAMGASIZ (eski cekim, ETH-tavan hatali donem olabilir)"
    rc = os.path.join(dirpath, "run_config.txt")
    if os.path.exists(rc):
        txt = open(rc, encoding="utf-8").read()
        for line in txt.splitlines():
            if line.startswith("cmd:"):
                info["cmd"] = line[4:].strip()
            if line.startswith("fetched:"):
                info["era"] = line.strip()
            if line.startswith("data: SYNTHETIC"):
                info["era"] = "SYNTHETIC"
    return info


def main(argv=None):
    p = argparse.ArgumentParser(description="backtest_out deney defteri uret")
    p.add_argument("--out-root", type=str, default="backtest_out")
    args = p.parse_args(argv)

    notes = {}
    notes_path = os.path.join(args.out_root, "notes.json")
    if os.path.exists(notes_path):
        notes = json.load(open(notes_path, encoding="utf-8"))

    runs = []
    for d in sorted(os.listdir(args.out_root)):
        full = os.path.join(args.out_root, d)
        if os.path.isdir(full):
            r = _read_run(full)
            if r:
                runs.append(r)
            # sweep alt klasorleri (thr_*)
            for sub in sorted(os.listdir(full)) if os.path.isdir(full) else []:
                sf = os.path.join(full, sub)
                if os.path.isdir(sf) and sub.startswith("thr_"):
                    r2 = _read_run(sf)
                    if r2:
                        r2["name"] = f"{d}/{sub}"
                        runs.append(r2)

    # caga gore grupla; cag icinde tarihe gore sirala
    eras: dict[str, list] = {}
    for r in runs:
        eras.setdefault(r["era"], []).append(r)
    for v in eras.values():
        v.sort(key=lambda x: x["mtime"])

    lines = []
    lines.append("# DENEY DEFTERI (otomatik: `python -m backtest.ledger`)\n")
    lines.append(f"_guncellendi: {datetime.now().strftime('%Y-%m-%d %H:%M')}_\n")
    lines.append("## Kurallar\n")
    lines.append("1. **Farkli veri damgali kosular KIYASLANAMAZ** — damga degisti mi once yeni s0 referansi kosulur.")
    lines.append("2. Tek kosu = tek degisken; kiyas her zaman ayni damgali referansa karsi.")
    lines.append("3. Toplam-R farki ~2 SE (≈±40R) altindaysa GURULTU; karar icin beklenti + iki-yari tutarliligi + (cikis katmani icin) eslesmis test.")
    lines.append("4. Kanit esigi: her iki YARIDA ve her iki SEMBOLDE ayni yonde fark.")
    lines.append("5. Karar verilen kosunun notu `notes.json`a yazilir; klasor silinmez.\n")

    for era in sorted(eras, key=lambda e: max(r["mtime"] for r in eras[e]), reverse=True):
        lines.append(f"## Veri cagi: {era}\n")
        lines.append("| kosu | tarih | islem | topR | bekl | WR% | BTC | ETH | yari1 | yari2 | pencere | not |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for r in eras[era]:
            note = notes.get(r["name"], "")
            lines.append(
                f"| {r['name']} | {r['mtime'].strftime('%m-%d %H:%M')} | {r['n']} "
                f"| {r['tot']:+.1f} | {r['exp']:+.3f} | {r['wr']:.0f} "
                f"| {r['btc']:+.1f} | {r['eth']:+.1f} | {r['h1']:+.1f} | {r['h2']:+.1f} "
                f"| {r['first']}→{r['last']} | {note} |")
        lines.append("")

    out = "\n".join(lines)
    md_path = os.path.join(args.out_root, "LEDGER.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(out)
    print(out)
    print(f"\n[ledger] {md_path} yazildi ({len(runs)} kosu)")


if __name__ == "__main__":
    main()
