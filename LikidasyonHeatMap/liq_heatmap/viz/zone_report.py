"""
Zone report — primary user-facing output.
=========================================
Replaces the heatmap visualization. For each (coin, timeframe):
  1. Text report  → readable summary (printed + saved as .txt)
  2. JSON file    → machine-readable for bot consumption
  3. PNG/HTML viz → minimal bar chart, no time axis, no heatmap
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from core.zone_extractor import Zone, nearest_zone_above, nearest_zone_below

logger = logging.getLogger("zonereport")


# ============================================================
# Formatting helpers
# ============================================================
def _fmt_money(n: float) -> str:
    """Format dollars in human-readable form: $1.23B, $456M, $7.8M, $123k."""
    n = float(n)
    if n >= 1e9:  return f"${n/1e9:.2f}B"
    if n >= 1e6:  return f"${n/1e6:.1f}M"
    if n >= 1e3:  return f"${n/1e3:.0f}k"
    return f"${n:.0f}"


def _fmt_price(p: float, coin: str) -> str:
    """Format price with appropriate precision per coin."""
    if coin == "XRP":
        return f"${p:,.4f}"
    if coin == "ETH":
        return f"${p:,.2f}"
    return f"${p:,.0f}"


def _bar(dollars: float, max_dollars: float, width: int = 20) -> str:
    """ASCII bar of width `width` proportional to dollars / max_dollars."""
    if max_dollars <= 0:
        return ""
    filled = int(round(width * dollars / max_dollars))
    return "█" * max(filled, 1) + "░" * (width - max(filled, 1))


# ============================================================
# Text report
# ============================================================
def format_text_report(
    coin: str,
    tf: str,
    current_price: float,
    total_oi_usd: float,
    funding: float,
    long_ratio: float,
    zones: dict[str, list[Zone]],
) -> str:
    longs  = zones.get("long",  [])
    shorts = zones.get("short", [])
    max_dollars = max(
        [z.dollars for z in longs + shorts] or [1.0]
    )

    lines = []
    lines.append("═" * 70)
    lines.append(f"  {coin}  —  {tf} görünüm")
    lines.append(f"  Şu anki fiyat: {_fmt_price(current_price, coin)}"
                 f"    |    Toplam izlenen OI: {_fmt_money(total_oi_usd)}")
    lines.append(f"  Funding: {funding*100:+.4f}%    "
                 f"|    Long oranı: {long_ratio*100:.1f}%")
    lines.append("═" * 70)

    # LONG liquidations (below current — bearish magnets)
    lines.append("")
    lines.append("↓  LONG LİKİDASYONLARI  (fiyat aşağı çekilirse buralarda likide olur)")
    lines.append("")
    if not longs:
        lines.append("    (anlamlı bir bölge tespit edilmedi)")
    else:
        for z in longs:
            lines.append(
                f"    Bölge {z.rank}:  "
                f"{_fmt_price(z.price_low, coin)}  –  "
                f"{_fmt_price(z.price_high, coin)}   "
                f"→  {_fmt_money(z.dollars):>9}   "
                f"{_bar(z.dollars, max_dollars)}"
            )
            lines.append(
                f"              merkez {_fmt_price(z.price_center, coin)}  "
                f"|  {z.pct_from_spot:+.2f}% spot'tan uzak"
            )

    # SHORT liquidations (above current — bullish magnets)
    lines.append("")
    lines.append("↑  SHORT LİKİDASYONLARI  (fiyat yukarı çekilirse buralarda likide olur)")
    lines.append("")
    if not shorts:
        lines.append("    (anlamlı bir bölge tespit edilmedi)")
    else:
        for z in shorts:
            lines.append(
                f"    Bölge {z.rank}:  "
                f"{_fmt_price(z.price_low, coin)}  –  "
                f"{_fmt_price(z.price_high, coin)}   "
                f"→  {_fmt_money(z.dollars):>9}   "
                f"{_bar(z.dollars, max_dollars)}"
            )
            lines.append(
                f"              merkez {_fmt_price(z.price_center, coin)}  "
                f"|  {z.pct_from_spot:+.2f}% spot'tan uzak"
            )

    # Footer: nearest magnets
    near_up   = nearest_zone_above(shorts, current_price)
    near_down = nearest_zone_below(longs,  current_price)
    lines.append("")
    lines.append("─" * 70)
    lines.append("  EN YAKIN MIKNATISLAR")
    if near_up:
        dist_up = near_up.price_center - current_price
        lines.append(
            f"    ↑  {_fmt_price(near_up.price_center, coin)}   "
            f"({_fmt_money(near_up.dollars)}, {near_up.pct_from_spot:+.2f}%)"
        )
    else:
        lines.append("    ↑  (yukarıda tespit edilen bölge yok)")
    if near_down:
        lines.append(
            f"    ↓  {_fmt_price(near_down.price_center, coin)}   "
            f"({_fmt_money(near_down.dollars)}, {near_down.pct_from_spot:+.2f}%)"
        )
    else:
        lines.append("    ↓  (aşağıda tespit edilen bölge yok)")
    lines.append("═" * 70)

    return "\n".join(lines)


# ============================================================
# JSON output
# ============================================================
def format_json(
    coin: str,
    tf: str,
    current_price: float,
    total_oi_usd: float,
    funding: float,
    long_ratio: float,
    zones: dict[str, list[Zone]],
) -> dict:
    longs  = zones.get("long",  [])
    shorts = zones.get("short", [])
    near_up   = nearest_zone_above(shorts, current_price)
    near_down = nearest_zone_below(longs,  current_price)
    return {
        "coin": coin,
        "tf": tf,
        "current_price": current_price,
        "total_oi_usd": total_oi_usd,
        "funding": funding,
        "long_ratio": long_ratio,
        "zones": {
            "long":  [z.to_dict() for z in longs],
            "short": [z.to_dict() for z in shorts],
        },
        "nearest_magnets": {
            "above": near_up.to_dict()   if near_up   else None,
            "below": near_down.to_dict() if near_down else None,
        },
    }


# ============================================================
# Minimal visual:  zones as horizontal bars on a vertical price axis
# ============================================================
LONG_COLOR  = "#ef4444"        # red  — longs get rekt below
SHORT_COLOR = "#22c55e"        # green — shorts get rekt above
CURRENT_COLOR = "#fafafa"
NEUTRAL_BG = "#0a0e2c"


def render_slice_diverging(packet: dict, coin: str, tf: str) -> go.Figure:
    """
    Diverging horizontal bar chart from a fixed 1%-slice packet.

    Fixes the two visual problems with variable-width zones:
      • NO OVERLAP — long-liq extends LEFT (red), short-liq extends RIGHT
        (green). They live on opposite sides of the zero axis, so they can
        never sit on top of each other.
      • EQUAL THICKNESS — every slice is exactly 1% wide → uniform bar height.
    """
    slices = packet.get("slices", [])
    cp = packet.get("current_price", 0)
    if not slices or cp <= 0:
        fig = go.Figure()
        fig.add_annotation(text="No slice data", showarrow=False, font_color="#888")
        fig.update_layout(template="plotly_dark", paper_bgcolor=NEUTRAL_BG,
                          plot_bgcolor=NEUTRAL_BG)
        return fig

    mids   = [s["price_mid"] for s in slices]
    longs  = [-s["long_liq_usd"]  for s in slices]   # negative → extends left
    shorts = [ s["short_liq_usd"] for s in slices]   # positive → extends right
    # uniform bar thickness in price units = 1% of spot
    bar_w = cp * packet.get("slice_pct", 0.01) * 0.9

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=mids, x=longs, orientation="h",
        marker_color=LONG_COLOR, name="LONG patlama",
        width=bar_w,
        hovertemplate="$%{y:,.2f}<br>LONG liq: $%{customdata:,.0f}<extra></extra>",
        customdata=[s["long_liq_usd"] for s in slices],
    ))
    fig.add_trace(go.Bar(
        y=mids, x=shorts, orientation="h",
        marker_color=SHORT_COLOR, name="SHORT patlama",
        width=bar_w,
        hovertemplate="$%{y:,.2f}<br>SHORT liq: $%{x:,.0f}<extra></extra>",
    ))
    fig.add_hline(y=cp, line=dict(color=CURRENT_COLOR, width=1.5, dash="dot"),
                  annotation_text=f" spot {cp:,.4f}", annotation_position="right",
                  annotation_font=dict(color=CURRENT_COLOR, size=10))
    fig.add_vline(x=0, line=dict(color="#666", width=1))

    fig.update_layout(
        title=f"{coin} — {tf}  likidasyon mıknatısları (← LONG | SHORT →)",
        template="plotly_dark",
        barmode="overlay",
        bargap=0.15,
        xaxis=dict(title="Likide olacak $  (← long   |   short →)",
                   showgrid=True, gridcolor="#222"),
        yaxis=dict(title="Fiyat (USD)", showgrid=True, gridcolor="#222"),
        height=720,
        paper_bgcolor=NEUTRAL_BG, plot_bgcolor=NEUTRAL_BG,
        margin=dict(l=80, r=120, t=50, b=50),
        legend=dict(orientation="h", y=1.02, x=0),
    )
    return fig


def render_zone_chart(
    coin: str,
    tf: str,
    current_price: float,
    zones: dict[str, list[Zone]],
) -> go.Figure:
    """One panel per (coin, tf) — vertical price axis, horizontal $ bars."""
    longs  = zones.get("long",  [])
    shorts = zones.get("short", [])
    all_zones = longs + shorts
    if not all_zones:
        fig = go.Figure()
        fig.add_annotation(text="No significant zones detected",
                           showarrow=False, font_color="#888")
        fig.update_layout(template="plotly_dark", paper_bgcolor=NEUTRAL_BG,
                          plot_bgcolor=NEUTRAL_BG)
        return fig

    # y-axis bounds: span all zones + current price with 5% padding
    y_lo = min([z.price_low for z in all_zones] + [current_price]) * 0.98
    y_hi = max([z.price_high for z in all_zones] + [current_price]) * 1.02

    fig = go.Figure()

    # Bars: horizontal rectangles, x = $ amount, y_band = [price_low, price_high]
    for z in all_zones:
        color = LONG_COLOR if z.side == "long" else SHORT_COLOR
        fig.add_shape(
            type="rect",
            x0=0, x1=z.dollars,
            y0=z.price_low, y1=z.price_high,
            fillcolor=color, opacity=0.75,
            line=dict(color=color, width=1),
        )
        fig.add_annotation(
            x=z.dollars, y=z.price_center,
            text=f"  {_fmt_money(z.dollars)}",
            showarrow=False, xanchor="left",
            font=dict(color=color, size=11),
        )

    # Current price line
    fig.add_hline(
        y=current_price, line=dict(color=CURRENT_COLOR, width=1.5, dash="dot"),
        annotation_text=f"  spot {_fmt_price(current_price, coin)}",
        annotation_position="right",
        annotation_font=dict(color=CURRENT_COLOR, size=10),
    )

    max_dollars = max(z.dollars for z in all_zones)
    fig.update_layout(
        title=f"{coin} — {tf}  likidasyon bölgeleri",
        template="plotly_dark",
        xaxis=dict(
            title="Likide olacak para ($)",
            range=[0, max_dollars * 1.18],
            showgrid=True, gridcolor="#222",
        ),
        yaxis=dict(
            title="Fiyat (USD)",
            range=[y_lo, y_hi],
            showgrid=True, gridcolor="#222",
        ),
        height=520,
        paper_bgcolor=NEUTRAL_BG,
        plot_bgcolor=NEUTRAL_BG,
        margin=dict(l=80, r=120, t=50, b=50),
        showlegend=False,
    )
    return fig


def render_coin_dashboard(coin: str, current_price: float,
                          zones_by_tf: dict[str, dict[str, list[Zone]]],
                          ranges_by_tf: dict[str, float] | None = None) -> go.Figure:
    """4-panel grid (one per timeframe) showing zone bars.

    ranges_by_tf: optional {tf_label: pct_range}. When given, each panel's
    y-axis is fixed to current_price × (1 ± pct_range) so the visible WIDTH
    matches the configured per-timeframe band (24h tight, 1m wide) instead of
    auto-scaling to wherever the zones happen to fall."""
    labels = list(zones_by_tf.keys())
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=labels,
        horizontal_spacing=0.08, vertical_spacing=0.10,
    )

    for idx, tf in enumerate(labels):
        r = idx // 2 + 1
        c = idx % 2 + 1
        zones = zones_by_tf[tf]
        longs, shorts = zones.get("long", []), zones.get("short", [])
        all_zones = longs + shorts
        if not all_zones:
            continue

        # Fixed band from config if available, else fall back to zone extent.
        if ranges_by_tf and tf in ranges_by_tf:
            rng = ranges_by_tf[tf]
            y_lo = current_price * (1.0 - rng)
            y_hi = current_price * (1.0 + rng)
        else:
            y_lo = min([z.price_low for z in all_zones] + [current_price]) * 0.98
            y_hi = max([z.price_high for z in all_zones] + [current_price]) * 1.02
        max_dollars = max(z.dollars for z in all_zones)

        for z in all_zones:
            color = LONG_COLOR if z.side == "long" else SHORT_COLOR
            fig.add_shape(
                type="rect",
                x0=0, x1=z.dollars,
                y0=z.price_low, y1=z.price_high,
                fillcolor=color, opacity=0.75,
                line=dict(color=color, width=1),
                row=r, col=c,
            )
            fig.add_annotation(
                x=z.dollars, y=z.price_center,
                text=f" {_fmt_money(z.dollars)}",
                showarrow=False, xanchor="left",
                font=dict(color=color, size=9),
                row=r, col=c,
            )
        # current price line
        fig.add_hline(
            y=current_price, line=dict(color=CURRENT_COLOR, width=1, dash="dot"),
            row=r, col=c,
        )
        fig.update_xaxes(range=[0, max_dollars * 1.25], row=r, col=c,
                         gridcolor="#222", title_text="")
        fig.update_yaxes(range=[y_lo, y_hi], row=r, col=c,
                         gridcolor="#222", title_text="")

    fig.update_layout(
        title=f"{coin} likidasyon bölgeleri  —  spot {_fmt_price(current_price, coin)}",
        template="plotly_dark",
        paper_bgcolor=NEUTRAL_BG,
        plot_bgcolor=NEUTRAL_BG,
        height=900, showlegend=False,
    )
    return fig


# ============================================================
# Orchestration: produce all outputs
# ============================================================
def write_all_outputs(engine, out_dir: Path) -> dict:
    """For each (coin, tf): write txt + json. Also build per-coin HTML dashboard.
    Returns a summary dict."""
    out_dir.mkdir(parents=True, exist_ok=True)
    zones_dir = out_dir / "zones"
    zones_dir.mkdir(exist_ok=True)

    summary = {"text_files": [], "json_files": [], "html_files": []}

    for coin in engine.coins:
        current_price = engine.current_price(coin)
        if current_price <= 0:
            continue
        ctx = engine.context_summary(coin)
        funding   = ctx["avg_funding"]
        lr        = ctx["avg_long_ratio"]
        total_oi  = ctx["total_oi_usd"]
        zones_by_tf = {}

        for tf in [t.label for t in engine.timeframes]:
            zones = engine.extract_zones_for(coin, tf)
            zones_by_tf[tf] = zones

            txt = format_text_report(coin, tf, current_price, total_oi,
                                      funding, lr, zones)
            txt_path = zones_dir / f"{coin.lower()}_{tf}.txt"
            txt_path.write_text(txt, encoding="utf-8")
            summary["text_files"].append(txt_path)

            j = format_json(coin, tf, current_price, total_oi, funding, lr, zones)
            json_path = zones_dir / f"{coin.lower()}_{tf}.json"
            json_path.write_text(json.dumps(j, indent=2), encoding="utf-8")
            summary["json_files"].append(json_path)

        # Per-coin HTML dashboard (4 timeframes), y-axis fixed to each TF's band
        ranges_by_tf = {t.label: t.pct_range for t in engine.timeframes}
        fig = render_coin_dashboard(coin, current_price, zones_by_tf, ranges_by_tf)
        html_path = out_dir / f"{coin.lower()}_zones.html"
        fig.write_html(str(html_path), include_plotlyjs="cdn", full_html=True)
        summary["html_files"].append(html_path)

    return summary
