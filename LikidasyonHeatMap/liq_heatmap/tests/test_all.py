"""
Full test suite for the v2 engine.
Run:  python tests/test_all.py
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

import config
from core.math_models  import (
    liq_price_long, liq_price_short, decay_weight,
    split_long_short, oi_weights, project_liquidations,
)
from core.exact_liq    import exact_liq_long, exact_liq_short, implied_leverage
from core.percent_bins import PercentBins
from core.timeframe    import load_timeframes


def approx(a, b, eps=1e-4):
    return abs(a - b) <= eps * max(1.0, abs(b))


# ============================================================
# Math models
# ============================================================
def test_liq_long():
    # exact: 50000 × (1 − 0.1)/(1 − 0.004) = 45180.72
    px = liq_price_long(50_000, 10, 0.004)
    assert approx(px, 45180.72, eps=1e-4)


def test_liq_short():
    # exact: 50000 × (1 + 0.1)/(1 + 0.004) = 54780.88
    px = liq_price_short(50_000, 10, 0.004)
    assert approx(px, 54780.88, eps=1e-4)


def test_decay_log_taming():
    # whale-spike compression — 100x volume should NOT produce 100x weight
    big   = decay_weight(1e8, 0, 0.2)
    small = decay_weight(1e6, 0, 0.2)
    assert 1.0 < (big / small) < 2.0


def test_decay_5h_halflife_check():
    # half-life ≈ 5h with λ=0.1386
    fresh = decay_weight(1e6, 0, 0.1386)
    aged  = decay_weight(1e6, 5, 0.1386)
    assert 0.45 < aged / fresh < 0.55, f"got {aged/fresh}"


def test_oi_weights_sum_one():
    w = oi_weights({"binance": 5e9, "bybit": 3e9, "okx": 2e9})
    assert approx(sum(w.values()), 1.0)
    assert approx(w["binance"], 0.5)


def test_split_long_short():
    lp, sp = split_long_short(100.0, 0.6)
    assert approx(lp, 60.0)
    assert approx(sp, 40.0)


def test_project_arrays_align():
    lev = [5, 10, 25, 50]
    dist = {5: 0.25, 10: 0.25, 25: 0.25, 50: 0.25}
    lp, lw, sp, sw = project_liquidations(
        50_000, 100, 100, lev, dist, 0.004,
    )
    assert len(lp) == len(lw) == len(sp) == len(sw) == 4
    assert (lp < 50_000).all()
    assert (sp > 50_000).all()
    assert approx(lw.sum(), 100.0)
    assert approx(sw.sum(), 100.0)


# ============================================================
# Exact liquidation (DEX path)
# ============================================================
def test_exact_liq_long_basic():
    # 10x long: $1000 collateral, 1 BTC, entry $10k, MMR 1%
    liq = exact_liq_long(entry=10_000, size=1.0, collateral=1000, mmr=0.01)
    # (1·10000 − 1000)/(1·(1 − 0.01)) = 9000/0.99 = 9090.91
    assert approx(liq, 9090.909, eps=1e-3)


def test_exact_liq_short_basic():
    liq = exact_liq_short(entry=10_000, size=1.0, collateral=1000, mmr=0.01)
    # (1000 + 10000)/(1·1.01) = 10891.09
    assert approx(liq, 10891.089, eps=1e-3)


def test_implied_leverage():
    assert approx(implied_leverage(10_000, 1.0, 1000), 10.0)
    assert implied_leverage(0, 1.0, 1000) == 0.0


# ============================================================
# PercentBins
# ============================================================
def test_percent_bins_construction():
    pb = PercentBins(spot=50_000, pct_range=0.2, pct_bucket=0.01, n_time_steps=240)
    assert pb.n_price_bins == 40
    assert pb.long_mat.shape == (240, 40)
    assert approx(pb.min_price, 40_000)
    assert approx(pb.max_price, 60_000)


def test_percent_bins_scatter_in_range():
    pb = PercentBins(spot=50_000, pct_range=0.2, pct_bucket=0.01, n_time_steps=10)
    prices  = np.array([45_000, 47_500, 50_000, 52_500, 55_000], dtype=np.float64)
    weights = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    written = pb.add_long_vec(5, prices, weights)
    assert written == 5
    assert approx(pb.long_mat[5].sum(), 15.0)


def test_percent_bins_clip_out_of_range():
    pb = PercentBins(spot=50_000, pct_range=0.2, pct_bucket=0.01, n_time_steps=5)
    prices  = np.array([30_000, 50_000, 70_000], dtype=np.float64)
    weights = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    written = pb.add_long_vec(0, prices, weights)
    assert written == 1
    assert approx(pb.long_mat.sum(), 2.0)


def test_percent_bins_recenter_small():
    pb = PercentBins(spot=50_000, pct_range=0.2, pct_bucket=0.01, n_time_steps=10,
                     recenter_trigger=0.05)
    pb.add_long_vec(0, np.array([50_000.0]), np.array([7.0]))
    # spot moves 5% up → triggers recenter; mass should still be in matrix
    changed = pb.maybe_recenter(52_500)
    assert changed
    assert approx(pb.anchor_spot, 52_500)


def test_percent_bins_recenter_huge_resets():
    pb = PercentBins(spot=50_000, pct_range=0.2, pct_bucket=0.01, n_time_steps=10)
    pb.add_long_vec(0, np.array([50_000.0]), np.array([7.0]))
    # spot drops 50% — outside 2·PCT_RANGE window → full clear
    pb.maybe_recenter(25_000)
    assert pb.long_mat.sum() == 0.0
    assert approx(pb.anchor_spot, 25_000)


def test_percent_bins_roll_time_drops_oldest():
    pb = PercentBins(spot=50_000, pct_range=0.2, pct_bucket=0.01, n_time_steps=3)
    pb.add_long_vec(0, np.array([50_000.0]), np.array([7.0]))
    pb.roll_time()
    assert pb.long_mat[0].sum() == 0.0   # original row 0 dropped


def test_percent_bins_hottest_zones():
    pb = PercentBins(spot=50_000, pct_range=0.2, pct_bucket=0.01, n_time_steps=5)
    pb.add_long_vec(0, np.array([45_000, 55_000]), np.array([10.0, 25.0]))
    top = pb.hottest_zones("long", top_k=2)
    assert len(top) == 2
    assert top[0][1] > top[1][1]                # sorted desc by mass
    assert 54_000 < top[0][0] < 56_000           # near 55k


# ============================================================
# Timeframes
# ============================================================
def test_timeframe_loading():
    tfs = load_timeframes(config.TIMEFRAMES)
    assert len(tfs) == 4
    labels = [t.label for t in tfs]
    assert labels == ["24h", "3d", "1w", "1m"]
    for tf in tfs:
        # consistency
        assert tf.window_sec == tf.candle_sec * tf.n_candles
        # half-life calibration: short TFs ~20%, long TFs up to ~55% of window
        ratio = tf.half_life_hours / tf.window_hours
        assert 0.15 < ratio < 0.55, f"{tf.label}: {ratio}"


def test_timeframe_candle_bucketing():
    tfs = load_timeframes(config.TIMEFRAMES)
    tf24 = tfs[0]  # 24h with 6-min candle
    # 6-minute candle = 360s = 360,000 ms
    ts = 1_700_000_000_123
    bucket = tf24.candle_index_for_ts(ts)
    assert bucket % 360_000 == 0
    assert bucket <= ts < bucket + 360_000


# ============================================================
# Zone extraction
# ============================================================
def test_zones_two_clear_peaks():
    from core.zone_extractor import extract_zones
    pb = PercentBins(spot=65_000, pct_range=0.2, pct_bucket=0.001, n_time_steps=10)
    # Inject two distinct dollar peaks: one below ($61k long-liq), one above ($69k short-liq)
    pb.add_long_vec(5,  np.array([61_000.0, 61_050.0, 61_100.0]),
                        np.array([5e6, 8e6, 5e6]))
    pb.add_short_vec(5, np.array([69_000.0, 69_050.0, 69_100.0]),
                        np.array([4e6, 6e6, 4e6]))
    zones = extract_zones(pb, current_price=65_000,
                          total_oi_usd=1e9, long_ratio=0.55)
    assert len(zones["long"])  >= 1
    assert len(zones["short"]) >= 1
    z_long  = zones["long"][0]
    z_short = zones["short"][0]
    assert 60_500 < z_long.price_center  < 61_500
    assert 68_500 < z_short.price_center < 69_500


def test_zones_dollar_calibration():
    from core.zone_extractor import extract_zones
    pb = PercentBins(spot=50_000, pct_range=0.2, pct_bucket=0.001, n_time_steps=5)
    pb.add_long_vec(0, np.array([47_500.0]), np.array([100.0]))  # raw
    zones = extract_zones(pb, current_price=50_000,
                          total_oi_usd=1e9, long_ratio=0.6)
    # 60% of 1B → long side total = $600M  →  must show up in zones
    total_dollars = sum(z.dollars for z in zones["long"])
    assert 0.95 * 6e8 < total_dollars < 1.05 * 6e8


def test_zones_min_share_filter():
    from core.zone_extractor import extract_zones
    pb = PercentBins(spot=50_000, pct_range=0.2, pct_bucket=0.001, n_time_steps=5)
    # Two peaks: one huge, one tiny (1%)
    pb.add_long_vec(0, np.array([47_500.0, 47_550.0]), np.array([100.0, 100.0]))
    pb.add_long_vec(0, np.array([45_000.0]), np.array([2.0]))
    zones = extract_zones(pb, current_price=50_000,
                          total_oi_usd=1e9, long_ratio=0.5,
                          min_dollar_share=0.05)
    # tiny peak should be filtered out
    assert len(zones["long"]) == 1


def test_nearest_zone_helpers():
    from core.zone_extractor import (
        Zone, nearest_zone_above, nearest_zone_below
    )
    shorts = [
        Zone("short", 66_000, 66_500, 66_250, 1e7, 1, 1.92),
        Zone("short", 68_000, 68_500, 68_250, 5e7, 2, 5.0),
    ]
    longs = [
        Zone("long", 63_000, 63_500, 63_250, 2e7, 1, -2.69),
        Zone("long", 60_000, 60_500, 60_250, 4e7, 2, -7.31),
    ]
    nu = nearest_zone_above(shorts, 65_000)
    nd = nearest_zone_below(longs,  65_000)
    assert nu.price_center == 66_250          # closer above
    assert nd.price_center == 63_250          # closer below


def test_zones_dex_exact_overlay():
    """DEX exact dollars should add on top of calibrated CEX, undistorted."""
    from core.zone_extractor import extract_zones
    pb = PercentBins(spot=50_000, pct_range=0.2, pct_bucket=0.001, n_time_steps=5)
    # Small CEX mass below
    pb.add_long_vec(0, np.array([47_500.0]), np.array([10.0]))
    # DEX: one big exact long position at $46,000 worth $200M
    dex = [(46_000.0, 200e6, True)]
    zones = extract_zones(pb, current_price=50_000,
                          total_oi_usd=1e8, long_ratio=0.5,
                          dex_positions=dex)
    # The DEX position must appear as a zone near $46k with ≈$200M
    dex_zone = [z for z in zones["long"] if 45_500 < z.price_center < 46_500]
    assert dex_zone, "DEX position did not produce a zone"
    assert dex_zone[0].dollars >= 200e6 * 0.95


def test_zones_dex_only_no_cex():
    """If CEX matrix is empty but DEX has positions, zones still form."""
    from core.zone_extractor import extract_zones
    pb = PercentBins(spot=50_000, pct_range=0.2, pct_bucket=0.001, n_time_steps=5)
    dex = [(48_000.0, 50e6, True), (52_000.0, 30e6, False)]
    zones = extract_zones(pb, current_price=50_000,
                          total_oi_usd=0.0, long_ratio=0.5,
                          dex_positions=dex)
    assert len(zones["long"])  >= 1
    assert len(zones["short"]) >= 1


def test_physical_constraint_zones():
    """Long-liq zones must be below spot; short-liq zones above. Already-swept
    levels on the wrong side must be removed."""
    from core.zone_extractor import extract_zones
    pb = PercentBins(spot=66_000, pct_range=0.2, pct_bucket=0.001, n_time_steps=5)
    # Inject a LONG-liq mass ABOVE spot ($72k) — physically already swept
    pb.add_long_vec(0, np.array([72_000.0]), np.array([100.0]))
    # And a valid LONG-liq mass BELOW spot ($60k)
    pb.add_long_vec(0, np.array([60_000.0]), np.array([100.0]))
    # A SHORT-liq mass BELOW spot ($60k) — already swept
    pb.add_short_vec(0, np.array([60_000.0]), np.array([100.0]))
    # A valid SHORT-liq mass ABOVE spot ($72k)
    pb.add_short_vec(0, np.array([72_000.0]), np.array([100.0]))
    zones = extract_zones(pb, current_price=66_000,
                          total_oi_usd=1e9, long_ratio=0.5)
    # Every long zone must be BELOW spot
    for z in zones["long"]:
        assert z.price_center < 66_000, f"long zone above spot: {z.price_center}"
    # Every short zone must be ABOVE spot
    for z in zones["short"]:
        assert z.price_center > 66_000, f"short zone below spot: {z.price_center}"


def test_physical_constraint_slices():
    """In the slice packet, below-spot slices carry only long-liq, above-spot
    slices only short-liq."""
    from core.slice_packet import build_slice_packet
    pb = PercentBins(spot=66_000, pct_range=0.2, pct_bucket=0.001, n_time_steps=5)
    pb.add_long_vec(0,  np.array([72_000.0, 60_000.0]), np.array([100.0, 100.0]))
    pb.add_short_vec(0, np.array([60_000.0, 72_000.0]), np.array([100.0, 100.0]))
    pkt = build_slice_packet(pb, current_price=66_000,
                             total_oi_usd=1e9, long_ratio=0.5)
    for s in pkt["slices"]:
        if s["idx"] >= 0:                      # above spot
            assert s["long_liq_usd"] == 0.0, "long-liq leaked above spot"
        else:                                  # below spot
            assert s["short_liq_usd"] == 0.0, "short-liq leaked below spot"


# ============================================================
# Run
# ============================================================
if __name__ == "__main__":
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  ✓ {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  ✗ {t.__name__}: {e}")
        except Exception as e:
            failed += 1
            print(f"  ✗ {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
