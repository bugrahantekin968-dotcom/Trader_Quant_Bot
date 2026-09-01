"""
Percent-based bin manager — anchored to current spot.
=====================================================
Grid: [spot · (1 − PCT_RANGE), spot · (1 + PCT_RANGE)]
      divided into 2·PCT_RANGE / PCT_BUCKET equal-percent cells.

When spot drifts more than RECENTER_TRIGGER from the anchor, the entire
matrix is shifted in price space (np.roll), preserving overlapping mass.

For larger moves (>2·PCT_RANGE) the grid is cleared — old projections
have lost meaning anyway because they were computed against a far-away
entry price.

Hot path is O(1) scatter into a pre-allocated NumPy matrix.
"""
from __future__ import annotations

import numpy as np


class PercentBins:
    """
    Two parallel matrices (long-liq, short-liq) of shape (n_time_steps, n_price_bins).

    Time axis: oldest at index 0, newest at index n-1.
    Price axis: lowest at index 0, highest at index n_price_bins-1.
    """

    def __init__(
        self,
        spot: float,
        pct_range: float = 0.20,
        pct_bucket: float = 0.01,
        n_time_steps: int = 240,
        recenter_trigger: float = 0.05,
    ):
        if pct_range <= 0 or pct_bucket <= 0:
            raise ValueError("pct_range and pct_bucket must be positive")
        if n_time_steps <= 0:
            raise ValueError("n_time_steps must be positive")

        self.anchor_spot = float(spot)
        self.pct_range = pct_range
        self.pct_bucket = pct_bucket
        self.recenter_trigger = recenter_trigger
        self.n_time_steps = n_time_steps
        self.n_price_bins = int(round(2 * pct_range / pct_bucket))

        shape = (n_time_steps, self.n_price_bins)
        self.long_mat  = np.zeros(shape, dtype=np.float64)
        self.short_mat = np.zeros(shape, dtype=np.float64)

        # cached tick array  (mid price of each cell, in absolute USD)
        self._refresh_ticks()

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------
    def _refresh_ticks(self) -> None:
        offsets = -self.pct_range + (np.arange(self.n_price_bins) + 0.5) * self.pct_bucket
        self.price_ticks = self.anchor_spot * (1.0 + offsets)

    @property
    def min_price(self) -> float:
        return self.anchor_spot * (1.0 - self.pct_range)

    @property
    def max_price(self) -> float:
        return self.anchor_spot * (1.0 + self.pct_range)

    def price_to_bin(self, price: float) -> int:
        pct_offset = (price / self.anchor_spot) - 1.0
        if pct_offset < -self.pct_range or pct_offset >= self.pct_range:
            return -1
        return int((pct_offset + self.pct_range) / self.pct_bucket)

    # ------------------------------------------------------------------
    # Hot-path scatter-add
    # ------------------------------------------------------------------
    def add_long_vec(self, t_idx: int, prices: np.ndarray, weights: np.ndarray) -> int:
        """Vectorized scatter of (price, weight) pairs into the LONG matrix.
        Returns # cells actually written (after range-clipping)."""
        return self._add(self.long_mat, t_idx, prices, weights)

    def add_short_vec(self, t_idx: int, prices: np.ndarray, weights: np.ndarray) -> int:
        return self._add(self.short_mat, t_idx, prices, weights)

    def _add(self, mat: np.ndarray, t_idx: int, prices, weights) -> int:
        if not (0 <= t_idx < self.n_time_steps):
            return 0
        prices  = np.asarray(prices,  dtype=np.float64)
        weights = np.asarray(weights, dtype=np.float64)
        if prices.size == 0:
            return 0
        pct_offsets = (prices / self.anchor_spot) - 1.0
        bins = ((pct_offsets + self.pct_range) / self.pct_bucket).astype(np.int64)
        mask = (bins >= 0) & (bins < self.n_price_bins)
        if not mask.any():
            return 0
        np.add.at(mat[t_idx], bins[mask], weights[mask])
        return int(mask.sum())

    # ------------------------------------------------------------------
    # Window maintenance
    # ------------------------------------------------------------------
    def roll_time(self) -> None:
        """Drop oldest candle, open new empty slot at the right edge."""
        self.long_mat[:-1]  = self.long_mat[1:]
        self.long_mat[-1]   = 0.0
        self.short_mat[:-1] = self.short_mat[1:]
        self.short_mat[-1]  = 0.0

    def maybe_recenter(self, new_spot: float) -> bool:
        """
        Re-anchor the grid to new_spot if drift exceeds recenter_trigger.

        Strategy:
          - If drift > 2·pct_range  →  clear (old data outside new window)
          - Else  →  np.roll by the cell-count corresponding to the drift,
                     zeroing the cells that came in from the wrap.

        Returns True if the grid was modified.
        """
        if new_spot <= 0:
            return False
        drift = (new_spot - self.anchor_spot) / self.anchor_spot
        if abs(drift) < self.recenter_trigger:
            return False

        if abs(drift) > 2 * self.pct_range:
            # Full reset — old projections relate to a far-away entry price
            self.long_mat.fill(0.0)
            self.short_mat.fill(0.0)
            self.anchor_spot = float(new_spot)
            self._refresh_ticks()
            return True

        # Cell-count to shift. Positive drift → spot moved UP → old mass
        # appears LOWER on new grid → shift cells DOWN (negative shift).
        shift_cells = int(round(drift / self.pct_bucket))
        if shift_cells == 0:
            return False

        self.long_mat  = np.roll(self.long_mat,  -shift_cells, axis=1)
        self.short_mat = np.roll(self.short_mat, -shift_cells, axis=1)
        # Zero the wrapped-in tail
        if shift_cells > 0:
            self.long_mat[:,  -shift_cells:] = 0.0
            self.short_mat[:, -shift_cells:] = 0.0
        else:
            self.long_mat[:,  :(-shift_cells)] = 0.0
            self.short_mat[:, :(-shift_cells)] = 0.0

        self.anchor_spot = float(new_spot)
        self._refresh_ticks()
        return True

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------
    def combined(self) -> np.ndarray:
        return self.long_mat + self.short_mat

    def column_mass(self) -> np.ndarray:
        """Total mass per price cell, summed over time."""
        return (self.long_mat + self.short_mat).sum(axis=0)

    def hottest_zones(self, side: str = "both", top_k: int = 5):
        if side == "long":
            mat = self.long_mat
        elif side == "short":
            mat = self.short_mat
        else:
            mat = self.long_mat + self.short_mat
        col_sums = mat.sum(axis=0)
        if col_sums.sum() <= 0:
            return []
        order = np.argsort(col_sums)[::-1][:top_k]
        return [(float(self.price_ticks[i]), float(col_sums[i])) for i in order if col_sums[i] > 0]

    def snapshot(self) -> dict:
        """JSON-serializable snapshot for persistence / bot consumption."""
        return {
            "anchor_spot": self.anchor_spot,
            "pct_range":   self.pct_range,
            "pct_bucket":  self.pct_bucket,
            "n_time_steps": self.n_time_steps,
            "n_price_bins": self.n_price_bins,
            "price_ticks": self.price_ticks.tolist(),
            "long_mat":  self.long_mat.tolist(),
            "short_mat": self.short_mat.tolist(),
        }
