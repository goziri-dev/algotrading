"""Higher-High / Lower-Low structural pivot indicator.

Streaming port of the TradingView study ``Higher High Lower Low Strategy``
by LonesomeThecolor.blue (MPL-2.0, https://mozilla.org/MPL/2.0/).
"""
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from .indicator import (
    Indicator,
    IndicatorMarkerSpec,
    IndicatorPlotSpec,
    IndicatorTraceSpec,
)


class HHLLOutput(NamedTuple):
    trend: int
    support: float
    resistance: float
    pivot_type: int
    pivot_price: float


class HHLL(Indicator):
    """Higher-high / lower-low pivot classifier with trend and structural S/R.

    Pivots are confirmed ``right_bars`` after the fact, so ``pivot_type`` and
    ``pivot_price`` are emitted with that delay. ``trend`` flips on the bar the
    close breaks the current support or resistance — it does not lag.

    Outputs per bar (also available as arrays via properties):
      * ``trend``        +1 up, -1 down, 0 undetermined
      * ``support``      last structural support price, NaN until first set
      * ``resistance``   last structural resistance price, NaN until first set
      * ``pivot_type``   :pyattr:`HH` / :pyattr:`HL` / :pyattr:`LL` /
                          :pyattr:`LH`, or 0 when no new pivot / unclassified
      * ``pivot_price``  pivot price when a pivot is confirmed this bar, else NaN
    """

    HH = 1
    HL = 2
    LL = -1
    LH = -2

    _FIELDS = [
        ("_value",       np.float64),
        ("_support",     np.float64),
        ("_resistance",  np.float64),
        ("_pivot_type",  np.int8),
        ("_pivot_price", np.float64),
    ]

    _value: NDArray[np.float64]
    _support: NDArray[np.float64]
    _resistance: NDArray[np.float64]
    _pivot_type: NDArray[np.int8]
    _pivot_price: NDArray[np.float64]

    def __init__(self, left_bars: int = 5, right_bars: int = 5):
        super().__init__()
        if left_bars < 1 or right_bars < 1:
            raise ValueError("left_bars and right_bars must be >= 1")
        self._lb = int(left_bars)
        self._rb = int(right_bars)
        window = self._lb + self._rb + 1
        self._high_buf = np.full(window, np.nan)
        self._low_buf = np.full(window, np.nan)
        self._buf_count = 0

        self._pivots: list[tuple[int, float]] = []
        self._last_hl: int = 0
        self._last_zz: float = np.nan

        self._trend: int = 0
        self._sup: float = np.nan
        self._res: float = np.nan

    def __call__(self, high: float, low: float, close: float) -> HHLLOutput:  # type: ignore[override]
        self._high_buf[:-1] = self._high_buf[1:]
        self._low_buf[:-1] = self._low_buf[1:]
        self._high_buf[-1] = high
        self._low_buf[-1] = low
        self._buf_count += 1

        pivot_type = 0
        pivot_price = np.nan

        if self._buf_count >= self._lb + self._rb + 1:
            idx = self._lb
            cand_high = self._high_buf[idx]
            cand_low = self._low_buf[idx]
            is_ph = bool(
                (cand_high > self._high_buf[:idx]).all()
                and (cand_high > self._high_buf[idx + 1:]).all()
            )
            is_pl = bool(
                (cand_low < self._low_buf[:idx]).all()
                and (cand_low < self._low_buf[idx + 1:]).all()
            )

            new_hl = 0
            new_zz = np.nan
            if is_ph:
                new_hl = 1
                new_zz = cand_high
            elif is_pl:
                new_hl = -1
                new_zz = cand_low

            if new_hl != 0:
                pivot_type, pivot_price = self._ingest_pivot(new_hl, new_zz)

        if pivot_type == HHLL.LH:
            self._res = pivot_price
        if pivot_type == HHLL.HL:
            self._sup = pivot_price

        if not np.isnan(self._res) and close > self._res:
            self._trend = 1
        elif not np.isnan(self._sup) and close < self._sup:
            self._trend = -1

        if self._trend == 1:
            if pivot_type == HHLL.HH:
                self._res = pivot_price
            elif pivot_type == HHLL.HL:
                self._sup = pivot_price
        elif self._trend == -1:
            if pivot_type == HHLL.LH:
                self._res = pivot_price
            elif pivot_type == HHLL.LL:
                self._sup = pivot_price

        super().update(
            _value=float(self._trend),
            _support=self._sup,
            _resistance=self._res,
            _pivot_type=pivot_type,
            _pivot_price=pivot_price,
        )
        return HHLLOutput(
            trend=self._trend,
            support=float(self._sup),
            resistance=float(self._res),
            pivot_type=pivot_type,
            pivot_price=float(pivot_price),
        )

    def _ingest_pivot(self, new_hl: int, new_zz: float) -> tuple[int, float]:
        # Filter 1: consecutive same-direction pivot that doesn't extend the extreme.
        # Pine leaves hl intact on this path, so _last_hl advances but _last_zz doesn't.
        cancel_zz = False
        if self._last_hl != 0 and not np.isnan(self._last_zz):
            if new_hl == -1 and self._last_hl == -1 and new_zz > self._last_zz:
                cancel_zz = True
            elif new_hl == 1 and self._last_hl == 1 and new_zz < self._last_zz:
                cancel_zz = True

        effective_zz = np.nan if cancel_zz else new_zz

        # Filter 2: direction flipped but price went the wrong way (a "low" above
        # the previous high, or a "high" below the previous low). Skipped when
        # filter 1 already cancelled zz, matching Pine's NA-short-circuit.
        cancel_hl = False
        if not cancel_zz and self._last_hl != 0 and not np.isnan(self._last_zz):
            if new_hl == -1 and self._last_hl == 1 and effective_zz > self._last_zz:
                cancel_hl = True
            elif new_hl == 1 and self._last_hl == -1 and effective_zz < self._last_zz:
                cancel_hl = True

        if cancel_hl:
            self._last_hl = new_hl
            return 0, np.nan

        if cancel_zz:
            self._last_hl = new_hl
            return 0, np.nan

        self._last_hl = new_hl
        self._last_zz = float(effective_zz)
        self._pivots.append((new_hl, float(effective_zz)))
        return self._classify_pivot(), float(effective_zz)

    def _classify_pivot(self) -> int:
        n = len(self._pivots)
        if n < 1:
            return 0
        a_dir, a = self._pivots[-1]

        # Walk backwards picking the nearest alternating-direction predecessors
        # for (b, c, d, e). Same as findprevious() in the Pine source.
        targets: list[float | None] = [None, None, None, None]
        needed = -a_dir
        slot = 0
        i = n - 2
        while i >= 0 and slot < 4:
            p_dir, p_val = self._pivots[i]
            if p_dir == needed:
                targets[slot] = p_val
                slot += 1
                needed = -needed
            i -= 1
        b, c, d, e = targets

        have_bcd = b is not None and c is not None and d is not None
        have_bcde = have_bcd and e is not None

        if have_bcd and a > b and a > c and c > b and c > d:  # type: ignore[operator]
            return HHLL.HH
        if have_bcd and a < b and a < c and c < b and c < d:  # type: ignore[operator]
            return HHLL.LL
        if have_bcde and a >= c and b > c and b > d and d > c and d > e:  # type: ignore[operator]
            return HHLL.HL
        if have_bcde and a <= c and b < c and b < d and d < c and d < e:  # type: ignore[operator]
            return HHLL.LH
        if have_bcd and a < b and a > c and b < d:  # type: ignore[operator]
            return HHLL.HL
        if have_bcd and a > b and a < c and b > d:  # type: ignore[operator]
            return HHLL.LH
        return 0

    @property
    def trend(self) -> NDArray[np.int8]:
        return self._value[:self._size].astype(np.int8)

    @property
    def support(self) -> NDArray[np.float64]:
        return self._support[:self._size]

    @property
    def resistance(self) -> NDArray[np.float64]:
        return self._resistance[:self._size]

    @property
    def pivot_type(self) -> NDArray[np.int8]:
        return self._pivot_type[:self._size]

    @property
    def pivot_price(self) -> NDArray[np.float64]:
        return self._pivot_price[:self._size]

    def _marker_array(self, kind: int) -> NDArray[np.float64]:
        """Price per bar for a given classification, anchored at the *pivot* bar.

        Pivots are emitted at bar ``N`` but refer to bar ``N - right_bars``, so
        we shift the price back by ``right_bars`` — same as Pine's
        ``plotshape(..., offset=-rb)``.
        """
        size = self._size
        out = np.full(size, np.nan, dtype=np.float64)
        if size == 0:
            return out
        pt = self._pivot_type[:size]
        pp = self._pivot_price[:size]
        src = np.where(pt == kind)[0]
        if len(src) == 0:
            return out
        dst = src - self._rb
        valid = dst >= 0
        out[dst[valid]] = pp[src[valid]]
        return out

    @property
    def hh_price(self) -> NDArray[np.float64]:
        return self._marker_array(HHLL.HH)

    @property
    def hl_price(self) -> NDArray[np.float64]:
        return self._marker_array(HHLL.HL)

    @property
    def ll_price(self) -> NDArray[np.float64]:
        return self._marker_array(HHLL.LL)

    @property
    def lh_price(self) -> NDArray[np.float64]:
        return self._marker_array(HHLL.LH)

    def plot_spec(self) -> IndicatorPlotSpec:
        return IndicatorPlotSpec(
            kind="overlay",
            traces=(
                IndicatorTraceSpec(attr="resistance", label="Resistance", color="#dc2626"),
                IndicatorTraceSpec(attr="support", label="Support", color="#16a34a"),
            ),
            markers=(
                IndicatorMarkerSpec(attr="hh_price", label="HH", shape="arrow_down", color="#16a34a", position="above"),
                IndicatorMarkerSpec(attr="hl_price", label="HL", shape="arrow_up",   color="#16a34a", position="below"),
                IndicatorMarkerSpec(attr="ll_price", label="LL", shape="arrow_up",   color="#dc2626", position="below"),
                IndicatorMarkerSpec(attr="lh_price", label="LH", shape="arrow_down", color="#dc2626", position="above"),
            ),
        )
