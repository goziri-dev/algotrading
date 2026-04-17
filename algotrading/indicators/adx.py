from typing import NamedTuple

import numpy as np

from .indicator import Indicator, IndicatorPlotSpec, IndicatorTraceSpec
from ._rma import _RMA


class ADXOutput(NamedTuple):
    adx: float
    plus_di: float
    minus_di: float


class ADX(Indicator):
    _FIELDS = [
        ("_value",    np.float64),
        ("_plus_di",  np.float64),
        ("_minus_di", np.float64),
    ]

    def __init__(self, di_length: int = 14, adx_smoothing: int = 14):
        super().__init__()
        self._rma_tr       = _RMA(di_length)
        self._rma_plus_dm  = _RMA(di_length)
        self._rma_minus_dm = _RMA(di_length)
        self._rma_dx       = _RMA(adx_smoothing)
        self._prev_high  = np.nan
        self._prev_low   = np.nan
        self._prev_close = np.nan

    def __call__(self, high: float, low: float, close: float) -> ADXOutput:  # type: ignore[override]
        if np.isnan(self._prev_close):
            plus_di = minus_di = adx = np.nan
        else:
            tr   = max(high - low, abs(high - self._prev_close), abs(low - self._prev_close))
            up   = high - self._prev_high
            down = self._prev_low - low
            plus_dm  = up   if (up > down   and up   > 0) else 0.0
            minus_dm = down if (down > up   and down > 0) else 0.0

            rma_tr    = self._rma_tr.update(tr)
            rma_plus  = self._rma_plus_dm.update(plus_dm)
            rma_minus = self._rma_minus_dm.update(minus_dm)

            if np.isnan(rma_tr) or rma_tr == 0.0:
                plus_di = minus_di = adx = np.nan
            else:
                plus_di  = 100.0 * rma_plus  / rma_tr
                minus_di = 100.0 * rma_minus / rma_tr
                di_sum   = plus_di + minus_di
                dx       = 100.0 * abs(plus_di - minus_di) / (di_sum if di_sum != 0.0 else 1.0)
                adx      = self._rma_dx.update(dx)

        self._prev_high  = high
        self._prev_low   = low
        self._prev_close = close
        super().update(_value=adx, _plus_di=plus_di, _minus_di=minus_di)
        return ADXOutput(adx, plus_di, minus_di)

    @property
    def adx(self):
        return self._value[:self._size]

    @property
    def plus_di(self):
        return self._plus_di[:self._size]

    @property
    def minus_di(self):
        return self._minus_di[:self._size]

    def plot_spec(self) -> IndicatorPlotSpec:
        return IndicatorPlotSpec(
            kind="panel",
            panel_title="ADX",
            traces=(
                IndicatorTraceSpec(attr="adx", label="ADX", color="#111827"),
                IndicatorTraceSpec(attr="plus_di", label="+DI", color="#16a34a"),
                IndicatorTraceSpec(attr="minus_di", label="-DI", color="#dc2626"),
            ),
        )
