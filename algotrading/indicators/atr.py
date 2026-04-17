import numpy as np

from .indicator import Indicator, IndicatorPlotSpec, IndicatorTraceSpec
from ._rma import _RMA


class ATR(Indicator):
    def __init__(self, period: int = 14):
        super().__init__()
        self._rma = _RMA(period)
        self._prev_close = np.nan

    def __call__(self, high: float, low: float, close: float) -> "ATR":  # type: ignore[override]
        if np.isnan(self._prev_close):
            tr = high - low
        else:
            tr = max(high - low, abs(high - self._prev_close), abs(low - self._prev_close))
        self._prev_close = close
        super().update(_value=self._rma.update(tr))
        return self

    def plot_spec(self) -> IndicatorPlotSpec:
        return IndicatorPlotSpec(
            kind="panel",
            panel_title="ATR",
            traces=(IndicatorTraceSpec(attr="value", label="ATR", color="#b45309"),),
        )
