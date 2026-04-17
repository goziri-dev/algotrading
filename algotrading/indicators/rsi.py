import numpy as np

from ._rma import _RMA
from .indicator import Indicator, IndicatorPlotSpec, IndicatorTraceSpec


class RSI(Indicator):
    def __init__(self, period: int = 14):
        super().__init__()
        self._period = period
        self._rma_gain = _RMA(period)
        self._rma_loss = _RMA(period)
        self._prev_price = np.nan

    def __call__(self, price: float) -> "RSI":  # type: ignore[override]
        if np.isnan(self._prev_price):
            result = np.nan
        else:
            change = price - self._prev_price
            up   = self._rma_gain.update(max(change,  0.0))
            down = self._rma_loss.update(max(-change, 0.0))
            if np.isnan(up):
                result = np.nan
            elif down == 0.0:
                result = 100.0
            elif up == 0.0:
                result = 0.0
            else:
                result = 100.0 - 100.0 / (1.0 + up / down)

        self._prev_price = price
        super().update(_value=result)
        return self

    def plot_spec(self) -> IndicatorPlotSpec:
        return IndicatorPlotSpec(
            kind="panel",
            panel_title="RSI",
            traces=(IndicatorTraceSpec(attr="value", label="RSI", color="#2563eb"),),
        )
