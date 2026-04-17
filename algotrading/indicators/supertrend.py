from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from ._rma import _RMA
from .indicator import Indicator, IndicatorPlotSpec, IndicatorTraceSpec


class SupertrendOutput(NamedTuple):
    trend: float
    direction: int


class Supertrend(Indicator):
    _FIELDS = [
        ("_value", np.float64),
        ("_direction", np.int8),
    ]

    _value: NDArray[np.float64]
    _direction: NDArray[np.int8]

    def __init__(self, atr_period: int = 10, factor: float = 3.0):
        super().__init__()
        self._factor = factor
        self._rma = _RMA(atr_period)
        self._prev_close = np.nan
        self._prev_atr = np.nan
        self._prev_upper = np.nan
        self._prev_lower = np.nan
        self._prev_trend = np.nan

    def __call__(self, high: float, low: float, close: float) -> SupertrendOutput:  # type: ignore[override]
        if np.isnan(self._prev_close):
            tr = high - low
        else:
            tr = max(high - low, abs(high - self._prev_close), abs(low - self._prev_close))
        atr = self._rma.update(tr)

        if np.isnan(atr):
            self._prev_close = close
            super().update(_value=np.nan, _direction=1)
            return SupertrendOutput(np.nan, 1)

        hl2 = (high + low) / 2.0
        upper = hl2 + self._factor * atr
        lower = hl2 - self._factor * atr

        if not np.isnan(self._prev_lower):
            if not (lower > self._prev_lower or self._prev_close < self._prev_lower):
                lower = self._prev_lower
        if not np.isnan(self._prev_upper):
            if not (upper < self._prev_upper or self._prev_close > self._prev_upper):
                upper = self._prev_upper

        if np.isnan(self._prev_atr):
            direction = 1
        elif self._prev_trend == self._prev_upper:
            direction = -1 if close > upper else 1
        else:
            direction = 1 if close < lower else -1

        trend = lower if direction == -1 else upper

        self._prev_close = close
        self._prev_atr = atr
        self._prev_upper = upper
        self._prev_lower = lower
        self._prev_trend = trend

        super().update(_value=trend, _direction=direction)
        return SupertrendOutput(float(trend), direction)

    @property
    def trend(self) -> NDArray[np.float64]:
        return self._value[:self._size]

    @property
    def direction(self) -> NDArray[np.int8]:
        return self._direction[:self._size]

    def plot_spec(self) -> IndicatorPlotSpec:
        return IndicatorPlotSpec(
            kind="overlay",
            traces=(
                IndicatorTraceSpec(attr="trend", label="Supertrend", color="#16a34a"),
            ),
        )
