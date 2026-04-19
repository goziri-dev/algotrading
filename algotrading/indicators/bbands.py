import math
from typing import NamedTuple
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from ._rma import _RMA
from .indicator import Indicator, IndicatorPlotSpec, IndicatorTraceSpec


class BBOutput(NamedTuple):
    mid: float
    upper: float
    lower: float


class BBANDS(Indicator):
    _FIELDS = [
        ("_value", np.float64),
        ("_upper", np.float64),
        ("_lower", np.float64),
        ("_input", np.float64),
    ]

    _value: NDArray[np.float64]
    _upper: NDArray[np.float64]
    _lower: NDArray[np.float64]
    _input: NDArray[np.float64]

    def __init__(
        self,
        period: int = 20,
        mult: float = 2.0,
        ma_type: Literal["SMA", "SMMA (RMA)"] = "SMA",
    ):
        super().__init__()
        self._period = period
        self._mult = mult
        self._ma_type = ma_type.strip().upper()
        self._rma = _RMA(period)
        self._running_sum = 0.0
        self._running_sum_sq = 0.0

        if self._ma_type not in {"SMA", "SMMA (RMA)", "SMMA", "RMA"}:
            raise ValueError(
                "Unsupported ma_type for BBANDS. Supported values: "
                "'SMA', 'SMMA (RMA)'"
            )

    def __call__(self, price: float) -> BBOutput:  # type: ignore[override]
        self._running_sum += price
        self._running_sum_sq += price * price
        if self._size + 1 > self._period:
            oldest = self._input[self._size - self._period]
            self._running_sum -= oldest
            self._running_sum_sq -= oldest * oldest

        if self._size + 1 >= self._period:
            mean = self._running_sum / self._period
            # population variance (Pine ta.stdev biased=true / ddof=0)
            variance = self._running_sum_sq / self._period - mean * mean
            std = math.sqrt(max(variance, 0.0))  # guard against float precision drift
            if self._ma_type == "SMA":
                mid = mean
            else:
                mid = self._rma.update(price)
            upper = mid + self._mult * std
            lower = mid - self._mult * std
        else:
            if self._ma_type != "SMA":
                self._rma.update(price)
            mid = upper = lower = np.nan

        super().update(_input=price, _value=mid, _upper=upper, _lower=lower)
        return BBOutput(mid, upper, lower)

    @property
    def mid(self):
        return self._value[:self._size]

    @property
    def upper(self):
        return self._upper[:self._size]

    @property
    def lower(self):
        return self._lower[:self._size]

    def plot_spec(self) -> IndicatorPlotSpec:
        return IndicatorPlotSpec(
            kind="overlay",
            traces=(
                IndicatorTraceSpec(attr="mid", label="BB Mid", color="#6b7280"),
                IndicatorTraceSpec(attr="upper", label="BB Upper", color="#4f46e5"),
                IndicatorTraceSpec(attr="lower", label="BB Lower", color="#4f46e5"),
            ),
        )
