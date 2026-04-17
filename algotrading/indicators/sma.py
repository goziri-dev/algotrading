import numpy as np
from numpy.typing import NDArray

from .indicator import Indicator


class SMA(Indicator):
    _FIELDS = [("_value", np.float64), ("_input", np.float64)]

    _input: NDArray[np.float64]

    def __init__(self, period: int=9):
        super().__init__()
        self._period = period
        self._running_sum = 0.0

    def __call__(self, price: float) -> "SMA":  # type: ignore[override]
        self._running_sum += price
        if self._size + 1 >= self._period:
            if self._size + 1 > self._period:
                self._running_sum -= self._input[self._size - self._period]
            result = self._running_sum / self._period
        else:
            result = np.nan
        super().update(_input=price, _value=result)
        return self
