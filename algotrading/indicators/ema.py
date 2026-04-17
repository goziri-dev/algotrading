import numpy as np
from numpy.typing import NDArray

from .indicator import Indicator


class EMA(Indicator):
    _FIELDS = [("_value", np.float64), ("_input", np.float64)]

    _input: NDArray[np.float64]

    def __init__(self, period: int=9):
        super().__init__()
        self._period = period
        self._k = 2.0 / (period + 1)

    def __call__(self, price: float) -> "EMA":  # type: ignore[override]
        if self._size + 1 < self._period:
            result = np.nan
        elif self._size + 1 == self._period:
            result = float((np.sum(self._input[: self._size]) + price) / self._period)
        else:
            result = price * self._k + self._value[self._size - 1] * (1.0 - self._k)
        super().update(_input=price, _value=result)
        return self
