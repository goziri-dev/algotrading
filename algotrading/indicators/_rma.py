import numpy as np


class _RMA:
    """Wilder's smoothed moving average — lightweight scalar helper.

    Seeds with the SMA of the first `period` values, then applies
    ``rma = (prev * (period - 1) + value) / period``.
    Returns ``np.nan`` during warmup.
    """

    def __init__(self, period: int):
        self._period = period
        self._value = np.nan
        self._sum = 0.0
        self._count = 0

    def update(self, value: float) -> float:
        if np.isnan(value):
            return np.nan
        self._count += 1
        if self._count < self._period:
            self._sum += value
            return np.nan
        if self._count == self._period:
            self._sum += value
            self._value = self._sum / self._period
        else:
            self._value = (self._value * (self._period - 1) + value) / self._period
        return float(self._value)
