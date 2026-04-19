from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal, Self

import numpy as np
from numpy.typing import NDArray

from algotrading.core.growable import Growable


@dataclass(frozen=True)
class IndicatorTraceSpec:
    attr: str
    label: str | None = None
    color: str | None = None
    width: int = 2


@dataclass(frozen=True)
class IndicatorPlotSpec:
    kind: Literal["overlay", "panel"] = "overlay"
    panel_title: str | None = None
    traces: tuple[IndicatorTraceSpec, ...] = (IndicatorTraceSpec(attr="value"),)


class Indicator(Growable, ABC):
    _FIELDS = [("_value", np.float64)]

    _value: NDArray[np.float64]

    def __init__(self):
        super().__init__(fields=self._FIELDS)

    @property
    def value(self) -> NDArray[np.float64]:
        """Full output history as a slice — no copy."""
        return self._value[:self._size]

    def __getitem__(self, index: int) -> float:
        """Standard Python indexing: ``[-1]`` = current bar, ``[-2]`` = one bar ago."""
        if self._size == 0 or -index > self._size:
            return np.nan
        return float(self.value[index])

    def output(self, attr: str) -> "IndicatorOutput":
        """Reference a named output of this indicator (e.g. ``BBANDS.upper``).

        Use as a ``source`` for another indicator::

            self.ema = self.I(EMA(200), source=self.bbands.output("upper"))
        """
        return IndicatorOutput(self, attr)

    @abstractmethod
    def __call__(self, *args: Any, **kwds: Any) -> Self:
        """Compute and store the next value, return ``self`` for chaining."""
        ...

    def plot_spec(self) -> IndicatorPlotSpec:
        """Default plotting blueprint.

        Indicators default to an overlay line using their ``value`` output.
        Subclasses can override this to route traces to a panel or expose
        multi-trace outputs (e.g. BBANDS, ADX).
        """
        return IndicatorPlotSpec()


class IndicatorOutput:
    """Reference to a named output of a multi-output indicator.

    Exposes ``[-1]`` indexing over the selected attribute so it can be used
    as a ``source`` wherever an :class:`Indicator` is accepted.
    """
    __slots__ = ("_indicator", "_attr")

    def __init__(self, indicator: Indicator, attr: str):
        self._indicator = indicator
        self._attr = attr

    def __getitem__(self, index: int) -> float:
        arr = getattr(self._indicator, self._attr)
        if len(arr) == 0 or -index > len(arr):
            return float("nan")
        return float(arr[index])
