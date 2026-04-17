from typing import Literal

import numpy as np
from numpy.typing import NDArray

from .indicator import Indicator


class COTIndex(Indicator):
    """Rolling stochastic of the speculator-commercial spread.

    Measures where the current net speculator position (noncomm_net - comm_net)
    sits relative to its range over the last ``lookback`` weeks, normalised to
    0–100.  A flat range returns 50 (neutral).

    Typical usage inside a COT strategy::

        self.cot_index = self.I(COTIndex(lookback=26), source=???)

    Because the source is a derived bar field rather than a single bar column,
    wire it manually in ``on_bar``::

        def on_bar(self, ...):
            self.bars.update(...)
            spread = int(self.bars.noncomm_net[-1]) - int(self.bars.comm_net[-1])
            self.cot_index(spread)
            return self._run_indicators_and_next({...})
    """

    _FIELDS = Indicator._FIELDS + [("_spread", np.float64)]

    _spread: NDArray[np.float64]

    def __init__(self, lookback: Literal[26, 52, 156] = 26):
        super().__init__()
        self._lookback = lookback

    def __call__(self, spread: float) -> "COTIndex":  # type: ignore[override]
        """Update the indicator with the latest spread value and return itself.
        
        Args:
            spread: The current net speculator position (noncomm_net - comm_net)
        """
        n = self._size
        if n >= self._lookback - 1:
            window = np.empty(self._lookback)
            window[:-1] = self._spread[n - self._lookback + 1 : n]
            window[-1] = spread
            lo, hi = window.min(), window.max()
            rng = hi - lo
            result = (spread - lo) / rng * 100.0 if rng != 0.0 else 50.0
        else:
            result = np.nan

        super().update(_value=result, _spread=spread)
        return self
