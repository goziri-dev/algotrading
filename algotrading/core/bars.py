from datetime import datetime, timezone

import numpy as np
from numpy.typing import NDArray

from .growable import Growable


class Bars(Growable):
    _FIELDS = [
        ("_open",  np.float64),
        ("_high",  np.float64),
        ("_low",   np.float64),
        ("_close", np.float64),
        ("_time",  "datetime64[ns]"),
    ]

    _open:  NDArray[np.float64]
    _high:  NDArray[np.float64]
    _low:   NDArray[np.float64]
    _close: NDArray[np.float64]
    _time:  NDArray[np.datetime64]

    def __init__(self):
        super().__init__(fields=self._FIELDS)

    @property
    def open(self):
        return self._open[:self._size]

    @property
    def high(self):
        return self._high[:self._size]

    @property
    def low(self):
        return self._low[:self._size]

    @property
    def close(self):
        return self._close[:self._size]

    @property
    def time(self):
        return self._time[:self._size]

    def __getitem__(self, name: str) -> np.ndarray:
        """Access a dynamically added field by its public name.

        Example::

            bars.add_field("noncomm_net", int)
            bars["noncomm_net"]  # returns sliced array up to current size
        """
        attr = f"_{name}"
        if not hasattr(self, attr):
            raise KeyError(f"No field '{name}' on {type(self).__name__}")
        return getattr(self, attr)[:self._size]

    def append(self, **extra) -> None:  # type: ignore[override]
        """Mutate extra fields on the last appended row in place.

        Accepts public field names (without leading underscore)::

            bars.append(noncomm_net=500, spread=0.0003)
        """
        super().append(**{f'_{k}': v for k, v in extra.items()})

    _BASE_FIELD_NAMES: frozenset[str] = frozenset(name for name, _ in _FIELDS)

    def update(self, time: datetime | np.datetime64, open: float, high: float, low: float, close: float, **extra) -> None:  # type: ignore[override]
        """Add a new bar to the arrays, increasing capacity if needed.

        Extra keyword arguments are forwarded as dynamic fields using their
        public names (without leading underscore).

        Dynamic fields registered via ``add_field()`` that are not provided in
        ``extra`` are forward-filled from the previous bar's value. On the very
        first bar (no previous value available), they fall back to ``nan``
        (float fields) or ``0`` (integer fields).

        **WARNING!** If capacity is increased, the underlying arrays are replaced
        with new ones, so any references to the old arrays will become stale.
        """
        if isinstance(time, datetime) and time.tzinfo is not None:
            time = time.astimezone(timezone.utc).replace(tzinfo=None)
        super().update(
            _time=time,
            _open=open,
            _high=high,
            _low=low,
            _close=close,
            **{f'_{k}': v for k, v in extra.items()}
        )
        # Forward-fill any dynamic fields not provided in this call.
        # If a previous value exists, carry it forward; otherwise fall back to nan/0.
        if len(self._fields) > len(self._BASE_FIELD_NAMES):
            provided = frozenset(f'_{k}' for k in extra)
            for name, _ in self._fields:
                if name not in self._BASE_FIELD_NAMES and name not in provided:
                    arr = getattr(self, name)
                    if self._size >= 2:
                        arr[self._size - 1] = arr[self._size - 2]
                    else:
                        arr[self._size - 1] = np.nan if np.issubdtype(arr.dtype, np.floating) else 0
