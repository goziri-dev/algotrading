from datetime import datetime, time, timezone
from itertools import compress
from typing import Literal, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from algotrading.indicators.indicator import Indicator

# Supported input types for crossover / crossunder
_Numeric = Union[Indicator, "NDArray[np.float64]", pd.Series, float, int]


def _last_two(x: _Numeric) -> tuple[float, float]:
    """Return (prev_bar, curr_bar) from any supported numeric type.

    A scalar is treated as a constant — both slots return the same value.
    """
    if isinstance(x, (int, float)):
        v = float(x)
        return v, v
    if isinstance(x, pd.Series):
        if len(x) < 2:
            return np.nan, np.nan
        return float(x.iloc[-2]), float(x.iloc[-1])
    # Indicator or ndarray — both support negative integer indexing
    if len(x) < 2:
        return np.nan, np.nan
    return float(x[-2]), float(x[-1])


def crossover(
        a: Union[Indicator, "NDArray[np.float64]", pd.Series, float, int], 
        b: Union[Indicator, "NDArray[np.float64]", pd.Series, float, int]
    ) -> bool:
    """Return True if `a` crossed above `b` on the last bar.

    Accepts any combination of Indicator, numpy array, pandas Series,
    or a scalar float/int (treated as a flat constant across all bars).
    Returns False when either input has fewer than two data points.
    """
    a_prev, a_curr = _last_two(a)
    b_prev, b_curr = _last_two(b)
    if any(np.isnan(v) for v in (a_prev, a_curr, b_prev, b_curr)):
        return False
    return a_prev < b_prev and a_curr > b_curr


def crossunder(
        a: Union[Indicator, "NDArray[np.float64]", pd.Series, float, int], 
        b: Union[Indicator, "NDArray[np.float64]", pd.Series, float, int]
    ) -> bool:
    """Return True if `a` crossed below `b` on the last bar.

    Accepts any combination of Indicator, numpy array, pandas Series,
    or a scalar float/int (treated as a flat constant across all bars).
    Returns False when either input has fewer than two data points.
    """
    a_prev, a_curr = _last_two(a)
    b_prev, b_curr = _last_two(b)
    if any(np.isnan(v) for v in (a_prev, a_curr, b_prev, b_curr)):
        return False
    return a_prev > b_prev and a_curr < b_curr


def bars_since(
        condition: Union[Indicator, "NDArray[np.bool_]", pd.Series, list[bool]], 
        default: float = np.inf
    ) -> int | float:
    """Return the number of bars since `condition` was last True.

    Returns `default` (``np.inf``) if the condition has never been True.
    Index 0 means the current (most recent) bar.

    Accepts Indicator, numpy bool array, pandas Series, or a plain bool list.
    When an Indicator is passed its ``.value`` slice is used directly.

        >>> bars_since([False, True, False, False])
        2
        >>> bars_since([False, False, False])
        inf
    """
    if isinstance(condition, Indicator):
        seq: NDArray[np.bool_] = condition.value.astype(bool)
    elif isinstance(condition, pd.Series):
        seq = condition.to_numpy(dtype=bool)
    else:
        seq = np.asarray(condition, dtype=bool)
    return next(compress(range(len(seq)), seq[::-1]), default)


def risk_to_qty(
    equity: float,
    risk_pct: float,
    entry: float,
    stop_loss: float,
    value_per_point: float,
) -> float:
    """Calculate quantity based on how much of the equity you're willing to risk.

    ``value_per_point`` is required so sizing stays symbol-aware.
    risk_pct is a percentage value: 1.0 = 1%, 2.5 = 2.5%

    Example: equity=10_000, risk_pct=1.0, entry=95_000, stop_loss=94_500,
    value_per_point=1.0 → risk_amount = 100, stop_distance = 500, quantity = 0.2
    """
    if entry == stop_loss:
        raise ValueError("Entry and stop loss cannot be the same price")
    if risk_pct <= 0 or risk_pct >= 100:
        raise ValueError("risk_pct must be between 0 and 100 (exclusive)")
    if value_per_point <= 0:
        raise ValueError("value_per_point must be positive")
    risk_amount = equity * (risk_pct / 100)
    stop_distance = abs(entry - stop_loss)
    return risk_amount / (stop_distance * value_per_point)


def fraction_to_qty(
    equity: float,
    fraction_pct: float,
    entry: float,
    value_per_point: float,
) -> float:
    """Calculate quantity as a fraction of account equity divided by entry price.

    ``value_per_point`` is required so notional sizing stays symbol-aware.
    fraction_pct is a percentage value: 10.0 = 10%

    Example: equity=10_000, fraction_pct=10.0, entry=95_000,
    value_per_point=1.0 → notional = 1_000, quantity = 0.0105

    Raises: ValueError if fraction_pct is not between 0 and 100,
        or if entry or value_per_point are not positive.
    """
    if fraction_pct <= 0 or fraction_pct >= 100:
        raise ValueError("fraction_pct must be between 0 and 100 (exclusive)")
    if entry <= 0:
        raise ValueError("entry price must be positive")
    if value_per_point <= 0:
        raise ValueError("value_per_point must be positive")
    return (equity * (fraction_pct / 100)) / (entry * value_per_point)


def session(
    at: datetime | time | np.datetime64 | pd.Timestamp,
) -> Literal["sydney", "tokyo", "london", "new_york"]:
    """Map a UTC time to a major FX trading session.

    Session boundaries are fixed in UTC and deterministic across DST:
    - tokyo: 00:00-08:59
    - london: 08:00-16:59
    - new_york: 13:00-21:59
    - sydney: all remaining times
    """
    if isinstance(at, np.datetime64):
        at = pd.Timestamp(at).to_pydatetime()
    elif isinstance(at, pd.Timestamp):
        at = at.to_pydatetime()

    if isinstance(at, datetime):
        if at.tzinfo is not None:
            hour = at.astimezone(timezone.utc).hour
        else:
            hour = at.hour
    elif isinstance(at, time):
        hour = at.hour
    else:
        raise TypeError(f"Unsupported type for session(): {type(at).__name__}")

    if 13 <= hour < 22:
        return "new_york"
    if 8 <= hour < 17:
        return "london"
    if 0 <= hour < 9:
        return "tokyo"
    return "sydney"
