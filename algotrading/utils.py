import json
from datetime import datetime, time, timezone
from itertools import compress
from pathlib import Path
from typing import Callable, Literal, Union

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


class MonthlyDrawdownTracker:
    """Track peak equity and max drawdown within each calendar month.

    Pass ``equity_source`` as a zero-arg callable returning current equity.
    For multi-strategy accounts, use ``lambda: strategy.strategy_equity``
    (strategy-scoped — ignores other strategies' PnL); otherwise
    ``lambda: strategy.equity`` reads the whole-account view.

    Call :meth:`update` once per bar with the bar time; read :meth:`breached`
    to gate new entries.

    Pass ``state_path`` to persist state to disk (JSON) on every update so
    drawdown history survives restarts within the same month. Between a
    backtest warmup and a live run, call :meth:`reset` so the warmup's
    synthetic equity history doesn't poison the live session.
    """

    def __init__(
        self,
        equity_source: Callable[[], float],
        state_path: str | Path | None = None,
    ):
        self._equity_source = equity_source
        self._state_path = Path(state_path) if state_path is not None else None
        self._month_key: tuple[int, int] | None = None
        self._peak: float = 0.0
        self._max_dd_pct: float = 0.0
        if self._state_path is not None:
            self._load_state()

    def update(self, now: datetime) -> None:
        key = (now.year, now.month)
        equity = self._equity_source()
        if key != self._month_key:
            self._month_key = key
            self._peak = equity
            self._max_dd_pct = 0.0
            self._save_state()
            return
        if equity > self._peak:
            self._peak = equity
        if self._peak > 0.0:
            dd_pct = (self._peak - equity) / self._peak * 100.0
            if dd_pct > self._max_dd_pct:
                self._max_dd_pct = dd_pct
        self._save_state()

    @property
    def max_dd_pct(self) -> float:
        return self._max_dd_pct

    def breached(self, limit_pct: float) -> bool:
        return self._max_dd_pct > limit_pct

    def reset(self) -> None:
        """Clear in-memory state and remove any persisted state file.

        Call between warmup and live so the live session starts from a clean
        slate rather than inheriting the backtest broker's synthetic peak.
        """
        self._month_key = None
        self._peak = 0.0
        self._max_dd_pct = 0.0
        if self._state_path is not None and self._state_path.exists():
            self._state_path.unlink()

    def _load_state(self) -> None:
        assert self._state_path is not None
        if not self._state_path.exists():
            return
        try:
            data = json.loads(self._state_path.read_text(encoding="utf-8"))
            month = data.get("month")
            if month is not None and len(month) == 2:
                self._month_key = (int(month[0]), int(month[1]))
            self._peak = float(data.get("peak", 0.0))
            self._max_dd_pct = float(data.get("max_dd_pct", 0.0))
        except (json.JSONDecodeError, ValueError, KeyError, TypeError, OSError):
            self._month_key = None
            self._peak = 0.0
            self._max_dd_pct = 0.0

    def _save_state(self) -> None:
        if self._state_path is None:
            return
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "month": list(self._month_key) if self._month_key is not None else None,
            "peak": self._peak,
            "max_dd_pct": self._max_dd_pct,
        }
        self._state_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
