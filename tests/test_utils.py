import pytest
from datetime import datetime, time, timezone, timedelta
import numpy as np
import pandas as pd

from algotrading.utils import fraction_to_qty, risk_to_qty, session


def test_risk_to_qty_uses_value_per_point() -> None:
    qty = risk_to_qty(
        equity=2_730.0,
        risk_pct=1.0,
        entry=2_062.47,
        stop_loss=2_060.97,
        value_per_point=0.8528,
    )
    assert qty == pytest.approx(21.34146341463415)


def test_fraction_to_qty_uses_value_per_point() -> None:
    qty = fraction_to_qty(
        equity=10_000.0,
        fraction_pct=10.0,
        entry=2_000.0,
        value_per_point=0.5,
    )
    assert qty == pytest.approx(1.0)


def test_risk_to_qty_rejects_nonpositive_value_per_point() -> None:
    with pytest.raises(ValueError, match="value_per_point must be positive"):
        risk_to_qty(10_000.0, 1.0, 100.0, 99.0, value_per_point=0.0)


def test_fraction_to_qty_rejects_nonpositive_value_per_point() -> None:
    with pytest.raises(ValueError, match="value_per_point must be positive"):
        fraction_to_qty(10_000.0, 10.0, 100.0, value_per_point=0.0)


def test_session_classifies_utc_time_boundaries() -> None:
    assert session(time(0, 0)) == "tokyo"
    assert session(time(8, 0)) == "london"
    assert session(time(13, 0)) == "new_york"
    assert session(time(22, 0)) == "sydney"


def test_session_uses_utc_for_timezone_aware_datetimes() -> None:
    utc_plus_2 = timezone(timedelta(hours=2))
    # 10:00 in UTC+2 is 08:00 UTC -> london
    at = datetime(2026, 4, 15, 10, 0, tzinfo=utc_plus_2)
    assert session(at) == "london"


def test_session_accepts_numpy_datetime64() -> None:
    at = np.datetime64("2026-04-15T14:00:00")
    assert session(at) == "new_york"


def test_session_accepts_pandas_timestamp() -> None:
    at = pd.Timestamp("2026-04-15 06:30:00")
    assert session(at) == "tokyo"