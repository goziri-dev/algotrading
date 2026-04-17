from datetime import datetime, timezone

import pytest

from algotrading.backtest.backtest_broker import BacktestBroker
from algotrading.core.broker import BrokerView
from algotrading.core.feed import feed_bars, mt5_timeframe_duration
from algotrading.core.strategy import Strategy, StrategyParams
from algotrading.indicators import SMA


def _bar(ts: int, close: float) -> dict[str, float | int]:
    return {
        "time": ts,
        "open": close,
        "high": close,
        "low": close,
        "close": close,
    }


class _ProbeStrategy(Strategy[StrategyParams]):
    def __init__(self, symbol: str):
        super().__init__(symbol)
        self.h4 = self.new_bars(240)
        self.fast = self.I(SMA(1), source="close")
        self.trend = self.I(SMA(1), source="close", bars=self.h4)
        self.samples: list[tuple[datetime, float]] = []

    def next(self) -> None:
        t = self.bars.time[-1].astype("datetime64[s]").astype("int64")
        dt = datetime.fromtimestamp(int(t), tz=timezone.utc)
        self.samples.append((dt, float(self.trend[-1])))


def _attach_broker(strategy: Strategy[StrategyParams]) -> None:
    BrokerView(BacktestBroker(initial_balance=10_000), strategy)


def test_secondary_bar_visible_when_it_closes_within_primary_bar() -> None:
    """Secondary bars closing by primary close should be visible for that bar's decision."""
    s = _ProbeStrategy("XAUUSD")
    _attach_broker(s)

    # Primary M15 bars at t=0 and t=900.
    primary = [_bar(0, 100.0), _bar(900, 101.0)]

    # H4 bar closes exactly at t=900: open=-13_500, duration=14_400.
    secondary = {240: [_bar(-13_500, 200.0)]}

    feed_bars("XAUUSD", [s], primary, secondary, primary_duration=900)

    assert len(s.samples) == 2
    assert s.samples[0][1] == pytest.approx(200.0)


def test_secondary_bar_not_visible_before_it_has_closed() -> None:
    """No look-ahead: secondary bar stays hidden until its close time."""
    s = _ProbeStrategy("XAUUSD")
    _attach_broker(s)

    # Primary M15 bars at t=0 and t=900.
    primary = [_bar(0, 100.0), _bar(900, 101.0)]

    # H4 bar closes at t=1_800 (> first primary close at t=900).
    secondary = {240: [_bar(-12_600, 300.0)]}

    feed_bars("XAUUSD", [s], primary, secondary, primary_duration=900)

    assert len(s.samples) == 1
    assert s.samples[0][1] == pytest.approx(300.0)


def test_mt5_timeframe_duration_handles_hourly_enum_constants() -> None:
    # MT5 H4 enum value is 16388, not 240.
    assert mt5_timeframe_duration(16388) == 14_400
    assert mt5_timeframe_duration(16385) == 3_600


def test_mt5_timeframe_duration_handles_string_timeframes() -> None:
    assert mt5_timeframe_duration("H4") == 14_400
    assert mt5_timeframe_duration("D1") == 86_400
