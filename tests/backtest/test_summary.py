from datetime import datetime, timezone

from algotrading.backtest.backtest_broker import BacktestBroker, ClosedTrade, EquityPoint
from algotrading.backtest.summary import calculate_backtest_stats, print_backtest_summary
from algotrading.core.position import Position
from algotrading.core.signal import SignalDirection


def _ts(value: str) -> datetime:
    return datetime.fromisoformat(value).replace(tzinfo=timezone.utc)


def test_calculate_backtest_stats_includes_daily_ratios() -> None:
    broker = BacktestBroker(initial_balance=1_000.0)
    broker._balance = 1_210.0
    broker.equity_curve = [
        EquityPoint(time=_ts("2024-01-01T12:00:00"), equity=1_100.0, balance=1_100.0),
        EquityPoint(time=_ts("2024-01-02T12:00:00"), equity=1_000.0, balance=1_000.0),
        EquityPoint(time=_ts("2024-01-03T12:00:00"), equity=1_210.0, balance=1_210.0),
    ]

    stats = calculate_backtest_stats(broker)

    assert stats.cagr_pct is not None
    assert stats.sharpe_ratio is not None
    assert stats.sortino_ratio is not None
    assert stats.calmar_ratio is not None
    assert stats.max_drawdown_pct < 0


def test_print_backtest_summary_includes_trade_return_percentages(capsys) -> None:
    broker = BacktestBroker(initial_balance=1_000.0)
    broker._balance = 998.0
    broker.trade_log = [
        ClosedTrade(
            position=Position(
                id=1,
                symbol="XAUUSD",
                strategy_id=1,
                direction=SignalDirection.LONG,
                qty=2.0,
                entry_time=_ts("2024-01-01T12:00:00"),
                entry_price=100.0,
            ),
            exit_time=_ts("2024-01-01T16:00:00"),
            exit_price=103.0,
            pnl=10.0,
            spread_cost=0.5,
            slippage_cost=0.25,
            commission_cost=0.25,
        ),
        ClosedTrade(
            position=Position(
                id=2,
                symbol="XAUUSD",
                strategy_id=1,
                direction=SignalDirection.SHORT,
                qty=2.0,
                entry_time=_ts("2024-01-02T12:00:00"),
                entry_price=100.0,
            ),
            exit_time=_ts("2024-01-02T16:00:00"),
            exit_price=102.0,
            pnl=-12.0,
            spread_cost=0.5,
            slippage_cost=0.25,
            commission_cost=0.25,
        ),
    ]
    broker.equity_curve = [
        EquityPoint(time=_ts("2024-01-01T12:00:00"), equity=1_000.0, balance=1_000.0),
        EquityPoint(time=_ts("2024-01-02T16:00:00"), equity=998.0, balance=998.0),
    ]

    print_backtest_summary(broker)

    output = capsys.readouterr().out
    assert "Final balance   : 998.00" in output
    assert "Avg win         : +10.00 (+5.00%)" in output
    assert "Avg loss        : -12.00 (-6.00%)" in output
    assert "Expectancy/trade: -1.00 (-0.50%)" in output