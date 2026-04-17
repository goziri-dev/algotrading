from datetime import datetime, timedelta, timezone

import numpy as np

from algotrading.backtest.backtest_broker import BacktestBroker, ClosedTrade
from algotrading.backtest.monte_carlo import simulate_monte_carlo_from_broker
from algotrading.backtest.plotting import plot_monte_carlo_paths
from algotrading.core.position import Position
from algotrading.core.signal import SignalDirection


def _sample_broker() -> BacktestBroker:
    broker = BacktestBroker(initial_balance=1_000.0)
    t0 = datetime(2024, 1, 1, tzinfo=timezone.utc)

    pos1 = Position(
        id=1,
        symbol="XAUUSD",
        strategy_id=1,
        direction=SignalDirection.LONG,
        qty=1.0,
        entry_time=t0,
        entry_price=2_000.0,
    )
    pos2 = Position(
        id=2,
        symbol="XAUUSD",
        strategy_id=1,
        direction=SignalDirection.SHORT,
        qty=1.0,
        entry_time=t0 + timedelta(days=1),
        entry_price=2_010.0,
    )

    broker.trade_log = [
        ClosedTrade(
            position=pos1,
            exit_time=t0 + timedelta(days=1),
            exit_price=2_010.0,
            pnl=50.0,
            spread_cost=2.0,
            slippage_cost=1.0,
            commission_cost=1.0,
        ),
        ClosedTrade(
            position=pos2,
            exit_time=t0 + timedelta(days=2),
            exit_price=2_000.0,
            pnl=-30.0,
            spread_cost=2.0,
            slippage_cost=1.0,
            commission_cost=1.0,
        ),
    ]
    return broker


def test_simulate_monte_carlo_from_broker_is_seeded_and_shaped() -> None:
    broker = _sample_broker()
    report_a = simulate_monte_carlo_from_broker(broker, n_paths=100, horizon_trades=5, random_state=7)
    report_b = simulate_monte_carlo_from_broker(broker, n_paths=100, horizon_trades=5, random_state=7)

    assert report_a.paths.shape == (100, 6)
    assert report_a.sampled_returns.shape == (100, 5)
    assert np.allclose(report_a.paths, report_b.paths)


def test_plot_monte_carlo_paths_smoke() -> None:
    broker = _sample_broker()
    report = simulate_monte_carlo_from_broker(broker, n_paths=120, horizon_trades=8, random_state=11)

    fig, ax = plot_monte_carlo_paths(report)

    assert fig is not None
    assert ax is not None
    assert len(ax.lines) >= 2
