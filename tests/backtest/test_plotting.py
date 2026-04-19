from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

from algotrading.backtest.backtest_broker import BacktestBroker, ClosedTrade, EquityPoint
from algotrading.backtest.plotting import (
    plot_drawdown,
    plot_equity_and_drawdown,
    plot_equity_curve,
    plot_trade_pnl_distribution,
    plot_equity_vs_benchmark,
    plot_equity_vs_symbols,
    plot_symbol_return_correlation,
    plot_monthly_returns_heatmap,
    plot_price_with_trades,
    plot_price_with_trades_interactive,
    save_swap_history_interactive_html,
    save_trade_history_interactive_html,
)
from algotrading.core.position import Position
from algotrading.core.signal import SignalDirection


def _sample_broker() -> BacktestBroker:
    broker = BacktestBroker(initial_balance=1_000)
    t0 = datetime(2024, 1, 1, tzinfo=timezone.utc)

    broker.equity_curve = [
        EquityPoint(time=t0 + timedelta(days=0), equity=1_000.0, balance=1_000.0),
        EquityPoint(time=t0 + timedelta(days=1), equity=1_020.0, balance=1_010.0),
        EquityPoint(time=t0 + timedelta(days=2), equity=980.0, balance=990.0),
        EquityPoint(time=t0 + timedelta(days=3), equity=1_050.0, balance=1_030.0),
    ]

    pos_long = Position(
        id=1,
        symbol="XAUUSD",
        strategy_id=1,
        direction=SignalDirection.LONG,
        qty=1.0,
        entry_time=t0 + timedelta(days=1),
        entry_price=2_000.0,
    )
    pos_short = Position(
        id=2,
        symbol="XAUUSD",
        strategy_id=1,
        direction=SignalDirection.SHORT,
        qty=1.0,
        entry_time=t0 + timedelta(days=2),
        entry_price=2_010.0,
    )

    broker.trade_log = [
        ClosedTrade(
            position=pos_long,
            exit_time=t0 + timedelta(days=2),
            exit_price=2_008.0,
            pnl=8.0,
            spread_cost=1.0,
            slippage_cost=1.0,
            commission_cost=0.0,
        ),
        ClosedTrade(
            position=pos_short,
            exit_time=t0 + timedelta(days=3),
            exit_price=2_020.0,
            pnl=-10.0,
            spread_cost=1.0,
            slippage_cost=1.0,
            commission_cost=0.0,
        ),
    ]

    return broker


def test_plot_equity_curve_smoke():
    broker = _sample_broker()
    fig, ax = plot_equity_curve(broker)
    assert fig is not None
    assert ax is not None
    assert len(ax.lines) >= 1


def test_plot_drawdown_smoke():
    broker = _sample_broker()
    fig, ax = plot_drawdown(broker)
    assert fig is not None
    assert ax is not None


def test_plot_equity_vs_benchmark_smoke():
    broker = _sample_broker()
    bt = [point.time for point in broker.equity_curve]
    bp = [100.0, 101.0, 99.0, 102.0]
    fig, ax = plot_equity_vs_benchmark(broker, benchmark_times=bt, benchmark_prices=bp)
    assert fig is not None
    assert ax is not None
    assert len(ax.lines) == 2


def test_plot_equity_vs_symbols_smoke():
    broker = _sample_broker()
    times = [point.time for point in broker.equity_curve]
    series = {
        "XAUUSD": (times, [100.0, 101.0, 103.0, 104.0]),
        "XAGUSD": (times, [50.0, 49.0, 51.0, 52.0]),
    }
    fig, ax = plot_equity_vs_symbols(broker, series)
    assert fig is not None
    assert ax is not None
    assert len(ax.lines) == 3


def test_plot_symbol_return_correlation_smoke():
    broker = _sample_broker()
    times = [point.time for point in broker.equity_curve]
    series = {
        "XAUUSD": (times, [100.0, 102.0, 101.0, 104.0]),
        "XAGUSD": (times, [50.0, 51.0, 52.0, 53.0]),
    }
    fig, ax = plot_symbol_return_correlation(series)
    assert fig is not None
    assert ax is not None
    assert len(ax.images) == 1


def test_plot_price_with_trades_smoke():
    broker = _sample_broker()
    times = [point.time for point in broker.equity_curve]
    prices = [2_000.0, 2_006.0, 2_004.0, 2_018.0]
    fig, ax = plot_price_with_trades(times, prices, broker.trade_log, symbol="XAUUSD")
    assert fig is not None
    assert ax is not None


def test_plot_equity_and_drawdown_smoke():
    broker = _sample_broker()
    fig, axes = plot_equity_and_drawdown(broker)
    assert fig is not None
    assert len(axes) == 2


def test_plot_monthly_returns_heatmap_smoke():
    broker = _sample_broker()
    fig, ax = plot_monthly_returns_heatmap(broker)
    assert fig is not None
    assert ax is not None
    assert len(ax.images) == 1
    assert any("%" in text.get_text() for text in ax.texts)


def test_plot_trade_pnl_distribution_smoke():
    broker = _sample_broker()
    fig, ax = plot_trade_pnl_distribution(broker)
    assert fig is not None
    assert ax is not None
    assert len(ax.patches) > 0
    assert ax.get_xlabel() == "Trade PnL (%)"
    legend = ax.get_legend()
    assert legend is not None
    labels = [text.get_text() for text in legend.get_texts()]
    assert any(label.startswith("Mean ") and "% (" in label for label in labels)
    assert any(label.startswith("Median ") and "% (" in label for label in labels)


def test_plot_price_with_trades_interactive_uses_candles_when_ohlc_provided():
    broker = _sample_broker()
    times = [point.time for point in broker.equity_curve]
    open_prices = [2_000.0, 2_010.0, 2_005.0, 2_012.0]
    high_prices = [2_015.0, 2_020.0, 2_018.0, 2_025.0]
    low_prices = [1_995.0, 2_000.0, 1_998.0, 2_000.0]
    close_prices = [2_006.0, 2_004.0, 2_018.0, 2_014.0]

    fig = plot_price_with_trades_interactive(
        price_times=times,
        prices=close_prices,
        trades=broker.trade_log,
        symbol="XAUUSD",
        price_open=open_prices,
        price_high=high_prices,
        price_low=low_prices,
        price_close=close_prices,
    )

    assert fig is not None
    assert fig.metadata["engine"] == "lightweight-charts"
    assert fig.metadata["mode"] == "candlestick"
    assert "lightweight-charts" in fig.html


def test_save_trade_history_interactive_html_creates_file(tmp_path: Path):
    broker = _sample_broker()
    output_path = save_trade_history_interactive_html(
        trades=broker.trade_log,
        output_path=tmp_path / "trade_history.html",
        title="Trade History Test",
    )

    assert output_path.exists()
    content = output_path.read_text(encoding="utf-8")
    assert "Trade History Test" in content
    assert "Trade #" in content
    assert "Gross PnL" in content
    assert "Net PnL" in content
    assert "Total Costs" in content
    assert "Swap" in content
    assert "<span style=" not in content


def test_save_swap_history_interactive_html_creates_file(tmp_path: Path):
    broker = _sample_broker()
    output_path = save_swap_history_interactive_html(
        trades=broker.trade_log,
        output_path=tmp_path / "swap_history.html",
        title="Swap History Test",
    )

    assert output_path.exists()
    content = output_path.read_text(encoding="utf-8")
    assert "Swap History Test" in content
    assert "total_swap=" in content


def test_plot_price_with_trades_interactive_preserves_marker_times():
    broker = _sample_broker()
    times = [point.time for point in broker.equity_curve]
    close_prices = [2_006.0, 2_004.0, 2_018.0, 2_014.0]

    shifted_trades = [
        ClosedTrade(
            position=Position(
                id=trade.position.id,
                symbol=trade.position.symbol,
                strategy_id=trade.position.strategy_id,
                direction=trade.position.direction,
                qty=trade.position.qty,
                entry_time=trade.position.entry_time + timedelta(hours=6),
                entry_price=trade.position.entry_price,
            ),
            exit_time=trade.exit_time + timedelta(hours=6),
            exit_price=trade.exit_price,
            pnl=trade.pnl,
            spread_cost=trade.spread_cost,
            slippage_cost=trade.slippage_cost,
            commission_cost=trade.commission_cost,
        )
        for trade in broker.trade_log
    ]

    fig = plot_price_with_trades_interactive(
        price_times=times,
        prices=close_prices,
        trades=shifted_trades,
        symbol="XAUUSD",
        render_mode="line",
    )

    expected_times = [
        trade.position.entry_time
        for trade in shifted_trades
        if trade.position.is_long
    ]
    assert fig.metadata["mode"] == "line"
    assert fig.metadata["marker_times"]["long_entry"] == expected_times


def test_plot_price_with_trades_interactive_uses_closest_hover_and_trade_labels():
    broker = _sample_broker()
    times = [point.time for point in broker.equity_curve]
    close_prices = [2_006.0, 2_004.0, 2_018.0, 2_014.0]

    fig = plot_price_with_trades_interactive(
        price_times=times,
        prices=close_prices,
        trades=broker.trade_log,
        symbol="XAUUSD",
        render_mode="line",
    )

    assert fig.metadata["engine"] == "lightweight-charts"
    assert fig.metadata["title"].startswith("XAUUSD")
    assert "PnL" in fig.html


def test_plot_price_with_trades_interactive_staggered_layout_offsets_markers_from_candles():
    broker = _sample_broker()
    times = [point.time for point in broker.equity_curve]
    open_prices = [2_000.0, 2_010.0, 2_005.0, 2_012.0]
    high_prices = [2_015.0, 2_020.0, 2_018.0, 2_025.0]
    low_prices = [1_995.0, 2_000.0, 1_998.0, 2_000.0]
    close_prices = [2_006.0, 2_004.0, 2_018.0, 2_014.0]

    fig = plot_price_with_trades_interactive(
        price_times=times,
        prices=close_prices,
        trades=broker.trade_log,
        symbol="XAUUSD",
        price_open=open_prices,
        price_high=high_prices,
        price_low=low_prices,
        price_close=close_prices,
        render_mode="candlestick",
        marker_layout="staggered",
    )

    assert fig.metadata["marker_layout"] == "staggered"
    assert fig.metadata["marker_positions"]["entry"] == "above"
    assert fig.metadata["marker_positions"]["exit"] == "below"


def test_plot_price_with_trades_interactive_infers_rangebreaks_for_missing_bars():
    broker = _sample_broker()
    t0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
    times = [
        t0,
        t0 + timedelta(minutes=15),
        t0 + timedelta(minutes=30),
        t0 + timedelta(minutes=60),
    ]
    close_prices = [2_000.0, 2_001.0, 2_002.0, 2_003.0]

    fig = plot_price_with_trades_interactive(
        price_times=times,
        prices=close_prices,
        trades=broker.trade_log,
        symbol="XAUUSD",
        render_mode="line",
    )

    assert fig.metadata["rangebreak_count"] >= 1


def test_plot_price_with_trades_interactive_renders_overlay_and_panel_indicators():
    broker = _sample_broker()
    times = [point.time for point in broker.equity_curve]
    close_prices = [2_006.0, 2_004.0, 2_018.0, 2_014.0]
    overlays = [
        {
            "name": "Fast SMA",
            "values": np.array([2_005.0, 2_006.0, 2_007.0, 2_008.0], dtype=np.float64),
            "color": "#1d4ed8",
        }
    ]
    panels = [
        {
            "title": "ATR",
            "traces": [
                {
                    "name": "ATR",
                    "values": np.array([1.0, 1.1, 1.2, 1.3], dtype=np.float64),
                    "color": "#b45309",
                }
            ],
        }
    ]

    fig = plot_price_with_trades_interactive(
        price_times=times,
        prices=close_prices,
        trades=broker.trade_log,
        symbol="XAUUSD",
        render_mode="line",
        indicator_overlays=overlays,
        indicator_panels=panels,
    )

    assert fig.metadata["overlay_indicators"] == ["Fast SMA"]
    assert fig.metadata["panel_indicators"] == ["ATR"]
    assert "Fast SMA" in fig.html
    assert "ATR" in fig.html


