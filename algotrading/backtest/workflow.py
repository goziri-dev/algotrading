from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Any, Sequence

import MetaTrader5 as mt5

from algotrading.core.strategy import Strategy

from .backtest_session import BacktestSession
from .plotter import BacktestPlotter
from .plotting import (
    plot_equity_vs_symbols,
    plot_monthly_returns_heatmap,
    plot_symbol_return_correlation,
    save_swap_history_interactive_html,
    save_trade_history_interactive_html,
)
from .report import BacktestReport


def _try_fetch_benchmark(
    symbol: str,
    timeframe: int,
    date_from: datetime | None = None,
    date_to: datetime | None = None,
    primary_count: int | None = None,
) -> tuple[list[datetime], list[float]] | None:
    if primary_count is not None:
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 1, primary_count)  # type: ignore
    elif date_from is not None and date_to is not None:
        rates = mt5.copy_rates_range(symbol, timeframe, date_from, date_to)  # type: ignore
    else:
        return None

    if rates is None or len(rates) == 0:
        return None

    times = [datetime.fromtimestamp(int(row["time"])) for row in rates]
    closes = [float(row["close"]) for row in rates]
    return times, closes


def _as_python_datetime(value: object) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime.combine(value, datetime.min.time())
    item = getattr(value, "item", None)
    if callable(item):
        converted = item()
        if isinstance(converted, datetime):
            return converted
        if isinstance(converted, date):
            return datetime.combine(converted, datetime.min.time())
    return None


def save_backtest_plots(
    bt: BacktestSession,
    strategy: Strategy[Any] | Sequence[Strategy[Any]],
    primary_tf: int,
    date_from: datetime | None,
    date_to: datetime | None,
    output_dir: Path,
    primary_count: int | None = None,
    benchmark_symbol: str | None = None,
    benchmark_name: str | None = None,
) -> None:
    strategies = [strategy] if isinstance(strategy, Strategy) else list(strategy)
    if not strategies:
        raise ValueError("At least one strategy is required to save plots.")

    if len(strategies) == 1:
        _save_single_strategy_plots(
            bt=bt,
            strategy=strategies[0],
            primary_tf=primary_tf,
            date_from=date_from,
            date_to=date_to,
            output_dir=output_dir,
            primary_count=primary_count,
            benchmark_symbol=benchmark_symbol,
            benchmark_name=benchmark_name,
        )
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    symbol_total: dict[str, int] = {}
    for item in strategies:
        symbol_total[item.symbol] = symbol_total.get(item.symbol, 0) + 1
    symbol_seen: dict[str, int] = {}

    for item in strategies:
        symbol_seen[item.symbol] = symbol_seen.get(item.symbol, 0) + 1
        suffix = (
            f"_{symbol_seen[item.symbol]:02d}"
            if symbol_total[item.symbol] > 1
            else ""
        )
        strategy_dir = output_dir / f"strategy_{item.symbol}{suffix}"
        _save_single_strategy_plots(
            bt=bt,
            strategy=item,
            primary_tf=primary_tf,
            date_from=date_from,
            date_to=date_to,
            output_dir=strategy_dir,
            primary_count=primary_count,
            benchmark_symbol=item.symbol,
            benchmark_name=f"{item.symbol} B&H",
        )

    _save_multi_symbol_unified_outputs(
        bt=bt,
        strategies=strategies,
        output_dir=output_dir,
    )


def _save_single_strategy_plots(
    *,
    bt: BacktestSession,
    strategy: Strategy[Any],
    primary_tf: int,
    date_from: datetime | None,
    date_to: datetime | None,
    output_dir: Path,
    primary_count: int | None,
    benchmark_symbol: str | None,
    benchmark_name: str | None,
) -> None:
    broker = bt.broker

    if not broker.equity_curve:
        return

    has_strategy_times = len(strategy.bars.time) > 0
    strategy_start = _as_python_datetime(strategy.bars.time[0]) if has_strategy_times else None
    strategy_end = _as_python_datetime(strategy.bars.time[-1]) if has_strategy_times else None

    resolved_benchmark_symbol = benchmark_symbol or strategy.symbol
    resolved_benchmark_name = benchmark_name or f"{resolved_benchmark_symbol} B&H"

    benchmark = _try_fetch_benchmark(
        symbol=resolved_benchmark_symbol,
        timeframe=primary_tf,
        date_from=strategy_start or date_from,
        date_to=strategy_end or date_to,
        primary_count=primary_count,
    )
    benchmark_times = benchmark[0] if benchmark is not None else None
    benchmark_prices = benchmark[1] if benchmark is not None else None

    report = BacktestReport.from_strategy(
        broker=broker,
        strategy=strategy,
        benchmark_times=benchmark_times,
        benchmark_prices=benchmark_prices,
        benchmark_name=resolved_benchmark_name,
    )
    strategy_trades = [
        trade
        for trade in broker.trade_log
        if trade.position.strategy_id == strategy.id
    ]
    BacktestPlotter(report, trades=strategy_trades).save_standard_bundle(
        output_dir=output_dir,
        max_markers_per_group=0,
        max_line_points=0,
    )


def _collect_symbol_series(
    strategies: Sequence[Strategy[Any]],
    symbols: set[str],
) -> dict[str, tuple[Sequence[datetime] | Sequence[Any], Sequence[float] | Sequence[Any]]]:
    series: dict[str, tuple[Sequence[datetime] | Sequence[Any], Sequence[float] | Sequence[Any]]] = {}
    for strategy in strategies:
        if strategy.symbol not in symbols:
            continue
        if len(strategy.bars.time) == 0 or len(strategy.bars.close) == 0:
            continue
        current = series.get(strategy.symbol)
        if current is None or len(strategy.bars.time) > len(current[0]):
            series[strategy.symbol] = (strategy.bars.time, strategy.bars.close)
    return series


def _save_multi_symbol_unified_outputs(
    *,
    bt: BacktestSession,
    strategies: Sequence[Strategy[Any]],
    output_dir: Path,
) -> None:
    broker = bt.broker
    configured_symbols = {strategy.symbol for strategy in strategies}
    traded_symbols = {trade.position.symbol for trade in broker.trade_log}

    if broker.equity_curve:
        fig_monthly, _ = plot_monthly_returns_heatmap(broker)
        fig_monthly.savefig(
            output_dir / "monthly_returns_heatmap.png", dpi=140, bbox_inches="tight"
        )

    symbol_series = _collect_symbol_series(strategies, configured_symbols)
    if len(symbol_series) >= 2:
        fig_vs, _ = plot_equity_vs_symbols(
            broker,
            symbol_series,
            title="Normalized Equity vs Configured Symbols",
        )
        fig_vs.savefig(output_dir / "equity_vs_symbols.png", dpi=140, bbox_inches="tight")

        fig_corr, _ = plot_symbol_return_correlation(
            symbol_series,
            title="Configured Symbol Return Correlation",
        )
        fig_corr.savefig(output_dir / "symbol_return_correlation.png", dpi=140, bbox_inches="tight")

    if len(traded_symbols) < 2:
        return

    unified_trades = sorted(broker.trade_log, key=lambda trade: trade.exit_time)
    if unified_trades:
        save_trade_history_interactive_html(
            trades=unified_trades,
            output_path=output_dir / "trade_history_table_unified.html",
            title="Unified Trade History Table",
        )
        save_swap_history_interactive_html(
            trades=unified_trades,
            output_path=output_dir / "swap_history_table_unified.html",
            title="Unified Swap History Table",
        )
