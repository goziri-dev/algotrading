from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Sequence

from algotrading.core.strategy import Strategy

from .backtest_broker import BacktestBroker
from .summary import BacktestStats, calculate_backtest_stats


@dataclass(frozen=True)
class BacktestReport:
    broker: BacktestBroker
    strategy: Strategy[Any]
    stats: BacktestStats
    symbol: str
    price_times: Sequence[datetime] | Sequence[Any]
    price_open: Sequence[float] | Sequence[Any]
    price_high: Sequence[float] | Sequence[Any]
    price_low: Sequence[float] | Sequence[Any]
    price_close: Sequence[float] | Sequence[Any]
    benchmark_times: Sequence[datetime] | None = None
    benchmark_prices: Sequence[float] | None = None
    benchmark_name: str = "Benchmark"

    @classmethod
    def from_strategy(
        cls,
        *,
        broker: BacktestBroker,
        strategy: Strategy[Any],
        benchmark_times: Sequence[datetime] | None = None,
        benchmark_prices: Sequence[float] | None = None,
        benchmark_name: str = "Benchmark",
    ) -> "BacktestReport":
        return cls(
            broker=broker,
            strategy=strategy,
            stats=calculate_backtest_stats(broker),
            symbol=strategy.symbol,
            price_times=strategy.bars.time,
            price_open=strategy.bars.open,
            price_high=strategy.bars.high,
            price_low=strategy.bars.low,
            price_close=strategy.bars.close,
            benchmark_times=benchmark_times,
            benchmark_prices=benchmark_prices,
            benchmark_name=benchmark_name,
        )
