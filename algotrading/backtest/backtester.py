from collections.abc import Sequence
from typing import Any, Callable

from algotrading.core.feed import feed_bars, feed_bars_aligned, mt5_timeframe_duration
from algotrading.core.strategy import Strategy


class Backtester:
    """Feeds pre-fetched historical bars into one or more strategies.

    The backtester does not own the broker — attach a ``BrokerView`` and
    ``BacktestBroker`` to each strategy before calling ``run()``, then query
    the broker directly for results afterwards::

        broker = BacktestBroker(initial_balance=10_000, symbol_specs={...})
        BrokerView(broker, strategy)

        Backtester([strategy]).run(
            primary_rates={"BTCUSD": m1_rates},
            secondary_rates={"BTCUSD": {mt5.TIMEFRAME_M15: m15_rates}},
        )

        print(broker.trade_log)

    Strategies are grouped by symbol internally; to backtest a multi-symbol
    portfolio pass all strategies in one call so their bars are processed in
    the same pass.

    Note: bars for different symbols are processed symbol-by-symbol, not
    interleaved by timestamp.  True cross-symbol temporal alignment is a
    planned future feature.
    """

    def __init__(
        self,
        strategies: Sequence[Strategy],
        timeframe_duration: Callable[[Any], int] = mt5_timeframe_duration,
    ):
        if not strategies:
            raise ValueError("At least one strategy is required")
        self._strategies = strategies
        self._timeframe_duration = timeframe_duration

        self._by_symbol: dict[str, list[Strategy]] = {}
        for s in strategies:
            self._by_symbol.setdefault(s.symbol, []).append(s)

    def run(
        self,
        primary_rates: dict[str, Any],
        secondary_rates: dict[str, dict[Any, Any]] | None = None,
        primary_timeframes: dict[str, Any] | None = None,
    ) -> None:
        """Feed bars into all strategies.

        Args:
            primary_rates:   Mapping of symbol → bar iterable (completed bars only).
            secondary_rates: Optional mapping of symbol → {timeframe → bar iterable}.
            primary_timeframes: Optional mapping of symbol → primary timeframe key.
        """
        sec = secondary_rates or {}
        prim_tfs = primary_timeframes or {}

        if len(self._by_symbol) == 1:
            symbol, symbol_strategies = next(iter(self._by_symbol.items()))
            p_rates = primary_rates.get(symbol)
            if p_rates is not None:
                primary_duration = self._timeframe_duration(prim_tfs[symbol]) if symbol in prim_tfs else None
                feed_bars(
                    symbol,
                    symbol_strategies,
                    p_rates,
                    sec.get(symbol),
                    self._timeframe_duration,
                    primary_duration,
                    show_progress=True,
                )
        else:
            feed_bars_aligned(
                symbols=list(self._by_symbol.keys()),
                by_symbol=self._by_symbol,
                primary_rates=primary_rates,
                secondary_rates=sec,
                primary_timeframes=prim_tfs,
                timeframe_duration=self._timeframe_duration,
                show_progress=True,
            )
