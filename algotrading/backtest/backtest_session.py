from collections.abc import Sequence
from typing import Any

from algotrading.backtest.backtest_broker import BacktestBroker, SymbolSpec
from algotrading.backtest.backtester import Backtester
from algotrading.core.broker import BrokerView
from algotrading.core.strategy import Strategy


class BacktestSession:
    """Runs a backtest against pre-fetched bar data and exposes the broker for results.

    This class is data-source agnostic — pass any mapping of bars (from MT5,
    CSV, a custom data provider, etc.).  For the common case of fetching data
    directly from MetaTrader 5, use :meth:`MT5Session.backtest` which handles
    data fetching and returns a ``BacktestSession``.

    Args:
        strategies:    One or more strategy instances.  Strategies for different
                       symbols can be mixed freely.
        initial_balance: Starting account balance for the simulated broker.
        symbol_specs:  Per-symbol contract specifications (spread, slippage,
                       value_per_point, etc.).  Required for realistic PnL.

    Example — standalone usage with pre-fetched data::

        from algotrading.backtest import BacktestSession, SymbolSpec

        spec = SymbolSpec(value_per_point=1.0, spread_pct=0.02, slippage_pct=0.002)
        bt = BacktestSession(
            strategies=[strategy],
            initial_balance=10_000,
            symbol_specs={"XAUUSD": spec},
        )
        bt.run(primary_rates={"XAUUSD": bars}, secondary_rates={"XAUUSD": {H4: h4_bars}})

        print_backtest_summary(bt.broker)
    """

    def __init__(
        self,
        strategies: Sequence[Strategy],
        initial_balance: float,
        symbol_specs: dict[str, SymbolSpec],
        leverage: float = 1 / 100,
    ):
        if not strategies:
            raise ValueError("At least one strategy is required")
        self._strategies = strategies
        self._broker = BacktestBroker(
            initial_balance=initial_balance,
            symbol_specs=symbol_specs,
            leverage=leverage,
        )

    @property
    def broker(self) -> BacktestBroker:
        """The backtest broker — inspect trade log and equity curve after :meth:`run`."""
        return self._broker

    def run(
        self,
        primary_rates: dict[str, Any],
        secondary_rates: dict[str, dict[Any, Any]] | None = None,
        primary_timeframes: dict[str, Any] | None = None,
    ) -> None:
        """Feed bars into all strategies and execute the backtest.

        Args:
            primary_rates:      Mapping of symbol → bar iterable (completed bars only).
            secondary_rates:    Optional mapping of symbol → {timeframe → bar iterable}.
            primary_timeframes: Optional mapping of symbol → primary timeframe key.
        """
        Strategy.clear_shared_state()
        for s in self._strategies:
            BrokerView(self._broker, s)
        Backtester(self._strategies).run(
            primary_rates=primary_rates,
            secondary_rates=secondary_rates,
            primary_timeframes=primary_timeframes,
        )
