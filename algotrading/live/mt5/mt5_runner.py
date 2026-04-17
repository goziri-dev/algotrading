from collections.abc import Sequence
from typing import Any

import MetaTrader5 as mt5

from algotrading.core.runner import LiveRunner
from algotrading.core.strategy import Strategy


class MT5LiveRunner(LiveRunner):
    """Live runner that fetches bars from MetaTrader 5.

    Args:
        strategies:      One or more strategy instances with brokers already attached.
                         Strategies trading different symbols can be mixed freely.
        primary_tf:      MT5 timeframe constant for the primary timeframe.  Pass a
                         single ``int`` to apply the same timeframe to every symbol,
                         or a ``dict[str, int]`` to set a different timeframe per
                         symbol (e.g. M1 for BTCUSD, M5 for EURUSD).
        primary_count:   Number of completed bars to fetch per poll per symbol.
                         Position 1 is used so the forming bar is always skipped.
        secondary_count: Number of completed bars to fetch for each secondary
                         timeframe registered on any strategy.
        poll_interval:   Seconds to sleep between full portfolio polls.

    Example — single symbol::

        runner = MT5LiveRunner([strategy], primary_tf=mt5.TIMEFRAME_M1)
        runner.run()

    Example — multi-symbol portfolio::

        runner = MT5LiveRunner(
            [btc_strategy, eur_strategy, gbp_strategy],
            primary_tf={"BTCUSD": mt5.TIMEFRAME_M1, "EURUSD": mt5.TIMEFRAME_M5, "GBPUSD": mt5.TIMEFRAME_M5},
        )
        runner.run()
    """

    def __init__(
        self,
        strategies: Sequence[Strategy],
        primary_tf: int | dict[str, int],
        primary_count: int = 10,
        secondary_count: int = 30,
        poll_interval: float = 1.0,
    ):
        super().__init__(strategies, poll_interval=poll_interval)
        if isinstance(primary_tf, int):
            self._primary_tf: dict[str, int] = {s.symbol: primary_tf for s in strategies}
        else:
            self._primary_tf = primary_tf
        self._primary_count = primary_count
        self._secondary_count = secondary_count

    def fetch_primary_bars(self, symbol: str) -> Any | None:
        tf = self._primary_tf.get(symbol)
        if tf is None:
            return None
        return mt5.copy_rates_from_pos(symbol, tf, 1, self._primary_count)  # type: ignore

    def fetch_secondary_bars(self, symbol: str, timeframe: Any) -> Any | None:
        return mt5.copy_rates_from_pos(symbol, timeframe, 1, self._secondary_count)  # type: ignore

    def primary_timeframe(self, symbol: str) -> Any | None:
        return self._primary_tf.get(symbol)
