import time as time_module
from abc import ABC, abstractmethod
from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Any

from .feed import feed_bars, mt5_timeframe_duration
from .strategy import Strategy


class LiveRunner(ABC):
    """Base class for live trading runners.

    A single runner manages all symbols and strategies in a portfolio.  Strategies
    are grouped internally by symbol; each poll tick fetches and feeds bars for
    every symbol, keeping all strategies on the same clock — which is required for
    portfolio-level correlation analysis and cross-symbol signals.

    Subclasses implement :meth:`fetch_primary_bars` and
    :meth:`fetch_secondary_bars`; this class owns the event loop, secondary bar
    alignment, and the ``broker.on_bar → strategy.on_bar`` dispatch.

    The starting position for each symbol is derived automatically from the
    strategies' existing bar state — whatever was fed during a preceding backtest
    warmup is already in ``strategy.bars``, so the runner resumes from where the
    warmup left off without any manual bookkeeping.

    Example::

        runner = MT5LiveRunner(
            [btc_strategy, eur_strategy],
            primary_tf={
                "BTCUSD": mt5.TIMEFRAME_M1,
                "EURUSD": mt5.TIMEFRAME_M5,
            },
        )
        runner.run()
    """

    def __init__(self, strategies: Sequence[Strategy], poll_interval: float = 1.0):
        if not strategies:
            raise ValueError("At least one strategy is required")
        self._strategies = strategies
        self._poll_interval = poll_interval

        # Group strategies by symbol for efficient bar dispatch
        self._by_symbol: dict[str, list[Strategy]] = {}
        for s in strategies:
            self._by_symbol.setdefault(s.symbol, []).append(s)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Start the live loop.  Blocks until stopped or an unhandled exception.

        The starting position per symbol is inferred from ``strategy.bars`` — the
        runner picks up exactly where the backtest warmup left off.
        """
        bar_times = self._initial_bar_times()
        last_sec_times = {s: self._last_secondary_times(s) for s in self._strategies}

        while True:
            for s in self._strategies:
                s.broker.poll_sl_tp(s.symbol)

            for symbol, symbol_strategies in self._by_symbol.items():
                primary_rates = self.fetch_primary_bars(symbol)
                if primary_rates is None:
                    continue

                last_t = bar_times.get(symbol)
                new_primary = [
                    r for r in primary_rates
                    if last_t is None or r["time"] > last_t
                ]
                if not new_primary:
                    continue

                # Union of secondary timeframes across all strategies for this symbol
                all_tfs: set[Any] = set()
                for s in symbol_strategies:
                    all_tfs.update(s.secondary_bars.keys())

                secondary_rates: dict[Any, Any] = {}
                for tf in all_tfs:
                    # Fetch since the earliest last-seen time so every strategy
                    # (regardless of warmup depth) is caught up
                    relevant = [
                        last_sec_times[s].get(tf)
                        for s in symbol_strategies if tf in s.secondary_bars
                    ]
                    min_t = min((t for t in relevant if t is not None), default=None)
                    sec_rates = self.fetch_secondary_bars(symbol, tf)
                    if sec_rates is None:
                        continue
                    new_sec = [r for r in sec_rates if min_t is None or r["time"] > min_t]
                    if new_sec:
                        secondary_rates[tf] = new_sec

                for bar in new_primary:
                    self.on_bar(symbol, bar)

                result = self._feed_bars(symbol, symbol_strategies, new_primary, secondary_rates or None)
                if result is not None:
                    bar_times[symbol] = result

            last_sec_times = {s: self._last_secondary_times(s) for s in self._strategies}
            time_module.sleep(self._poll_interval)

    def on_bar(self, symbol: str, bar: Any) -> None:
        """Called once per new primary bar before it is fed.  Override to log or react."""
        bar_time = datetime.fromtimestamp(bar["time"], tz=timezone.utc)
        print(
            f"[{symbol}] {bar_time}"
            f"  O={bar['open']}  H={bar['high']}  L={bar['low']}  C={bar['close']}"
        )

    # ------------------------------------------------------------------
    # Abstract — subclasses supply the data
    # ------------------------------------------------------------------

    @abstractmethod
    def fetch_primary_bars(self, symbol: str) -> Any | None:
        """Return the most recent completed primary bars for ``symbol``, or None on failure.

        Must skip the currently forming bar.  Each element must support
        ``bar["time"]`` (unix seconds int), ``bar["open"]``, ``bar["high"]``,
        ``bar["low"]``, ``bar["close"]``.
        """

    @abstractmethod
    def fetch_secondary_bars(self, symbol: str, timeframe: Any) -> Any | None:
        """Return the most recent completed bars for ``symbol`` at ``timeframe``,
        or None on failure."""

    # ------------------------------------------------------------------
    # Internal bar feeding
    # ------------------------------------------------------------------

    def _feed_bars(
        self,
        symbol: str,
        strategies: Sequence[Strategy],
        primary_rates: Any,
        secondary_rates: dict[Any, Any] | None = None,
    ) -> int | None:
        primary_tf = self.primary_timeframe(symbol)
        primary_duration = self.timeframe_duration(primary_tf) if primary_tf is not None else None
        return feed_bars(
            symbol,
            strategies,
            primary_rates,
            secondary_rates,
            self.timeframe_duration,
            primary_duration,
        )

    def primary_timeframe(self, symbol: str) -> Any | None:
        """Return the primary timeframe key for ``symbol`` if known."""
        return None

    def timeframe_duration(self, timeframe: Any) -> int:
        """Bar duration in seconds for ``timeframe``.

        Default treats the value as an MT5-style integer constant in minutes
        (``TIMEFRAME_M1=1``, ``TIMEFRAME_M15=15``, ``TIMEFRAME_H1=60``, …).
        Override for other conventions.
        """
        return mt5_timeframe_duration(timeframe)

    def _initial_bar_times(self) -> dict[str, int | None]:
        """Derive the starting bar time per symbol from current strategy state.

        Takes the maximum last primary bar time across strategies for each symbol
        so the runner resumes without re-processing bars any strategy has seen.
        """
        result: dict[str, int | None] = {}
        for symbol, strategies in self._by_symbol.items():
            times = [
                int(s.bars.time[-1].astype("datetime64[s]").astype("int64"))
                for s in strategies if len(s.bars) > 0
            ]
            result[symbol] = max(times) if times else None
        return result

    def _last_secondary_times(self, strategy: Strategy) -> dict[Any, int | None]:
        result: dict[Any, int | None] = {}
        for tf, bars_obj in strategy.secondary_bars.items():
            if len(bars_obj) == 0:
                result[tf] = None
            else:
                result[tf] = int(bars_obj.time[-1].astype("datetime64[s]").astype("int64"))
        return result
