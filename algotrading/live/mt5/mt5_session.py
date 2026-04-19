from collections.abc import Sequence
from dataclasses import asdict, replace
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from typing import Any

import MetaTrader5 as mt5

from algotrading.backtest.backtest_broker import SymbolSpec
from algotrading.backtest.backtest_session import BacktestSession
from algotrading.core.broker import BrokerView
from algotrading.core.feed import mt5_timeframe_duration
from algotrading.core.strategy import Strategy

from .mt5_broker import MT5Broker
from .mt5_runner import MT5LiveRunner


class MT5Session:
    """Orchestrates MT5 data fetching, backtesting, and the live handoff.

    Connects to the MetaTrader 5 terminal on construction.  Call
    :meth:`backtest` to run a historical simulation — it returns a
    :class:`~algotrading.backtest.BacktestSession` whose broker holds the full
    trade log and equity curve.  Then call :meth:`go_live` to switch the same
    (already warmed-up) strategies to live trading without re-instantiation.

    Args:
        strategies:            One or more strategy instances.  Strategies for
                               different symbols can be mixed freely.
        primary_tf:            MT5 timeframe constant for the primary bars.
                               Pass a single ``int`` to apply the same timeframe
                               to every symbol, or ``dict[str, int]`` for
                               per-symbol timeframes.
        poll_interval:         Seconds between live bar polls.
        symbol_specs_cache_path: Where to persist fetched symbol specs between
                               runs.  Defaults to ``.algotrading/symbol_specs.json``.

    Example::

        session = MT5Session(
            strategies=[strategy],
            primary_tf=mt5.TIMEFRAME_M15,
        )
        bt = session.backtest(
            initial_balance=10_000,
            slippage_pct=0.0024,
            date_from=datetime(2024, 1, 1),
            secondary={mt5.TIMEFRAME_H4: 300},
        )
        print_backtest_summary(bt.broker)
        session.go_live()
    """

    _DEFAULT_SYMBOL_SPECS_CACHE = Path(".algotrading") / "symbol_specs.json"

    def __init__(
        self,
        strategies: Sequence[Strategy],
        primary_tf: int | dict[str, int],
        poll_interval: float = 1.0,
        symbol_specs_cache_path: str | Path | None = None,
    ):
        if not strategies:
            raise ValueError("At least one strategy is required")

        self._strategies = strategies
        self._primary_tf = primary_tf
        self._poll_interval = poll_interval
        self._symbol_specs_cache_path = (
            Path(symbol_specs_cache_path)
            if symbol_specs_cache_path is not None
            else self._DEFAULT_SYMBOL_SPECS_CACHE
        )
        self._live_broker = MT5Broker()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def backtest(
        self,
        initial_balance: float,
        slippage_pct: float = 0.0,
        commission_per_lot: float = 0.0,
        primary_count: int | None = None,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
        secondary: dict[Any, int] | None = None,
        symbol_specs: dict[str, SymbolSpec] | None = None,
        refresh_symbol_specs: bool = False,
        leverage: float = 1 / 100,
    ) -> BacktestSession:
        """Fetch historical bars from MT5 and run a backtest.

        Provide exactly one of ``primary_count`` or ``date_from``.

        Args:
            initial_balance:    Starting balance for the simulated broker.
            slippage_pct:       Additional execution slippage as a percentage of
                                price, applied on every backtest fill.
            commission_per_lot: Per-side commission in account currency.
            primary_count:      Fetch the last N completed primary bars per symbol.
            date_from:          Fetch primary bars starting from this UTC datetime.
            date_to:            End of the primary range (default: now).  Only
                                used with ``date_from``.
            secondary:          Mapping of ``{timeframe: indicator_warmup_bars}``.
                                With ``date_from``, warmup bars are fetched
                                *before* ``date_from`` so indicators are primed
                                when the first primary bar arrives.
            symbol_specs:       Explicit per-symbol specs that bypass MT5 lookup
                                and the cache entirely.
            refresh_symbol_specs: Force re-fetch of symbol specs from MT5,
                                ignoring the on-disk cache.

        Returns:
            A :class:`~algotrading.backtest.BacktestSession` whose
            :attr:`~algotrading.backtest.BacktestSession.broker` holds the
            completed trade log and equity curve.

        Examples::

            # last 1 000 M15 bars
            bt = session.backtest(initial_balance=10_000, primary_count=1000)
            print_backtest_summary(bt.broker)

            # full year 2024 with H4 indicator warmup
            bt = session.backtest(
                initial_balance=10_000,
                slippage_pct=0.0024,
                date_from=datetime(2024, 1, 1, tzinfo=timezone.utc),
                date_to=datetime(2025, 1, 1, tzinfo=timezone.utc),
                secondary={mt5.TIMEFRAME_H4: 300},
            )
        """
        if primary_count is None and date_from is None:
            raise ValueError("Provide either primary_count or date_from")
        if primary_count is not None and date_from is not None:
            raise ValueError("Provide either primary_count or date_from, not both")

        symbols = {s.symbol for s in self._strategies}
        resolved_specs = self._resolve_symbol_specs(
            symbols=symbols,
            explicit_symbol_specs=symbol_specs,
            slippage_pct=slippage_pct,
            commission_per_lot=commission_per_lot,
            refresh_symbol_specs=refresh_symbol_specs,
        )

        now = datetime.now(timezone.utc)
        end = date_to or now
        primary_rates: dict[str, Any] = {}
        secondary_rates: dict[str, dict[Any, Any]] = {}

        for symbol in symbols:
            tf = self._tf_for(symbol)

            if primary_count is not None:
                rates = mt5.copy_rates_from_pos(symbol, tf, 1, primary_count)  # type: ignore
            else:
                rates = mt5.copy_rates_range(symbol, tf, date_from, end)  # type: ignore

            primary_rates[symbol] = self._rates_with_historical_spread_pct(symbol, rates)

            if secondary:
                secondary_rates[symbol] = {}
                for sec_tf, count in secondary.items():
                    if primary_count is not None:
                        rates = mt5.copy_rates_from_pos(symbol, sec_tf, 1, count)  # type: ignore
                    else:
                        sec_duration = mt5_timeframe_duration(sec_tf)
                        warmup_start = date_from - timedelta(seconds=sec_duration * count)  # type: ignore
                        rates = mt5.copy_rates_range(symbol, sec_tf, warmup_start, end)  # type: ignore
                    if rates is not None:
                        secondary_rates[symbol][sec_tf] = self._rates_with_historical_spread_pct(symbol, rates)

        bt = BacktestSession(
            strategies=self._strategies,
            initial_balance=initial_balance,
            symbol_specs=resolved_specs,
            leverage=leverage,
        )
        bt.run(
            primary_rates=primary_rates,
            secondary_rates=secondary_rates or None,
            primary_timeframes={symbol: self._tf_for(symbol) for symbol in symbols},
        )
        return bt

    def go_live(self) -> None:
        """Switch strategies to the live broker and start the live loop.

        Blocks until stopped or an unhandled exception.  Shuts down the MT5
        connection on exit.
        """
        for s in self._strategies:
            BrokerView(self._live_broker, s)
        try:
            MT5LiveRunner(
                self._strategies,
                primary_tf=self._primary_tf,
                poll_interval=self._poll_interval,
            ).run()
        finally:
            for s in self._strategies:
                s.on_finish()
            self._live_broker.shutdown()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _tf_for(self, symbol: str) -> int:
        if isinstance(self._primary_tf, int):
            return self._primary_tf
        return self._primary_tf[symbol]

    def _rates_with_historical_spread_pct(self, symbol: str, rates: Any) -> list[dict[str, float | int]]:
        """Convert MT5 rate rows to dict bars and attach spread_pct per bar."""
        if rates is None:
            return []
        info = mt5.symbol_info(symbol)  # type: ignore
        point = float(info.point) if info is not None and getattr(info, "point", 0) else 0.0

        bars: list[dict[str, float | int]] = []
        for row in rates:
            open_price = float(row["open"])
            bar: dict[str, float | int] = {
                "time": int(row["time"]),
                "open": open_price,
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
            }
            spread_points = float(row["spread"]) if "spread" in row.dtype.names else 0.0
            if point > 0 and open_price > 0 and spread_points >= 0:
                bar["spread_pct"] = (spread_points * point / open_price) * 100.0
            bars.append(bar)
        return bars

    def _resolve_symbol_specs(
        self,
        symbols: set[str],
        explicit_symbol_specs: dict[str, SymbolSpec] | None,
        slippage_pct: float,
        commission_per_lot: float,
        refresh_symbol_specs: bool,
    ) -> dict[str, SymbolSpec]:
        cached_specs = {} if refresh_symbol_specs else self._load_symbol_specs_cache(self._symbol_specs_cache_path)
        resolved: dict[str, SymbolSpec] = {}
        fetched_specs: dict[str, SymbolSpec] = {}
        explicit_symbol_specs = explicit_symbol_specs or {}

        for symbol in symbols:
            if symbol in explicit_symbol_specs:
                resolved[symbol] = explicit_symbol_specs[symbol]
                continue

            cached = cached_specs.get(symbol)
            if cached is not None:
                resolved[symbol] = replace(
                    cached,
                    slippage_pct=slippage_pct,
                    commission_per_lot=commission_per_lot,
                )
                continue

            fetched = self._live_broker.get_symbol_spec(
                symbol,
                slippage_pct=slippage_pct,
                commission_per_lot=commission_per_lot,
            )
            resolved[symbol] = fetched
            fetched_specs[symbol] = fetched

        if fetched_specs or explicit_symbol_specs:
            merged_cache = cached_specs | resolved
            self._save_symbol_specs_cache(self._symbol_specs_cache_path, merged_cache)

        return resolved

    @staticmethod
    def _load_symbol_specs_cache(path: Path) -> dict[str, SymbolSpec]:
        if not path.exists():
            return {}
        data = json.loads(path.read_text(encoding="utf-8"))
        return {
            symbol: SymbolSpec(**spec_data)
            for symbol, spec_data in data.get("symbols", {}).items()
        }

    @staticmethod
    def _save_symbol_specs_cache(path: Path, symbol_specs: dict[str, SymbolSpec]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        serialized = {
            "symbols": {
                symbol: asdict(spec)
                for symbol, spec in sorted(symbol_specs.items())
            }
        }
        path.write_text(json.dumps(serialized, indent=2, sort_keys=True), encoding="utf-8")
