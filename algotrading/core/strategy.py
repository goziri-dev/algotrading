from abc import ABC, abstractmethod
from dataclasses import dataclass
import hashlib
import math
from typing import TYPE_CHECKING, Any
from algotrading.indicators.indicator import Indicator

from .bars import Bars
from .broker import BrokerView, SignalExecutionErrorReason, PositionCloseErrorReason
from .position import Position
from .signal import EntrySignal, PendingSignal, SignalType

if TYPE_CHECKING:
    pass


@dataclass
class StrategyParams:
    """Contains parameters that define the strategy and are used
    to generate a unique ID for the strategy instance.
    """
    ...


class Strategy[T: StrategyParams](ABC):
    shared_state: dict = {}

    @classmethod
    def clear_shared_state(cls) -> None:
        """Clear all shared state across every strategy instance.

        Called automatically at the start of each ``BacktestSession.run()`` so
        that consecutive backtests in the same process do not inherit stale state.
        """
        cls.shared_state.clear()

    def __init__(self, symbol: str, params: T | None = None):
        self._symbol = symbol
        self._params = params
        self._id = self._generate_id()
        self._bars = Bars()
        self._secondary_bars: dict[str, Bars] = {}
        self._indicators: list[tuple[Indicator, str | Indicator | tuple[str | Indicator, ...], Bars]] = []
        self._indicator_plot_visibility: dict[int, bool] = {}
        self._indicator_plot_labels: dict[int, str] = {}
        # (id(bars), id(indicator)) -> count of secondary bars already fed.
        # This must be per-indicator, not per-bars, so multiple indicators
        # bound to the same secondary Bars all advance correctly.
        self._secondary_bars_consumed: dict[tuple[int, int], int] = {}
        self._broker: BrokerView | None = None

    def I[IndT: Indicator](
        self,
        ind: IndT,
        source: str | Indicator | tuple[str | Indicator, ...],
        bars: Bars | None = None,
    ) -> IndT:
        """Register an indicator and bind it to a data source.

        The indicator will be automatically updated on every `on_bar` call
        before `next` is invoked.  `source` can be:

        - Any bar field name, including dynamic ones: `'close'`, `'noncomm_net'`
        - Another indicator (reads its latest value each bar)
        - A tuple of the above for multi-input indicators

        `bars` defaults to the primary timeframe. Pass a secondary bars
        instance (from `new_bars()`) to source from a different timeframe::

            self.trend = self.I(SMA(200), source='close', bars=self.bars_d1)

        Examples:
            ```python
            self.sma = self.I(SMA(period=20), source='close')
            self.atr = self.I(ATR(period=14), source=('high', 'low', 'close'))
            self.cot = self.I(COTIndex(26), source='spec_comm_spread')
            self.trend = self.I(SMA(200), source='close', bars=self.bars_d1)
            ```
        """
        self._indicators.append((ind, source, bars if bars is not None else self._bars))
        self._indicator_plot_visibility[id(ind)] = True
        return ind

    def set_indicator_plot_visible(self, ind: Indicator, visible: bool) -> None:
        """Enable or disable plotting for a registered indicator."""
        self._indicator_plot_visibility[id(ind)] = visible

    def hide_indicator_plot(self, ind: Indicator) -> None:
        """Hide an indicator from backtest plots."""
        self.set_indicator_plot_visible(ind, False)

    def show_indicator_plot(self, ind: Indicator) -> None:
        """Show an indicator in backtest plots."""
        self.set_indicator_plot_visible(ind, True)

    def is_indicator_plot_visible(self, ind: Indicator) -> bool:
        """Return whether an indicator is currently configured to be plotted."""
        return self._indicator_plot_visibility.get(id(ind), True)

    def set_indicator_plot_label(self, ind: Indicator, label: str) -> None:
        """Override chart legend label for a registered indicator."""
        self._indicator_plot_labels[id(ind)] = label

    def get_indicator_plot_label(self, ind: Indicator) -> str | None:
        """Return legend label override for an indicator, if set."""
        return self._indicator_plot_labels.get(id(ind))

    @abstractmethod
    def next(self) -> None:
        """Called after every new bar is fed and all indicators are updated."""
        ...

    def on_bar(
        self,
        time,
        open: float,
        high: float,
        low: float,
        close: float,
        **extra,
    ) -> None:
        """Feed a bar into the strategy.

        Lifecycle:
        1. Append OHLC (+ any ``extra`` cols the engine already knows) to bars
        2. Call ``update_bars()`` — user hook to fill in remaining dynamic fields
        3. Run all registered indicators
        4. Call ``next()``
        """
        self._bars.update(time=time, open=open, high=high, low=low, close=close, **extra)
        self.update_bars()
        self._run_indicators_and_next()

    def update_bars(self) -> None:
        """Hook called after OHLC is appended, before indicators and ``next()``.

        Override to fill in dynamic bar fields from an external source::

            class COTStrategy(Strategy[Params]):
                def update_bars(self):
                    data = self._cot_cache          # populated by a background thread
                    self.bars.append(noncomm_net=data["noncomm_net"])
        """

    def _run_indicators_and_next(self) -> None:
        """Update all registered indicators from their bound bars then call ``next``.

        Primary-bars indicators are fed once (the current bar), as before.

        Secondary-bars indicators replay *every* bar that arrived since the last
        primary bar fired.  The engine may feed several D1 bars before the next
        H1 bar closes; each one must advance the indicator, not just the latest.
        We track ``_secondary_bars_consumed[(id(bars), id(indicator))]`` — the
        count of secondary bars already fed to each indicator — and step
        forward from there.

        Empty secondary bars are skipped silently (indicator stays ``nan``).
        ``next()`` is only called once ALL indicators (primary and secondary)
        have a finite value, preventing look-ahead bias when a secondary
        timeframe starts later than the primary (e.g. primary 2009, D1 2010).
        """
        for ind, source, bars in self._indicators:
            if bars is self._bars:
                # Primary bars: feed current bar value once.
                if isinstance(source, Indicator):
                    val = source[-1]
                    if not math.isnan(val):
                        ind(val)
                elif isinstance(source, str):
                    ind(bars[source][-1])
                else:
                    ind(*(bars[s][-1] if isinstance(s, str) else s[-1] for s in source))
            else:
                # Secondary bars: replay any bars that arrived since last primary.
                bars_id = id(bars)
                feed_key = (bars_id, id(ind))
                already_fed = self._secondary_bars_consumed.get(feed_key, 0)
                current_len = len(bars)
                for i in range(already_fed, current_len):
                    if isinstance(source, Indicator):
                        val = source[-1]
                        if not math.isnan(val):
                            ind(val)
                    elif isinstance(source, str):
                        ind(bars[source][i])
                    else:
                        ind(*(bars[s][i] if isinstance(s, str) else s[-1] for s in source))
                self._secondary_bars_consumed[feed_key] = current_len

        if all(not math.isnan(ind[-1]) for ind, _, _ in self._indicators):
            self.next()

    @property
    def params(self) -> T:
        """Return the strategy parameters."""
        if self._params is None:
            raise ValueError("Strategy parameters not set")
        return self._params

    @property
    def symbol(self) -> str:
        """Return the symbol this strategy trades."""
        return self._symbol

    def new_bars(self, timeframe: Any) -> Bars:
        """Create and register a secondary Bars instance for a different timeframe.

        The engine uses ``secondary_bars`` to discover which timeframes to feed.
        Call this in ``__init__`` before any data is fed::

            self.bars_d1 = self.new_bars('D1')
            self.trend   = self.I(SMA(200), source='close', bars=self.bars_d1)
        """
        if timeframe in self._secondary_bars:
            raise ValueError(f"Bars for timeframe {timeframe} already exists")
        bars = Bars()
        self._secondary_bars[timeframe] = bars
        return bars

    @property
    def secondary_bars(self) -> dict[Any, Bars]:
        """Secondary timeframe bars keyed by timeframe identifier."""
        return self._secondary_bars
    
    @property
    def price(self) -> float:
        """Convenience property to get the latest close price from the primary bars."""
        return self._bars.close[-1]
    
    @property
    def equity(self) -> float:
        """Convenience property to get the current equity from the broker view."""
        return self.broker.equity

    @property
    def vpp(self) -> float:
        """Convenience property to get the current value per point for the strategy symbol."""
        return self.broker.value_per_point(self._symbol)
    
    @property
    def positions(self) -> list[Position]:
        """Convenience property to get the current open positions from the broker view."""
        return self.broker.positions
    
    def update_tp(self, position_id: int, tp_price: float) -> None:
        """Convenience method to update the take-profit price of an open position."""
        self.broker.update_tp(position_id, tp_price)

    def update_sl(self, position_id: int, sl_price: float) -> None:
        """Convenience method to update the stop-loss price of an open position."""
        self.broker.update_sl(position_id, sl_price)

    def close_positions(self, position_ids: list[int] | None = None) -> None:
        """Close all open positions for this strategy, or a specific subset by ID."""
        self.broker.close_positions(position_ids)

    @property
    def pending_signals(self) -> list[PendingSignal]:
        """Pending LIMIT/STOP signals queued for this strategy."""
        return self.broker.pending_signals

    def cancel_signal(self, signal_id: int) -> bool:
        """Cancel a specific pending signal by ID. Returns True if it existed."""
        return self.broker.cancel_signal(signal_id)

    def cancel_pending_signals(self) -> int:
        """Cancel all pending signals for this strategy. Returns the count cancelled."""
        return self.broker.cancel_pending_signals()

    def pnl(self, position_id: int | None = None) -> float:
        """Unrealized PnL in account currency.

        If ``position_id`` is given, returns PnL for that position only.
        Otherwise returns the sum across all open positions for this strategy.
        """
        return self.broker.pnl(position_id)

    def pnl_pct(self, position_id: int | None = None) -> float:
        """Unrealized PnL as a percentage of cost basis.

        If ``position_id`` is given, returns PnL% for that position only.
        Otherwise returns PnL% across all open positions for this strategy.
        """
        return self.broker.pnl_pct(position_id)

    @property
    def bars(self) -> Bars:
        """Return the primary bars object."""
        return self._bars

    @property
    def indicators(self) -> list[Indicator]:
        """Registered indicators in declaration order."""
        return [ind for ind, _, _ in self._indicators]

    @property
    def indicator_bindings(self) -> list[tuple[Indicator, Bars]]:
        """Registered indicator bindings as (indicator, source_bars)."""
        return [(ind, bars) for ind, _, bars in self._indicators]

    @property
    def plotted_indicator_bindings(self) -> list[tuple[Indicator, Bars]]:
        """Indicator bindings filtered by strategy plot visibility settings."""
        return [(ind, bars) for ind, bars in self.indicator_bindings if self.is_indicator_plot_visible(ind)]

    @property
    def id(self) -> int:
        """Return the unique ID of this strategy instance."""
        return self._id
    
    @property
    def broker(self) -> BrokerView:
        """Return the attached BrokerView for this strategy."""
        if self._broker is None:
            raise RuntimeError("No BrokerView attached. Create BrokerView(broker, strategy) before running.")
        return self._broker

    @broker.setter
    def broker(self, broker_view: BrokerView):
        """Attach a BrokerView to this strategy."""
        self._broker = broker_view

    def write_state(self, key: str, value) -> None:
        """Write a value to the shared state for this strategy instance."""
        Strategy.shared_state.setdefault(self._id, {})[key] = value

    def read_state(self, strategy: 'Strategy', key: str):
        """Read a value from the shared state for a given strategy instance."""
        return Strategy.shared_state.get(strategy._id, {}).get(key)

    def _generate_id(self) -> int:
        key = f"{self.__class__.__name__}:{self._symbol}:{self._params}"
        digest = hashlib.md5(key.encode()).digest()
        return int.from_bytes(digest[:4], byteorder='big')

    def buy(
        self,
        qty: float,
        *,
        entry_price: float | None = None,
        sl_price: float | None = None,
        tp_price: float | None = None,
        exclusive: bool = True
    ) -> None:
        """Create a LONG entry signal and submit it via the broker view."""
        current_price = self.bars.close[-1]
        qty = min(qty, self.broker.max_affordable_qty(self._symbol, current_price) * 0.99)
        if qty <= 0:
            return
        signal_type = SignalType.from_price(entry_price, current_price) if entry_price is not None else SignalType.MARKET
        self.broker.submit_signal(EntrySignal(
            strategy_id=self._id,
            symbol=self._symbol,
            direction='LONG',
            type=signal_type,
            exclusive=exclusive,
            qty=qty,
            price=current_price if entry_price is None else entry_price,
            sl=sl_price,
            tp=tp_price,
        ))

    def sell(
        self,
        qty: float,
        *,
        entry_price: float | None = None,
        sl_price: float | None = None,
        tp_price: float | None = None,
        exclusive: bool = True
    ) -> None:
        """Create a SHORT entry signal and submit it via the broker view."""
        current_price = self.bars.close[-1]
        qty = min(qty, self.broker.max_affordable_qty(self._symbol, current_price) * 0.99)
        if qty <= 0:
            return
        signal_type = SignalType.from_price(entry_price, current_price) if entry_price is not None else SignalType.MARKET
        self.broker.submit_signal(EntrySignal(
            strategy_id=self._id,
            symbol=self._symbol,
            direction='SHORT',
            type=signal_type,
            exclusive=exclusive,
            qty=qty,
            price=current_price if entry_price is None else entry_price,
            sl=sl_price,
            tp=tp_price,
        ))

    def on_signal_execution_error(self, signal: EntrySignal, reason: SignalExecutionErrorReason) -> None:
        """Called by BrokerView when a signal fails to execute. Override to retry via buy()/sell()."""

    def on_signal_execution_success(self, signal: EntrySignal, position: Position) -> None:
        """Called by BrokerView when a signal is successfully executed and a position is opened."""

    def on_signal_queued(self, pending: PendingSignal) -> None:
        """Called by BrokerView when a LIMIT or STOP signal is accepted into the pending queue."""

    def on_position_close_success(self, position: Position) -> None:
        """Called by BrokerView when a position is successfully closed (including SL/TP hits)."""

    def on_sl_hit(self, position: Position, exit_price: float) -> None:
        """Called by BrokerView when a stop-loss is triggered. Fires before on_position_close_success."""

    def on_tp_hit(self, position: Position, exit_price: float) -> None:
        """Called by BrokerView when a take-profit is triggered. Fires before on_position_close_success."""

    def on_position_close_error(self, position: Position, reason: PositionCloseErrorReason) -> None:
        """Called by BrokerView when a position fails to close."""
        print(f"Error closing position: {position}, reason: {reason}")
