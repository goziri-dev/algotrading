from abc import ABC, abstractmethod
from datetime import datetime
from enum import StrEnum
from typing import TYPE_CHECKING

from .position import Position
from .signal import EntrySignal, ExitSignal, PendingSignal, SignalType

if TYPE_CHECKING:
    from .strategy import Strategy


class BrokerError(Exception):
    """Custom exception for broker-related errors."""
    pass


class SignalExecutionErrorReason(StrEnum):
    REQUOTE = "requote"           # price moved; safe to retry immediately
    INVALID_PARAMS = "invalid_params"   # bad volume/price/stops; don't retry without fixing signal
    MARKET_CLOSED = "market_closed"     # retry when market reopens
    INSUFFICIENT_FUNDS = "insufficient_funds"  # don't retry
    CONNECTION_ERROR = "connection_error"       # retry after reconnect
    BROKER_REJECTED = "broker_rejected"         # broker declined for unspecified reason
    UNKNOWN = "unknown"                         # None result or unrecognized retcode


class PositionCloseErrorReason(StrEnum):
    REQUOTE = "requote"                   # price moved; retry
    POSITION_NOT_FOUND = "position_not_found"  # already closed or never existed
    CONNECTION_ERROR = "connection_error"       # retry after reconnect
    BROKER_REJECTED = "broker_rejected"         # broker declined for unspecified reason
    UNKNOWN = "unknown"                         # None result or unrecognized retcode


class SignalExecutionError(BrokerError):
    """Raised when there is an error executing a signal."""
    def __init__(self, message: str, reason: SignalExecutionErrorReason = SignalExecutionErrorReason.UNKNOWN):
        super().__init__(message)
        self.reason = reason


class PositionCloseError(BrokerError):
    """Raised when there is an error closing a position."""
    def __init__(self, message: str, reason: PositionCloseErrorReason = PositionCloseErrorReason.UNKNOWN):
        super().__init__(message)
        self.reason = reason


class PositionModifyErrorReason(StrEnum):
    POSITION_NOT_FOUND = "position_not_found"
    INVALID_PARAMS = "invalid_params"     # e.g. SL/TP violates broker rules
    CONNECTION_ERROR = "connection_error"
    BROKER_REJECTED = "broker_rejected"
    UNKNOWN = "unknown"


class PositionModifyError(BrokerError):
    """Raised when updating SL or TP on an open position fails."""
    def __init__(self, message: str, reason: PositionModifyErrorReason = PositionModifyErrorReason.UNKNOWN):
        super().__init__(message)
        self.reason = reason


class Broker(ABC):
    def __init__(self):
        pass

    @property
    @abstractmethod
    def balance(self) -> float:
        """Current cash balance available in the account."""
        pass

    @property
    @abstractmethod
    def equity(self) -> float:
        """Current equity in the account (balance + unrealized PnL)."""
        pass

    @abstractmethod
    def execute_signal(self, signal: EntrySignal) -> Position:
        """Execute the given entry signal by placing an order with the broker.

        Returns the opened Position on success.
        Raises:
            SignalExecutionError: If the order fails.
        """
        pass

    @abstractmethod
    def shutdown(self):
        """Clean up any resources or connections when shutting down the broker."""
        pass

    @abstractmethod
    def get_positions(self, symbol: str, strategy_id: int) -> list[Position]:
        """Get open positions for a given symbol and strategy ID."""
        pass

    @abstractmethod
    def close_positions(self, exit_signal: ExitSignal) -> list[tuple[Position, PositionCloseError]]:
        """Close positions described by the exit signal.

        Returns a list of (position, error) pairs for any positions that failed to close.
        """
        pass

    @abstractmethod
    def pnl(self, strategy_id: int, position_id: int | None = None) -> float:
        """Return broker-reported unrealized PnL.

        If position_id is given, returns PnL for that position only.
        Otherwise returns the sum across all open positions for the strategy.
        """
        pass

    @abstractmethod
    def pnl_pct(self, strategy_id: int, position_id: int | None = None) -> float:
        """Return unrealized PnL as a percentage of cost basis.

        If position_id is given, returns PnL% for that position only.
        Otherwise returns PnL% across all open positions for the strategy.
        """
        pass

    def realized_pnl(self, strategy_id: int) -> float:
        """Cumulative realized PnL (net of costs) for closed trades of this strategy.

        Default returns 0.0 for live brokers that don't track a trade log locally.
        BacktestBroker overrides this with the actual sum.
        """
        return 0.0

    def value_per_point(self, symbol: str) -> float:
        """Account-currency value of a 1-point move for one engine qty unit."""
        return 1.0

    def max_affordable_qty(self, symbol: str, price: float) -> float:
        """Maximum quantity currently affordable by margin at the given price.

        Backtest brokers can override this to provide an exact cap. Live brokers
        return infinity by default because precise margin checks are broker-side.
        """
        return float("inf")

    def update_price(self, symbol: str, price: float, time: datetime | None = None) -> None:
        """Update internal price for a symbol. No-op for live brokers (they read prices directly)."""

    def prepare_bar(self, symbol: str, open: float, time: datetime | None = None) -> None:
        """Prepare broker state for the start of a completed bar before fills.

        Backtest brokers can override this to mark the bar-open price/time without
        recording an end-of-bar equity point.
        """

    def set_bar_spread_pct(self, symbol: str, spread_pct: float | None) -> None:
        """Set historical spread override for the current bar.

        Backtest brokers can override this to consume per-bar spread estimates.
        Live brokers can ignore it.
        """

    def check_sl_tp(
        self,
        symbol: str,
        open: float,
        high: float,
        low: float,
        strategy_id: int,
    ) -> list[tuple[Position, float, bool]]:
        """Check if any open positions had SL/TP triggered within this bar.

        Returns list of ``(position, exit_price, is_sl)`` for each hit.
        Positions are already closed when returned.
        ``is_sl=True`` → stop-loss hit; ``False`` → take-profit hit.
        Default no-op: live brokers handle SL/TP server-side.
        """
        return []

    @abstractmethod
    def update_sl(self, position_id: int, new_sl_price: float) -> Position | None:
        """Update the stop loss on an open position.

        Returns the updated Position, or None if the position is already closed
        (e.g. it was closed by another SL/TP hit earlier in the same bar).
        Raises PositionModifyError for live brokers if the modification fails.
        """
        pass

    @abstractmethod
    def update_tp(self, position_id: int, new_tp_price: float) -> Position | None:
        """Update the take profit on an open position.

        Returns the updated Position, or None if the position is already closed.
        Raises PositionModifyError for live brokers if the modification fails.
        """
        pass

    @property
    def defers_market_orders(self) -> bool:
        """True if MARKET orders are deferred to the next bar's open (backtest mode).

        Live brokers return False — MARKET signals execute immediately.
        BacktestBroker returns True to avoid look-ahead bias.
        """
        return False

    def queue_signal(self, signal: EntrySignal) -> PendingSignal:
        """Add a LIMIT or STOP signal to the pending queue.

        Returns the created PendingSignal.
        Raises NotImplementedError for brokers that do not support pending signals
        (e.g. live brokers pending MT5 integration).
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support pending signals. "
            "Override queue_signal to add support."
        )

    def cancel_signal(self, signal_id: int) -> bool:
        """Cancel a pending signal by ID.  Returns True if found and cancelled."""
        return False

    def get_pending_signals(self, symbol: str, strategy_id: int) -> list[PendingSignal]:
        """Return pending signals for the given symbol and strategy."""
        return []

    def fill_pending_signals(
        self,
        symbol: str,
        open: float,
        high: float,
        low: float,
        strategy_id: int,
    ) -> list[tuple[PendingSignal, "Position | SignalExecutionError"]]:
        """Check pending signals against the current bar and fill any that trigger.

        Returns a list of ``(pending_signal, result)`` for each triggered signal,
        where ``result`` is the opened ``Position`` on success or a
        ``SignalExecutionError`` on failure (e.g. insufficient margin).
        Triggered signals are removed from the queue regardless of fill outcome.
        Default no-op: returns [] (live brokers handle fills server-side).
        """
        return []


class BrokerView:
    """Intermediary between a Broker and a Strategy.

    Handles signal submission with retry logic and routes broker error callbacks
    back to the strategy. Auto-attaches to the strategy on construction.
    """
    def __init__(self, broker: Broker, strategy: "Strategy"):
        self._broker = broker
        self._strategy = strategy
        strategy.broker = self

    @property
    def balance(self) -> float:
        return self._broker.balance

    @property
    def equity(self) -> float:
        return self._broker.equity

    @property
    def positions(self) -> list[Position]:
        return self._broker.get_positions(self._strategy.symbol, self._strategy.id)

    @property
    def realized_pnl(self) -> float:
        return self._broker.realized_pnl(self._strategy.id)

    def close_positions(self, position_ids: list[int] | None = None) -> None:
        """Close positions for this strategy, optionally restricted to specific IDs."""
        all_positions = self.positions
        attempted = (
            all_positions if not position_ids
            else [p for p in all_positions if p.id in position_ids]
        )
        exit_signal = ExitSignal(
            strategy_id=self._strategy.id,
            symbol=self._strategy.symbol,
            position_ids=position_ids or [],
        )
        failures = self._broker.close_positions(exit_signal)
        failed_ids = {pos.id for pos, _ in failures}
        for pos in attempted:
            if pos.id not in failed_ids:
                self._strategy.on_position_close_success(pos)
        for pos, error in failures:
            self._strategy.on_position_close_error(pos, error.reason)

    def on_bar(
        self,
        symbol: str,
        open: float,
        high: float,
        low: float,
        close: float,
        time: datetime | None = None,
        spread_pct: float | None = None,
    ) -> None:
        """Update price and process pending signals then SL/TP for the completed bar.

        Order of operations (before strategy.on_bar):
        1. In backtests, prepare the broker with the bar's open price/time.
        2. Fill any pending signals at the earliest executable price for the bar.
        3. Loop SL/TP checks until no new hits remain (handles callback-modified stops).
        4. Mark the broker to the bar close.
        """
        self._broker.set_bar_spread_pct(symbol, spread_pct)
        if self._broker.defers_market_orders:
            self._broker.prepare_bar(symbol, open, time)
            self._dispatch_pending_fills(symbol, open, high, low)
            self._dispatch_sl_tp_hits(symbol, open, high, low)
            self._broker.update_price(symbol, close, time)
            return

        self._broker.update_price(symbol, close, time)
        self._dispatch_pending_fills(symbol, open, high, low)
        self._dispatch_sl_tp_hits(symbol, open, high, low)

    def _dispatch_pending_fills(self, symbol: str, open: float, high: float, low: float) -> None:
        """Fill pending signals that triggered within this bar and dispatch callbacks."""
        if self._broker.defers_market_orders:
            should_close = any(
                pending.signal.type == SignalType.MARKET and pending.signal.exclusive
                for pending in self.pending_signals
            )
            if should_close and self.positions:
                self.close_positions()
        results = self._broker.fill_pending_signals(symbol, open, high, low, self._strategy.id)
        for pending, result in results:
            if isinstance(result, SignalExecutionError):
                self._strategy.on_signal_execution_error(pending.signal, result.reason)
            else:
                self._strategy.on_signal_execution_success(pending.signal, result)

    def _dispatch_sl_tp_hits(self, symbol: str, open: float, high: float, low: float) -> None:
        """Dispatch SL/TP callbacks until no further hits remain."""
        while True:
            hits = self._broker.check_sl_tp(symbol, open, high, low, self._strategy.id)
            if not hits:
                break
            for pos, exit_price, is_sl in hits:
                if is_sl:
                    self._strategy.on_sl_hit(pos, exit_price)
                else:
                    self._strategy.on_tp_hit(pos, exit_price)
                self._strategy.on_position_close_success(pos)

    def poll_sl_tp(self, symbol: str | None = None) -> None:
        """Poll broker for server-side SL/TP closes between completed bars.

        Useful for live brokers where SL/TP is executed asynchronously by the
        broker server and callbacks should fire faster than bar cadence.
        """
        self._dispatch_sl_tp_hits(symbol or self._strategy.symbol, 0.0, 0.0, 0.0)

    def submit_signal(self, signal: EntrySignal) -> None:
        """Submit a signal to the broker, handling exclusive closes and pending queuing.

        MARKET signals on live brokers execute immediately via ``execute_signal``.
        On backtest brokers (``defers_market_orders=True``) MARKET signals are
        queued and fill at the next bar's open to avoid look-ahead bias.
        LIMIT and STOP signals are always queued for future bar-level triggering.

        For live brokers and non-deferred orders, ``exclusive=True`` closes
        existing positions at submission time. For deferred backtest MARKET
        orders, the close is deferred to the next bar open alongside the fill.
        """
        should_close_immediately = not (
            signal.exclusive
            and signal.type == SignalType.MARKET
            and self._broker.defers_market_orders
        )
        if signal.exclusive and should_close_immediately:
            self.close_positions()
        if signal.type == SignalType.MARKET and not self._broker.defers_market_orders:
            try:
                position = self._broker.execute_signal(signal)
                self._strategy.on_signal_execution_success(signal, position)
            except SignalExecutionError as e:
                self._strategy.on_signal_execution_error(signal, e.reason)
        else:
            pending = self._broker.queue_signal(signal)
            self._strategy.on_signal_queued(pending)

    def pnl(self, position_id: int | None = None) -> float:
        """Broker-reported unrealized PnL for this strategy.

        If position_id is given, returns PnL for that position only.
        Otherwise returns the sum across all open positions.
        """
        return self._broker.pnl(self._strategy.id, position_id)

    def pnl_pct(self, position_id: int | None = None) -> float:
        """Broker-reported unrealized PnL as a percentage of cost basis.

        If position_id is given, returns PnL% for that position only.
        Otherwise returns PnL% across all open positions.
        """
        return round(self._broker.pnl_pct(self._strategy.id, position_id), 2)

    def value_per_point(self, symbol: str) -> float:
        """Account-currency value of a 1-point move for one engine qty unit."""
        return self._broker.value_per_point(symbol)

    def max_affordable_qty(self, symbol: str, price: float) -> float:
        """Maximum quantity currently affordable by margin at the given price."""
        return self._broker.max_affordable_qty(symbol, price)

    def update_sl(self, position_id: int, new_sl_price: float) -> Position | None:
        """Update the stop loss price for an open position."""
        return self._broker.update_sl(position_id, new_sl_price)

    def update_tp(self, position_id: int, new_tp_price: float) -> Position | None:
        """Update the take profit price for an open position."""
        return self._broker.update_tp(position_id, new_tp_price)

    @property
    def pending_signals(self) -> list[PendingSignal]:
        """Pending LIMIT/STOP signals for this strategy."""
        return self._broker.get_pending_signals(self._strategy.symbol, self._strategy.id)

    def cancel_signal(self, signal_id: int) -> bool:
        """Cancel a specific pending signal by ID.  Returns True if it existed."""
        return self._broker.cancel_signal(signal_id)

    def cancel_pending_signals(self) -> int:
        """Cancel all pending signals for this strategy.  Returns the count cancelled."""
        cancelled = 0
        for ps in self.pending_signals:
            if self._broker.cancel_signal(ps.id):
                cancelled += 1
        return cancelled


class WarmupBrokerView(BrokerView):
    """No-op BrokerView used during the historical warmup phase.

    Attach this before feeding historical bars so strategies can access
    broker methods (get_positions, pnl, etc.) without firing real orders.
    Replace with a real BrokerView before going live.

    Usage::

        WarmupBrokerView(strategy)
        feed_historical_data(strategy, rates)
        BrokerView(live_broker, strategy)   # overwrites warmup view
        feed_live_data(strategy, ...)
    """
    def __init__(self, strategy: "Strategy"):
        self._strategy = strategy
        strategy._broker = self

    @property
    def balance(self) -> float:
        return 0.0

    @property
    def equity(self) -> float:
        return 0.0

    @property
    def positions(self) -> list[Position]:
        return []

    def on_bar(self, symbol, open, high, low, close, time=None, spread_pct=None) -> None:
        pass

    def close_positions(self, position_ids: list[int] | None = None) -> None:
        pass

    def submit_signal(self, signal: EntrySignal) -> None:
        pass

    def poll_sl_tp(self, symbol: str | None = None) -> None:
        pass

    def pnl(self, position_id: int | None = None) -> float:
        return 0.0

    def pnl_pct(self, position_id: int | None = None) -> float:
        return 0.0

    def value_per_point(self, symbol: str) -> float:
        return 1.0

    def max_affordable_qty(self, symbol: str, price: float) -> float:
        return float("inf")

    @property
    def pending_signals(self) -> list[PendingSignal]:
        return []

    def cancel_signal(self, signal_id: int) -> bool:
        return False

    def cancel_pending_signals(self) -> int:
        return 0