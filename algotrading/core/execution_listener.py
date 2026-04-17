from __future__ import annotations
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from .position import Position
from .broker import SignalExecutionErrorReason, PositionCloseErrorReason

if TYPE_CHECKING:
    from .signal import EntrySignal


@runtime_checkable
class ExecutionListener(Protocol):
    def on_signal_execution_error(self, signal: 'EntrySignal', reason: SignalExecutionErrorReason) -> None:
        """Called when there is an error executing a signal."""
        pass

    def on_position_close_success(self, position: Position) -> None:
        """Called when a position is successfully closed."""
        pass

    def on_position_close_error(self, position: Position, reason: PositionCloseErrorReason) -> None:
        """Called when there is an error closing a position."""
        pass

    def on_signal_execution_success(self, signal: 'EntrySignal', position: Position) -> None:
        """Called when a signal is successfully executed and a position is opened."""
        pass