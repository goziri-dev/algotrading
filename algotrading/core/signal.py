from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Literal


class SignalDirection(StrEnum):
    """Defines the direction of the entry signal: LONG or SHORT."""
    LONG = 'long'
    SHORT = 'short'


class SignalType(StrEnum):
    """Defines the type of entry signal based on the relationship between the
    entry price and current price.

    - MARKET: Entry price is equal to current price (immediate execution)
    - LIMIT: Entry price is below the current price (buy) or above the current price (sell)
    - STOP: Entry price is above the current price (buy) or below the current price (sell)
    """
    MARKET = 'market'
    LIMIT = 'limit'
    STOP = 'stop'

    @staticmethod
    def from_price(entry_price: float, current_price: float) -> 'SignalType':
        """Determine the signal type based on the entry price and current price."""
        if entry_price < current_price:
            return SignalType.LIMIT
        elif entry_price > current_price:
            return SignalType.STOP
        else:
            return SignalType.MARKET


@dataclass
class Signal:
    """Base class for trading signals."""
    strategy_id: int
    symbol: str


@dataclass
class EntrySignal(Signal):
    """Represents a signal to enter a trade, including direction, type, quantity, price,
    and optional stop loss and take profit levels.
    """
    direction: Literal["LONG", "SHORT"]
    type: SignalType
    qty: float
    price: float
    exclusive: bool
    sl: float | None = None
    tp: float | None = None
    max_slippage: float | None = None

    def __str__(self):
        return (
            f"EntrySignal(direction={self.direction}, type={self.type}, "
            f"qty={self.qty}, price={self.price}, sl={self.sl}, tp={self.tp}, "
            f"max_slippage={self.max_slippage})"
        )


@dataclass
class ExitSignal(Signal):
    """Represents a signal to exit a trade, including the position ID to close."""
    position_ids: list[int]


@dataclass
class PendingSignal:
    """A LIMIT or STOP entry signal that has been accepted into the broker's
    pending queue and is waiting to be triggered by price.

    ``id`` is assigned by the broker and is unique within a broker instance.
    ``signal`` is the original EntrySignal.
    ``queued_time`` is the bar time at which the signal was submitted.
    """
    id: int
    signal: EntrySignal
    queued_time: datetime
