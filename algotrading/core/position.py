from dataclasses import dataclass
from datetime import datetime

from .signal import SignalDirection


@dataclass
class Position:
    id: int
    symbol: str
    strategy_id: int
    direction: SignalDirection
    qty: float
    entry_time: datetime
    entry_price: float
    sl_price: float | None = None
    tp_price: float | None = None

    @property
    def is_long(self) -> bool:
        return self.direction == SignalDirection.LONG

    @property
    def is_short(self) -> bool:
        return self.direction == SignalDirection.SHORT
    
    @property
    def sl_distance(self) -> float | None:
        if self.sl_price is None:
            return None
        return abs(self.entry_price - self.sl_price)
