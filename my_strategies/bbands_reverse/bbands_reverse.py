from dataclasses import dataclass

from algotrading import Strategy, StrategyParams
from algotrading.indicators import BBANDS, RSI, EMA, ATR
from algotrading.utils import MonthlyDrawdownTracker, crossover, session, risk_to_qty


@dataclass
class BBANDSReverseParams(StrategyParams):
    bbands_period: int = 50
    ema_period: int = 20
    monthly_drawdown_limit_pct: float = 6.0


class BBANDSReverse(Strategy):
    def __init__(self, symbol: str="EURUSD.i", params: BBANDSReverseParams=BBANDSReverseParams()):
        super().__init__(symbol, params)
        self._bbands = self.I(BBANDS(period=self.params.bbands_period, ma_type="SMMA (RMA)"), source="close")
        self._ema_upper = self.I(EMA(period=self.params.ema_period), source=self._bbands.output("upper"))
        self._ema_lower = self.I(EMA(period=self.params.ema_period), source=self._bbands.output("lower"))
        self._rsi = self.I(RSI(period=14), source="close")
        self._atr = self.I(ATR(period=14), source=("high", "low", "close"))
        self._buy_signal_count: int = 0
        self._dd_tracker = MonthlyDrawdownTracker(lambda: self.strategy_equity)

    @property
    def upper_bbands_crossover_ema(self) -> bool:
        upper_bbands = self._bbands.upper
        return crossover(upper_bbands, self._ema_upper)

    @property
    def price_above_upper_bbands_and_ema(self) -> bool:
        upper_bbands = self._bbands.upper
        return self.price > upper_bbands[-1] and self.price > self._ema_upper[-1]

    @property
    def long_rsi_exit(self) -> bool:
        return self._rsi[-1] > 80

    @property
    def not_sydney_session(self) -> bool:
        return session(self.time) in {"london", "new_york", "tokyo"}

    @property
    def end_of_day(self) -> bool:
        return self.time.hour == 22 and self.time.minute >= 45

    def get_quantity(self, scale: float=3) -> float:
        risk_pct: float = 1.0
        stop_loss = self.price - self._atr[-1] * 4
        return risk_to_qty(self.strategy_equity, risk_pct, self.price, stop_loss, self.vpp) * scale

    def next(self) -> None:
        self._dd_tracker.update(self.time)

        if self.positions and self.end_of_day:
            self.close_positions()

        if self._dd_tracker.breached(self.params.monthly_drawdown_limit_pct):
            return

        should_buy = self.upper_bbands_crossover_ema and self.price_above_upper_bbands_and_ema and self.not_sydney_session
        if should_buy and not self.positions:
            self._buy_signal_count += 1
            sl = self.price - self._atr[-1] * 4
            tp = self.price + self._atr[-1] * 12
            self.buy(qty=self.get_quantity(), sl_price=sl, tp_price=tp)
