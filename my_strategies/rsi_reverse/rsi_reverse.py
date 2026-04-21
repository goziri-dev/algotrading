from dataclasses import dataclass

import MetaTrader5 as mt5

from algotrading import Strategy, StrategyParams
from algotrading.core.position import Position
from algotrading.indicators import RSI, SMA, ATR, HHLL
from algotrading.utils import crossover, crossunder, risk_to_qty

@dataclass
class RSIReverseParams(StrategyParams):
    rsi_period: int = 14
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0
    sma_period: int = 30
    atr_period: int = 14
    atr_multiplier: float = 3.0

class RSIReverse(Strategy):
    def __init__(self, params: RSIReverseParams = RSIReverseParams()):
        super().__init__(symbol="EURUSD.i", params=params)
        self._sma_ltf = self.I(SMA(period=self.params.sma_period), source="close")
        self._atr_ltf = self.I(ATR(period=self.params.atr_period), source=("high", "low", "close"))
        self._hhll = self.I(HHLL(), source=("high", "low", "close"))
        self.hide_indicator_plot(self._atr_ltf)
        self._bars_htf = self.new_bars(mt5.TIMEFRAME_H1)
        self._rsi_1h = self.I(RSI(period=self.params.rsi_period), source="close", bars=self._bars_htf)

        self._buy_signals: int = 0
        self._sell_signals: int = 0

    @property
    def rsi_overbought(self) -> bool:
        return round(self._rsi_1h[-1]) >= self.params.rsi_overbought

    @property
    def rsi_oversold(self) -> bool:
        return round(self._rsi_1h[-1]) <= self.params.rsi_oversold

    @property
    def price_crossover_sma(self) -> bool:
        return crossover(self.bars.close, self._sma_ltf)

    @property
    def price_crossunder_sma(self) -> bool:
        return crossunder(self.bars.close, self._sma_ltf)
    
    @property
    def should_exit(self) -> bool:
        if not self.positions:
            return False
        pos = self.positions[0]
        if pos.is_long:
            return crossunder(self._rsi_1h, self.params.rsi_overbought)
        else:
            return crossover(self._rsi_1h, self.params.rsi_oversold)
        
    def trail_stop(self) -> None:
        pt = int(self._hhll.pivot_type[-1])
        if pt != HHLL.HL and pt != HHLL.LH:
            return
        pivot_price = float(self._hhll.pivot_price[-1])
        for pos in self.positions:
            if self.pnl_pct(pos.id) < 0.1:
                continue
            if pos.is_long and pt == HHLL.HL:
                if pos.sl_price is None or pivot_price > pos.sl_price:
                    self.update_sl(pos.id, sl_price=pivot_price)
            elif pos.is_short and pt == HHLL.LH:
                if pos.sl_price is None or pivot_price < pos.sl_price:
                    self.update_sl(pos.id, sl_price=pivot_price)

    def next(self) -> None:
        self.trail_stop()

        if self.should_exit:
            self.close_positions()
            return

        if self.positions or self.pending_signals:
            return

        should_buy = self.rsi_overbought and self.price_crossover_sma
        should_sell = self.rsi_oversold and self.price_crossunder_sma
        if should_buy and not self.positions:
            if self.pending_signals:
                self.cancel_pending_signals()
            self._buy_signals += 1
            entry = self._sma_ltf[-1]
            sl = entry - self._atr_ltf[-1] * self.params.atr_multiplier
            qty = risk_to_qty(self.equity, .5, entry, sl, self.vpp)
            self.buy(qty, entry_price=entry, sl_price=sl)
            self.buy(qty, sl_price = sl, exclusive=False)
        elif should_sell and not self.positions:
            if self.pending_signals:
                self.cancel_pending_signals()
            self._sell_signals += 1
            entry = self._sma_ltf[-1]
            sl = entry + self._atr_ltf[-1] * self.params.atr_multiplier
            qty = risk_to_qty(self.equity, .5, entry, sl, self.vpp)
            self.sell(qty, entry_price=entry, sl_price=sl)
            self.sell(qty, sl_price=sl, exclusive=False)

    def on_position_close_success(self, position: Position) -> None:
        self.cancel_pending_signals()

    def on_finish(self) -> None:
        print(f"Total buy signals: {self._buy_signals}")
        print(f"Total sell signals: {self._sell_signals}")
        print(f"Total signals: {self._buy_signals + self._sell_signals}")