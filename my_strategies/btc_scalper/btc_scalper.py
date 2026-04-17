from dataclasses import dataclass, field
from typing import Literal

import MetaTrader5 as mt5

from algotrading import Strategy, StrategyParams
from algotrading.core.position import Position
from algotrading.core.signal import EntrySignal
from algotrading.utils import risk_to_qty
from algotrading.indicators import Supertrend, ATR, ADX, EMA


@dataclass
class BTCScalperParams(StrategyParams):
    adx_threshold: float = 30.0
    sl_atr_multiplier: float = 2.5
    tp_atr_multiplier: float = 3.2
    htf = mt5.TIMEFRAME_H4
    risk_perc: float = 1.0


class BTCScalper(Strategy[BTCScalperParams]):
    def __init__(self, symbol: str="BTCUSD", params: BTCScalperParams = BTCScalperParams()):
        super().__init__(symbol=symbol, params=params)
        hlc_source = ("high", "low", "close")
        self.st = self.I(Supertrend(), source=hlc_source)
        self.atr = self.I(ATR(), source=hlc_source)
        self.adx = self.I(ADX(), source=hlc_source)

        ### Multi-timeframe:
        self.bars_htf = self.new_bars(timeframe=self.params.htf)
        self.st_htf = self.I(Supertrend(), source=hlc_source, bars=self.bars_htf)
        self.ema_htf = self.I(EMA(period=200), source='close', bars=self.bars_htf)

    @property
    def short_trend(self) -> Literal["up", "down"]:
        if self.price > self.st.trend[-1]:
            return "up"
        else:
            return "down"
        
    @property
    def long_trend(self) -> Literal["up", "down"]:
        if self.price > self.st_htf.trend[-1]:
            return "up"
        else:
            return "down"
        
    @property
    def strong_trend(self) -> bool:
        return self.adx[-1] >= self.params.adx_threshold
    
    @property
    def above_htf_ema(self) -> bool:
        return self.price > self.ema_htf[-1]

    def next(self) -> None:
        if self.positions:
            return

        should_long = self.short_trend == "up" and self.strong_trend and self.long_trend == "up" and self.above_htf_ema
        should_short = self.short_trend == "down" and self.strong_trend and self.long_trend == "down" and not self.above_htf_ema

        if should_long:
            stop_loss = self.price - self.atr[-1] * self.params.sl_atr_multiplier
            qty = risk_to_qty(
                self.equity, 
                self.params.risk_perc,
                self.price, 
                stop_loss,
                self.vpp,
            )
            self.buy(qty, sl_price=stop_loss)
        elif should_short:
            stop_loss = self.price + self.atr[-1] * self.params.sl_atr_multiplier
            qty = risk_to_qty(
                self.equity, 
                self.params.risk_perc,
                self.price,
                stop_loss, 
                self.vpp,
            )
            self.sell(qty, sl_price=stop_loss)

    def on_signal_execution_success(self, signal: EntrySignal, position: Position) -> None:
        if position.is_long:
            take_profit = self.price + self.atr[-1] * self.params.tp_atr_multiplier
        else:
            take_profit = self.price - self.atr[-1] * self.params.tp_atr_multiplier
        self.update_tp(position.id, take_profit)
