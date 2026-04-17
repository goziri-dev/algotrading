from dataclasses import dataclass
from datetime import datetime, timezone
import math
from pathlib import Path
from typing import Literal

import MetaTrader5 as mt5

from algotrading import COTClient, Strategy, StrategyParams
from algotrading.backtest import BacktestSession, print_backtest_summary, save_backtest_plots
from algotrading.core.broker import PositionCloseErrorReason, SignalExecutionErrorReason
from algotrading.core.position import Position
from algotrading.core.signal import EntrySignal, PendingSignal
from algotrading.live.mt5 import MT5Session
from algotrading.utils import crossover, crossunder, fraction_to_qty
from algotrading.indicators import SMA, ATR


@dataclass
class SMACrossParams(StrategyParams):
    fast_period: int = 10
    slow_period: int = 30
    trend_period: int = 50
    dd_exit_threshold_pct: float = 8.0
    qty_multiplier: float = 1
    pnl_pct_exit_threshold: float = 3
    pnl_pct_sl_exit_threshold: float = -1
    plot_fast_sma: bool = True
    plot_slow_sma: bool = True
    plot_trend_htf: bool = True
    plot_atr: bool = True


class SMACross(Strategy[SMACrossParams]):
    def __init__(
        self,
        symbol: str = "XAUUSD",
        params: SMACrossParams = SMACrossParams(),
        *,
        cot_start: datetime | None = None,
        cot_end: datetime | None = None,
        cot_client: COTClient | None = None,
        contract_code: str = "088691",
    ):
        super().__init__(symbol=symbol, params=params)
        self.fast_sma = self.I(SMA(period=params.fast_period), source='close')
        self.slow_sma = self.I(SMA(period=params.slow_period), source='close')
        self.atr = self.I(ATR(period=14), source=('high', 'low', 'close'))
        self.bars_htf = self.new_bars(timeframe=mt5.TIMEFRAME_H4)
        self.trend_htf = self.I(SMA(period=params.trend_period), source='close', bars=self.bars_htf)
        # Strategy-level plotting policy: choose exactly which indicators appear in charts.
        self.set_indicator_plot_visible(self.fast_sma, params.plot_fast_sma)
        self.set_indicator_plot_visible(self.slow_sma, params.plot_slow_sma)
        self.set_indicator_plot_visible(self.trend_htf, params.plot_trend_htf)
        self.set_indicator_plot_visible(self.atr, params.plot_atr)
        self.set_indicator_plot_label(self.fast_sma, "Fast SMA")
        self.set_indicator_plot_label(self.slow_sma, "Slow SMA")
        self.set_indicator_plot_label(self.trend_htf, "Trend SMA (HTF)")
        self.set_indicator_plot_label(self.atr, "ATR")
        self.bars.add_field("noncomm_longs", int)
        self.bars.add_field("noncomm_shorts", int)
        self.bars.add_field("noncomm_net", int)
        self.bars.add_field("cot_spread", int)
        self.bars.add_field("noncomm_longs_pct", float)
        self.bars.add_field("noncomm_short_pct", float)
        self.bars.add_field("noncomm_longs_oi_pct", float)
        self.bars.add_field("noncomm_short_oi_pct", float)
        self.bars.add_field("open_interest", int)
        self.bars.add_field("chng_long", int)
        self.bars.add_field("chng_short", int)
        self._cot_client = cot_client or COTClient(contract_code=contract_code)
        if cot_client is None:
            self._cot_client.fetch_historical(start_date=cot_start, end_date=cot_end)

        self._on_signal_queued_logged = False
        self._on_signal_execution_success_logged = False
        self._on_signal_execution_error_logged = False
        self._on_position_close_success_logged = False
        self._on_position_close_error_logged = False

    def update_bars(self) -> None:
        # Keep strategy usable in both backtest (cached history) and live mode (periodic refresh).
        report = self._cot_client.report_at(self.bars.time[-1], auto_live_refresh=True)
        if report is None:
            return

        self.bars.append(**report.to_strategy_data())

    def next(self) -> None:
        trend_up = self.price > self.trend_htf[-1]
        vpp = self.broker.value_per_point(self.symbol)
        long_pct = float(self.bars["noncomm_longs_pct"][-1])
        short_pct = float(self.bars["noncomm_short_pct"][-1])

        if not math.isfinite(long_pct) or not math.isfinite(short_pct):
            return

        cot_bullish = long_pct > 60.0
        cot_bearish = short_pct > 60.0

        if self.exit_when_drawdown_exceeds(self.params.dd_exit_threshold_pct):
            self.close_positions()
            return

        if crossover(self.fast_sma, self.slow_sma) and trend_up and cot_bullish:
            if not self.should_buy():
                return
            qty = fraction_to_qty(self.equity, 99.9, entry=self.price, value_per_point=vpp)
            self.buy(qty=qty * self.params.qty_multiplier)
        elif crossunder(self.fast_sma, self.slow_sma) and not trend_up and cot_bearish:
            if not self.should_sell():
                return
            qty = fraction_to_qty(self.equity, 99.9, entry=self.price, value_per_point=vpp)
            self.sell(qty=qty * self.params.qty_multiplier)

    def should_buy(self) -> bool:
        current_pos = self.positions[-1] if self.positions else None
        if not current_pos:
            return True
        pnl_pct = self.pnl_pct(current_pos.id)
        if current_pos.is_long and pnl_pct < self.params.pnl_pct_exit_threshold:
            return False
        return True

    def should_sell(self) -> bool:
        current_pos = self.positions[-1] if self.positions else None
        if not current_pos:
            return True
        pnl_pct = self.pnl_pct(current_pos.id)
        if current_pos.is_short and pnl_pct < self.params.pnl_pct_exit_threshold:
            return False
        return True

    def exit_when_drawdown_exceeds(self, threshold_pct: float) -> bool:
        if threshold_pct <= 0:
            raise ValueError("threshold_pct should be positive and expressed in percent (e.g. 5 for 5%).")
        current_pos = self.positions[-1] if self.positions else None
        if not current_pos:
            return False
        return self.broker.pnl_pct(current_pos.id) < -threshold_pct

    def on_signal_queued(self, pending: PendingSignal) -> None:
        if self._on_signal_queued_logged:
            return
        self._on_signal_queued_logged = True
        signal = pending.signal
        print(
            f"[{self.symbol}] queued signal id={pending.id} time={pending.queued_time} "
            f"dir={signal.direction} type={signal.type} qty={signal.qty:,.6f} price={signal.price:,.2f} "
            f"exclusive={signal.exclusive}"
        )

    def on_signal_execution_success(self, signal: EntrySignal, position: Position) -> None:
        if self._on_signal_execution_success_logged:
            return
        self._on_signal_execution_success_logged = True
        print(
            f"[{self.symbol}] execution success dir={signal.direction} type={signal.type} "
            f"req_qty={signal.qty:,.6f} fill_qty={position.qty:,.6f} fill_price={position.entry_price:,.2f} "
            f"position_id={position.id}"
        )

    def on_signal_execution_error(self, signal: EntrySignal, reason: SignalExecutionErrorReason) -> None:
        if self._on_signal_execution_error_logged:
            return
        self._on_signal_execution_error_logged = True
        max_qty = self.broker.max_affordable_qty(self.symbol, self.price)
        print(
            f"[{self.symbol}] execution error reason={reason} dir={signal.direction} type={signal.type} "
            f"req_qty={signal.qty:,.6f} price={signal.price:,.2f} current_price={self.price:,.2f} "
            f"equity={self.equity:,.2f} max_affordable_qty={max_qty:,.6f} open_positions={len(self.positions)}"
        )

    def on_position_close_success(self, position: Position) -> None:
        if self._on_position_close_success_logged:
            return
        self._on_position_close_success_logged = True
        print(
            f"[{self.symbol}] position closed id={position.id} dir={position.direction.value} "
            f"entry={position.entry_price:,.2f} qty={position.qty:,.6f}"
        )

    def on_position_close_error(self, position: Position, reason: PositionCloseErrorReason) -> None:
        if self._on_position_close_error_logged:
            return
        self._on_position_close_error_logged = True
        print(
            f"[{self.symbol}] close error position_id={position.id} "
            f"dir={position.direction.value} reason={reason}"
        )


def run_backtest() -> None:
    date_from = datetime(2024, 1, 1, tzinfo=timezone.utc)
    date_to = datetime(2026, 4, 14, tzinfo=timezone.utc)
    primary_tf = mt5.TIMEFRAME_M15
    secondary_tf = mt5.TIMEFRAME_H4
    strategy = SMACross(cot_start=date_from, cot_end=date_to)

    session = MT5Session(
        strategies=[strategy],
        primary_tf=primary_tf,
    )
    bt = session.backtest(
        initial_balance=2_730,
        slippage_pct=0.0024,
        date_from=date_from,
        date_to=date_to,
        secondary={secondary_tf: 300},
    )
    print_backtest_summary(bt.broker)
    save_backtest_plots(
        bt=bt,
        strategy=[strategy],
        primary_tf=primary_tf,
        date_from=date_from,
        date_to=date_to,
        output_dir=Path("plots/sma_cross2"),
    )


def run_live() -> None:
    cot_start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    primary_tf = mt5.TIMEFRAME_M15
    secondary_tf = mt5.TIMEFRAME_H4
    # cot_end=None fetches COT data up to today
    strategy = SMACross(cot_start=cot_start, cot_end=None)

    session = MT5Session(
        strategies=[strategy],
        primary_tf=primary_tf,
    )
    # Warmup: feed enough bars so all indicators (slowest: SMA-50 on H4) are primed
    # before the live loop starts. primary_count avoids a date_from requirement.
    session.backtest(
        initial_balance=0,
        primary_count=100,
        secondary={secondary_tf: 300},
    )

    info = mt5.account_info() # type: ignore
    if info is not None:
        print(
            f"\n--- MT5 Live Account ---\n"
            f"  Login:    {info.login}\n"
            f"  Name:     {info.name}\n"
            f"  Server:   {info.server}\n"
            f"  Currency: {info.currency}\n"
            f"  Balance:  {info.balance:,.2f}\n"
            f"  Equity:   {info.equity:,.2f}\n"
            f"  Margin:   {info.margin:,.2f}\n"
            f"  Free margin: {info.margin_free:,.2f}\n"
            f"  Leverage: 1:{info.leverage}\n"
            f"  Trade allowed: {bool(info.trade_allowed)}\n"
            f"------------------------\n"
        )

    session.go_live()


def main(mode: Literal["backtest", "live"]) -> None:
    if mode == "backtest":
        run_backtest()
    else:
        run_live()


if __name__ == "__main__":
    main("live")
