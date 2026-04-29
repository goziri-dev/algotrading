from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import MetaTrader5 as mt5

from algotrading.backtest import print_backtest_summary, save_backtest_plots
from algotrading.live.mt5 import MT5Session

from main import SMACross
from my_strategies.bbands_reverse.bbands_reverse import BBANDSReverse


def run_backtest() -> None:
    date_from = datetime(2024, 1, 1, tzinfo=timezone.utc)
    date_to = datetime(2026, 4, 14, tzinfo=timezone.utc)
    primary_tf = mt5.TIMEFRAME_M15
    secondary_tf = mt5.TIMEFRAME_H4

    sma_cross = SMACross(symbol="XAUUSD", cot_start=date_from, cot_end=date_to)
    bbands_reverse = BBANDSReverse(symbol="EURUSD.i")

    session = MT5Session(
        strategies=[sma_cross, bbands_reverse],
        primary_tf=primary_tf,
    )
    bt = session.backtest(
        initial_balance=5_730.0,
        slippage_pct=0.0024,
        date_from=date_from,
        date_to=date_to,
        secondary={secondary_tf: 300},
        leverage=1 / 200,
    )

    print_backtest_summary(bt.broker)
    save_backtest_plots(
        bt=bt,
        strategy=[sma_cross, bbands_reverse],
        primary_tf=primary_tf,
        date_from=date_from,
        date_to=date_to,
        output_dir=Path("plots/multi_strategy"),
    )


def run_live() -> None:
    cot_start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    primary_tf = mt5.TIMEFRAME_M15
    secondary_tf = mt5.TIMEFRAME_H4

    sma_cross = SMACross(symbol="XAUUSD", cot_start=cot_start, cot_end=None)
    bbands_reverse = BBANDSReverse(symbol="EURUSD.i")

    session = MT5Session(
        strategies=[bbands_reverse],
        primary_tf=primary_tf,
    )
    # Warmup: prime indicators from cached history before the live loop starts.
    # Slowest indicator is SMA-50 on H4 (SMACross trend filter) — 300 H4 bars covers it.
    session.backtest(
        initial_balance=0,
        primary_count=100,
        secondary={secondary_tf: 300},
    )

    info = mt5.account_info()  # type: ignore
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
