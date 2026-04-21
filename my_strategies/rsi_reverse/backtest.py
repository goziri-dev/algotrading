from datetime import datetime, timezone
from pathlib import Path

import MetaTrader5 as mt5

from algotrading.backtest import print_backtest_summary, save_backtest_plots
from algotrading.live.mt5 import MT5Session

from .rsi_reverse import RSIReverse


def run_backtest():
    strategy = RSIReverse()
    primary_tf = mt5.TIMEFRAME_M5

    date_from = datetime(2025, 10, 20, tzinfo=timezone.utc)
    date_to = datetime(2026, 4, 14, tzinfo=timezone.utc)

    session = MT5Session(
        strategies=[strategy],
        primary_tf=primary_tf,
    )

    bt = session.backtest(
        initial_balance=3_000.0,
        date_from=date_from,
        date_to=date_to,
        slippage_pct=0.0024,
        commission_per_lot=7.0,
        secondary={mt5.TIMEFRAME_H1: 300},
    )
    
    print_backtest_summary(bt.broker)
    save_backtest_plots(
        bt=bt,
        strategy=[strategy],
        primary_tf=primary_tf,
        date_from=date_from,
        date_to=date_to,
        output_dir=Path("plots/rsi_reverse"),
    )

if __name__ == "__main__":
    run_backtest()