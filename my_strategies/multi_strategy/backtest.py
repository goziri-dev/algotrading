from datetime import datetime, timezone
from pathlib import Path

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


if __name__ == "__main__":
    run_backtest()
