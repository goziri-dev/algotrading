"""Visual-debug harness for the HHLL indicator.

Runs a no-op strategy (never trades) over a short window so the produced
``plots/hhll_debug/trade_history.html`` can be eyeballed against the original
TradingView ``Higher High Lower Low Strategy`` study.
"""
from datetime import datetime, timezone
from pathlib import Path

import MetaTrader5 as mt5

from algotrading import Strategy, StrategyParams
from algotrading.backtest import print_backtest_summary, save_backtest_plots
from algotrading.indicators import HHLL
from algotrading.live.mt5 import MT5Session


class HHLLDebug(Strategy):
    def __init__(self, symbol: str = "EURUSD.i", left_bars: int = 5, right_bars: int = 5):
        super().__init__(symbol, StrategyParams())
        self._hhll = self.I(
            HHLL(left_bars=left_bars, right_bars=right_bars),
            source=("high", "low", "close"),
        )

    def next(self) -> None:
        pass


def main() -> None:
    DATE_FROM = datetime(2026, 4, 1, tzinfo=timezone.utc)
    DATE_TO = datetime(2026, 4, 21, tzinfo=timezone.utc)
    SYMBOL = "EURUSD.i"
    PRIMARY_TF = mt5.TIMEFRAME_M15
    PLOT_DIR = Path("plots/hhll_debug")

    strategy = HHLLDebug(symbol=SYMBOL, left_bars=5, right_bars=5)

    session = MT5Session(strategies=[strategy], primary_tf=PRIMARY_TF)
    bt = session.backtest(
        initial_balance=10_000.0,
        date_from=DATE_FROM,
        date_to=DATE_TO,
    )

    print_backtest_summary(bt.broker)
    save_backtest_plots(
        bt=bt,
        strategy=[strategy],
        primary_tf=PRIMARY_TF,
        date_from=DATE_FROM,
        date_to=DATE_TO,
        output_dir=PLOT_DIR,
    )
    print(f"\nOpen {PLOT_DIR / 'trade_history.html'} to compare against TradingView.")


if __name__ == "__main__":
    main()
