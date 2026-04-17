from datetime import datetime, timezone
from pathlib import Path
import MetaTrader5 as mt5

from algotrading.live.mt5 import MT5Session
from algotrading.backtest import print_backtest_summary, save_backtest_plots


from .btc_scalper import BTCScalper


def main():
    DATE_FROM_IS = datetime(2024, 1, 1, tzinfo=timezone.utc)
    DATE_TO_IS = datetime(2025, 1, 1, tzinfo=timezone.utc)
    DATE_FROM_OS = datetime(2025, 1, 1, tzinfo=timezone.utc)
    DATE_TO_OS = datetime(2026, 4, 1, tzinfo=timezone.utc)
    PLOT_DIR = Path("plots/btc_scalper")
    SYMBOL = "BTCUSD"
    STRATEGIES = [BTCScalper(symbol=SYMBOL)]
    PRIMARY_TF = mt5.TIMEFRAME_M15
    SECONDARY_TF = mt5.TIMEFRAME_H4
    INITIAL_BALANCE = 3_000.0

    session = MT5Session(
        strategies=STRATEGIES,
        primary_tf=PRIMARY_TF,
    )

    bt = session.backtest(
        initial_balance=INITIAL_BALANCE,
        slippage_pct=0.0024,
        date_from=DATE_FROM_OS,
        date_to=DATE_TO_OS,
        secondary={SECONDARY_TF: 300},
    )

    print_backtest_summary(bt.broker)
    save_backtest_plots(
        bt=bt,
        strategy=STRATEGIES,
        primary_tf=PRIMARY_TF,
        date_from=DATE_FROM_OS,
        date_to=DATE_TO_OS,
        output_dir=PLOT_DIR,
    )


if __name__ == "__main__":
    main()