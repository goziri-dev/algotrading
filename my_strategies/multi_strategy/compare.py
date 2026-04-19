from datetime import datetime, timezone

import MetaTrader5 as mt5

from algotrading.backtest import print_backtest_summary
from algotrading.live.mt5 import MT5Session

from main import SMACross
from my_strategies.bbands_reverse.bbands_reverse import BBANDSReverse


def run() -> None:
    date_from = datetime(2024, 1, 1, tzinfo=timezone.utc)
    date_to = datetime(2026, 4, 14, tzinfo=timezone.utc)
    primary_tf = mt5.TIMEFRAME_M15
    secondary_tf = mt5.TIMEFRAME_H4
    initial_balance = 5_730.0
    slippage_pct = 0.0024
    leverage = 1 / 200

    # ---- SOLO: SMACross on XAUUSD ----
    sma_only = SMACross(symbol="XAUUSD", cot_start=date_from, cot_end=date_to)
    bt_sma = MT5Session(strategies=[sma_only], primary_tf=primary_tf).backtest(
        initial_balance=initial_balance,
        slippage_pct=slippage_pct,
        date_from=date_from,
        date_to=date_to,
        secondary={secondary_tf: 300},
        leverage=leverage,
    )
    print("\n\n===== SOLO: SMACross on XAUUSD =====")
    print_backtest_summary(bt_sma.broker)

    # ---- SOLO: BBANDSReverse on EURUSD.i ----
    bb_only = BBANDSReverse(symbol="EURUSD.i")
    bt_bb = MT5Session(strategies=[bb_only], primary_tf=primary_tf).backtest(
        initial_balance=initial_balance,
        slippage_pct=slippage_pct,
        date_from=date_from,
        date_to=date_to,
        secondary={secondary_tf: 300},
        leverage=leverage,
    )
    print("\n\n===== SOLO: BBANDSReverse on EURUSD.i =====")
    print_backtest_summary(bt_bb.broker)

    # ---- COMBINED: both on one portfolio ----
    sma_combo = SMACross(symbol="XAUUSD", cot_start=date_from, cot_end=date_to)
    bb_combo = BBANDSReverse(symbol="EURUSD.i")
    bt_combo = MT5Session(
        strategies=[sma_combo, bb_combo],
        primary_tf=primary_tf,
    ).backtest(
        initial_balance=initial_balance,
        slippage_pct=slippage_pct,
        date_from=date_from,
        date_to=date_to,
        secondary={secondary_tf: 300},
        leverage=leverage,
    )
    print("\n\n===== COMBINED: SMACross + BBANDSReverse =====")
    print_backtest_summary(bt_combo.broker)


if __name__ == "__main__":
    run()
