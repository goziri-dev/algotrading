from datetime import datetime, timezone

import MetaTrader5 as mt5

from algotrading.backtest import (
    BacktestStats,
    calculate_backtest_stats,
    run_walk_forward,
)
from algotrading.live.mt5 import MT5Session

from .bbands_reverse import BBANDSReverse, BBANDSReverseParams


SYMBOL = "EURUSD.i"
PRIMARY_TF = mt5.TIMEFRAME_M15
DATE_FROM = datetime(2024, 1, 1, tzinfo=timezone.utc)
DATE_TO = datetime(2026, 4, 14, tzinfo=timezone.utc)
INITIAL_BALANCE = 3_000.0
SLIPPAGE_PCT = 0.0024

SEARCH_VALUES = [2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0]

# M15 bar counts: ~96/day → ~2_016/month (21 trading days)
TRAIN_BARS = 6 * 2_016   # ~6 months in-sample
TEST_BARS = 3 * 2_016    # ~3 months out-of-sample


def _fetch_bar_times() -> list[datetime]:
    rates = mt5.copy_rates_range(SYMBOL, PRIMARY_TF, DATE_FROM, DATE_TO)  # type: ignore
    if rates is None or len(rates) == 0:
        raise RuntimeError(f"No bars fetched for {SYMBOL}")
    return [datetime.fromtimestamp(int(r["time"]), tz=timezone.utc) for r in rates]


def _evaluate(params: dict, date_from: datetime, date_to: datetime) -> BacktestStats:
    strategy = BBANDSReverse(
        params=BBANDSReverseParams(
            monthly_drawdown_limit_pct=float(params["monthly_drawdown_limit_pct"]),
        ),
    )
    session = MT5Session(strategies=[strategy], primary_tf=PRIMARY_TF)
    bt = session.backtest(
        initial_balance=INITIAL_BALANCE,
        date_from=date_from,
        date_to=date_to,
        slippage_pct=SLIPPAGE_PCT,
    )
    return calculate_backtest_stats(bt.broker)


def run_wfo() -> None:
    MT5Session(strategies=[BBANDSReverse()], primary_tf=PRIMARY_TF)

    bar_times = _fetch_bar_times()
    total_size = len(bar_times)
    print(f"Total M15 bars: {total_size}  ({bar_times[0]:%Y-%m-%d} → {bar_times[-1]:%Y-%m-%d})")
    print(f"Train size: {TRAIN_BARS} bars,  test size: {TEST_BARS} bars")

    def evaluate_window(params: dict, start: int, end: int) -> BacktestStats:
        return _evaluate(params, bar_times[start], bar_times[end - 1])

    def calmar(stats: BacktestStats) -> float:
        return stats.calmar_ratio if stats.calmar_ratio is not None else float("-inf")

    report = run_walk_forward(
        param_space={"monthly_drawdown_limit_pct": SEARCH_VALUES},
        total_size=total_size,
        train_size=TRAIN_BARS,
        test_size=TEST_BARS,
        evaluate_window=evaluate_window,
        objective=calmar,
        maximize=True,
    )

    print("\n" + "=" * 120)
    print("WALK-FORWARD OPTIMISATION — monthly_drawdown_limit_pct (maximise Calmar)")
    print("=" * 120)
    header = (
        f"{'#':>3}  {'train':>25}  {'test':>25}  "
        f"{'best_lim':>9}  {'IS_calmar':>10}  {'OOS_calmar':>11}  "
        f"{'OOS_|dd|%':>10}  {'OOS_bal':>10}  {'OOS_tr':>6}"
    )
    print(header)
    print("-" * 120)
    for i, step in enumerate(report.steps, start=1):
        w = step.window
        train_range = f"{bar_times[w.train_start]:%Y-%m-%d}→{bar_times[w.train_end - 1]:%Y-%m-%d}"
        test_range = f"{bar_times[w.test_start]:%Y-%m-%d}→{bar_times[w.test_end - 1]:%Y-%m-%d}"
        best_lim = step.in_sample_best.params["monthly_drawdown_limit_pct"]
        is_calmar = step.in_sample_best.evaluation.calmar_ratio
        oos = step.out_of_sample_evaluation
        is_str = f"{is_calmar:>10.2f}" if is_calmar is not None else f"{'—':>10}"
        oos_calmar_str = f"{oos.calmar_ratio:>11.2f}" if oos.calmar_ratio is not None else f"{'—':>11}"
        print(
            f"{i:>3}  {train_range:>25}  {test_range:>25}  "
            f"{best_lim:>9.2f}  {is_str}  {oos_calmar_str}  "
            f"{abs(oos.max_drawdown_pct):>10.2f}  {oos.final_balance:>10.2f}  {oos.trade_count:>6d}"
        )
    print("=" * 120)

    mean_oos = report.mean_out_of_sample_score
    if mean_oos is not None:
        print(f"\nMean OOS Calmar: {mean_oos:.2f}")
    chosen = [step.in_sample_best.params["monthly_drawdown_limit_pct"] for step in report.steps]
    print(f"Limits chosen per step: {chosen}")


if __name__ == "__main__":
    run_wfo()
