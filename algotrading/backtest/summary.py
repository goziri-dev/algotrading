from dataclasses import dataclass
from datetime import timedelta
import math

from .backtest_broker import BacktestBroker, ClosedTrade


_TRADING_DAYS_PER_YEAR = 252.0


@dataclass(frozen=True)
class BacktestStats:
    initial_balance: float
    final_balance: float
    total_pnl: float
    total_gross: float
    total_costs: float
    total_spread_cost: float
    total_slippage_cost: float
    total_commission_cost: float
    total_swap_cost: float
    peak_equity: float
    trough_equity: float
    max_drawdown: float
    max_drawdown_pct: float
    trade_count: int
    win_count: int
    loss_count: int
    win_rate: float | None
    avg_win: float | None
    avg_loss: float | None
    avg_win_pct: float | None
    avg_loss_pct: float | None
    profit_factor: float | None
    expectancy_per_trade: float | None
    expectancy_per_trade_pct: float | None
    avg_holding_time: timedelta | None
    turnover: float
    avg_turnover_per_trade: float | None
    turnover_multiple: float | None
    cost_ratio_pct: float | None
    exposure_pct: float | None
    cagr_pct: float | None
    sharpe_ratio: float | None
    sortino_ratio: float | None
    calmar_ratio: float | None
    trades: list[ClosedTrade]


def _format_duration(duration: timedelta) -> str:
    total_seconds = int(duration.total_seconds())
    if total_seconds <= 0:
        return "0m"
    days, remainder = divmod(total_seconds, 86_400)
    hours, remainder = divmod(remainder, 3_600)
    minutes, _ = divmod(remainder, 60)
    parts: list[str] = []
    if days:
        parts.append(f"{days}d")
    if hours:
        parts.append(f"{hours}h")
    if minutes or not parts:
        parts.append(f"{minutes}m")
    return " ".join(parts)


def _merged_trade_duration(trades: list[ClosedTrade]) -> timedelta:
    if not trades:
        return timedelta(0)
    intervals = sorted((t.position.entry_time, t.exit_time) for t in trades)
    merged_start, merged_end = intervals[0]
    total = timedelta(0)
    for start, end in intervals[1:]:
        if start <= merged_end:
            if end > merged_end:
                merged_end = end
            continue
        total += merged_end - merged_start
        merged_start, merged_end = start, end
    total += merged_end - merged_start
    return total


def _backtest_window(broker: BacktestBroker, trades: list[ClosedTrade]) -> timedelta:
    if broker.equity_curve:
        return broker.equity_curve[-1].time - broker.equity_curve[0].time
    if trades:
        return max(t.exit_time for t in trades) - min(t.position.entry_time for t in trades)
    return timedelta(0)


def _daily_equity_returns(broker: BacktestBroker) -> list[float]:
    if not broker.equity_curve:
        return []
    daily_equity: dict[object, float] = {}
    for point in broker.equity_curve:
        daily_equity[point.time.date()] = point.equity
    returns: list[float] = []
    previous_equity = broker._initial_balance
    for equity in daily_equity.values():
        if previous_equity > 0:
            returns.append((equity / previous_equity) - 1.0)
        previous_equity = equity
    return returns


def _sample_std(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    if variance <= 0:
        return None
    return math.sqrt(variance)


def _downside_deviation(values: list[float]) -> float | None:
    if not values:
        return None
    downside = [min(value, 0.0) for value in values]
    downside_variance = sum(value ** 2 for value in downside) / len(downside)
    if downside_variance <= 0:
        return None
    return math.sqrt(downside_variance)


def _trade_pnl_pct(trade: ClosedTrade, broker: BacktestBroker) -> float:
    cost_basis = (
        trade.position.entry_price
        * trade.position.qty
        * broker.value_per_point(trade.position.symbol)
    )
    return (trade.pnl / cost_basis * 100.0) if cost_basis else 0.0


def calculate_backtest_stats(broker: BacktestBroker) -> BacktestStats:
    trades = broker.trade_log
    wins = [trade for trade in trades if trade.is_win]
    losses = [trade for trade in trades if not trade.is_win]
    total_pnl = sum(trade.pnl for trade in trades)

    if broker.equity_curve:
        peak_equity = broker._initial_balance
        trough_equity = broker._initial_balance
        running_peak = broker._initial_balance
        max_drawdown = 0.0
        for point in broker.equity_curve:
            equity = point.equity
            if equity > peak_equity:
                peak_equity = equity
            if equity < trough_equity:
                trough_equity = equity
            if equity > running_peak:
                running_peak = equity
            drawdown = equity - running_peak
            if drawdown < max_drawdown:
                max_drawdown = drawdown
    else:
        peak_equity = broker._initial_balance
        trough_equity = broker._initial_balance
        max_drawdown = 0.0
    max_drawdown_pct = (max_drawdown / peak_equity * 100.0) if peak_equity else 0.0

    total_gross = sum(trade.gross_pnl for trade in trades)
    total_costs = sum(trade.costs for trade in trades)
    total_spread_cost = sum(trade.spread_cost for trade in trades)
    total_slippage_cost = sum(trade.slippage_cost for trade in trades)
    total_commission_cost = sum(trade.commission_cost for trade in trades)
    total_swap_cost = sum(trade.swap_cost for trade in trades)
    gross_profit = sum(trade.pnl for trade in wins)
    gross_loss = -sum(trade.pnl for trade in losses)
    profit_factor = (gross_profit / gross_loss) if gross_loss else None
    expectancy_per_trade = (total_pnl / len(trades)) if trades else None
    win_pcts = [_trade_pnl_pct(trade, broker) for trade in wins]
    loss_pcts = [_trade_pnl_pct(trade, broker) for trade in losses]
    trade_pcts = [_trade_pnl_pct(trade, broker) for trade in trades]
    avg_holding_time = (
        sum((trade.exit_time - trade.position.entry_time for trade in trades), timedelta(0)) / len(trades)
        if trades else None
    )
    turnover = sum(
        (trade.position.entry_price + trade.exit_price)
        * trade.position.qty
        * broker.value_per_point(trade.position.symbol)
        for trade in trades
    )
    avg_turnover_per_trade = (turnover / len(trades)) if trades else None
    turnover_multiple = (turnover / broker._initial_balance) if broker._initial_balance > 0 else None
    cost_ratio_pct = (total_costs / total_gross * 100.0) if total_gross > 0 else None
    backtest_window = _backtest_window(broker, trades)
    exposure_pct = (
        _merged_trade_duration(trades) / backtest_window * 100.0
        if backtest_window > timedelta(0)
        else None
    )
    years = backtest_window.total_seconds() / (365.25 * 86_400) if backtest_window > timedelta(0) else 0.0
    if years > 0 and broker._initial_balance > 0 and broker._balance > 0:
        cagr_pct = (((broker._balance / broker._initial_balance) ** (1 / years)) - 1.0) * 100.0
    else:
        cagr_pct = None
    daily_returns = _daily_equity_returns(broker)
    if daily_returns:
        mean_daily_return = sum(daily_returns) / len(daily_returns)
        daily_std = _sample_std(daily_returns)
        downside_deviation = _downside_deviation(daily_returns)
        sharpe_ratio = (
            mean_daily_return / daily_std * math.sqrt(_TRADING_DAYS_PER_YEAR)
            if daily_std is not None else None
        )
        sortino_ratio = (
            mean_daily_return / downside_deviation * math.sqrt(_TRADING_DAYS_PER_YEAR)
            if downside_deviation is not None else None
        )
    else:
        sharpe_ratio = None
        sortino_ratio = None
    max_drawdown_fraction = abs(max_drawdown_pct) / 100.0
    calmar_ratio = (
        (cagr_pct / 100.0) / max_drawdown_fraction
        if cagr_pct is not None and max_drawdown_fraction > 0 else None
    )

    return BacktestStats(
        initial_balance=broker._initial_balance,
        final_balance=broker._balance,
        total_pnl=total_pnl,
        total_gross=total_gross,
        total_costs=total_costs,
        total_spread_cost=total_spread_cost,
        total_slippage_cost=total_slippage_cost,
        total_commission_cost=total_commission_cost,
        total_swap_cost=total_swap_cost,
        peak_equity=peak_equity,
        trough_equity=trough_equity,
        max_drawdown=max_drawdown,
        max_drawdown_pct=max_drawdown_pct,
        trade_count=len(trades),
        win_count=len(wins),
        loss_count=len(losses),
        win_rate=(len(wins) / len(trades) * 100.0) if trades else None,
        avg_win=(sum(trade.pnl for trade in wins) / len(wins)) if wins else None,
        avg_loss=(sum(trade.pnl for trade in losses) / len(losses)) if losses else None,
        avg_win_pct=(sum(win_pcts) / len(win_pcts)) if win_pcts else None,
        avg_loss_pct=(sum(loss_pcts) / len(loss_pcts)) if loss_pcts else None,
        profit_factor=profit_factor,
        expectancy_per_trade=expectancy_per_trade,
        expectancy_per_trade_pct=(sum(trade_pcts) / len(trade_pcts)) if trade_pcts else None,
        avg_holding_time=avg_holding_time,
        turnover=turnover,
        avg_turnover_per_trade=avg_turnover_per_trade,
        turnover_multiple=turnover_multiple,
        cost_ratio_pct=cost_ratio_pct,
        exposure_pct=exposure_pct,
        cagr_pct=cagr_pct,
        sharpe_ratio=sharpe_ratio,
        sortino_ratio=sortino_ratio,
        calmar_ratio=calmar_ratio,
        trades=trades,
    )


def print_backtest_summary(broker: BacktestBroker, show_trades: bool = False) -> None:
    stats = calculate_backtest_stats(broker)
    if stats.initial_balance > 0:
        total_pnl_pct = stats.total_pnl / stats.initial_balance * 100.0
        total_gross_pct = stats.total_gross / stats.initial_balance * 100.0
    else:
        total_pnl_pct = 0.0
        total_gross_pct = 0.0
    print("\n" + "=" * 50)
    print("BACKTEST SUMMARY")
    print("=" * 50)
    print(f"  Initial balance : {stats.initial_balance:,.2f}")
    print(f"  Final balance   : {stats.final_balance:,.2f}")
    print(f"  Total PnL (net) : {stats.total_pnl:+,.2f} ({total_pnl_pct:+.2f}%)")
    print(f"  Total PnL (gross): {stats.total_gross:+,.2f} ({total_gross_pct:+.2f}%)")
    print(
        f"  Total costs     : {stats.total_costs:,.2f}"
        f"  (spread={stats.total_spread_cost:,.2f}"
        f"  slippage={stats.total_slippage_cost:,.2f}"
        f"  commission={stats.total_commission_cost:,.2f}"
        f"  swap={stats.total_swap_cost:,.2f})"
    )
    print(f"  Peak equity     : {stats.peak_equity:,.2f}")
    print(f"  Trough equity   : {stats.trough_equity:,.2f}")
    print(f"  Max drawdown    : {stats.max_drawdown_pct:.2f}% ({stats.max_drawdown:,.2f})")
    print(f"  Trades          : {stats.trade_count}  (W:{stats.win_count} / L:{stats.loss_count})")
    if stats.trade_count:
        print(f"  Win rate        : {stats.win_rate:.1f}%" if stats.win_rate is not None else "  Win rate        : —")
        print(
            f"  Avg win         : {stats.avg_win:+,.2f} ({stats.avg_win_pct:+.2f}%)"
            if stats.avg_win is not None and stats.avg_win_pct is not None else
            "  Avg win         : —"
        )
        print(
            f"  Avg loss        : {stats.avg_loss:+,.2f} ({stats.avg_loss_pct:+.2f}%)"
            if stats.avg_loss is not None and stats.avg_loss_pct is not None else
            "  Avg loss        : —"
        )
        print(f"  Profit factor   : {stats.profit_factor:.2f}" if stats.profit_factor is not None else "  Profit factor   : —")
        print(
            f"  Expectancy/trade: {stats.expectancy_per_trade:+,.2f} ({stats.expectancy_per_trade_pct:+.2f}%)"
            if stats.expectancy_per_trade is not None and stats.expectancy_per_trade_pct is not None else
            "  Expectancy/trade: —"
        )
        print(
            f"  Avg hold time   : {_format_duration(stats.avg_holding_time)}"
            if stats.avg_holding_time is not None else
            "  Avg hold time   : —"
        )
        print(f"  Turnover        : {stats.turnover:,.2f}")
        print(
            f"  Avg turnover    : {stats.avg_turnover_per_trade:,.2f} / trade"
            if stats.avg_turnover_per_trade is not None else
            "  Avg turnover    : —"
        )
        print(
            f"  Turnover / init : {stats.turnover_multiple:,.1f}x"
            if stats.turnover_multiple is not None else
            "  Turnover / init : —"
        )
        print(f"  Cost / gross    : {stats.cost_ratio_pct:.1f}%" if stats.cost_ratio_pct is not None else "  Cost / gross    : —")
        print(f"  Exposure        : {stats.exposure_pct:.1f}%" if stats.exposure_pct is not None else "  Exposure        : —")
        print(f"  CAGR            : {stats.cagr_pct:.1f}%" if stats.cagr_pct is not None else "  CAGR            : —")
        print(f"  Sharpe (daily)  : {stats.sharpe_ratio:.2f}" if stats.sharpe_ratio is not None else "  Sharpe (daily)  : —")
        print(f"  Sortino (daily) : {stats.sortino_ratio:.2f}" if stats.sortino_ratio is not None else "  Sortino (daily) : —")
        print(f"  Calmar          : {stats.calmar_ratio:.2f}" if stats.calmar_ratio is not None else "  Calmar          : —")
    if show_trades:
        print("-" * 50)
        print("  TRADES")
        print("-" * 50)
        for trade in stats.trades:
            direction = trade.position.direction.upper()
            print(
                f"  [{direction}] {trade.position.entry_time:%Y-%m-%d %H:%M} → {trade.exit_time:%Y-%m-%d %H:%M}"
                f"  entry={trade.position.entry_price:.5f}  exit={trade.exit_price:.5f}"
                f"  qty={trade.position.qty}  pnl={trade.pnl:+,.2f}"
                f"  gross={trade.gross_pnl:+,.2f}  costs={trade.costs:.2f}"
            )
    print("=" * 50 + "\n")