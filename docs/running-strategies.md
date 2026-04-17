# Running Strategies

This guide covers how to run your strategies in backtest mode and live mode.

## Overview

Two classes handle the full workflow:

| Class | Import | Responsibility |
|---|---|---|
| `MT5Session` | `algotrading.live.mt5` | Connects to MT5, fetches data, transitions to live |
| `BacktestSession` | `algotrading.backtest` | Holds the simulated broker; exposes results |

`MT5Session.backtest()` returns a `BacktestSession` — so everything related to results (trade log, equity curve, plots) stays in `algotrading.backtest`.

---

## Running a Backtest

```python
from datetime import datetime
import MetaTrader5 as mt5

from algotrading.live.mt5 import MT5Session
from algotrading.backtest import print_backtest_summary

session = MT5Session(
    strategies=[strategy],
    primary_tf=mt5.TIMEFRAME_M15,
)
bt = session.backtest(
    initial_balance=10_000,
    slippage_pct=0.0024,
    date_from=datetime(2024, 1, 1),
    secondary={mt5.TIMEFRAME_H4: 300},
)
print_backtest_summary(bt.broker)
```

### `MT5Session` constructor

| Parameter | Type | Default | Description |
|---|---|---|---|
| `strategies` | `list[Strategy]` | required | One or more strategy instances. Strategies for different symbols can be mixed. |
| `primary_tf` | `int \| dict[str, int]` | required | MT5 timeframe for primary bars. Single constant for all symbols, or `{"SYMBOL": tf}` per symbol. |
| `poll_interval` | `float` | `1.0` | Seconds between bar polls in live mode. |
| `symbol_specs_cache_path` | `str \| Path \| None` | `.algotrading/symbol_specs.json` | Where to persist fetched symbol specs between runs. |

### `MT5Session.backtest()` parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `initial_balance` | `float` | required | Starting account balance for the simulated broker. |
| `slippage_pct` | `float` | `0.0` | Extra execution slippage as a percentage of price (e.g. `0.0024` = 0.24%). |
| `commission_per_lot` | `float` | `0.0` | Per-side commission in account currency, applied to every fill. |
| `primary_count` | `int \| None` | `None` | Fetch the last N completed primary bars. Mutually exclusive with `date_from`. |
| `date_from` | `datetime \| None` | `None` | Start of the backtest range (UTC). Mutually exclusive with `primary_count`. |
| `date_to` | `datetime \| None` | `None` | End of the range (UTC). Defaults to now. Only used with `date_from`. |
| `secondary` | `dict[int, int] \| None` | `None` | Secondary timeframes for indicators: `{timeframe: warmup_bar_count}`. With `date_from`, warmup bars are fetched *before* `date_from`. |
| `symbol_specs` | `dict[str, SymbolSpec] \| None` | `None` | Explicit symbol specs that bypass MT5 lookup and the cache. |
| `refresh_symbol_specs` | `bool` | `False` | Force re-fetch of symbol specs from MT5, ignoring the cache. |

Provide **exactly one** of `primary_count` or `date_from`.

### Date range backtest

```python
from datetime import timezone

bt = session.backtest(
    initial_balance=10_000,
    slippage_pct=0.0024,
    date_from=datetime(2024, 1, 1, tzinfo=timezone.utc),
    date_to=datetime(2026, 1, 1, tzinfo=timezone.utc),
    secondary={mt5.TIMEFRAME_H4: 300},
)
```

### Last-N-bars backtest

```python
bt = session.backtest(
    initial_balance=10_000,
    primary_count=2000,
    secondary={mt5.TIMEFRAME_H4: 300},
)
```

### Accessing results

```python
from algotrading.backtest import print_backtest_summary

broker = bt.broker

print_backtest_summary(broker)   # full stats to stdout
print(broker.trade_log)          # list[ClosedTrade]
print(broker.equity_curve)       # list[EquityPoint]
```

### Saving plots

All plotting functions are importable from `algotrading.backtest`.

#### Static matplotlib charts

```python
from algotrading.backtest import (
    plot_equity_curve,
    plot_drawdown,
    plot_equity_and_drawdown,
    plot_equity_vs_benchmark,
    plot_monthly_returns_heatmap,
    plot_trade_pnl_distribution,
    plot_price_with_trades,
    plot_monte_carlo_paths,
    show,
)

broker = bt.broker

# Combined equity + drawdown (stacked subplots)
fig, (ax1, ax2) = plot_equity_and_drawdown(broker)
fig.savefig("plots/equity_drawdown.png", dpi=140, bbox_inches="tight")

# Standalone equity curve or drawdown
fig, ax = plot_equity_curve(broker)
fig, ax = plot_drawdown(broker)

fig, ax = plot_monthly_returns_heatmap(broker)
fig.savefig("plots/monthly_returns.png", dpi=140, bbox_inches="tight")

fig, ax = plot_trade_pnl_distribution(broker)
fig.savefig("plots/trade_pnl_distribution.png", dpi=140, bbox_inches="tight")

# Price line with entry/exit markers (matplotlib)
fig, ax = plot_price_with_trades(
    price_times=strategy.bars.time,
    prices=strategy.bars.close,
    trades=broker.trade_log,
    symbol=strategy.symbol,
)
fig.savefig("plots/price_trades.png", dpi=140, bbox_inches="tight")

# Equity vs benchmark — requires benchmark price series
fig, ax = plot_equity_vs_benchmark(
    broker,
    benchmark_times=bench_times,   # list[datetime] or ndarray
    benchmark_prices=bench_prices,  # list[float] or ndarray
    benchmark_name="Gold B&H",
)
fig.savefig("plots/equity_vs_benchmark.png", dpi=140, bbox_inches="tight")

# Monte Carlo paths (requires simulate_monte_carlo_from_broker result)
from algotrading.backtest import simulate_monte_carlo_from_broker, plot_monte_carlo_paths
mc = simulate_monte_carlo_from_broker(broker, n_simulations=1000)
fig, ax = plot_monte_carlo_paths(mc)

show()  # display all open matplotlib figures (calls plt.show())
```

#### Interactive HTML charts

```python
from algotrading.backtest import (
    plot_price_with_trades_interactive,
    plot_trade_history_table_interactive,
    save_interactive_figure_html,
    save_trade_history_interactive_html,
    save_trade_chunks_interactive_html,
)

# Full-range interactive price chart with trade markers
# Renders as candlesticks when OHLC is provided and bar count ≤ max_candles.
fig = plot_price_with_trades_interactive(
    price_times=strategy.bars.time,
    prices=strategy.bars.close,
    trades=broker.trade_log,
    symbol=strategy.symbol,
    price_open=strategy.bars.open,
    price_high=strategy.bars.high,
    price_low=strategy.bars.low,
    price_close=strategy.bars.close,
    render_mode="auto",         # "auto" | "candlestick" | "line"
    max_candles=1_500,          # fall back to line chart above this count
    marker_layout="staggered",  # "staggered" | "fill"
    # Optional indicator overlays on the price pane:
    # indicator_overlays=[{"label": "SMA50", "values": sma50_array}],
    # Optional sub-panels below the price pane:
    # indicator_panels=[{"label": "RSI", "values": rsi_array}],
)
save_interactive_figure_html(fig, "plots/price_trades.html", auto_open=True)

# Scrollable trade history table (Plotly)
fig = plot_trade_history_table_interactive(broker.trade_log)
save_interactive_figure_html(fig, "plots/trade_history.html")

# Or use the one-shot helper:
save_trade_history_interactive_html(
    trades=broker.trade_log,
    output_path="plots/trade_history.html",
)

# Chunked interactive HTML — one file per N trades, with surrounding price context
save_trade_chunks_interactive_html(
    price_times=strategy.bars.time,
    prices=strategy.bars.close,
    trades=broker.trade_log,
    output_dir="plots/trade_chunks",
    symbol=strategy.symbol,
    chunk_size=50,
    pad_bars=150,
    price_open=strategy.bars.open,
    price_high=strategy.bars.high,
    price_low=strategy.bars.low,
    price_close=strategy.bars.close,
    marker_layout="staggered",  # "staggered" | "fill"
    # indicator_overlays / indicator_panels sliced per chunk automatically
)
```

#### One-shot bundle via `save_backtest_plots`

`save_backtest_plots` is a convenience wrapper that saves the full standard set of plots (equity/drawdown, monthly returns, trade PnL, benchmark comparison, interactive trade chunks) to a directory in one call. It requires an active MT5 connection to fetch the benchmark series.

```python
from algotrading.backtest import save_backtest_plots
from pathlib import Path

save_backtest_plots(
    bt=bt,
    strategy=strategy,
    primary_tf=mt5.TIMEFRAME_M15,
    date_from=datetime(2024, 1, 1),
    date_to=datetime(2026, 4, 1),
    output_dir=Path("plots"),
)
```

---

## Running Live

Call `session.go_live()` after `session.backtest()`. The same strategy instances are re-wired to the live MT5 broker — their indicators and bar history are already warmed up. The call **blocks until stopped or an exception occurs**. MT5 is shut down automatically on exit.

```python
bt = session.backtest(...)
print_backtest_summary(bt.broker)  # review before going live

session.go_live()                  # blocks — Ctrl+C to stop
```

The live runner polls MT5 every `poll_interval` seconds. On each poll it:

1. Fetches completed primary bars since the last known bar time
2. Fetches completed secondary bars for each registered secondary timeframe
3. Updates indicators and calls `strategy.next()` for each new bar
4. Polls open positions to detect server-side SL/TP fills between bars

> **Note:** Always run `session.backtest()` before `session.go_live()`. The backtest warms up indicators so `next()` receives valid values from the very first live bar.

---

## Multi-Symbol Setup

Pass multiple strategy instances — one per symbol — to a single session:

```python
gold = SMACross(symbol="XAUUSD")
btc  = SMACross(symbol="BTCUSD", params=SMACrossParams(fast_period=5, slow_period=20))

session = MT5Session(
    strategies=[gold, btc],
    primary_tf=mt5.TIMEFRAME_M15,
)
bt = session.backtest(
    initial_balance=10_000,
    slippage_pct=0.0024,
    date_from=datetime(2024, 1, 1),
    secondary={mt5.TIMEFRAME_H4: 300},
)
session.go_live()
```

For per-symbol timeframes:

```python
session = MT5Session(
    strategies=[gold, btc],
    primary_tf={
        "XAUUSD": mt5.TIMEFRAME_M15,
        "BTCUSD": mt5.TIMEFRAME_M1,
    },
)
```

---

## Symbol Specs

Symbol specs (spread, point value, contract size) are fetched from MT5 on the first `backtest()` call for each symbol and cached to `.algotrading/symbol_specs.json`.

**Force a refresh** after a broker changes contract specs:

```python
bt = session.backtest(
    initial_balance=10_000,
    refresh_symbol_specs=True,
    date_from=...,
)
```

**Override specs manually** (useful for testing or unsupported instruments):

```python
from algotrading.backtest import SymbolSpec

bt = session.backtest(
    initial_balance=10_000,
    symbol_specs={
        "XAUUSD": SymbolSpec(
            value_per_point=1.0,
            spread_pct=0.02,
            slippage_pct=0.0024,
            contract_size=100.0,
            commission_per_lot=0.0,
        )
    },
    date_from=...,
)
```

---

## Backtest Without MT5 Data

If you have bars from another source (CSV, a custom API, etc.), use `BacktestSession` directly — no MT5 connection required:

```python
from algotrading.backtest import BacktestSession, SymbolSpec, print_backtest_summary

bt = BacktestSession(
    strategies=[strategy],
    initial_balance=10_000,
    symbol_specs={"XAUUSD": SymbolSpec(value_per_point=1.0, spread_pct=0.02)},
)
bt.run(
    primary_rates={"XAUUSD": my_bars},
    secondary_rates={"XAUUSD": {"H4": my_h4_bars}},
    primary_timeframes={"XAUUSD": "M15"},
)
print_backtest_summary(bt.broker)
```

Timeframe identifiers can be MT5 constants (`mt5.TIMEFRAME_H4`) or strings (`"M1"`, `"M15"`, `"H4"`, `"D1"`, …) — the engine resolves either to a duration. `primary_timeframes` is optional but recommended, since it lets the engine align secondary bars to the primary cadence without inferring the duration from timestamps.

Each bar in `primary_rates` / `secondary_rates` must be a dict with at least `time`, `open`, `high`, `low`, `close` keys (where `time` is a Unix timestamp in seconds). An optional `spread_pct` key is used for per-bar spread simulation.

---

## Putting It All Together — Minimal Example

```python
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import MetaTrader5 as mt5

from algotrading import Strategy, StrategyParams
from algotrading.backtest import print_backtest_summary, save_backtest_plots
from algotrading.live.mt5 import MT5Session
from algotrading.indicators import SMA
from algotrading.utils import crossover, crossunder, fraction_to_qty


@dataclass
class MyParams(StrategyParams):
    fast: int = 10
    slow: int = 30


class MyStrategy(Strategy[MyParams]):
    def __init__(self, symbol: str, params: MyParams = MyParams()):
        super().__init__(symbol=symbol, params=params)
        self.fast_sma = self.I(SMA(params.fast), source="close")
        self.slow_sma = self.I(SMA(params.slow), source="close")

    def next(self) -> None:
        vpp = self.broker.value_per_point(self.symbol)
        qty = fraction_to_qty(self.equity, 99.9, self.price, vpp)
        if crossover(self.fast_sma, self.slow_sma):
            self.buy(qty=qty)
        elif crossunder(self.fast_sma, self.slow_sma):
            self.sell(qty=qty)


date_from = datetime(2024, 1, 1, tzinfo=timezone.utc)
date_to = datetime(2026, 4, 1, tzinfo=timezone.utc)
primary_tf = mt5.TIMEFRAME_M15
secondary_tf = mt5.TIMEFRAME_H4

strategy = MyStrategy(symbol="XAUUSD")

session = MT5Session(
    strategies=[strategy],
    primary_tf=primary_tf,
)

bt = session.backtest(
    initial_balance=10_000,
    slippage_pct=0.0024,
    date_from=date_from,
    date_to=date_to,
    secondary={secondary_tf: 300},
)

print_backtest_summary(bt.broker)
save_backtest_plots(
    bt=bt,
    strategy=strategy,
    primary_tf=primary_tf,
    date_from=date_from,
    date_to=date_to,
    output_dir=Path("plots/my_strategy"),
)

# Uncomment to go live after reviewing backtest results:
# session.go_live()
```
