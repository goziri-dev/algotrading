# algotrading

A Python framework for designing, backtesting, optimising, and live-trading systematic strategies through MetaTrader 5. Built from scratch to be a single, cohesive environment where the same `Strategy` class runs identically in a backtest and against a live broker.

---

## Highlights

- **Unified backtest and live execution.** The same strategy instance can be backtested and handed over to the live runner. Indicators, bar history, and open positions all survive the transition.
- **Multi-symbol, multi-timeframe, multi-strategy.** Run a portfolio of strategies — each on its own symbol and timeframe mix — against one simulated or live broker.
- **Realistic execution model.** Spread, slippage, per-lot commission, margin, SL/TP fills, limit/stop queues, requote handling, and symbol-specific contract sizing.
- **Equity-scoped risk sizing.** Size positions by portfolio equity *or* per-strategy equity so one strategy's drawdown cannot starve another's sizing.
- **Parameter optimisation with walk-forward validation.** Grid search, random search, time-series cross-validation, and walk-forward analysis with rolling or anchored training windows.
- **Monte Carlo robustness testing** of the equity curve to separate skill from path luck.
- **Rich reporting.** 15+ summary statistics, static matplotlib charts, interactive Plotly HTML, and chunked trade-by-trade interactive views for audit.
- **COT data feed.** Built-in ingestion of CFTC Commitment of Traders reports for regime/sentiment filters.
- **Fully typed, fully tested.** Python 3.13, generics, and a pytest suite covering broker, execution, indicators, optimisation, plotting, and live session plumbing.

---

## Example: SMACross on XAUUSD (Jan 2024 → Apr 2026)

```
==================================================
BACKTEST SUMMARY
==================================================
  Initial balance : 2,730.00
  Final balance   : 6,832.65
  Total PnL (net) : +4,102.65 (+150.28%)
  Total PnL (gross): +4,117.77 (+150.83%)
  Total costs     : 15.13  (spread=6.57  slippage=8.56  commission=0.00)
  Peak equity     : 7,476.86
  Trough equity   : 2,648.39
  Max drawdown    : -12.95% (-968.30)
  Trades          : 36  (W:33 / L:3)
  Win rate        : 91.7%
  Avg win         : +176.25 (+3.65%)
  Avg loss        : -571.15 (-8.39%)
  Profit factor   : 3.39
  Expectancy/trade: +113.96 (+2.64%)
  Avg hold time   : 21d 8h 41m
  Turnover        : 356,626.72
  Avg turnover    : 9,906.30 / trade
  Turnover / init : 130.6x
  Cost / gross    : 0.4%
  Exposure        : 92.3%
  CAGR            : 49.5%
  Sharpe (daily)  : 2.25
  Sortino (daily) : 3.33
  Calmar          : 3.82
==================================================
```

Summary produced by `algotrading.backtest.print_backtest_summary`. Figures reflect a simulated broker with realistic spread, slippage, and margin accounting — not a frictionless replay. The strategy source lives in [main.py](main.py).

---

## Quick Start

### Install

```bash
# uv is the project's pinned package manager — see uv.lock
uv sync
```

Requires Python 3.13+ and a MetaTrader 5 terminal installed locally (Windows).

### A minimal strategy

```python
from dataclasses import dataclass
from datetime import datetime, timezone
import MetaTrader5 as mt5

from algotrading import Strategy, StrategyParams
from algotrading.backtest import print_backtest_summary
from algotrading.live.mt5 import MT5Session
from algotrading.indicators import SMA
from algotrading.utils import crossover, crossunder, fraction_to_qty


@dataclass
class MAParams(StrategyParams):
    fast: int = 10
    slow: int = 30


class MovingAverageCross(Strategy[MAParams]):
    def __init__(self, symbol: str = "XAUUSD", params: MAParams = MAParams()):
        super().__init__(symbol=symbol, params=params)
        self.fast = self.I(SMA(params.fast), source="close")
        self.slow = self.I(SMA(params.slow), source="close")

    def next(self) -> None:
        vpp = self.broker.value_per_point(self.symbol)
        qty = fraction_to_qty(self.equity, 99.9, self.price, vpp)
        if crossover(self.fast, self.slow):
            self.buy(qty=qty)
        elif crossunder(self.fast, self.slow):
            self.sell(qty=qty)


session = MT5Session(
    strategies=[MovingAverageCross()],
    primary_tf=mt5.TIMEFRAME_M15,
)

bt = session.backtest(
    initial_balance=10_000,
    slippage_pct=0.0024,
    date_from=datetime(2024, 1, 1, tzinfo=timezone.utc),
)
print_backtest_summary(bt.broker)

# Promote the warm session to live trading — same instances, same state:
# session.go_live()
```

---

## Architecture

```
algotrading/
├── core/          Strategy base class, Bars feed, Broker interface, Position,
│                  EntrySignal / PendingSignal, execution listeners, runner.
├── backtest/      Simulated broker, execution engine, optimisation (grid,
│                  random, walk-forward, time-series CV), Monte Carlo,
│                  summary statistics, matplotlib + Plotly reporting.
├── live/mt5/      MT5 broker adapter, live session loop, symbol-spec cache.
├── indicators/    SMA, EMA, RSI, ATR, BBANDS, ADX, Supertrend, HHLL, COTIndex.
│                  Supports chained indicators and multi-output (BBANDS, ADX).
├── data/          COT (Commitment of Traders) client + report model.
└── utils.py       crossover / crossunder / bars_since, session filter,
                   fraction_to_qty, risk_to_qty, MonthlyDrawdownTracker.

my_strategies/     Reference strategies — bbands_reverse, rsi_reverse,
                   btc_scalper, hhll_debug, multi_strategy.

tests/             pytest suite — backtest/, core/, data/, indicators/, live/.

docs/              Long-form guides (see below).
```

### Design notes

- **One Strategy API for two execution modes.** `Strategy.next()` is invoked identically by the backtester and the live runner. The only thing that changes between modes is which `Broker` implementation the strategy talks to.
- **Bars grow forward only.** Indicators and strategies use negative indexing (`self.fast[-1]`) which maps unambiguously onto completed history. A bar is never revised once `next()` has seen it.
- **Signals are values, not side effects.** `self.buy(...)` and `self.sell(...)` emit `EntrySignal` objects that the broker executes. Limit and stop orders live in a pending queue that can be inspected and cancelled — the broker resolves triggers against bar highs/lows in-simulation and against real fills in live.
- **Sizing is decoupled from signals.** `fraction_to_qty` and `risk_to_qty` are pure functions of equity, price, and `value_per_point`, so the same sizing logic works for every symbol without special-casing FX vs. metals vs. crypto.
- **Strategy IDs are deterministic.** `self.id` is hashed from `(class, symbol, params)`, which is how the live broker routes orders for re-runs of the same strategy back to the correct position set.

---

## Documentation

Full guides live under [docs/](docs/):

- [Writing strategies](docs/writing-strategies.md) — the complete `Strategy` API: indicators, multi-timeframe, sizing, SL/TP, pending orders, callbacks, shared state.
- [Running strategies](docs/running-strategies.md) — `MT5Session` and `BacktestSession`, multi-symbol setups, symbol specs, plotting, and how to backtest without an MT5 connection.
- [Optimisation](docs/optimisation.md) — grid and random search, walk-forward analysis, time-series cross-validation, Monte Carlo.

---

## Testing

```bash
uv run pytest
```

The suite exercises:

- Simulated broker execution, including SL/TP fills, margin checks, and limit/stop queue resolution.
- Every indicator against reference values.
- The full optimisation and walk-forward pipelines.
- Monte Carlo path generation.
- Summary statistics and both matplotlib and Plotly plotting paths.
- MT5 live session plumbing against a mocked broker.

---

## Tech stack

Python 3.13 · MetaTrader5 · numpy · pandas · matplotlib · plotly · lightweight-charts · tqdm · sodapy (CFTC Socrata API) · uv · pytest.
