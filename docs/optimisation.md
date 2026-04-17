# Optimisation & Walk-Forward

This guide covers parameter optimisation, walk-forward validation, and time-series cross-validation.

---

## The Core Pattern

All optimisation functions are data-agnostic. They don't know about strategies, bars, or brokers — they only know about parameter dicts and scores. You wire in the backtest through a callback:

```
param_space → parameter combinations → evaluate(params) → score → ranked results
```

The `evaluate` callback is yours to write. It receives `params: dict[str, object]`, runs a backtest, and returns any value that can be scored. The most natural return type is `BacktestStats`.

```python
from algotrading.backtest import BacktestSession, SymbolSpec, calculate_backtest_stats
from algotrading.backtest import optimize_parameters, parameter_grid

# Pre-fetch bars once — reused across every evaluate() call
all_bars = [...]   # list of bar dicts from MT5 or CSV
spec = SymbolSpec(value_per_point=1.0, spread_pct=0.02, slippage_pct=0.0024)

def evaluate(params: dict) -> BacktestStats:
    strategy = MyStrategy(
        symbol="XAUUSD",
        params=MyParams(
            fast_period=params["fast_period"],
            slow_period=params["slow_period"],
        ),
    )
    bt = BacktestSession(
        strategies=[strategy],
        initial_balance=10_000,
        symbol_specs={"XAUUSD": spec},
    )
    bt.run(primary_rates={"XAUUSD": all_bars})
    return calculate_backtest_stats(bt.broker)
```

---

## Grid Search

### Building a Parameter Space

`parameter_grid` builds the cartesian product of all candidate values:

```python
from algotrading.backtest import parameter_grid

grid = parameter_grid({
    "fast_period": range(5, 20, 5),    # 5, 10, 15
    "slow_period": range(20, 60, 10),  # 20, 30, 40, 50
})
# → 12 combinations
```

### Running a Grid Search

`optimize_parameters` evaluates every combination and returns results ranked by objective:

```python
from algotrading.backtest import optimize_parameters

report = optimize_parameters(
    param_space={
        "fast_period": range(5, 20, 5),
        "slow_period": range(20, 60, 10),
    },
    evaluate=evaluate,
    objective="sharpe_ratio",   # any BacktestStats field name
    maximize=True,
)
```

### Reading Results

```python
report.best                          # top-ranked OptimizationResult
report.best.params                   # {"fast_period": 10, "slow_period": 30}
report.best.score                    # 1.43 (the objective value)
report.best.evaluation               # full BacktestStats

# All results in ranked order
for result in report.results:
    print(result.params, result.score)
```

---

## Constraints

Use `constraint` to filter out nonsensical combinations before evaluation. It receives the `params` dict and returns `True` to keep the combination:

```python
report = optimize_parameters(
    param_space={
        "fast_period": range(5, 30, 5),
        "slow_period": range(10, 60, 10),
    },
    evaluate=evaluate,
    objective="sharpe_ratio",
    constraint=lambda p: p["fast_period"] < p["slow_period"],  # must be true
)
```

Constraints are applied before evaluation — filtered combinations are never passed to `evaluate`.

---

## Random Search

When the search space is large, exhaustive grid search is slow. `optimize_parameters_random` samples a random subset:

```python
from algotrading.backtest import optimize_parameters_random

report = optimize_parameters_random(
    param_space={
        "fast_period":  range(3, 50),
        "slow_period":  range(10, 200),
        "atr_period":   range(7, 28),
        "sl_atr_mult":  [1.0, 1.5, 2.0, 2.5, 3.0],
    },
    evaluate=evaluate,
    objective="sharpe_ratio",
    n_iter=200,           # evaluate 200 random combinations
    random_state=42,      # reproducible
    constraint=lambda p: p["fast_period"] < p["slow_period"],
)
```

`replace=False` (the default) ensures no combination is evaluated twice.

---

## Choosing an Objective

Any `BacktestStats` field name can be passed as a string objective. A callable is also accepted for custom scoring.

| Objective | Field name | Good for |
|---|---|---|
| Sharpe ratio | `"sharpe_ratio"` | Risk-adjusted return vs volatility |
| Sortino ratio | `"sortino_ratio"` | Penalises downside volatility only |
| Calmar ratio | `"calmar_ratio"` | Return relative to max drawdown |
| Profit factor | `"profit_factor"` | Gross profit / gross loss |
| CAGR | `"cagr_pct"` | Compounded annual growth rate |
| Expectancy | `"expectancy_per_trade"` | Average PnL per trade |

Custom objective via callable:

```python
def my_objective(stats: BacktestStats) -> float:
    if stats.sharpe_ratio is None or stats.max_drawdown_pct < -20:
        return float("-inf")   # disqualify
    return stats.sharpe_ratio

report = optimize_parameters(
    ...,
    objective=my_objective,
    maximize=True,
)
```

**Warning:** optimising directly on in-sample data produces overfit parameters. Always validate on out-of-sample data — use walk-forward or time-series CV for this.

---

## Walk-Forward Analysis

Walk-forward analysis is the standard way to validate that optimised parameters generalise. The dataset is split into a series of windows. In each window, parameters are optimised on the **training** period and then evaluated on the **test** period that immediately follows. The test period is never seen during optimisation.

```
Total data: [===================================]
Window 1:   [TRAIN──────────][TEST──]
Window 2:         [TRAIN──────────][TEST──]
Window 3:               [TRAIN──────────][TEST──]
```

### Window Parameters

`walk_forward_windows` generates the index ranges:

```python
from algotrading.backtest import walk_forward_windows

windows = walk_forward_windows(
    total_size=1000,   # total number of bars
    train_size=600,    # bars in each training window
    test_size=100,     # bars in each test window
    step_size=100,     # how far to advance each step (defaults to test_size)
    anchored=False,    # True = expanding train window, False = rolling
)
# → list[WalkForwardWindow]
```

**Rolling** (`anchored=False`): train window slides forward — same size every step, older data dropped.

**Anchored** (`anchored=True`): train window always starts at bar 0 — grows with each step, no data discarded.

```python
window = windows[0]
window.train_start   # 0
window.train_end     # 600    (exclusive)
window.test_start    # 600
window.test_end      # 700    (exclusive)
```

### Running Walk-Forward

The `evaluate_window` callback receives `(params, start, end)` — index bounds into your dataset:

```python
from algotrading.backtest import run_walk_forward

# Pre-fetch all bars as an indexed list
all_bars = [...]

def evaluate_window(params: dict, start: int, end: int) -> BacktestStats:
    strategy = MyStrategy(
        symbol="XAUUSD",
        params=MyParams(
            fast_period=params["fast_period"],
            slow_period=params["slow_period"],
        ),
    )
    bt = BacktestSession(
        strategies=[strategy],
        initial_balance=10_000,
        symbol_specs={"XAUUSD": spec},
    )
    bt.run(primary_rates={"XAUUSD": all_bars[start:end]})
    return calculate_backtest_stats(bt.broker)


wf_report = run_walk_forward(
    param_space={
        "fast_period": range(5, 25, 5),
        "slow_period": range(20, 60, 10),
    },
    total_size=len(all_bars),
    train_size=600,
    test_size=100,
    evaluate_window=evaluate_window,
    objective="sharpe_ratio",
    maximize=True,
    constraint=lambda p: p["fast_period"] < p["slow_period"],
)
```

To use random search instead of exhaustive grid in each window, pass `optimizer`:

```python
from algotrading.backtest import optimize_parameters_random
import functools

wf_report = run_walk_forward(
    ...,
    optimizer=functools.partial(optimize_parameters_random, n_iter=50, random_state=42),
)
```

### Reading Walk-Forward Results

```python
# Aggregate out-of-sample performance — float | None (None if all steps are non-finite)
print(wf_report.mean_out_of_sample_score)

# Per-window breakdown
for step in wf_report.steps:
    w = step.window
    print(
        f"Train [{w.train_start}:{w.train_end}]  "
        f"Test [{w.test_start}:{w.test_end}]  "
        f"Best params: {step.in_sample_best.params}  "
        f"OOS score: {step.out_of_sample_score:.3f}"
    )

# Full in-sample OptimizationReport for a step (all ranked combinations)
step = wf_report.steps[0]
all_in_sample = step.optimization.results   # list[OptimizationResult]

# Full BacktestStats for the out-of-sample period of each step
oos_stats = step.out_of_sample_evaluation   # BacktestStats
print(oos_stats.sharpe_ratio)
print(oos_stats.max_drawdown_pct)
```

**Interpreting results:** if in-sample scores are strong but `mean_out_of_sample_score` is weak or negative, the parameters are overfit. A healthy strategy shows consistent (if lower) performance in both periods.

---

## Time-Series Cross-Validation

`optimize_parameters_random_cv` combines random search with time-series CV. Instead of optimising on a single training window, each parameter set is scored across multiple chronological folds and the results are aggregated. This gives a more robust estimate of generalisation.

```python
from algotrading.backtest import optimize_parameters_random_cv

cv_report = optimize_parameters_random_cv(
    param_space={
        "fast_period": range(3, 30),
        "slow_period": range(10, 100),
    },
    evaluate_window=evaluate_window,
    objective="sharpe_ratio",
    total_size=len(all_bars),
    min_train_size=400,
    validation_size=100,
    n_iter=100,
    maximize=True,
    random_state=42,
    anchored=True,       # expanding train window (default)
    constraint=lambda p: p["fast_period"] < p["slow_period"],
)

cv_report.best.params        # best params by aggregated CV score
cv_report.best.score         # mean score across folds
cv_report.best.evaluation    # TimeSeriesCVEvaluation
```

The `evaluation` field is a `TimeSeriesCVEvaluation`:

```python
cv_eval = cv_report.best.evaluation

cv_eval.fold_scores          # list[float] — score per fold
cv_eval.aggregated_score     # mean of finite fold scores
cv_eval.fold_evaluations     # list[BacktestStats] — one per fold
cv_eval.fold_windows         # list[WalkForwardWindow]
```

A custom `fold_aggregator` can replace the default mean — useful if you want the minimum fold score (worst-case robustness) rather than the average:

```python
cv_report = optimize_parameters_random_cv(
    ...,
    fold_aggregator=min,   # prefer params that hold up in the worst fold
)
```

---

## Full End-to-End Example

```python
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import MetaTrader5 as mt5

from algotrading import Strategy, StrategyParams
from algotrading.backtest import (
    BacktestSession,
    calculate_backtest_stats,
    run_walk_forward,
)
from algotrading.indicators import SMA, ATR
from algotrading.utils import crossover, crossunder, risk_to_qty
from algotrading.live.mt5 import MT5Session


@dataclass
class SMACrossParams(StrategyParams):
    fast_period: int   = 10
    slow_period: int   = 30
    sl_atr_mult: float = 2.0


class SMACross(Strategy[SMACrossParams]):
    def __init__(self, symbol, params=SMACrossParams()):
        super().__init__(symbol=symbol, params=params)
        p = params
        self.fast = self.I(SMA(p.fast_period), source="close")
        self.slow = self.I(SMA(p.slow_period), source="close")
        self.atr  = self.I(ATR(14),            source=("high", "low", "close"))

    def next(self) -> None:
        vpp = self.broker.value_per_point(self.symbol)
        atr = self.atr[-1]
        if crossover(self.fast, self.slow):
            sl  = self.price - self.params.sl_atr_mult * atr
            qty = risk_to_qty(self.equity, 1.0, self.price, sl, vpp)
            self.buy(qty=qty, sl_price=sl)
        elif crossunder(self.fast, self.slow):
            sl  = self.price + self.params.sl_atr_mult * atr
            qty = risk_to_qty(self.equity, 1.0, self.price, sl, vpp)
            self.sell(qty=qty, sl_price=sl)


# ── 1. Fetch data once ───────────────────────────────────────────────────
session = MT5Session(strategies=[SMACross("XAUUSD")], primary_tf=mt5.TIMEFRAME_M15)
bt_init = session.backtest(
    initial_balance=10_000,
    slippage_pct=0.0024,
    date_from=datetime(2022, 1, 1),
    date_to=datetime(2026, 1, 1),
)
spec     = bt_init.broker._symbol_specs["XAUUSD"]
all_bars = ...   # store the raw rates used for the initial backtest


# ── 2. Define evaluate_window ────────────────────────────────────────────
def evaluate_window(params: dict, start: int, end: int):
    strategy = SMACross(
        "XAUUSD",
        SMACrossParams(
            fast_period=int(params["fast_period"]),
            slow_period=int(params["slow_period"]),
            sl_atr_mult=float(params["sl_atr_mult"]),
        ),
    )
    bt = BacktestSession(
        strategies=[strategy],
        initial_balance=10_000,
        symbol_specs={"XAUUSD": spec},
    )
    bt.run(primary_rates={"XAUUSD": all_bars[start:end]})
    return calculate_backtest_stats(bt.broker)


param_space = {
    "fast_period": range(5, 25, 5),
    "slow_period": range(15, 55, 10),
    "sl_atr_mult": [1.5, 2.0, 2.5, 3.0],
}
constraint = lambda p: p["fast_period"] < p["slow_period"]


# ── 3. Walk-forward ──────────────────────────────────────────────────────
wf_report = run_walk_forward(
    param_space=param_space,
    total_size=len(all_bars),
    train_size=3_000,
    test_size=500,
    evaluate_window=evaluate_window,
    objective="sharpe_ratio",
    maximize=True,
    constraint=constraint,
)

print(f"Mean OOS Sharpe: {wf_report.mean_out_of_sample_score:.3f}")
for step in wf_report.steps:
    print(
        f"  [{step.window.test_start}:{step.window.test_end}] "
        f"params={step.in_sample_best.params}  OOS={step.out_of_sample_score:.3f}"
    )
```

---

## Quick Reference

### Functions

| Function | Description |
|---|---|
| `parameter_grid(param_space, constraint)` | Cartesian product of all candidate values |
| `random_parameter_samples(param_space, n_iter, ...)` | Sample N random combinations without evaluating them |
| `optimize_parameters(param_space, evaluate, objective, maximize, constraint)` | Exhaustive grid search |
| `optimize_parameters_random(param_space, evaluate, objective, n_iter, ...)` | Random subset search |
| `walk_forward_windows(total_size, train_size, test_size, step_size, anchored)` | Generate window index ranges |
| `time_series_cv_windows(total_size, min_train_size, validation_size, ...)` | Generate CV fold windows (alias for `walk_forward_windows`) |
| `run_walk_forward(param_space, total_size, train_size, test_size, evaluate_window, ...)` | Full walk-forward run |
| `optimize_parameters_random_cv(param_space, evaluate_window, objective, ...)` | Random search with time-series CV |

### Result Types

| Type | Key fields |
|---|---|
| `OptimizationReport` | `.best`, `.results` |
| `OptimizationResult` | `.params`, `.score`, `.evaluation` |
| `WalkForwardReport` | `.steps`, `.mean_out_of_sample_score` (`float \| None`) |
| `WalkForwardStepResult` | `.window`, `.optimization`, `.in_sample_best`, `.out_of_sample_score`, `.out_of_sample_evaluation` |
| `WalkForwardWindow` | `.train_start`, `.train_end`, `.test_start`, `.test_end` |
| `TimeSeriesCVEvaluation` | `.fold_scores`, `.aggregated_score`, `.fold_evaluations`, `.fold_windows` |
